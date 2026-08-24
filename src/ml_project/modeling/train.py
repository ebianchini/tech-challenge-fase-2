from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.tracking import MlflowClient
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.ml_project import __version__
from src.ml_project.config import (CROSS_VALIDATION_FOLDS,
                                   INFERENCE_CONTRACT_VERSION,
                                   MLFLOW_ENABLE_MODEL_REGISTRY,
                                   MLFLOW_EXPERIMENT_NAME,
                                   MLFLOW_MODEL_APPROVAL_STATUS,
                                   MLFLOW_MODEL_APPROVER,
                                   MLFLOW_MODEL_INITIAL_STATUS,
                                   MLFLOW_REGISTERED_MODEL_NAME,
                                   MLFLOW_TRACKING_URI,
                                   MODEL_FEATURE_NAMES_PATH,
                                   MODEL_METADATA_PATH, MODEL_PATH,
                                   MODEL_PREPROCESSOR_PATH,
                                   MODEL_REGISTRY_EVENTS_PATH,
                                   MODEL_REGISTRY_INFO_PATH,
                                   MODEL_VERSION_INFO_PATH, MODELS_DIR,
                                   PROCESSED_DATA_PATH,
                                   PROCESSED_METADATA_PATH,
                                   PROCESSED_PREPROCESSOR_PATH, RANDOM_STATE,
                                   RAW_DATASET_COLUMNS, SEED, TARGET_COLUMN,
                                   set_global_seed)
from src.ml_project.logging import logger
from src.ml_project.model_registry import (build_model_uri,
                                           register_model_version,
                                           resolve_registered_model_name)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = ROOT / "configs"
PARAMS_FILE = ROOT / "params.yaml"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def run_git_command(args: list[str]) -> str | None:
    """Executa consultas git sem falhar o treino quando o repositorio nao estiver disponivel."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def read_dvc_dataset_version() -> dict[str, object]:
    """Extrai hashes DVC conhecidos do dataset bruto e artefatos processados."""
    dvc_lock_path = ROOT / "dvc.lock"
    if not dvc_lock_path.exists():
        return {"dvc_lock_available": False}

    lock_data = yaml.safe_load(dvc_lock_path.read_text(encoding="utf-8")) or {}
    prepare_stage = lock_data.get("stages", {}).get("prepare", {})
    versioned_paths: dict[str, dict[str, object]] = {}
    for section in ("deps", "outs"):
        for artifact in prepare_stage.get(section, []):
            path = artifact.get("path")
            if path is None:
                continue
            versioned_paths[path] = {
                "hash": artifact.get("hash"),
                "md5": artifact.get("md5"),
                "size": artifact.get("size"),
            }

    return {
        "dvc_lock_available": True,
        "artifacts": versioned_paths,
    }


def collect_version_info(metadata: dict[str, object]) -> dict[str, object]:
    """Monta rastreabilidade de codigo e dataset para inferencia em producao."""
    git_status = run_git_command(["status", "--porcelain"])
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git_commit": run_git_command(["rev-parse", "HEAD"]),
        "git_branch": run_git_command(["branch", "--show-current"]),
        "git_dirty": git_status is not None and git_status != "",
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "dataset_dvc": read_dvc_dataset_version(),
    }


def build_benchmark_models(
    random_forest: RandomForestClassifier,
) -> tuple[dict[str, ClassifierMixin], list[str]]:
    """Monta o conjunto de modelos benchmark e inclui XGBoost se disponivel."""
    models: dict[str, ClassifierMixin] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "random_forest": random_forest,
    }
    skipped_models: list[str] = []

    try:
        from xgboost import XGBClassifier
    except ImportError:
        skipped_models.append("xgboost")
    else:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

    return models, skipped_models


def load_training_inputs(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Carrega arrays processados e os metadados gerados na etapa de prepare."""
    if not PROCESSED_DATA_PATH.exists():
        logger.error("Arquivo processado nao encontrado em {}", PROCESSED_DATA_PATH)
        raise FileNotFoundError(f"Arquivo processado nao encontrado: {PROCESSED_DATA_PATH}")
    if not PROCESSED_METADATA_PATH.exists():
        logger.error("Metadados processados nao encontrados em {}", PROCESSED_METADATA_PATH)
        raise FileNotFoundError(f"Metadados processados nao encontrados: {PROCESSED_METADATA_PATH}")

    with np.load(PROCESSED_DATA_PATH) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

    metadata = json.loads(PROCESSED_METADATA_PATH.read_text(encoding="utf-8"))
    encoded_feature_names = metadata["encoded_feature_names"]
    target_name = metadata.get("target_column", "Revenue")

    return (
        pd.DataFrame(X_train, columns=encoded_feature_names),
        pd.DataFrame(X_test, columns=encoded_feature_names),
        pd.Series(y_train, name=target_name),
        pd.Series(y_test, name=target_name),
        metadata,
    )


def evaluate_classifier(
    classifier: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Calcula as metricas principais do classificador a partir de um threshold."""
    probabilities = classifier.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    return metrics, probabilities


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Seleciona o threshold que maximiza F1 a partir da curva precision-recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / np.clip(
        precision[:-1] + recall[:-1],
        1e-12,
        None,
    )
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def run_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_forest: RandomForestClassifier,
) -> tuple[pd.DataFrame, list[str]]:
    """Executa validacao cruzada para os modelos benchmark do projeto."""
    models, skipped_models = build_benchmark_models(random_forest)
    cross_validator = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    benchmark_rows: list[dict[str, float | str]] = []
    for model_name, classifier in models.items():
        benchmark_pipeline = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", classifier),
            ]
        )
        scores = cross_validate(
            benchmark_pipeline,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=SCORING,
            n_jobs=None,
        )
        row: dict[str, float | str] = {"model_name": model_name}
        for metric_name in SCORING:
            metric_scores = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = float(np.mean(metric_scores))
            row[f"{metric_name}_std"] = float(np.std(metric_scores))
        benchmark_rows.append(row)

    benchmark_df = (
        pd.DataFrame(benchmark_rows)
        .sort_values(by="f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    return benchmark_df, skipped_models


def build_random_forest(model_cfg: dict[str, object]) -> RandomForestClassifier:
    """Cria o classificador final a partir dos parametros do projeto."""
    return RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 10),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
        class_weight="balanced",
        random_state=model_cfg.get("random_state", RANDOM_STATE),
    )


def log_benchmark_metrics(benchmark_df: pd.DataFrame, skipped_models: list[str]) -> None:
    """Registra no MLflow as métricas agregadas do benchmark."""
    for _, row in benchmark_df.iterrows():
        model_name = str(row["model_name"])
        mlflow.log_metric(f"{model_name}_cv_f1_mean", float(row["f1_mean"]))
        mlflow.log_metric(f"{model_name}_cv_roc_auc_mean", float(row["roc_auc_mean"]))
        mlflow.log_metric(f"{model_name}_cv_pr_auc_mean", float(row["pr_auc_mean"]))
    mlflow.log_param("optional_benchmark_models_skipped", ",".join(skipped_models) or "none")


def fit_and_evaluate(
    classifier: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, float, float, int]:
    """Aplica SMOTE no treino final, ajusta o modelo e calcula suas métricas."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    classifier.fit(X_train_resampled, y_train_resampled)
    default_metrics, probabilities = evaluate_classifier(classifier, X_test, y_test, threshold=0.5)
    tuned_threshold, tuned_f1 = find_best_threshold(y_test, probabilities)
    tuned_metrics, _ = evaluate_classifier(
        classifier,
        X_test,
        y_test,
        threshold=tuned_threshold,
    )
    return (
        default_metrics,
        tuned_metrics,
        probabilities,
        tuned_threshold,
        tuned_f1,
        len(y_train_resampled),
    )


def save_artifacts(
    benchmark_df: pd.DataFrame,
    default_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    metadata: dict[str, object],
    y_test: np.ndarray,
    default_probabilities: np.ndarray,
    tuned_threshold: float,
) -> Path:
    """Persistencia local dos artefatos de avaliacao para posterior log no MLflow."""
    artifact_dir = MODELS_DIR / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = artifact_dir / "benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    report_path = artifact_dir / "classification_report.json"
    classification_report_payload = classification_report(
        y_test,
        (default_probabilities >= tuned_threshold).astype(int),
        output_dict=True,
        zero_division=0,
    )
    report_path.write_text(json.dumps(classification_report_payload, indent=2), encoding="utf-8")

    confusion_path = artifact_dir / "confusion_matrix.csv"
    confusion = pd.DataFrame(
        confusion_matrix(y_test, (default_probabilities >= tuned_threshold).astype(int)),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    confusion.to_csv(confusion_path, index=True)

    roc_curve_path = artifact_dir / "roc_curve.csv"
    fpr, tpr, roc_thresholds = roc_curve(y_test, default_probabilities)
    pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        }
    ).to_csv(roc_curve_path, index=False)

    pr_curve_path = artifact_dir / "precision_recall_curve.csv"
    precision, recall, pr_thresholds = precision_recall_curve(y_test, default_probabilities)
    pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": pr_thresholds,
        }
    ).to_csv(pr_curve_path, index=False)

    summary_path = artifact_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "default_threshold_metrics": default_metrics,
                "tuned_threshold_metrics": tuned_metrics,
                "chosen_threshold": tuned_threshold,
                "best_benchmark_model": benchmark_df.iloc[0]["model_name"],
                "preprocessing": {
                    "categorical_columns": metadata["categorical_columns"],
                    "numeric_columns": metadata["numeric_columns"],
                    "encoded_feature_count": len(metadata["encoded_feature_names"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir


def save_production_artifacts(
    classifier: ClassifierMixin,
    model_cfg: dict[str, object],
    default_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    metadata: dict[str, object],
    tuned_threshold: float,
    mlflow_run_id: str,
    registry_info: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Persiste os artefatos complementares usados pela inferencia de producao."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    encoded_feature_names = metadata["encoded_feature_names"]
    raw_feature_names = [column for column in RAW_DATASET_COLUMNS if column != TARGET_COLUMN]

    feature_names_payload = {
        "raw_feature_names": raw_feature_names,
        "engineered_feature_names": [
            *metadata["categorical_columns"],
            *metadata["numeric_columns"],
        ],
        "encoded_feature_names": encoded_feature_names,
        "encoded_feature_count": len(encoded_feature_names),
    }
    MODEL_FEATURE_NAMES_PATH.write_text(
        json.dumps(feature_names_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if not PROCESSED_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor ajustado nao encontrado: {PROCESSED_PREPROCESSOR_PATH}"
        )
    preprocessor = joblib.load(PROCESSED_PREPROCESSOR_PATH)
    joblib.dump(preprocessor, MODEL_PREPROCESSOR_PATH)

    version_info = collect_version_info(metadata)
    MODEL_VERSION_INFO_PATH.write_text(
        json.dumps(version_info, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    model_metadata_payload = {
        "model_artifact": MODEL_PATH.name,
        "preprocessor_artifact": MODEL_PREPROCESSOR_PATH.name,
        "feature_names_artifact": MODEL_FEATURE_NAMES_PATH.name,
        "version_info_artifact": MODEL_VERSION_INFO_PATH.name,
        "contract_version": INFERENCE_CONTRACT_VERSION,
        "model_type": classifier.__class__.__name__,
        "model_params": classifier.get_params(),
        "training_params": {
            "random_state": model_cfg.get("random_state", RANDOM_STATE),
            "seed": SEED,
            "test_size": metadata["test_size"],
            "cv_folds": CROSS_VALIDATION_FOLDS,
        },
        "mlflow_run_id": mlflow_run_id,
        "model_registry": registry_info,
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "target_column": metadata["target_column"],
        "feature_count": len(encoded_feature_names),
        "default_threshold_metrics": default_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "chosen_threshold": tuned_threshold,
        "preprocessing": {
            "categorical_columns": metadata["categorical_columns"],
            "numeric_columns": metadata["numeric_columns"],
            "encoded_feature_names": encoded_feature_names,
        },
        "version_info": version_info,
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(model_metadata_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "feature_names": MODEL_FEATURE_NAMES_PATH,
        "preprocessor": MODEL_PREPROCESSOR_PATH,
        "model_metadata": MODEL_METADATA_PATH,
        "version_info": MODEL_VERSION_INFO_PATH,
    }


def train() -> None:
    set_global_seed(SEED)
    logger.info("Iniciando treino do modelo com seed global fixa = {}", SEED)

    params_path = PARAMS_FILE if PARAMS_FILE.exists() else CONFIGS_DIR / "params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    model_cfg = params["model"]

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test, metadata = load_training_inputs()
    logger.info(
        "Dataset carregado: treino={} teste={} features={} fingerprint={}",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        metadata["dataset_fingerprint"],
    )

    classifier = build_random_forest(model_cfg)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "selected_model": "random_forest",
                "preprocessing_stage": "encoded_split_smote_per_fold",
            }
        )

        benchmark_df, skipped_models = run_benchmark(X_train, y_train, classifier)
        log_benchmark_metrics(benchmark_df, skipped_models)

        if skipped_models:
            logger.warning(
                "Modelos opcionais indisponiveis no benchmark: {}",
                ", ".join(skipped_models),
            )

        logger.info(
            "Benchmark concluido. Melhor F1 medio em CV: {} ({:.4f})",
            benchmark_df.iloc[0]["model_name"],
            benchmark_df.iloc[0]["f1_mean"],
        )

        logger.info("Executando treinamento final do RandomForestClassifier")
        (
            default_metrics,
            tuned_metrics,
            probabilities,
            tuned_threshold,
            tuned_f1,
            train_rows_resampled,
        ) = fit_and_evaluate(
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        metadata["train_rows_resampled"] = train_rows_resampled

        mlflow.log_params(
            {
                "n_estimators": model_cfg.get("n_estimators", 200),
                "max_depth": model_cfg.get("max_depth", 10),
                "min_samples_leaf": model_cfg.get("min_samples_leaf", 2),
                "random_state": model_cfg.get("random_state", RANDOM_STATE),
                "test_size": model_cfg.get("test_size", 0.2),
                "cv_folds": CROSS_VALIDATION_FOLDS,
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "encoded_feature_count": len(metadata["encoded_feature_names"]),
            }
        )
        mlflow.log_metrics(default_metrics)
        mlflow.log_metrics({f"tuned_{key}": value for key, value in tuned_metrics.items()})
        mlflow.log_metric("best_threshold", tuned_threshold)
        mlflow.log_metric("best_threshold_f1", tuned_f1)
        mlflow.log_metric("minority_class_ratio", float(metadata["minority_class_ratio"]))
        mlflow.log_metric("train_rows_resampled", float(metadata["train_rows_resampled"]))
        mlflow.sklearn.log_model(classifier, artifact_path="model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        registry_info: dict[str, object] | None = None
        if MLFLOW_ENABLE_MODEL_REGISTRY:
            registered_model_name = resolve_registered_model_name(
                experiment_name=experiment_name,
                configured_name=MLFLOW_REGISTERED_MODEL_NAME,
            )
            registry_info = register_model_version(
                client=MlflowClient(),
                registered_model_name=registered_model_name,
                model_uri=build_model_uri(run.info.run_id),
                run_id=run.info.run_id,
                experiment_name=experiment_name,
                experiment_id=run.info.experiment_id,
                status=os.getenv("MLFLOW_MODEL_INITIAL_STATUS", MLFLOW_MODEL_INITIAL_STATUS),
                approval_status=os.getenv(
                    "MLFLOW_MODEL_APPROVAL_STATUS",
                    MLFLOW_MODEL_APPROVAL_STATUS,
                ),
                approver=os.getenv("MLFLOW_MODEL_APPROVER", MLFLOW_MODEL_APPROVER) or None,
                metrics=tuned_metrics,
                metadata={
                    "dataset_fingerprint": metadata["dataset_fingerprint"],
                    "model_type": classifier.__class__.__name__,
                    "selected_model": "random_forest",
                },
                event_log_path=MODEL_REGISTRY_EVENTS_PATH,
            )
            MODEL_REGISTRY_INFO_PATH.write_text(
                json.dumps(registry_info, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            mlflow.set_tags(
                {
                    "registered_model_name": registry_info["registered_model_name"],
                    "registered_model_version": registry_info["version"],
                    "governance_status": registry_info["status"],
                    "approval_status": registry_info["approval_status"],
                }
            )
        else:
            registry_info = {
                "enabled": False,
                "reason": "MLFLOW_ENABLE_MODEL_REGISTRY=false",
                "run_id": run.info.run_id,
                "experiment_name": experiment_name,
            }
            MODEL_REGISTRY_INFO_PATH.write_text(
                json.dumps(registry_info, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            if not MODEL_REGISTRY_EVENTS_PATH.exists():
                MODEL_REGISTRY_EVENTS_PATH.write_text("[]", encoding="utf-8")

        joblib.dump(classifier, MODEL_PATH)

        production_artifacts = save_production_artifacts(
            classifier=classifier,
            model_cfg=model_cfg,
            default_metrics=default_metrics,
            tuned_metrics=tuned_metrics,
            metadata=metadata,
            tuned_threshold=tuned_threshold,
            mlflow_run_id=run.info.run_id,
            registry_info=registry_info,
        )

        artifact_dir = save_artifacts(
            benchmark_df=benchmark_df,
            default_metrics=default_metrics,
            tuned_metrics=tuned_metrics,
            metadata=metadata,
            y_test=y_test,
            default_probabilities=probabilities,
            tuned_threshold=tuned_threshold,
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="reports")
        mlflow.log_dict(metadata, "reports/preprocessing_metadata.json")
        for artifact_name, artifact_path in production_artifacts.items():
            mlflow.log_artifact(str(artifact_path), artifact_path="production")
            logger.info("Artefato de producao persistido: {}={}", artifact_name, artifact_path)
        if registry_info is not None:
            mlflow.log_artifact(str(MODEL_REGISTRY_INFO_PATH), artifact_path="registry")
            mlflow.log_artifact(str(MODEL_REGISTRY_EVENTS_PATH), artifact_path="registry")

        logger.info(
            "RUN_ID={} ACCURACY={:.4f} F1={:.4f} ROC_AUC={:.4f} PR_AUC={:.4f} THRESHOLD={:.4f}",
            run.info.run_id,
            tuned_metrics["accuracy"],
            tuned_metrics["f1"],
            tuned_metrics["roc_auc"],
            tuned_metrics["pr_auc"],
            tuned_threshold,
        )


if __name__ == "__main__":
    train()
