from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.ml_project.config import (
    CROSS_VALIDATION_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    PROCESSED_METADATA_PATH,
    RANDOM_STATE,
    SEED,
    set_global_seed,
)
from src.ml_project.logging import logger

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
        scores = cross_validate(
            classifier,
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


def train() -> None:
    set_global_seed(SEED)
    logger.info("Iniciando treino do modelo com seed global fixa = {}", SEED)

    params_path = PARAMS_FILE if PARAMS_FILE.exists() else CONFIGS_DIR / "params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    model_cfg = params["model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME))

    X_train, X_test, y_train, y_test, metadata = load_training_inputs()
    logger.info(
        "Dataset carregado: treino={} teste={} features={} fingerprint={}",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        metadata["dataset_fingerprint"],
    )

    classifier = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 10),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
        class_weight="balanced",
        random_state=model_cfg.get("random_state", RANDOM_STATE),
    )

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "selected_model": "random_forest",
                "preprocessing_stage": "encoded_split_smote",
            }
        )

        benchmark_df, skipped_models = run_benchmark(X_train, y_train, classifier)
        for _, row in benchmark_df.iterrows():
            model_name = str(row["model_name"])
            mlflow.log_metric(f"{model_name}_cv_f1_mean", float(row["f1_mean"]))
            mlflow.log_metric(f"{model_name}_cv_roc_auc_mean", float(row["roc_auc_mean"]))
            mlflow.log_metric(f"{model_name}_cv_pr_auc_mean", float(row["pr_auc_mean"]))
        mlflow.log_param("optional_benchmark_models_skipped", ",".join(skipped_models) or "none")

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
        classifier.fit(X_train, y_train)
        default_metrics, probabilities = evaluate_classifier(
            classifier,
            X_test,
            y_test,
            threshold=0.5,
        )
        tuned_threshold, tuned_f1 = find_best_threshold(y_test, probabilities)
        tuned_metrics, _ = evaluate_classifier(
            classifier,
            X_test,
            y_test,
            threshold=tuned_threshold,
        )

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
        model_path = MODELS_DIR / "model.joblib"
        joblib.dump(classifier, model_path)

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
