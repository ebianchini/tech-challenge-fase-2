from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.ml_project.config import (MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
                                   MODELS_DIR, RANDOM_STATE, SEED,
                                   set_global_seed)
from src.ml_project.logging import logger

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = ROOT / "configs"
PARAMS_FILE = ROOT / "params.yaml"


def train() -> None:
    set_global_seed(SEED)
    logger.info("Iniciando treino do modelo com seed global fixa = {}", SEED)

    params_path = PARAMS_FILE if PARAMS_FILE.exists() else CONFIGS_DIR / "params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    model_cfg = params["model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME))

    processed_path = ROOT / "data" / "processed" / "online_shoppers_processed.npz"
    if not processed_path.exists():
        logger.error("Arquivo processado não encontrado em {}", processed_path)
        raise FileNotFoundError(f"Arquivo processado não encontrado: {processed_path}")

    with np.load(processed_path) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

    logger.info(
        "Dataset carregado: treino={} teste={} features={} alvo={}",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        y_train.shape[0],
    )

    classifier = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 10),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
        class_weight="balanced",
        random_state=model_cfg.get("random_state", RANDOM_STATE),
    )

    with mlflow.start_run() as run:
        logger.info("Executando treinamento do RandomForestClassifier")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
        }

        mlflow.log_params(
            {
                "n_estimators": model_cfg.get("n_estimators", 200),
                "max_depth": model_cfg.get("max_depth", 10),
                "min_samples_leaf": model_cfg.get("min_samples_leaf", 2),
                "random_state": model_cfg.get("random_state", RANDOM_STATE),
                "test_size": model_cfg.get("test_size", 0.2),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(classifier, artifact_path="model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "model.joblib"
        joblib.dump(classifier, model_path)

        logger.info(
            "RUN_ID={} ACCURACY={:.4f} F1={:.4f} ROC_AUC={:.4f} PR_AUC={:.4f}",
            run.info.run_id,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["pr_auc"],
        )


if __name__ == "__main__":
    train()
