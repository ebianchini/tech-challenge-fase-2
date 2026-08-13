from __future__ import annotations

import os
from pathlib import Path

import mlflow
import yaml
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"


def train() -> None:
    params = yaml.safe_load((CONFIGS_DIR / "params.yaml").read_text(encoding="utf-8"))
    model_cfg = params["model"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "tech-challenge-fase-2")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
        stratify=iris.target,
    )

    with mlflow.start_run() as run:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", SVC(random_state=model_cfg["random_state"])),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score = accuracy_score(y_test, predictions)

        mlflow.log_param("test_size", model_cfg["test_size"])
        mlflow.log_param("random_state", model_cfg["random_state"])
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", env_manager="uv")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with (MODELS_DIR / "model.joblib").open("wb") as file:
            import joblib

            joblib.dump(pipeline, file)

        print(f"RUN_ID={run.info.run_id} ACCURACY={score:.4f}")


if __name__ == "__main__":
    train()
