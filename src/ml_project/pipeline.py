from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"


def prepare() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    prepared = {
        "X_train": X_train.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }

    with (processed_dir / "train.json").open("w", encoding="utf-8") as file:
        json.dump(prepared, file)


def train() -> None:
    params = yaml.safe_load((CONFIGS_DIR / "params.yaml").read_text(encoding="utf-8"))
    model_cfg = params["model"]

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
        stratify=iris.target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(random_state=model_cfg["random_state"])),
        ]
    )
    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / "model.joblib").open("wb") as file:
        import joblib

        joblib.dump(pipeline, file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline básico de ML")
    parser.add_argument("command", choices=["prepare", "train"])
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    else:
        train()


if __name__ == "__main__":
    main()
