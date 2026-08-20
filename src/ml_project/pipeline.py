from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ml_project.config import (
    DATA_DIR,
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    PROCESSED_METADATA_PATH,
    PROCESSED_PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.ml_project.dataset import prepare_interim_dataset
from src.ml_project.features import add_session_features
from src.ml_project.preprocessing import prepare_model_data


def prepare() -> Path:
    """Carrega, limpa, transforma e salva os dados processados para treino."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = prepare_interim_dataset()
    engineered = add_session_features(cleaned)
    X_train, X_test, y_train, y_test, metadata = prepare_model_data(
        engineered,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        preprocessor_output_path=PROCESSED_PREPROCESSOR_PATH,
    )

    np.savez(
        PROCESSED_DATA_PATH,
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
    )
    PROCESSED_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return PROCESSED_DATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de dados para Online Shoppers")
    parser.add_argument("command", choices=["prepare", "train"])
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    else:
        raise ValueError("Use o módulo src.ml_project.modeling.train para treinar o modelo.")


if __name__ == "__main__":
    main()
