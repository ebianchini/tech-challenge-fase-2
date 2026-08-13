from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.ml_project.config import MODELS_DIR


def predict(model_path: str | Path | None = None, dataframe: pd.DataFrame | None = None) -> pd.Series:
    """Carrega o modelo treinado e gera previsões em um DataFrame."""
    if model_path is None:
        model_path = MODELS_DIR / "model.joblib"
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}.")

    model = joblib.load(model_path)
    if dataframe is None:
        raise ValueError("É necessário passar um dataframe para previsão.")

    predictions = model.predict(dataframe)
    return pd.Series(predictions, name="predicted_revenue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predição de compra do modelo treinado")
    parser.add_argument("--model-path", type=str, default=str(MODELS_DIR / "model.joblib"))
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    result = predict(args.model_path, frame)
    print(result.to_string(index=False))
