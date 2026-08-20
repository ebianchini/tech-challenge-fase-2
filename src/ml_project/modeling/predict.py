from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.ml_project.config import MODELS_DIR, PROCESSED_METADATA_PATH
from src.ml_project.features import add_session_features
from src.ml_project.logging import logger
from src.ml_project.preprocessing import encode_inference_features, validate_inference_schema


def load_inference_metadata(metadata_path: str | Path | None = None) -> dict[str, object]:
    """Carrega os metadados do preprocessing usados para inferencia."""
    resolved_path = Path(metadata_path or PROCESSED_METADATA_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Metadados de inferencia nao encontrados em {resolved_path}.")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def validate_model_compatibility(model: object, inference_frame: pd.DataFrame) -> None:
    """Garante que o modelo salvo aceite o schema reconstruido para inferencia."""
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        expected = list(model_feature_names)
        received = inference_frame.columns.tolist()
        if expected != received:
            raise ValueError(
                "Schema de inferencia incompativel com o modelo persistido. "
                f"Esperado={expected} recebido={received}"
            )

    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != inference_frame.shape[1]:
        raise ValueError(
            "Quantidade de features incompativel com o modelo persistido. "
            f"Esperado={model_feature_count} recebido={inference_frame.shape[1]}"
        )


def prepare_inference_frame(
    dataframe: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Reaplica o pipeline de features e encoding para dados novos."""
    validated = validate_inference_schema(dataframe)
    engineered = add_session_features(validated)
    encoded = encode_inference_features(engineered, metadata)
    return encoded


def predict(
    model_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    metadata_path: str | Path | None = None,
) -> pd.Series:
    """Carrega o modelo treinado, reaplica o pipeline e gera previsoes."""
    if dataframe is None:
        raise ValueError("E necessario passar um dataframe para previsao.")

    resolved_model_path = Path(model_path or MODELS_DIR / "model.joblib")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {resolved_model_path}.")

    model = joblib.load(resolved_model_path)
    metadata = load_inference_metadata(metadata_path)
    inference_frame = prepare_inference_frame(dataframe, metadata)
    validate_model_compatibility(model, inference_frame)

    logger.info(
        "Executando inferencia com {} registros e {} features reconstruidas",
        inference_frame.shape[0],
        inference_frame.shape[1],
    )
    predictions = model.predict(inference_frame)
    return pd.Series(predictions, index=dataframe.index, name="predicted_revenue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicao de compra do modelo treinado")
    parser.add_argument("--model-path", type=str, default=str(MODELS_DIR / "model.joblib"))
    parser.add_argument("--metadata-path", type=str, default=str(PROCESSED_METADATA_PATH))
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    result = predict(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        dataframe=frame,
    )
    print(result.to_string(index=False))
