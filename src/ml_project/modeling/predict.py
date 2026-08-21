from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.ml_project.config import (
    MLFLOW_INFERENCE_MODEL_ALIAS,
    MLFLOW_INFERENCE_MODEL_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MODEL_PATH,
    MODEL_PREPROCESSOR_PATH,
    PROCESSED_METADATA_PATH,
    PROCESSED_PREPROCESSOR_PATH,
)
from src.ml_project.features import add_session_features
from src.ml_project.logging import logger
from src.ml_project.model_registry import build_registry_model_uri
from src.ml_project.preprocessing import encode_inference_features, validate_inference_schema


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def load_inference_metadata(metadata_path: str | Path | None = None) -> dict[str, object]:
    """Carrega os metadados do preprocessing usados para inferencia."""
    resolved_path = Path(metadata_path or PROCESSED_METADATA_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Metadados de inferencia nao encontrados em {resolved_path}.")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def load_preprocessor(preprocessor_path: str | Path | None = None) -> ColumnTransformer | None:
    """Carrega o transformador ajustado quando ele estiver disponivel."""
    candidate_paths = (
        [Path(preprocessor_path)]
        if preprocessor_path is not None
        else [MODEL_PREPROCESSOR_PATH, PROCESSED_PREPROCESSOR_PATH]
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return joblib.load(candidate_path)
    return None


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


def resolve_inference_model_uri(
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    registry_alias: str | None = None,
) -> str | None:
    """Resolve a URI do Model Registry quando a inferencia estiver configurada para isso."""
    explicit_model_uri = model_uri or os.getenv(
        "MLFLOW_INFERENCE_MODEL_URI",
        MLFLOW_INFERENCE_MODEL_URI or "",
    )
    if explicit_model_uri:
        return explicit_model_uri

    if not _env_flag("MLFLOW_USE_MODEL_REGISTRY_FOR_INFERENCE"):
        return None

    return build_registry_model_uri(
        registered_model_name=registered_model_name
        or os.getenv("MLFLOW_REGISTERED_MODEL_NAME", MLFLOW_REGISTERED_MODEL_NAME or ""),
        alias=registry_alias
        or os.getenv("MLFLOW_INFERENCE_MODEL_ALIAS", MLFLOW_INFERENCE_MODEL_ALIAS),
    )


def load_model_for_inference(
    model_path: str | Path | None = None,
    model_uri: str | None = None,
):
    """Carrega o modelo local ou uma versao governada pelo MLflow Model Registry."""
    if model_uri is not None:
        logger.info("Carregando modelo de inferencia via MLflow URI: {}", model_uri)
        return mlflow.sklearn.load_model(model_uri)

    resolved_model_path = Path(model_path or MODEL_PATH)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {resolved_model_path}.")
    return joblib.load(resolved_model_path)


def prepare_inference_frame(
    dataframe: pd.DataFrame,
    metadata: dict[str, object],
    preprocessor: ColumnTransformer | None = None,
) -> pd.DataFrame:
    """Reaplica o pipeline de features e encoding para dados novos."""
    validated = validate_inference_schema(dataframe)
    engineered = add_session_features(validated)
    if preprocessor is not None:
        encoded_feature_names = metadata.get("encoded_feature_names")
        if not isinstance(encoded_feature_names, list) or not encoded_feature_names:
            raise ValueError("Metadados sem encoded_feature_names validos para inferencia.")
        transformed = preprocessor.transform(engineered)
        return pd.DataFrame(transformed, columns=encoded_feature_names, index=engineered.index)

    encoded = encode_inference_features(engineered, metadata)
    return encoded


def predict(
    model_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    metadata_path: str | Path | None = None,
    preprocessor_path: str | Path | None = None,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    registry_alias: str | None = None,
) -> pd.Series:
    """Carrega o modelo treinado, reaplica o pipeline e gera previsoes."""
    if dataframe is None:
        raise ValueError("E necessario passar um dataframe para previsao.")

    resolved_model_uri = resolve_inference_model_uri(
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registry_alias=registry_alias,
    )
    model = load_model_for_inference(model_path=model_path, model_uri=resolved_model_uri)
    metadata = load_inference_metadata(metadata_path)
    preprocessor = load_preprocessor(preprocessor_path)
    inference_frame = prepare_inference_frame(dataframe, metadata, preprocessor)
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
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(PROCESSED_METADATA_PATH))
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--registered-model-name", type=str, default=None)
    parser.add_argument("--registry-alias", type=str, default=None)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    result = predict(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        preprocessor_path=args.preprocessor_path,
        model_uri=args.model_uri,
        registered_model_name=args.registered_model_name,
        registry_alias=args.registry_alias,
        dataframe=frame,
    )
    print(result.to_string(index=False))
