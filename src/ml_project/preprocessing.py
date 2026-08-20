from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.ml_project.config import (
    FEATURE_COLUMNS,
    MIN_TARGET_CLASS_RATIO,
    RANDOM_STATE,
    RAW_DATASET_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)

CATEGORICAL_COLUMNS = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]


def compute_dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """Gera um fingerprint estável do dataset para rastreabilidade em MLflow."""
    normalized = (
        dataset.sort_index(axis=1)
        .sort_values(by=dataset.columns.tolist())
        .reset_index(drop=True)
    )
    row_hashes = pd.util.hash_pandas_object(normalized, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def validate_dataset_quality(
    dataset: pd.DataFrame,
    expected_columns: Iterable[str] | None = None,
    target_column: str = TARGET_COLUMN,
    min_target_class_ratio: float = MIN_TARGET_CLASS_RATIO,
) -> dict[str, object]:
    """Valida schema, tipos e balanceamento mínimo do dataset antes do treino."""
    expected = list(expected_columns or [*FEATURE_COLUMNS, target_column])
    missing_columns = sorted(set(expected) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset com colunas ausentes: {missing_columns}")

    data = dataset[expected].copy()
    null_counts = data.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise ValueError(f"Dataset com valores nulos: {columns_with_nulls.to_dict()}")

    invalid_numeric_columns = [
        column
        for column in data.columns
        if column not in CATEGORICAL_COLUMNS
        and column != target_column
        and not pd.api.types.is_numeric_dtype(data[column])
    ]
    if invalid_numeric_columns:
        raise TypeError(f"Colunas numericas invalidas: {invalid_numeric_columns}")

    if target_column not in data.columns:
        raise KeyError(f"Coluna alvo '{target_column}' nao encontrada no dataset.")

    target = data[target_column].astype(int)
    class_distribution = target.value_counts(normalize=True).sort_index()
    minority_ratio = float(class_distribution.min())
    if minority_ratio < min_target_class_ratio:
        raise ValueError(
            "Distribuicao da variavel alvo abaixo do minimo esperado: "
            f"{class_distribution.to_dict()}"
        )

    return {
        "rows": int(len(data)),
        "columns": expected,
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "class_distribution": {str(key): float(value) for key, value in class_distribution.items()},
        "minority_class_ratio": minority_ratio,
        "dataset_fingerprint": compute_dataset_fingerprint(data),
    }


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Cria o transformador de features usado no treino e na validacao cruzada."""
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in features.columns]
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    return preprocessor, categorical_columns, numeric_columns


def validate_inference_schema(
    dataset: pd.DataFrame,
    expected_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Valida o schema bruto esperado em inferencia e remove a coluna alvo quando presente."""
    default_columns = [column for column in RAW_DATASET_COLUMNS if column != TARGET_COLUMN]
    expected = list(expected_columns or default_columns)
    missing_columns = sorted(set(expected) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset de inferencia com colunas ausentes: {missing_columns}")

    inference_frame = dataset[expected].copy()
    null_counts = inference_frame.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise ValueError(
            "Dataset de inferencia com valores nulos: "
            f"{columns_with_nulls.to_dict()}"
        )
    return inference_frame


def encode_inference_features(
    features: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Reconstroi o encoding do treino a partir dos metadados persistidos."""
    encoded_feature_names = metadata.get("encoded_feature_names")
    categorical_columns = metadata.get("categorical_columns")
    numeric_columns = metadata.get("numeric_columns")

    if not isinstance(encoded_feature_names, list) or not encoded_feature_names:
        raise ValueError("Metadados sem encoded_feature_names validos para inferencia.")
    if not isinstance(categorical_columns, list) or not isinstance(numeric_columns, list):
        raise ValueError("Metadados sem definicao valida de colunas categoricas e numericas.")

    missing_columns = sorted(set([*categorical_columns, *numeric_columns]) - set(features.columns))
    if missing_columns:
        raise ValueError(f"Features de inferencia com colunas ausentes: {missing_columns}")

    encoded_frame = pd.DataFrame(index=features.index)

    for column in numeric_columns:
        encoded_frame[f"numeric__{column}"] = pd.to_numeric(features[column], errors="raise")

    for column in categorical_columns:
        values = features[column].astype(str)
        prefix = f"categorical__{column}_"
        matching_feature_names = [
            feature_name
            for feature_name in encoded_feature_names
            if feature_name.startswith(prefix)
        ]
        for feature_name in matching_feature_names:
            category_value = feature_name[len(prefix) :]
            encoded_frame[feature_name] = values.eq(category_value).astype(float)

    missing_encoded_columns = sorted(set(encoded_feature_names) - set(encoded_frame.columns))
    if missing_encoded_columns:
        raise ValueError(
            "Nao foi possivel reconstruir todas as colunas codificadas para inferencia: "
            f"{missing_encoded_columns}"
        )

    return encoded_frame[encoded_feature_names]


def prepare_model_data(
    dataset: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_column: str = TARGET_COLUMN,
    preprocessor_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Aplica split, encoding e balanceamento SMOTE, retornando tambem metadados."""
    quality_report = validate_dataset_quality(
        dataset,
        expected_columns=[*FEATURE_COLUMNS, target_column],
    )
    data = dataset.copy()

    features = data.drop(columns=[target_column])
    target = data[target_column].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    preprocessor, categorical_columns, numeric_columns = build_preprocessor(features)
    X_train_encoded = preprocessor.fit_transform(X_train_raw)
    X_test_encoded = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if preprocessor_output_path is not None:
        output_path = Path(preprocessor_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, output_path)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

    metadata = {
        **quality_report,
        "target_column": target_column,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "encoded_feature_names": feature_names,
        "train_rows": int(X_train_raw.shape[0]),
        "test_rows": int(X_test_raw.shape[0]),
        "train_rows_resampled": int(len(y_train_resampled)),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    return (
        pd.DataFrame(X_train_resampled, columns=feature_names),
        pd.DataFrame(X_test_encoded, columns=feature_names),
        pd.Series(y_train_resampled, name=target_column),
        pd.Series(y_test.to_numpy(), name=target_column),
        metadata,
    )
