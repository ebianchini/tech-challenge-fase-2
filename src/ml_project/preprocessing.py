from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.ml_project.config import (
    FEATURE_COLUMNS,
    MIN_TARGET_CLASS_RATIO,
    RANDOM_STATE,
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


def prepare_model_data(
    dataset: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_column: str = TARGET_COLUMN,
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

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

    metadata = {
        **quality_report,
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
