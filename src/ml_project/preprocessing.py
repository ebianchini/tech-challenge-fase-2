from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.ml_project.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def prepare_model_data(
    dataset: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Aplica encoding categórico e balanceamento SMOTE antes do treinamento."""
    data = dataset.copy()

    if target_column not in data.columns:
        raise KeyError(f"Coluna alvo '{target_column}' não encontrada no dataset.")

    features = data.drop(columns=[target_column])
    target = data[target_column].astype(int)

    categorical_columns = [
        "Month",
        "VisitorType",
        "Weekend",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
    ]
    numeric_columns = [
        column
        for column in features.columns
        if column not in categorical_columns
    ]

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

    encoded_features = preprocessor.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features,
        target.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=target.to_numpy(),
    )

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return (
        pd.DataFrame(X_train_resampled),
        pd.DataFrame(X_test),
        pd.Series(y_train_resampled),
        pd.Series(y_test),
    )
