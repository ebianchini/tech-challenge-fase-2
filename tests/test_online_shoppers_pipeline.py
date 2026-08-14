from __future__ import annotations

import pandas as pd

from src.ml_project.dataset import drop_duplicate_rows, load_raw_dataset
from src.ml_project.features import add_session_features
from src.ml_project.modeling.train import train
from src.ml_project.preprocessing import prepare_model_data


def test_dataset_load_and_deduplication() -> None:
    df = load_raw_dataset()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 12330
    assert {"Revenue", "Month", "VisitorType"}.issubset(df.columns)

    cleaned = drop_duplicate_rows(df)

    assert cleaned.shape[0] < df.shape[0]
    assert not cleaned.duplicated().any()
    assert cleaned["Revenue"].dtype == bool


def test_feature_engineering_creates_session_metrics() -> None:
    df = load_raw_dataset()
    engineered = add_session_features(df)

    for column in [
        "TotalSessionTime",
        "TotalPagesVisited",
        "AdministrativeTimeRatio",
        "InformationalTimeRatio",
        "ProductRelatedTimeRatio",
        "AdministrativeRatio",
        "InformationalRatio",
        "ProductRelatedRatio",
    ]:
        assert column in engineered.columns

    assert engineered["TotalSessionTime"].ge(0).all()
    assert engineered["TotalPagesVisited"].ge(0).all()


def test_preprocessing_builds_train_test_split_and_targets() -> None:
    df = load_raw_dataset()
    engineered = add_session_features(df)
    X_train, X_test, y_train, y_test = prepare_model_data(engineered)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_training_logs_auc_metrics(capsys) -> None:
    train()
    output = capsys.readouterr().out

    assert "ROC_AUC=" in output
    assert "PR_AUC=" in output
