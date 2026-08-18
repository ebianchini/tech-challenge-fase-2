from __future__ import annotations

import json

import pandas as pd

from src.ml_project.config import PROCESSED_METADATA_PATH
from src.ml_project.dataset import drop_duplicate_rows, load_raw_dataset
from src.ml_project.features import add_session_features
from src.ml_project.modeling.train import build_benchmark_models, train
from src.ml_project.pipeline import prepare
from src.ml_project.preprocessing import prepare_model_data, validate_dataset_quality


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
    X_train, X_test, y_train, y_test, metadata = prepare_model_data(engineered)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
    assert metadata["dataset_fingerprint"]
    assert metadata["encoded_feature_names"]


def test_quality_validation_rejects_missing_columns() -> None:
    df = load_raw_dataset().drop(columns=["Month"])

    try:
        validate_dataset_quality(df)
    except ValueError as exc:
        assert "colunas ausentes" in str(exc)
    else:
        raise AssertionError("Era esperado erro de validacao para coluna ausente.")


def test_prepare_persists_metadata_file() -> None:
    prepare()

    assert PROCESSED_METADATA_PATH.exists()
    metadata = json.loads(PROCESSED_METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["dataset_fingerprint"]
    assert metadata["encoded_feature_names"]


def test_benchmark_models_include_optional_xgboost_when_available() -> None:
    from sklearn.ensemble import RandomForestClassifier

    models, skipped_models = build_benchmark_models(RandomForestClassifier(random_state=42))

    assert "logistic_regression" in models
    assert "random_forest" in models
    assert ("xgboost" in models) or ("xgboost" in skipped_models)


def test_training_logs_auc_metrics(capsys) -> None:
    prepare()
    train()
    output = capsys.readouterr().out

    assert "ROC_AUC=" in output
    assert "PR_AUC=" in output
    assert "THRESHOLD=" in output
