from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.ml_project.config import (
    MODEL_FEATURE_NAMES_PATH,
    MODEL_METADATA_PATH,
    MODEL_PREPROCESSOR_PATH,
    MODEL_VERSION_INFO_PATH,
    PROCESSED_METADATA_PATH,
    PROCESSED_PREPROCESSOR_PATH,
)
from src.ml_project.dataset import drop_duplicate_rows, load_raw_dataset
from src.ml_project.features import add_session_features
from src.ml_project.modeling.predict import load_inference_metadata, load_preprocessor, predict
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
    assert PROCESSED_PREPROCESSOR_PATH.exists()
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


def test_training_persists_production_artifacts() -> None:
    prepare()
    train()

    assert MODEL_FEATURE_NAMES_PATH.exists()
    assert MODEL_PREPROCESSOR_PATH.exists()
    assert MODEL_METADATA_PATH.exists()
    assert MODEL_VERSION_INFO_PATH.exists()

    feature_names = json.loads(MODEL_FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    model_metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    version_info = json.loads(MODEL_VERSION_INFO_PATH.read_text(encoding="utf-8"))

    assert feature_names["encoded_feature_names"]
    assert feature_names["encoded_feature_count"] == len(feature_names["encoded_feature_names"])
    assert model_metadata["preprocessor_artifact"] == MODEL_PREPROCESSOR_PATH.name
    assert model_metadata["feature_count"] == feature_names["encoded_feature_count"]
    assert model_metadata["dataset_fingerprint"] == version_info["dataset_fingerprint"]
    assert "package_version" in version_info
    assert "dataset_dvc" in version_info


def test_predict_reapplies_feature_pipeline_for_raw_input() -> None:
    prepare()
    train()
    raw_input = load_raw_dataset().head(5).drop(columns=["Revenue"])

    predictions = predict(dataframe=raw_input)

    assert len(predictions) == len(raw_input)
    assert predictions.name == "predicted_revenue"
    assert set(predictions.unique()).issubset({0, 1})


def test_predict_can_use_persisted_preprocessor() -> None:
    prepare()
    train()
    metadata = load_inference_metadata()
    preprocessor = load_preprocessor(MODEL_PREPROCESSOR_PATH)
    raw_input = load_raw_dataset().head(3).drop(columns=["Revenue"])

    assert preprocessor is not None
    predictions = predict(dataframe=raw_input, preprocessor_path=MODEL_PREPROCESSOR_PATH)

    assert len(predictions) == len(raw_input)
    assert len(metadata["encoded_feature_names"]) == preprocessor.transform(
        add_session_features(raw_input)
    ).shape[1]


def test_predict_validates_model_compatibility(tmp_path) -> None:
    prepare()
    invalid_model = LogisticRegression().fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    invalid_model_path = tmp_path / "invalid_model.joblib"
    joblib.dump(invalid_model, invalid_model_path)
    raw_input = load_raw_dataset().head(2).drop(columns=["Revenue"])

    with pytest.raises(ValueError, match="incompativel"):
        predict(model_path=invalid_model_path, dataframe=raw_input)
