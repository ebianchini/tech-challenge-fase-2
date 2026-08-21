from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from src.ml_project.model_registry import (
    build_model_uri,
    promote_model_version,
    register_model_version,
    resolve_registered_model_name,
    rollback_model_version,
)
from src.ml_project.modeling.predict import resolve_inference_model_uri


def _configure_tracking(tmp_path: Path, experiment_name: str) -> MlflowClient:
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)
    return MlflowClient()


def _log_dummy_model() -> str:
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(features, target)
    mlflow.sklearn.log_model(model, artifact_path="model")
    return mlflow.active_run().info.run_id


def _register_dummy_version(
    client: MlflowClient,
    registered_model_name: str,
    event_log_path: Path,
    experiment_name: str = "Registry Test",
) -> dict[str, object]:
    with mlflow.start_run() as run:
        run_id = _log_dummy_model()
        return register_model_version(
            client=client,
            registered_model_name=registered_model_name,
            model_uri=build_model_uri(run_id),
            run_id=run_id,
            experiment_name=experiment_name,
            experiment_id=run.info.experiment_id,
            status="Staging",
            approval_status="pending",
            metrics={"f1": 0.75},
            metadata={"dataset_fingerprint": "abc123"},
            event_log_path=event_log_path,
        )


def test_registered_model_name_defaults_to_experiment_scope() -> None:
    assert (
        resolve_registered_model_name(
            experiment_name="Online Shoppers Experiment",
            configured_name=None,
        )
        == "online-shoppers-experiment-random-forest"
    )
    assert (
        resolve_registered_model_name(
            experiment_name="Online Shoppers Experiment",
            configured_name="custom-model",
        )
        == "custom-model"
    )


def test_inference_model_uri_can_target_registry_alias(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_INFERENCE_MODEL_URI", raising=False)
    monkeypatch.setenv("MLFLOW_USE_MODEL_REGISTRY_FOR_INFERENCE", "true")

    assert (
        resolve_inference_model_uri(
            registered_model_name="online-shoppers-random-forest",
            registry_alias="Production",
        )
        == "models:/online-shoppers-random-forest@Production"
    )


def test_register_model_version_applies_status_alias_and_event_log(tmp_path: Path) -> None:
    client = _configure_tracking(tmp_path, "Registry Test")
    event_log_path = tmp_path / "model_registry_events.json"
    registered_model_name = resolve_registered_model_name(
        "Registry Test",
        configured_name=None,
    )

    result = _register_dummy_version(client, registered_model_name, event_log_path)

    model_version = client.get_model_version(
        registered_model_name,
        str(result["version"]),
    )
    staging_alias = client.get_model_version_by_alias(registered_model_name, "Staging")
    events = json.loads(event_log_path.read_text(encoding="utf-8"))

    assert result["status"] == "Staging"
    assert model_version.current_stage == "Staging"
    assert model_version.tags["governance_status"] == "Staging"
    assert model_version.tags["approval_status"] == "pending"
    assert str(staging_alias.version) == str(result["version"])
    assert events[-1]["event_type"] == "registered"
    assert events[-1]["registered_model_name"] == registered_model_name


def test_promote_and_rollback_model_versions(tmp_path: Path) -> None:
    client = _configure_tracking(tmp_path, "Registry Rollback Test")
    event_log_path = tmp_path / "model_registry_events.json"
    registered_model_name = resolve_registered_model_name(
        "Registry Rollback Test",
        configured_name=None,
    )

    first = _register_dummy_version(
        client,
        registered_model_name,
        event_log_path,
        experiment_name="Registry Rollback Test",
    )
    promote_model_version(
        client=client,
        registered_model_name=registered_model_name,
        version=str(first["version"]),
        target_status="Production",
        approver="qa",
        reason="Baseline aprovado",
        event_log_path=event_log_path,
    )

    second = _register_dummy_version(
        client,
        registered_model_name,
        event_log_path,
        experiment_name="Registry Rollback Test",
    )
    promote_model_version(
        client=client,
        registered_model_name=registered_model_name,
        version=str(second["version"]),
        target_status="Production",
        approver="qa",
        reason="Nova versao aprovada",
        event_log_path=event_log_path,
    )

    first_after_second_promotion = client.get_model_version(
        registered_model_name,
        str(first["version"]),
    )
    second_after_promotion = client.get_model_version(
        registered_model_name,
        str(second["version"]),
    )

    assert first_after_second_promotion.current_stage == "Archived"
    assert first_after_second_promotion.tags["rollback_candidate"] == "true"
    assert second_after_promotion.current_stage == "Production"

    rollback = rollback_model_version(
        client=client,
        registered_model_name=registered_model_name,
        target_version=str(first["version"]),
        approver="qa",
        reason="Regressao de metrica",
        event_log_path=event_log_path,
    )
    production_alias = client.get_model_version_by_alias(registered_model_name, "Production")
    events = json.loads(event_log_path.read_text(encoding="utf-8"))

    assert rollback["approval_status"] == "rolled_back"
    assert str(production_alias.version) == str(first["version"])
    assert client.get_model_version(
        registered_model_name,
        str(first["version"]),
    ).current_stage == "Production"
    assert client.get_model_version(
        registered_model_name,
        str(second["version"]),
    ).current_stage == "Archived"
    assert events[-1]["event_type"] == "rollback"
