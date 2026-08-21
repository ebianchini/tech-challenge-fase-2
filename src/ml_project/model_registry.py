from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.ml_project.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_INFERENCE_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MODEL_REGISTRY_EVENTS_PATH,
)
from src.ml_project.logging import logger

MODEL_STATUSES = ("Staging", "Production", "Archived")
APPROVAL_STATUSES = ("pending", "approved", "rejected", "rolled_back")
STATUS_TAG = "governance_status"
APPROVAL_STATUS_TAG = "approval_status"
APPROVED_BY_TAG = "approved_by"
APPROVED_AT_TAG = "approved_at_utc"
PROMOTION_REASON_TAG = "promotion_reason"
ROLLBACK_REASON_TAG = "rollback_reason"
ROLLBACK_CANDIDATE_TAG = "rollback_candidate"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_status(status: str) -> str:
    for allowed in MODEL_STATUSES:
        if status.lower() == allowed.lower():
            return allowed
    raise ValueError(f"Status de modelo invalido: {status}. Use {', '.join(MODEL_STATUSES)}.")


def normalize_approval_status(status: str) -> str:
    normalized = status.lower()
    if normalized not in APPROVAL_STATUSES:
        raise ValueError(
            "Status de aprovacao invalido: "
            f"{status}. Use {', '.join(APPROVAL_STATUSES)}."
        )
    return normalized


def slugify_model_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug.lower() or "model"


def resolve_registered_model_name(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    configured_name: str | None = MLFLOW_REGISTERED_MODEL_NAME,
    model_key: str = "random-forest",
) -> str:
    """Resolve o nome do registered model, versionando modelos por experimento."""
    if configured_name:
        return configured_name
    return f"{slugify_model_name(experiment_name)}-{slugify_model_name(model_key)}"


def build_model_uri(run_id: str, artifact_path: str = "model") -> str:
    return f"runs:/{run_id}/{artifact_path}"


def build_registry_model_uri(
    registered_model_name: str | None = None,
    alias: str = MLFLOW_INFERENCE_MODEL_ALIAS,
    version: str | None = None,
) -> str:
    """Monta a URI usada pela inferencia para carregar uma versao governada."""
    model_name = registered_model_name or resolve_registered_model_name()
    if version is not None:
        return f"models:/{model_name}/{version}"
    return f"models:/{model_name}@{alias}"


def ensure_registered_model(client: MlflowClient, name: str, experiment_name: str) -> None:
    try:
        client.create_registered_model(
            name,
            tags={
                "experiment_name": experiment_name,
                "governance": "mlflow_model_registry",
            },
            description=(
                "Modelo governado pelo MLflow Model Registry para o pipeline "
                "Online Shoppers Purchasing Intention."
            ),
        )
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise


def _set_model_version_tags(
    client: MlflowClient,
    name: str,
    version: str,
    tags: dict[str, Any],
) -> None:
    for key, value in tags.items():
        if value is None:
            continue
        client.set_model_version_tag(name=name, version=version, key=key, value=str(value))


def _transition_status(
    client: MlflowClient,
    name: str,
    version: str,
    status: str,
    archive_existing_versions: bool = False,
) -> ModelVersion:
    normalized_status = normalize_status(status)
    model_version = client.transition_model_version_stage(
        name=name,
        version=version,
        stage=normalized_status,
        archive_existing_versions=archive_existing_versions,
    )
    _set_model_version_tags(
        client,
        name,
        version,
        {
            STATUS_TAG: normalized_status,
            "status_updated_at_utc": utc_now(),
        },
    )
    client.set_registered_model_alias(name, normalized_status, version)
    return model_version


def _read_registry_events(event_log_path: Path) -> list[dict[str, Any]]:
    if not event_log_path.exists():
        return []
    return json.loads(event_log_path.read_text(encoding="utf-8"))


def append_registry_event(
    event: dict[str, Any],
    event_log_path: Path = MODEL_REGISTRY_EVENTS_PATH,
) -> dict[str, Any]:
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    events = _read_registry_events(event_log_path)
    payload = {
        "event_id": len(events) + 1,
        "created_at_utc": utc_now(),
        **event,
    }
    events.append(payload)
    event_log_path.write_text(json.dumps(events, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def _find_versions_by_status(
    client: MlflowClient,
    name: str,
    status: str,
) -> list[ModelVersion]:
    normalized_status = normalize_status(status)
    versions = client.search_model_versions(f"name='{name}'")
    filtered = [
        version
        for version in versions
        if version.current_stage == normalized_status
        or version.tags.get(STATUS_TAG) == normalized_status
    ]
    return sorted(filtered, key=lambda item: int(item.version), reverse=True)


def _current_status(model_version: ModelVersion) -> str:
    tag_status = model_version.tags.get(STATUS_TAG)
    if tag_status:
        return normalize_status(tag_status)
    if model_version.current_stage in MODEL_STATUSES:
        return model_version.current_stage
    return "Archived" if model_version.current_stage == "None" else str(model_version.current_stage)


def _validate_production_approval(
    target_status: str,
    approval_status: str,
    approver: str | None,
) -> None:
    if target_status != "Production":
        return
    if approval_status != "approved" or not approver:
        raise ValueError(
            "Promocao para Production exige approval_status='approved' e approver informado."
        )


def register_model_version(
    client: MlflowClient,
    registered_model_name: str,
    model_uri: str,
    run_id: str,
    experiment_name: str,
    experiment_id: str,
    status: str = "Staging",
    approval_status: str = "pending",
    approver: str | None = None,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
    event_log_path: Path = MODEL_REGISTRY_EVENTS_PATH,
) -> dict[str, Any]:
    """Registra uma versao no Model Registry e aplica tags de governanca."""
    normalized_status = normalize_status(status)
    normalized_approval = normalize_approval_status(approval_status)
    _validate_production_approval(normalized_status, normalized_approval, approver)

    ensure_registered_model(client, registered_model_name, experiment_name)

    version_tags: dict[str, Any] = {
        STATUS_TAG: normalized_status,
        APPROVAL_STATUS_TAG: normalized_approval,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "registered_from_uri": model_uri,
    }
    for key, value in (metrics or {}).items():
        version_tags[f"metric_{key}"] = value
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            version_tags[key] = value

    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run_id,
        tags=version_tags,
        description=(
            f"Run {run_id} do experimento {experiment_name}. "
            f"Status inicial: {normalized_status}; aprovacao: {normalized_approval}."
        ),
    )
    _transition_status(
        client,
        registered_model_name,
        model_version.version,
        normalized_status,
        archive_existing_versions=normalized_status == "Production",
    )
    _set_model_version_tags(
        client,
        registered_model_name,
        model_version.version,
        {
            APPROVAL_STATUS_TAG: normalized_approval,
            APPROVED_BY_TAG: approver,
            APPROVED_AT_TAG: utc_now() if normalized_approval == "approved" else None,
        },
    )

    event = append_registry_event(
        {
            "event_type": "registered",
            "registered_model_name": registered_model_name,
            "version": str(model_version.version),
            "from_status": None,
            "to_status": normalized_status,
            "approval_status": normalized_approval,
            "approved_by": approver,
            "run_id": run_id,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "model_uri": model_uri,
        },
        event_log_path=event_log_path,
    )
    logger.info(
        "Modelo registrado no MLflow Model Registry: name={} version={} status={}",
        registered_model_name,
        model_version.version,
        normalized_status,
    )
    return {
        "enabled": True,
        "registered_model_name": registered_model_name,
        "version": str(model_version.version),
        "status": normalized_status,
        "approval_status": normalized_approval,
        "model_uri": model_uri,
        "run_id": run_id,
        "experiment_name": experiment_name,
        "event": event,
    }


def promote_model_version(
    client: MlflowClient,
    registered_model_name: str,
    version: str,
    target_status: str,
    approver: str,
    reason: str,
    approval_status: str = "approved",
    event_log_path: Path = MODEL_REGISTRY_EVENTS_PATH,
) -> dict[str, Any]:
    """Promove uma versao entre Staging, Production e Archived."""
    normalized_status = normalize_status(target_status)
    normalized_approval = normalize_approval_status(approval_status)
    _validate_production_approval(normalized_status, normalized_approval, approver)

    previous_version = client.get_model_version(registered_model_name, version)
    previous_status = _current_status(previous_version)

    if normalized_status == "Production":
        current_production_versions = [
            item
            for item in _find_versions_by_status(client, registered_model_name, "Production")
            if str(item.version) != str(version)
        ]
        for current in current_production_versions:
            _transition_status(
                client,
                registered_model_name,
                str(current.version),
                "Archived",
                archive_existing_versions=False,
            )
            _set_model_version_tags(
                client,
                registered_model_name,
                str(current.version),
                {
                    ROLLBACK_CANDIDATE_TAG: "true",
                    "archived_by_version": version,
                    "archived_at_utc": utc_now(),
                },
            )

    _transition_status(
        client,
        registered_model_name,
        version,
        normalized_status,
        archive_existing_versions=normalized_status == "Production",
    )
    _set_model_version_tags(
        client,
        registered_model_name,
        version,
        {
            APPROVAL_STATUS_TAG: normalized_approval,
            APPROVED_BY_TAG: approver,
            APPROVED_AT_TAG: utc_now() if normalized_approval == "approved" else None,
            PROMOTION_REASON_TAG: reason,
            ROLLBACK_CANDIDATE_TAG: "false" if normalized_status == "Production" else None,
        },
    )

    event = append_registry_event(
        {
            "event_type": "promoted",
            "registered_model_name": registered_model_name,
            "version": str(version),
            "from_status": previous_status,
            "to_status": normalized_status,
            "approval_status": normalized_approval,
            "approved_by": approver,
            "reason": reason,
        },
        event_log_path=event_log_path,
    )
    logger.info(
        "Versao promovida no Model Registry: name={} version={} {} -> {}",
        registered_model_name,
        version,
        previous_status,
        normalized_status,
    )
    return {
        "registered_model_name": registered_model_name,
        "version": str(version),
        "from_status": previous_status,
        "to_status": normalized_status,
        "approval_status": normalized_approval,
        "approved_by": approver,
        "reason": reason,
        "event": event,
    }


def rollback_model_version(
    client: MlflowClient,
    registered_model_name: str,
    target_version: str | None,
    approver: str,
    reason: str,
    event_log_path: Path = MODEL_REGISTRY_EVENTS_PATH,
) -> dict[str, Any]:
    """Reativa uma versao arquivada ou a ultima candidata estavel."""
    if target_version is None:
        candidates = [
            version
            for version in _find_versions_by_status(client, registered_model_name, "Archived")
            if version.tags.get(ROLLBACK_CANDIDATE_TAG) == "true"
        ]
        if not candidates:
            raise ValueError("Nenhuma versao candidata a rollback foi encontrada.")
        target_version = str(candidates[0].version)

    current_production_versions = _find_versions_by_status(
        client,
        registered_model_name,
        "Production",
    )
    for current in current_production_versions:
        if str(current.version) == str(target_version):
            continue
        _transition_status(
            client,
            registered_model_name,
            str(current.version),
            "Archived",
            archive_existing_versions=False,
        )
        _set_model_version_tags(
            client,
            registered_model_name,
            str(current.version),
            {
                ROLLBACK_CANDIDATE_TAG: "true",
                "rolled_back_at_utc": utc_now(),
            },
        )

    result = promote_model_version(
        client=client,
        registered_model_name=registered_model_name,
        version=str(target_version),
        target_status="Production",
        approver=approver,
        reason=f"Rollback: {reason}",
        approval_status="approved",
        event_log_path=event_log_path,
    )
    _set_model_version_tags(
        client,
        registered_model_name,
        str(target_version),
        {
            APPROVAL_STATUS_TAG: "rolled_back",
            ROLLBACK_REASON_TAG: reason,
            ROLLBACK_CANDIDATE_TAG: "false",
        },
    )
    event = append_registry_event(
        {
            "event_type": "rollback",
            "registered_model_name": registered_model_name,
            "version": str(target_version),
            "from_status": result["from_status"],
            "to_status": "Production",
            "approval_status": "rolled_back",
            "approved_by": approver,
            "reason": reason,
        },
        event_log_path=event_log_path,
    )
    logger.info(
        "Rollback concluido no Model Registry: name={} version={} reason={}",
        registered_model_name,
        target_version,
        reason,
    )
    return {
        **result,
        "approval_status": "rolled_back",
        "event": event,
    }


def _build_client_from_args(args: argparse.Namespace) -> MlflowClient:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    return MlflowClient()


def main() -> None:
    parser = argparse.ArgumentParser(description="Governanca do MLflow Model Registry")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--registered-model-name",
        default=resolve_registered_model_name(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--target-status", choices=MODEL_STATUSES, default="Production")
    promote_parser.add_argument("--approver", required=True)
    promote_parser.add_argument("--reason", required=True)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--version", default=None)
    rollback_parser.add_argument("--approver", required=True)
    rollback_parser.add_argument("--reason", required=True)

    args = parser.parse_args()
    client = _build_client_from_args(args)
    if args.command == "promote":
        result = promote_model_version(
            client=client,
            registered_model_name=args.registered_model_name,
            version=args.version,
            target_status=args.target_status,
            approver=args.approver,
            reason=args.reason,
        )
    else:
        result = rollback_model_version(
            client=client,
            registered_model_name=args.registered_model_name,
            target_version=args.version,
            approver=args.approver,
            reason=args.reason,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
