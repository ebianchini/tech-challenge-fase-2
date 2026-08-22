from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from src.ml_project.config import DRIFT_REPORT_PATH, OPERATIONAL_METRICS_PATH
from src.ml_project.logging import logger

_METRICS_LOCK = Lock()


def record_operational_metric(event: str, **fields: Any) -> None:
    """Persiste um evento operacional em formato JSON Lines."""
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    OPERATIONAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_LOCK:
        with OPERATIONAL_METRICS_PATH.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        return 0.0

    minimum = min(reference_values.min(), current_values.min())
    maximum = max(reference_values.max(), current_values.max())
    if minimum == maximum:
        return 0.0
    bins = pd.interval_range(start=minimum, end=maximum, periods=10)
    reference_distribution = pd.cut(reference_values, bins=bins).value_counts(normalize=True)
    current_distribution = pd.cut(current_values, bins=bins).value_counts(normalize=True)
    total = 0.0
    for bucket in bins:
        reference_ratio = max(float(reference_distribution.get(bucket, 0.0)), 1e-6)
        current_ratio = max(float(current_distribution.get(bucket, 0.0)), 1e-6)
        total += (current_ratio - reference_ratio) * math.log(current_ratio / reference_ratio)
    return float(total)


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, Any]:
    """Compara distribuicoes numericas e categoricas entre dois datasets."""
    missing_columns = sorted(set(reference.columns) - set(current.columns))
    if missing_columns:
        raise ValueError(f"Dataset atual sem colunas da referencia: {missing_columns}")

    features: dict[str, dict[str, Any]] = {}
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            score = _population_stability_index(reference[column], current[column])
            features[column] = {
                "method": "psi",
                "score": score,
                "drifted": score >= threshold,
            }
        else:
            reference_distribution = reference[column].astype(str).value_counts(normalize=True)
            current_distribution = current[column].astype(str).value_counts(normalize=True)
            categories = set(reference_distribution.index) | set(current_distribution.index)
            score = sum(
                abs(
                    float(reference_distribution.get(category, 0.0))
                    - float(current_distribution.get(category, 0.0))
                )
                for category in categories
            )
            features[column] = {
                "method": "distribution_distance",
                "score": float(score),
                "drifted": score >= threshold,
            }

    drifted_features = [column for column, result in features.items() if result["drifted"]]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "threshold": threshold,
        "drifted": bool(drifted_features),
        "drifted_features": drifted_features,
        "features": features,
    }


def save_drift_report(report: dict[str, Any], path: str | Path = DRIFT_REPORT_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    record_operational_metric(
        "drift_check",
        drifted=report["drifted"],
        drifted_features=report["drifted_features"],
        report_path=str(output_path),
    )
    logger.info("Relatorio de drift persistido em {}", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara datasets para detectar drift")
    parser.add_argument("reference_csv")
    parser.add_argument("current_csv")
    parser.add_argument("--output", default=str(DRIFT_REPORT_PATH))
    args = parser.parse_args()
    report = compute_drift_report(
        pd.read_csv(args.reference_csv),
        pd.read_csv(args.current_csv),
    )
    save_drift_report(report, args.output)


if __name__ == "__main__":
    main()