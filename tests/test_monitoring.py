import pandas as pd

from src.ml_project.monitoring import compute_drift_report


def test_drift_report_detects_numeric_and_categorical_changes() -> None:
    reference = pd.DataFrame({"value": [0, 0, 0, 1], "kind": ["a", "a", "a", "b"]})
    current = pd.DataFrame({"value": [10, 10, 10, 11], "kind": ["b", "b", "b", "b"]})

    report = compute_drift_report(reference, current, threshold=0.2)

    assert report["drifted"] is True
    assert set(report["drifted_features"]) == {"value", "kind"}