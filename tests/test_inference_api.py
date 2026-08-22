from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from src.ml_project import api


def valid_payload() -> dict[str, object]:
    return {
        "contract_version": "1.0",
        "instances": [
            {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 1,
                "ProductRelated_Duration": 0.0,
                "BounceRates": 0.2,
                "ExitRates": 0.2,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Feb",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        ],
    }


def test_health_endpoint_reports_service_status() -> None:
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["contract_version"] == "1.0"
    assert "model_available" in payload
    assert "metadata_available" in payload


def test_predict_endpoint_returns_standard_response(monkeypatch) -> None:
    def fake_predict(model_path=None, dataframe: pd.DataFrame | None = None, metadata_path=None):
        assert model_path == api.DEFAULT_MODEL_PATH
        assert metadata_path is None
        assert dataframe is not None
        assert dataframe.shape[0] == 1
        return pd.Series([1], name="predicted_revenue")

    monkeypatch.setattr(api, "predict", fake_predict)
    client = TestClient(api.app)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload == {
        "contract_version": "1.0",
        "model_version": api.resolve_api_model_version(),
        "predictions": [{"prediction_id": "0", "predicted_revenue": 1}],
    }


def test_predict_endpoint_rejects_extra_fields() -> None:
    payload = valid_payload()
    payload["instances"][0]["Revenue"] = True  # type: ignore[index]
    client = TestClient(api.app)

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT_SCHEMA"
