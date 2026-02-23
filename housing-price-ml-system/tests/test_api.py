from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_predict_validation_rejects_invalid_area() -> None:
    client = TestClient(app)
    bad_payload = {
        "property_type": "House",
        "location": "DHA Defence",
        "city": "Islamabad",
        "province_name": "Islamabad Capital",
        "baths": 4,
        "bedrooms": 4,
        "date_added": "2023-05-11",
        "Total_Area": 0,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
