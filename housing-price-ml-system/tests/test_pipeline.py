from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src import api as api_module
from src.feature_engineering import engineer_features
from src.predict import predict_price
from src.preprocessing import build_preprocessor, select_training_columns
from src.train import train_and_select_best_model


def _sample_row() -> dict[str, object]:
    return {
        "property_type": "House",
        "location": "DHA Defence",
        "city": "Islamabad",
        "province_name": "Islamabad Capital",
        "baths": 4,
        "bedrooms": 4,
        "date_added": "2023-05-11",
        "Total_Area": 12.0,
        "price_in_lac": 120.0,
        "price": 12000000,
    }


def test_preprocessing_pipeline_handles_sample_input() -> None:
    frame = pd.DataFrame([_sample_row(), _sample_row()])
    engineered = engineer_features(frame, include_price_per_sqft=False)
    features = select_training_columns(engineered)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    assert transformed.shape[0] == 2


def test_train_produces_model_file(tmp_path: Path) -> None:
    rows = []
    for idx in range(30):
        row = _sample_row()
        row["location"] = "DHA Defence" if idx % 2 == 0 else "Bahria Town"
        row["bedrooms"] = 3 + (idx % 3)
        row["baths"] = 2 + (idx % 3)
        row["Total_Area"] = 8.0 + idx * 0.2
        row["price_in_lac"] = 60.0 + idx * 1.5
        row["price"] = int(row["price_in_lac"] * 100000)
        row["date_added"] = f"2023-05-{(idx % 28) + 1:02d}"
        rows.append(row)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    model_path = tmp_path / "best_model.pkl"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  csv_path: {csv_path.as_posix()}",
                "artifacts:",
                f"  model_output_path: {model_path.as_posix()}",
                f"  metrics_output_path: {metrics_path.as_posix()}",
                "training:",
                "  random_state: 42",
                "  cv_folds: 3",
                "models:",
                "  random_forest:",
                "    n_estimators: 20",
                "    max_depth: 6",
                "  gradient_boosting:",
                "    n_estimators: 20",
                "    learning_rate: 0.1",
                "  linear_regression: {}",
            ]
        ),
        encoding="utf-8",
    )

    train_and_select_best_model(config_path=config_path)
    assert model_path.exists()
    assert metrics_path.exists()


def test_predict_returns_numeric_price() -> None:
    payload = {
        "property_type": "House",
        "location": "DHA Defence",
        "city": "Islamabad",
        "province_name": "Islamabad Capital",
        "baths": 4,
        "bedrooms": 4,
        "date_added": "2023-05-11",
        "Total_Area": 12.0,
    }
    result = predict_price(payload, model_path="models/best_model.pkl")
    assert isinstance(result, float)


def test_api_predict_endpoint_status_and_structure(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "MODEL", object())
    monkeypatch.setattr(api_module, "predict_price", lambda *_args, **_kwargs: 123.456)

    client = TestClient(api_module.app)
    payload = {
        "property_type": "House",
        "location": "DHA Defence",
        "city": "Islamabad",
        "province_name": "Islamabad Capital",
        "baths": 4,
        "bedrooms": 4,
        "date_added": "2023-05-11",
        "Total_Area": 12.0,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_price_in_lac" in body
    assert isinstance(body["predicted_price_in_lac"], float)
