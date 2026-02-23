from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

try:
    from src.feature_engineering import engineer_features
    from src.preprocessing import select_training_columns
except ModuleNotFoundError:
    from feature_engineering import engineer_features
    from preprocessing import select_training_columns


DEFAULT_MODEL_PATH = Path("models/best_model.pkl")


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a persisted sklearn pipeline model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def predict_price(input_payload: Dict[str, Any], model_path: str | Path = DEFAULT_MODEL_PATH) -> float:
    """Predict housing price from a single input payload."""
    model = load_model(model_path)
    input_df = pd.DataFrame([input_payload])
    engineered = engineer_features(input_df, include_price_per_sqft=False)
    features = select_training_columns(engineered)
    prediction = model.predict(features)[0]
    return float(prediction)


if __name__ == "__main__":
    sample = {
        "property_type": "House",
        "location": "DHA Defence",
        "city": "Islamabad",
        "province_name": "Islamabad Capital",
        "baths": 4,
        "bedrooms": 4,
        "date_added": "2023-05-11",
        "Total_Area": 12.0,
    }
    print(predict_price(sample))
