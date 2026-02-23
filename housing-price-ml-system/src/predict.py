from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from pydantic import ValidationError

try:
    from src.feature_engineering import engineer_features
    from src.schema import HousePriceInput
    from src.preprocessing import select_training_columns
except ModuleNotFoundError:
    from feature_engineering import engineer_features
    from schema import HousePriceInput
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
    validated_payload = HousePriceInput(**input_payload).model_dump()
    input_df = pd.DataFrame([validated_payload])
    engineered = engineer_features(input_df, include_price_per_sqft=False)
    features = select_training_columns(engineered)
    prediction = model.predict(features)[0]
    return float(prediction)


def parse_input_payload(raw_json: str | None, json_file: str | None) -> Dict[str, Any]:
    """Parse prediction payload from CLI JSON string or JSON file."""
    if raw_json:
        return json.loads(raw_json)
    if json_file:
        return json.loads(Path(json_file).read_text(encoding="utf-8"))
    raise ValueError("Provide either --input-json or --input-file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict house price in lacs.")
    parser.add_argument("--input-json", type=str, default=None, help="Raw JSON payload string.")
    parser.add_argument("--input-file", type=str, default=None, help="Path to input JSON payload file.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    try:
        payload = parse_input_payload(args.input_json, args.input_file)
        prediction = predict_price(payload, model_path=args.model_path)
    except ValidationError as exc:
        raise SystemExit(f"Schema validation failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Prediction failed: {exc}") from exc

    print(json.dumps({"predicted_price_in_lac": prediction}, indent=2))
