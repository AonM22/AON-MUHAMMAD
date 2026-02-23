from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from src.feature_engineering import engineer_features
    from src.logger import get_logger
    from src.preprocessing import TARGET_COLUMN, select_training_columns
except ModuleNotFoundError:
    from feature_engineering import engineer_features
    from logger import get_logger
    from preprocessing import TARGET_COLUMN, select_training_columns


LOGGER = get_logger("housing-price-evaluation")


def metrics_to_frame(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert model metrics dictionary into a sorted DataFrame."""
    frame = pd.DataFrame.from_dict(metrics, orient="index").reset_index().rename(columns={"index": "model"})
    return frame.sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)


def save_metrics(metrics_payload: Dict[str, Any], output_path: str | Path) -> None:
    """Persist metrics payload to JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def print_model_comparison(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print clean model comparison table."""
    frame = metrics_to_frame(metrics)
    print("\nModel Comparison (5-fold CV):")
    print(frame.to_string(index=False))


def evaluate_saved_model(
    model_path: str | Path = "models/best_model.pkl",
    dataset_path: str | Path = "data/Islamabad houses.csv",
    metrics_path: str | Path = "models/metrics.json",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate persisted model on holdout split and save MAE/RMSE/R2 metrics."""
    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)

    engineered = engineer_features(data, include_price_per_sqft=False)
    features = select_training_columns(engineered)
    target = pd.to_numeric(engineered[TARGET_COLUMN], errors="coerce")

    valid_mask = target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    predictions = model.predict(x_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    payload: dict[str, Any] = {"mae": mae, "rmse": rmse, "r2": r2}
    save_metrics(payload, metrics_path)

    LOGGER.info("Evaluation complete")
    LOGGER.info("MAE: %.4f", mae)
    LOGGER.info("RMSE: %.4f", rmse)
    LOGGER.info("R2: %.4f", r2)
    LOGGER.info("Saved evaluation metrics to: %s", metrics_path)
    return {"mae": mae, "rmse": rmse, "r2": r2}


if __name__ == "__main__":
    evaluate_saved_model()
