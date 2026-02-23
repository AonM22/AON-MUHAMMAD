from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

try:
    from src.evaluate import print_model_comparison, save_metrics
    from src.feature_engineering import drop_unused_columns, engineer_features
    from src.preprocessing import (
        TARGET_COLUMN,
        build_preprocessor,
        load_dataset,
        select_training_columns,
        summarize_missing_values,
    )
except ModuleNotFoundError:
    from evaluate import print_model_comparison, save_metrics
    from feature_engineering import drop_unused_columns, engineer_features
    from preprocessing import TARGET_COLUMN, build_preprocessor, load_dataset, select_training_columns, summarize_missing_values


def load_train_config(config_path: str | Path) -> Dict[str, Any]:
    """Load training config from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def get_candidate_models(config: Dict[str, Any]) -> Dict[str, object]:
    """Return candidate regressors for comparison."""
    random_state = int(config["training"]["random_state"])
    model_cfg = config.get("models", {})
    rf_cfg = model_cfg.get("random_forest", {})
    gb_cfg = model_cfg.get("gradient_boosting", {})

    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 300)),
            max_depth=rf_cfg.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=int(gb_cfg.get("n_estimators", 100)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.1)),
            random_state=random_state,
        ),
    }


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering and return X/y suitable for modeling."""
    engineered = engineer_features(df, include_price_per_sqft=False)

    # Leakage guard: target-derived fields are excluded from train-time features.
    leakage_columns = ["price", "price_per_sqft"]
    metadata_columns = ["Unnamed: 0", "property_id", "location_id", "page_url"]

    cleaned = drop_unused_columns(
        engineered,
        columns=[TARGET_COLUMN, *leakage_columns, *metadata_columns],
    )
    features = select_training_columns(cleaned)
    target = engineered[TARGET_COLUMN]
    return features, target


def train_and_select_best_model(
    config_path: str | Path = "configs/train.yaml",
) -> None:
    """Train candidate models with 5-fold CV, save best model and metrics."""
    config = load_train_config(config_path)
    csv_path = str(config["data"]["csv_path"])
    model_output_path = str(config["artifacts"]["model_output_path"])
    metrics_output_path = str(config["artifacts"]["metrics_output_path"])
    random_state = int(config["training"]["random_state"])
    cv_folds = int(config["training"].get("cv_folds", 5))

    df = load_dataset(csv_path)
    missing_summary = summarize_missing_values(df)

    features, target = prepare_training_data(df)
    preprocessor = build_preprocessor(features)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    model_metrics: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for model_name, model in get_candidate_models(config=config).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        model_metrics[model_name] = {
            "cv_mae_mean": float(-np.mean(scores["test_mae"])),
            "cv_mae_std": float(np.std(-scores["test_mae"])),
            "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
            "cv_rmse_std": float(np.std(-scores["test_rmse"])),
        }

        pipeline.fit(features, target)
        fitted_pipelines[model_name] = pipeline

    best_model_name = min(model_metrics, key=lambda name: model_metrics[name]["cv_rmse_mean"])
    best_pipeline = fitted_pipelines[best_model_name]

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metrics_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "missing_summary": missing_summary,
        "target_column": TARGET_COLUMN,
        "training_config": config["training"],
        "best_model": best_model_name,
        "metrics": model_metrics,
    }

    save_metrics(metrics_payload, metrics_output_path)
    print_model_comparison(model_metrics)
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train housing price model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config YAML")
    args = parser.parse_args()
    train_and_select_best_model(config_path=args.config)
