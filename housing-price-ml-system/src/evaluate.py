from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


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
