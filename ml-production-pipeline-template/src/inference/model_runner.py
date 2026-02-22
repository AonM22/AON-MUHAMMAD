from __future__ import annotations

import random
from typing import Dict, List


def run_inference(batch: List[str]) -> List[Dict[str, float | str]]:
    """Run dummy inference and return prediction scores."""
    results: List[Dict[str, float | str]] = []
    for item in batch:
        results.append({
            "input": item,
            "prediction_score": random.random(),
        })
    return results
