from __future__ import annotations

from typing import Dict, List


def format_results(results: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """Format predictions into human-readable defect labels."""
    return [
        {
            "file": str(result["input"]),
            "label": "defect" if float(result["prediction_score"]) > 0.5 else "normal",
            "confidence": round(float(result["prediction_score"]), 3),
        }
        for result in results
    ]
