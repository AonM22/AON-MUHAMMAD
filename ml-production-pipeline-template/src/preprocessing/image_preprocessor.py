from __future__ import annotations

from pathlib import Path
from typing import List


def preprocess(images: List[Path]) -> List[str]:
    """Simulate preprocessing by annotating image paths."""
    return [f"{image}_processed" for image in images]
