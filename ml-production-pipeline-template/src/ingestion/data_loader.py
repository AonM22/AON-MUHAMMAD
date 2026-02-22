from __future__ import annotations

from pathlib import Path
from typing import Generator, List


def load_batch(input_dir: str | Path, batch_size: int) -> Generator[List[Path], None, None]:
    """Yield image files in fixed-size batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    files = sorted(Path(input_dir).glob("*.jpg"))
    for idx in range(0, len(files), batch_size):
        yield files[idx : idx + batch_size]
