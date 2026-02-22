from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return a Path object."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write records to newline-delimited JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return output
