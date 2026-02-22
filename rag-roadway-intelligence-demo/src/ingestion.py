from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_txt_documents(directory: str | Path) -> List[Dict[str, str]]:
    """Load all .txt documents from a directory with metadata."""
    base_path = Path(directory).expanduser().resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    documents: List[Dict[str, str]] = []
    txt_files = sorted(base_path.glob("*.txt"))
    logger.info("Discovered %d text files in %s", len(txt_files), base_path)

    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("Skipping empty file: %s", path)
                continue

            documents.append(
                {
                    "doc_id": path.stem,
                    "content": content,
                    "source_path": str(path),
                }
            )
        except Exception as exc:
            logger.exception("Failed to read %s: %s", path, exc)

    logger.info("Loaded %d documents", len(documents))
    return documents
