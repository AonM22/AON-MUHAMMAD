from __future__ import annotations

from typing import Dict, List


def _chunk_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[List[str]] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        window = words[start:end]
        if not window:
            continue
        chunks.append(window)
        if end >= len(words):
            break

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """Split documents into overlapping word chunks while preserving metadata."""
    all_chunks: List[Dict[str, str]] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        words = doc["content"].split()
        windows = _chunk_words(words, chunk_size=chunk_size, overlap=overlap)

        for idx, window in enumerate(windows):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                    "doc_id": doc_id,
                    "content": " ".join(window),
                }
            )

    return all_chunks
