from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class FaissVectorStore:
    """FAISS-backed vector store with metadata persistence."""

    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: List[Dict[str, str]] = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata lengths must match")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])

        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, object]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, top_k)

        results: List[Dict[str, object]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = dict(self.metadata[idx])
            result["distance"] = float(distance)
            results.append(result)

        return results

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save empty index")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
