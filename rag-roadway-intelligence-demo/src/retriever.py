from __future__ import annotations

from typing import Dict, List

import numpy as np

from vector_store import FaissVectorStore


class Retriever:
    """Vector retriever with similarity threshold filtering."""

    def __init__(self, vector_store: FaissVectorStore, similarity_threshold: float = 0.35) -> None:
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, object]]:
        raw_results = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
        filtered: List[Dict[str, object]] = []

        for item in raw_results:
            similarity = self._distance_to_similarity(float(item["distance"]))
            if similarity < self.similarity_threshold:
                continue

            filtered.append(
                {
                    "chunk_id": str(item["chunk_id"]),
                    "doc_id": str(item["doc_id"]),
                    "content": str(item["content"]),
                    "score": float(similarity),
                }
            )

        filtered.sort(key=lambda x: float(x["score"]), reverse=True)
        return filtered
