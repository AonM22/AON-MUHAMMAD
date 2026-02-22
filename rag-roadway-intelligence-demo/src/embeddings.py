from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around SentenceTransformer with production-style error handling."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded embedding model: %s", model_name)
        except Exception as exc:
            logger.exception("Failed to load embedding model %s", model_name)
            raise RuntimeError(f"Could not initialize embedding model '{model_name}'") from exc

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text inputs into float32 embeddings."""
        if not texts:
            raise ValueError("texts must not be empty")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise RuntimeError("Failed to encode texts") from exc

        return np.asarray(embeddings, dtype=np.float32)
