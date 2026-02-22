from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List

from chunking import chunk_documents
from embeddings import EmbeddingModel
from generator import OllamaGenerator, extractive_fallback
from ingestion import load_txt_documents
from prompt_builder import build_prompt
from retriever import Retriever
from vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end local RAG pipeline with citations and confidence scoring."""

    def __init__(
        self,
        data_dir: str | Path,
        index_path: str | Path,
        metadata_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_model: str = "mistral",
        similarity_threshold: float = 0.35,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.vector_store = FaissVectorStore(index_path=index_path, metadata_path=metadata_path)
        self.retriever = Retriever(self.vector_store, similarity_threshold=similarity_threshold)
        self.generator = OllamaGenerator(model_name=ollama_model)

    def build_or_load_index(self, rebuild: bool = False) -> None:
        if not rebuild:
            try:
                self.vector_store.load()
                logger.info("Loaded existing FAISS index")
                return
            except FileNotFoundError:
                logger.info("No existing index found. Building a new index.")

        documents = load_txt_documents(self.data_dir)
        if not documents:
            raise RuntimeError(f"No .txt documents found in {self.data_dir}")

        chunks = chunk_documents(documents)
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        metadata: List[Dict[str, str]] = []
        for chunk in chunks:
            metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "content": chunk["content"],
                }
            )

        self.vector_store.add_embeddings(embeddings, metadata)
        self.vector_store.save()
        logger.info("Built and persisted index with %d chunks", len(chunks))

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, object]:
        query_embedding = self.embedding_model.encode([user_query])[0]
        retrieved_chunks = self.retriever.retrieve(query_embedding=query_embedding, top_k=top_k)

        prompt = build_prompt(user_query, retrieved_chunks)

        try:
            answer = self.generator.generate(prompt)
        except RuntimeError:
            answer = extractive_fallback(user_query, retrieved_chunks)

        confidence = float(mean([item["score"] for item in retrieved_chunks])) if retrieved_chunks else 0.0
        sources = [f"{item['doc_id']}::{item['chunk_id']}" for item in retrieved_chunks]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }
