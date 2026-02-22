from __future__ import annotations

from typing import Dict, List


def build_prompt(query: str, retrieved_chunks: List[Dict[str, object]]) -> str:
    """Build a grounded prompt with explicit context and source IDs."""
    if not retrieved_chunks:
        return (
            "Answer the question using ONLY the context below.\n"
            "If the context is insufficient, say you do not have enough information.\n\n"
            "Context:\n[No retrieved context]\n\n"
            f"Question: {query}\n\n"
            "Required output:\n"
            "1) Direct answer focused only on the question scope\n"
            "2) Supporting evidence directly tied to the same scope\n"
            "3) Sources"
        )

    context_blocks = []
    source_ids = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source_id = f"{chunk['doc_id']}::{chunk['chunk_id']}"
        source_ids.append(source_id)
        context_blocks.append(
            f"[Chunk {idx}]\n"
            f"Source: {source_id}\n"
            f"Content: {chunk['content']}"
        )

    joined_context = "\n\n".join(context_blocks)
    joined_sources = "\n".join(f"- {src}" for src in source_ids)

    return (
        "Answer the question using ONLY the context below.\n"
        "If the answer cannot be derived from the context, explicitly say so.\n\n"
        "Constraints:\n"
        "- Stay strictly within the user question scope.\n"
        "- Do not add extra distress types, standards, or side notes unless the question asks for them.\n"
        "- If the question asks for a definition, return only the definition and threshold details.\n"
        "- Cite only sources that directly support the answer.\n\n"
        f"Context:\n{joined_context}\n\n"
        f"Question: {query}\n\n"
        "Return format:\n"
        "Answer: <concise grounded answer>\n"
        "Evidence: <short evidence summary>\n"
        "Sources:\n"
        f"{joined_sources}"
    )
