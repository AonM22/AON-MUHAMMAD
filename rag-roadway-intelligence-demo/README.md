# Roadway Intelligence RAG Demo

## Overview
This repository demonstrates a fully local, production-style retrieval-augmented generation (RAG) system for roadway pavement intelligence. It is designed around real-world RAG challenges: maintaining retrieval quality as data grows, preserving source traceability for auditability, and reducing hallucination risk by grounding generation in retrieved evidence.

A modular architecture is used so each stage can evolve independently. In practice, teams often replace embedding models, retrieval logic, or generation backends over time. Clear module boundaries reduce operational risk and make the system easier to test, maintain, and scale.

## Features
- Metadata-aware ingestion from local `.txt` files
- Configurable chunking with overlap controls
- Local embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- FAISS (`IndexFlatL2`) vector indexing with persistence
- Retrieval with configurable similarity threshold
- Prompt construction that enforces context grounding
- Citation support with `doc_id::chunk_id` traceability
- Confidence scoring from retrieved chunk similarity
- Fully local execution with Ollama (and local fallback mode)

## Repository Layout
```text
rag-roadway-intelligence-demo/
|-- data/sample_documents/
|-- src/
|-- evaluation/
|-- notebooks/
|-- architecture.png
|-- requirements.txt
`-- README.md
```

## Installation
```bash
pip install -r requirements.txt
```

Optional Ollama setup:
```bash
ollama pull mistral
ollama serve
```

## Quickstart
Build/load the local FAISS index and run inference:
```bash
python src/main.py --query "What defines high severity longitudinal cracking?"
```

For cleaner terminal output:
```bash
python src/main.py --query "What defines high severity longitudinal cracking?" --quiet
```

## Example Inference
Command used:
```bash
python src/main.py --query "What defines high severity longitudinal cracking?"
```

Example output (abbreviated):
```text
Answer:
High severity longitudinal cracking is defined as an average crack width greater
than 0.75 inch, and may include moderate to severe spalling, pumping, breakup,
or associated rutting.

Evidence:
- From DOT guideline sections on longitudinal crack severity thresholds.

Sources:
- inspection_report::inspection_report_chunk_0001
- pavement_guidelines::pavement_guidelines_chunk_0000
- pavement_guidelines::pavement_guidelines_chunk_0001

Confidence:
0.55
```

## CLI Options
- `--query`: user question (if omitted, interactive prompt is shown)
- `--top-k`: number of retrieved chunks (default `4`)
- `--similarity-threshold`: retrieval filter cutoff (default `0.35`)
- `--rebuild`: force index rebuild from source documents
- `--quiet`: minimal logs
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`

## Evaluation Seed Queries
Example evaluation prompts are provided in `evaluation/sample_queries.json`.

## Production Considerations
- Index persistence: version FAISS index and metadata together with the corpus snapshot.
- Re-embedding strategy: rebuild embeddings whenever corpus content, chunking policy, or model changes.
- Scaling to OpenSearch: keep retriever interface stable so the vector backend can be swapped.
- Caching layer: cache query embeddings and top retrievals for repeated operational questions.
- Multi-project corpora: isolate namespaces (district, route class, asset family) to limit retrieval drift.

## Notes
- First run may download the embedding model from Hugging Face.
- If Ollama is unavailable, the pipeline falls back to extractive response mode with citations.
