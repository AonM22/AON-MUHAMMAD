from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Roadway RAG Demo")
    parser.add_argument("--query", type=str, default=None, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Minimum similarity score required to keep a retrieved chunk",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force index rebuild")
    parser.add_argument("--data-dir", type=str, default="data/sample_documents")
    parser.add_argument("--index-path", type=str, default="artifacts/faiss.index")
    parser.add_argument("--metadata-path", type=str, default="artifacts/metadata.json")
    parser.add_argument("--ollama-model", type=str, default="mistral")
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Application log level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Force minimal logs (equivalent to --log-level ERROR)",
    )
    return parser


def configure_runtime(log_level: str) -> None:
    """Reduce third-party logging noise while keeping app logs configurable."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    warnings.filterwarnings(
        "ignore",
        message=".*huggingface_hub.*symlinks.*",
        category=UserWarning,
    )

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    for noisy_logger in (
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "huggingface_hub",
        "transformers",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)


def main() -> None:
    console = Console()
    args = build_parser().parse_args()
    effective_log_level = "ERROR" if args.quiet else args.log_level
    configure_runtime(effective_log_level)

    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        data_dir=Path(args.data_dir),
        index_path=Path(args.index_path),
        metadata_path=Path(args.metadata_path),
        ollama_model=args.ollama_model,
        similarity_threshold=args.similarity_threshold,
    )

    pipeline.build_or_load_index(rebuild=args.rebuild)

    user_query = args.query or console.input("[bold cyan]Enter your question:[/bold cyan] ").strip()
    if not user_query:
        raise ValueError("A query is required.")

    result = pipeline.query(user_query=user_query, top_k=args.top_k)

    console.print(Panel.fit(result["answer"], title="Answer", border_style="green"))

    sources_table = Table(title="Sources")
    sources_table.add_column("Source", style="yellow")
    for src in result["sources"]:
        sources_table.add_row(str(src))
    if not result["sources"]:
        sources_table.add_row("No sources retrieved")
    console.print(sources_table)

    console.print(Panel.fit(f"{result['confidence']:.4f}", title="Confidence", border_style="magenta"))


if __name__ == "__main__":
    main()
