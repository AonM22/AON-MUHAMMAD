from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    from src.ingestion.data_loader import load_batch
    from src.inference.model_runner import run_inference
    from src.monitoring.metrics import log_metrics
    from src.postprocessing.result_formatter import format_results
    from src.preprocessing.image_preprocessor import preprocess
    from src.utils.io import ensure_dir, write_jsonl
    from src.utils.logger import get_logger
    from src.utils.retry import retry
except ModuleNotFoundError:
    from ingestion.data_loader import load_batch
    from inference.model_runner import run_inference
    from monitoring.metrics import log_metrics
    from postprocessing.result_formatter import format_results
    from preprocessing.image_preprocessor import preprocess
    from utils.io import ensure_dir, write_jsonl
    from utils.logger import get_logger
    from utils.retry import retry


logger = get_logger("MLPipeline")


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def run_pipeline(config_path: str | Path = "src/config/config.yaml") -> List[Dict[str, float | str]]:
    """Execute a config-driven batch pipeline and return all formatted results."""
    config = _load_config(config_path)

    batch_size = int(config["batch"]["size"])
    max_retries = int(config["batch"].get("max_retries", 3))
    input_dir = Path(config["paths"]["input_dir"])
    output_dir = ensure_dir(config["paths"]["output_dir"])
    output_file = output_dir / "predictions.jsonl"

    global logger
    logger = get_logger("MLPipeline", level=str(config.get("logging", {}).get("level", "INFO")))

    all_results: List[Dict[str, float | str]] = []

    for batch in load_batch(input_dir=input_dir, batch_size=batch_size):
        logger.info("Processing batch of size %d", len(batch))

        def _process_batch() -> List[Dict[str, float | str]]:
            processed = preprocess(batch)
            predictions = run_inference(processed)
            return format_results(predictions)

        success = True
        try:
            formatted = retry(_process_batch, retries=max_retries, delay=1.0)
        except Exception:
            success = False
            formatted = []
            logger.exception("Batch failed after %d retries", max_retries)

        log_metrics(batch_size=len(batch), success=success)

        if formatted:
            all_results.extend(formatted)
            logger.info("Batch complete with %d predictions", len(formatted))

    write_jsonl(all_results, output_file)
    logger.info("Wrote %d predictions to %s", len(all_results), output_file)
    return all_results


if __name__ == "__main__":
    run_pipeline()
