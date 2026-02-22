from __future__ import annotations

from pathlib import Path

from src.pipeline import run_pipeline


def test_run_pipeline_writes_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        (input_dir / f"sample_{idx}.jpg").write_text("dummy", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "environment: test",
                "batch:",
                "  size: 2",
                "  max_retries: 2",
                "paths:",
                f"  input_dir: {input_dir.as_posix()}",
                f"  output_dir: {output_dir.as_posix()}",
                "logging:",
                "  level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    results = run_pipeline(config_path=config_path)

    assert len(results) == 3
    assert (output_dir / "predictions.jsonl").exists()
    assert {item["label"] for item in results}.issubset({"defect", "normal"})
