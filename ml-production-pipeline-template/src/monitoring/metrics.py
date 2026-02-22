from __future__ import annotations


def log_metrics(batch_size: int, success: bool = True) -> None:
    """Emit lightweight batch metrics for observability."""
    print(f"[METRICS] batch_size={batch_size}, success={success}")
