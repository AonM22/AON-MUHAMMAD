from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry(func: Callable[[], T], retries: int = 3, delay: float = 1.0) -> T:
    """Retry a function with fixed delay; raise final exception on exhaustion."""
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)

    raise RuntimeError("Retry exited unexpectedly")
