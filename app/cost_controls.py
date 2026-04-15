"""Runtime cost controls for paid external APIs."""

from __future__ import annotations

import threading
import time

_WINDOW_COUNTERS: dict[tuple[str, int], dict[str, int]] = {}
_WINDOW_COUNTERS_LOCK = threading.Lock()


def _window_bucket(window_s: int, now: float | None = None) -> int:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    current = time.time() if now is None else now
    return int(current // window_s)


def allow_call(bucket: str, limit: int, window_s: int, *, now: float | None = None) -> bool:
    """Return True and consume one call if the budget window still has capacity."""
    return allow_calls([(bucket, limit, window_s)], now=now)


def allow_calls(
    limits: list[tuple[str, int, int]],
    *,
    now: float | None = None,
) -> bool:
    """Return True and consume all counters only if every budget still has capacity."""
    if not limits:
        return True

    current = time.time() if now is None else now
    prepared: list[tuple[tuple[str, int], int, int]] = []
    for bucket, limit, window_s in limits:
        if limit <= 0:
            return False
        prepared.append(((bucket, window_s), _window_bucket(window_s, now=current), limit))

    with _WINDOW_COUNTERS_LOCK:
        touched: list[dict[str, int]] = []
        for counter_key, bucket_id, limit in prepared:
            counter = _WINDOW_COUNTERS.setdefault(counter_key, {"bucket": bucket_id, "count": 0})
            if counter["bucket"] != bucket_id:
                counter["bucket"] = bucket_id
                counter["count"] = 0
            if counter["count"] >= limit:
                return False
            touched.append(counter)

        for counter in touched:
            counter["count"] += 1
        return True
