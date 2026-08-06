"""Relay fresh local Push Statistics snapshots to the Render receiver."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.request

try:
    import certifi

    _TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - certifi is installed in production
    _TLS_CONTEXT = ssl.create_default_context()

LOCAL = os.environ.get("PUSH_API_BASE_LOCAL", "http://127.0.0.1:8050").rstrip("/")
RENDER = os.environ.get(
    "RENDER_SYNC_URL",
    "https://push-balancer.onrender.com",
).rstrip("/")
SECRET = os.environ.get("PUSH_SYNC_SECRET", "")
INTERVAL = max(30, int(os.environ.get("SYNC_INTERVAL", "30")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sync] %(message)s")
log = logging.getLogger(__name__)


def _fetch(path: str, *, timeout: int = 20, method: str = "GET"):
    try:
        request = urllib.request.Request(
            f"{LOCAL}{path}",
            headers={"Accept": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as exc:
        log.warning("Local fetch failed for %s: %s", path, type(exc).__name__)
        return None


def _push(
    messages: list,
    channels: list,
    *,
    source: str,
    snapshot_ts: float,
) -> bool:
    payload = json.dumps(
        {
            "secret": SECRET,
            "messages": messages,
            "channels": channels,
            "source": source,
            "snapshotTs": snapshot_ts,
        }
    ).encode()
    request = urllib.request.Request(
        f"{RENDER}/api/pushes/sync",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=20,
            context=_TLS_CONTEXT,
        ) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or not result.get("history_authoritative"):
            log.error("Sync failed: receiver did not confirm authoritative persistence")
            return False
        log.info("Sync OK: %d messages, %d channels", len(messages), len(channels))
        return True
    except Exception as exc:
        log.error("Sync failed: %s", type(exc).__name__)
        return False


def run_once() -> bool:
    refresh = _fetch("/api/pushes/refresh", method="POST", timeout=60)
    if not isinstance(refresh, dict) or not refresh.get("history_authoritative"):
        log.warning("Local Push Statistics snapshot is not authoritative")
        return False
    message_data = _fetch(
        "/api/push/statistics/message?relayCacheOnly=1",
    )
    channel_data = _fetch(
        "/api/push/statistics/message/channels?relayCacheOnly=1",
    )
    messages = (
        list(message_data.get("messages") or [])
        if isinstance(message_data, dict)
        else list(message_data or [])
        if isinstance(message_data, list)
        else []
    )
    channels = list(channel_data) if isinstance(channel_data, list) else []
    if not messages and not channels:
        log.warning("No local Push Statistics data available")
        return False
    source = (
        str(message_data.get("_source") or "unknown")
        if isinstance(message_data, dict)
        else "unknown"
    )
    snapshot_ts = (
        float(message_data.get("_snapshotTs") or 0.0)
        if isinstance(message_data, dict)
        else 0.0
    )
    if source not in {"live", "relay"} or snapshot_ts <= 0:
        log.warning("Local snapshot provenance is unavailable")
        return False
    return _push(
        messages,
        channels,
        source=source,
        snapshot_ts=snapshot_ts,
    )


def run() -> None:
    if not SECRET:
        raise RuntimeError("PUSH_SYNC_SECRET is required")
    if not RENDER.startswith("https://"):
        raise RuntimeError("RENDER_SYNC_URL must use HTTPS")
    log.info("Relay started; interval=%ds", INTERVAL)
    while True:
        run_once()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    run()
