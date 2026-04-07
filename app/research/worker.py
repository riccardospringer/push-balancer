"""app/research/worker.py — Autonomer Research Worker Thread.

Bündelt den Research-Worker und alle Hilfsfunktionen aus push-balancer-server.py.
Migration über Compat-Shim — direkte Migration folgt schrittweise.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("push-balancer")

# ── Globaler Research State (module-level) ─────────────────────────────────
_research_state: dict = {
    "push_data": [],
    "findings": {},
    "live_rules": [],
    "live_rules_version": 0,
    "rolling_accuracy": 0.0,
    "ticker_entries": [],
    "last_analysis": 0,
    "cutoff_24h": 0,
    "week_comparison": {},
    "research_memory": {},
    "research_modifiers": {},
    "_stacking_counter": 0,
    "_worker_first_log": True,
}
_research_state_lock = threading.Lock()

# ── Health State ───────────────────────────────────────────────────────────
_health_state: dict = {
    "status": "starting",
    "uptime_start": time.time(),
    "last_check": 0,
    "checks_ok": 0,
    "checks_fail": 0,
    "endpoints": {},
}

# ── Feed Cache ─────────────────────────────────────────────────────────────
_feed_cache: dict = {
    "competitors": {"data": None, "ts": 0},
    "international": {"data": None, "ts": 0},
    "sport_competitors": {"data": None, "ts": 0},
    "sport_europa": {"data": None, "ts": 0},
    "sport_global": {"data": None, "ts": 0},
}
_feed_cache_lock = threading.Lock()
_FEED_CACHE_TTL: int = 300  # 5 Minuten

# ── XOR Perf Cache ─────────────────────────────────────────────────────────
_xor_perf_cache: dict = {
    "word_perf": {},
    "cat_hour_perf": {},
    "eil_perf": {},
    "global_avg": 4.77,
    "built_at": 0,
}
_xor_perf_lock = threading.Lock()


def run_autonomous_analysis() -> None:
    """Analysiert Push-Daten autonom — wird vom Research-Worker alle 20s aufgerufen."""
    try:
        from push_balancer_server_compat import _run_autonomous_analysis  # type: ignore
        _run_autonomous_analysis()
    except ImportError:
        log.warning("[research] run_autonomous_analysis: Legacy-Import fehlgeschlagen")


def get_cached_feeds(feed_type: str) -> dict | list:
    """Liefert gecachte Feeds aus dem Background-Cache."""
    with _feed_cache_lock:
        entry = _feed_cache.get(feed_type, {})
        if entry.get("data") and (time.time() - entry.get("ts", 0)) < _FEED_CACHE_TTL * 3:
            return entry["data"]
    return {}


def update_residual_corrector() -> None:
    """Aktualisiert den Online-Bias-Korrektor aus der Prediction-Log-DB."""
    try:
        from push_balancer_server_compat import _update_residual_corrector  # type: ignore
        _update_residual_corrector()
    except ImportError:
        log.warning("[research] update_residual_corrector: Legacy-Import fehlgeschlagen")


def monitoring_tick() -> None:
    """Monitoring-Tick: prüft Drift, MAE-Spikes, A/B-Ergebnisse etc."""
    try:
        from push_balancer_server_compat import _monitoring_tick  # type: ignore
        _monitoring_tick()
    except ImportError:
        log.warning("[research] monitoring_tick: Legacy-Import fehlgeschlagen")


def build_xor_perf_cache() -> None:
    """Baut/aktualisiert den XOR-Performance-Cache für /api/competitor-xor."""
    try:
        from push_balancer_server_compat import _build_xor_perf_cache  # type: ignore
        _build_xor_perf_cache()
    except ImportError:
        log.warning("[research] build_xor_perf_cache: Legacy-Import fehlgeschlagen")
