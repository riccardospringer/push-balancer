"""app/ml/stats.py — History-Statistiken für GBRT- und LightGBM-Modelle.

Stellt _gbrt_build_history_stats und _ml_build_stats bereit.

IMPLEMENTIERUNGSHINWEIS:
    Vollständige Implementierungen aus push-balancer-server.py:
    - _gbrt_build_history_stats: Zeilen 2130–2800 (Bayesian-smoothed Aggregat-Stats
      für Kategorie, Stunde, Wochentag, Cat×Hour, Cat×Weekday, TF-IDF-Index,
      Entity-Index, XOR-Perf-Cache, etc.)
    - _ml_build_stats: Zeile 6523 (vereinfachte Stats für LightGBM)
"""
from __future__ import annotations

import logging

log = logging.getLogger("push-balancer")


def gbrt_build_history_stats(pushes: list, target_ts: int = 0) -> dict:
    """Baut vorberechnete Aggregat-Statistiken aus Push-Historie.

    Args:
        pushes: Liste von Push-Dicts aus push_db_load_all()
        target_ts: Optional — berechne nur Stats bis zu diesem Timestamp

    Returns:
        Dict mit aggregierten Statistiken (cat_stats, hour_stats, cat_hour_stats, etc.)

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py Zeilen 2130–2800
        (_gbrt_build_history_stats) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _gbrt_build_history_stats  # type: ignore
        return _gbrt_build_history_stats(pushes, target_ts)
    except ImportError:
        pass
    log.warning("[stats] gbrt_build_history_stats: Legacy-Import fehlgeschlagen")
    return {}


def ml_build_stats(pushes: list) -> dict:
    """Baut vereinfachte ML-Statistiken für LightGBM-Training.

    Args:
        pushes: Liste von Push-Dicts

    Returns:
        Dict mit ML-Stats

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py Zeile 6523
        (_ml_build_stats) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _ml_build_stats  # type: ignore
        return _ml_build_stats(pushes)
    except ImportError:
        pass
    log.warning("[stats] ml_build_stats: Legacy-Import fehlgeschlagen")
    return {}
