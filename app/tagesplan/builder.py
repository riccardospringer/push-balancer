"""app/tagesplan/builder.py — Tagesplan-Aufbau (ML-gestützt).

Bündelt _ml_build_tagesplan, _ml_build_tagesplan_inner und
_build_tagesplan_retro aus push-balancer-server.py.

IMPLEMENTIERUNGSHINWEIS:
    Vollständige Implementierungen aus push-balancer-server.py:
    - _ml_build_tagesplan(): Wrapper mit Background-Flag + Caching (Zeile 8014)
    - _ml_build_tagesplan_inner(): Kernlogik (Zeile 8065), erstellt 24-Stunden-Plan
      mit Slot-Empfehlungen, OR-Prognosen, Category-Rotation-Logik etc.
    - _build_tagesplan_retro(): Retro-Analyse vergangener Slots

Globaler Cache:
    _tagesplan_cache: Dict mit "plan", "ts", "mode" — wird von Research-Worker
    alle 5 Minuten im Hintergrund aufgefrischt.
"""
from __future__ import annotations

import logging

log = logging.getLogger("push-balancer")

# ── Tagesplan Cache (module-level global) ─────────────────────────────────
_tagesplan_cache: dict = {"plan": None, "ts": 0, "mode": "redaktion"}


def build_tagesplan(background: bool = False, mode: str = "redaktion") -> dict | None:
    """Erstellt oder gibt gecachten Tagesplan zurück.

    Args:
        background: True = im Hintergrund auffrischen (kein blocking)
        mode: "redaktion" oder "extended"

    Returns:
        Tagesplan-Dict oder None bei Fehler

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _ml_build_tagesplan()
        (Zeile 8014) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _ml_build_tagesplan  # type: ignore
        return _ml_build_tagesplan(background=background, mode=mode)
    except ImportError:
        pass
    log.warning("[tagesplan] build_tagesplan: Legacy-Import fehlgeschlagen")
    return None


def build_tagesplan_inner(now, current_hour: int, mode: str = "redaktion") -> dict | None:
    """Kernlogik des Tagesplan-Aufbaus.

    Args:
        now: datetime.datetime-Objekt
        current_hour: Aktuelle Stunde (0–23)
        mode: "redaktion" oder "extended"

    Returns:
        Tagesplan-Dict mit Slots, Empfehlungen, OR-Prognosen

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _ml_build_tagesplan_inner()
        (Zeile 8065) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _ml_build_tagesplan_inner  # type: ignore
        return _ml_build_tagesplan_inner(now, current_hour, mode)
    except ImportError:
        pass
    log.warning("[tagesplan] build_tagesplan_inner: Legacy-Import fehlgeschlagen")
    return None


def build_tagesplan_retro() -> dict | None:
    """Retro-Analyse: vergangene Slots des heutigen Tages mit tatsächlichen OR-Werten.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _build_tagesplan_retro()
    """
    try:
        from push_balancer_server_compat import _build_tagesplan_retro  # type: ignore
        return _build_tagesplan_retro()
    except ImportError:
        pass
    log.warning("[tagesplan] build_tagesplan_retro: Legacy-Import fehlgeschlagen")
    return None
