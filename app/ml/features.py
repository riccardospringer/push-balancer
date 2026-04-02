"""app/ml/features.py — Feature-Extraktion für GBRT- und LightGBM-Modelle.

Stellt _gbrt_extract_features und _ml_extract_features aus
push-balancer-server.py als importierbare Funktionen bereit.

Die vollständige Logik muss aus push-balancer-server.py übernommen werden:

GBRT (_gbrt_extract_features, Zeilen 1221–1630):
- ~80 Features: Text, Temporal, Category, Historical, Similarity, Embedding,
  Topic-Cluster, BILD-Titelstil, Volatilität, Sport-Kalender, App-Mix

ML (_ml_extract_features, Zeile 6528):
- Vereinfachter Feature-Vektor für LightGBM-Training

Diese Datei delegiert an die globalen Funktionen im Legacy-Modul während der
Migrations-Übergangsphase. Sobald alle Abhängigkeiten aufgelöst sind, wird
die Logik hier vollständig implementiert.
"""
from __future__ import annotations

import logging

log = logging.getLogger("push-balancer")


def gbrt_extract_features(push: dict, history_stats: dict, state: dict | None = None, fast_mode: bool = False) -> dict:
    """Extrahiert ~80 Features für das GBRT-Modell.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        history_stats: Vorberechnete Aggregat-Statistiken (von gbrt_build_history_stats)
        state: Optionaler Research-State für Kontext-Features
        fast_mode: True während Training (überspringt teure Ähnlichkeits-Suche)

    Returns:
        Dict mit Feature-Name → Float-Wert

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py Zeilen 1221–1630
        (_gbrt_extract_features) hierher migrieren.
    """
    # Delegiere an Legacy-Implementierung während Übergangsphase
    try:
        from push_balancer_server_compat import _gbrt_extract_features  # type: ignore
        return _gbrt_extract_features(push, history_stats, state, fast_mode)
    except ImportError:
        pass
    # Minimaler Stub-Fallback (gibt leeres Feature-Dict zurück)
    log.warning("[features] gbrt_extract_features: Legacy-Import fehlgeschlagen, verwende leeres Feature-Dict")
    return {}


def ml_extract_features(row: dict, stats: dict) -> dict:
    """Extrahiert vereinfachte Features für LightGBM-Training.

    Args:
        row: Push-Dict
        stats: Vorberechnete ML-Statistiken (von ml_build_stats)

    Returns:
        Dict mit Feature-Name → Float-Wert

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py Zeile 6528
        (_ml_extract_features) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _ml_extract_features  # type: ignore
        return _ml_extract_features(row, stats)
    except ImportError:
        pass
    log.warning("[features] ml_extract_features: Legacy-Import fehlgeschlagen")
    return {}
