"""app/ml/predict.py — predictOR-Pipeline (9 Methoden).

Bündelt alle Prediction-Methoden aus push-balancer-server.py:

1. GBRT-Prediction (_gbrt_predict)
2. LightGBM-Prediction (via _ml_state)
3. Unified/Stacking-Prediction (via _unified_state)
4. Heuristik-Fallback (Cat×Hour Baseline)
5. Keyword-Magnitude-Heuristik
6. LLM-Score-Prediction
7. Ensemble-Blending (GBRT × α + LightGBM × (1-α))
8. Online Residual Correction
9. Safety Envelope

SICHERHEITSHINWEIS:
    Alle Predictions sind ADVISORY_ONLY. Das System darf niemals autonom
    Push-Benachrichtigungen senden. Alle Ergebnisse enthalten
    advisory_only=True und action_allowed=False.

IMPLEMENTIERUNGSHINWEIS:
    Vollständige predictOR-Pipeline aus push-balancer-server.py:
    - predictOR(): Hauptfunktion (9-Methoden-Pipeline)
    - _safety_check() / _safety_envelope()
    - _update_residual_corrector()
    - Online Bias Correction via _residual_corrector
    - _model_selector_state für GBRT vs Ensemble-Auswahl
"""
from __future__ import annotations

import logging

from app.config import SAFETY_MODE

log = logging.getLogger("push-balancer")

# ── Safety Constants ───────────────────────────────────────────────────────
_SAFETY_ADVISORY_ONLY: bool = True  # Redundanter Guard


def safety_check() -> None:
    """Prüft beide Safety-Guards. Raises wenn nicht ADVISORY_ONLY."""
    if SAFETY_MODE != "ADVISORY_ONLY" or not _SAFETY_ADVISORY_ONLY:
        raise RuntimeError("SAFETY VIOLATION: System muss im ADVISORY_ONLY Modus laufen!")


def safety_envelope(result: dict | None) -> dict | None:
    """Fügt advisory_only und action_allowed zu jeder Prediction hinzu."""
    safety_check()
    if result is None:
        return None
    if isinstance(result, dict):
        result["advisory_only"] = True
        result["action_allowed"] = False
        result["safety_mode"] = SAFETY_MODE
    return result


def predict_or(push: dict, research_state: dict | None = None) -> dict | None:
    """Hauptfunktion der predictOR-Pipeline.

    Führt alle 9 Prediction-Methoden durch und gibt die beste Schätzung zurück.
    Ergebnis ist immer mit safety_envelope() gewrappt.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        research_state: Aktueller Research-Worker-State für Kontext

    Returns:
        Dict mit predicted_or, basis_method, confidence, q10, q90,
        advisory_only=True, action_allowed=False

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige predictOR-Pipeline aus push-balancer-server.py
        hierher migrieren.
    """
    try:
        from push_balancer_server_compat import predictOR  # type: ignore
        return predictOR(push, research_state)
    except ImportError:
        pass
    log.warning("[predict] predict_or: Legacy-Import fehlgeschlagen")
    return safety_envelope({
        "predicted_or": 5.0,
        "basis_method": "fallback_stub",
        "confidence": 0.0,
        "q10": 2.0,
        "q90": 8.0,
    })
