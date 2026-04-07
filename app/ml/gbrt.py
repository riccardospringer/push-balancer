"""app/ml/gbrt.py — GBRT-Modell (Gradient Boosted Regression Trees, reines Python).

Globaler State und Trainings-/Predict-Logik.
Migration über Compat-Shim — direkte Migration folgt schrittweise.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("push-balancer")

# ── Globaler GBRT State (module-level, thread-safe) ───────────────────────
_gbrt_model = None
_gbrt_model_direct = None
_gbrt_model_q10 = None
_gbrt_model_q90 = None
_gbrt_calibrator = None
_gbrt_feature_names: list = []
_gbrt_history_stats: dict = {}
_gbrt_ensemble_weights: list = []
_gbrt_model_type: str = "direct"
_gbrt_global_train_avg: float = 4.77
_gbrt_cat_hour_baselines: dict = {}
_gbrt_online_bias: float = 0.0

_gbrt_lock = threading.Lock()


def gbrt_predict(push: dict, state: dict | None = None) -> dict | None:
    """GBRT-Prediction für einen einzelnen Push.

    Returns:
        Dict mit predicted, confidence, q10, q90, features, importance
        oder None wenn kein Modell geladen.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py Zeilen 5681–5900
        (_gbrt_predict) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _gbrt_predict  # type: ignore
        return _gbrt_predict(push, state)
    except ImportError:
        pass
    with _gbrt_lock:
        if _gbrt_model is None:
            return None
    log.warning("[gbrt] gbrt_predict: Legacy-Import fehlgeschlagen")
    return None


def gbrt_train() -> None:
    """Trainiert das GBRT-Modell auf der aktuellen Push-Historie.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _gbrt_train()
    """
    try:
        from push_balancer_server_compat import _gbrt_train  # type: ignore
        _gbrt_train()
    except ImportError:
        log.warning("[gbrt] gbrt_train: Legacy-Import fehlgeschlagen")


def gbrt_load_model() -> bool:
    """Lädt gespeichertes GBRT-Modell von Disk.

    Returns:
        True wenn Modell erfolgreich geladen, False sonst.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _gbrt_load_model()
    """
    try:
        from push_balancer_server_compat import _gbrt_load_model  # type: ignore
        return _gbrt_load_model()
    except ImportError:
        log.warning("[gbrt] gbrt_load_model: Legacy-Import fehlgeschlagen")
        return False


def gbrt_online_update() -> None:
    """Online-Learning: aktualisiert GBRT-Bias anhand neuer Feedback-Daten.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _gbrt_online_update()
    """
    try:
        from push_balancer_server_compat import _gbrt_online_update  # type: ignore
        _gbrt_online_update()
    except ImportError:
        log.warning("[gbrt] gbrt_online_update: Legacy-Import fehlgeschlagen")


def gbrt_check_drift(state: dict) -> None:
    """Prüft auf Concept Drift und triggert ggf. Retraining.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _gbrt_check_drift()
    """
    try:
        from push_balancer_server_compat import _gbrt_check_drift  # type: ignore
        _gbrt_check_drift(state)
    except ImportError:
        log.warning("[gbrt] gbrt_check_drift: Legacy-Import fehlgeschlagen")
