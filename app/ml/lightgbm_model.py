"""app/ml/lightgbm_model.py — LightGBM-Training + Stacking Ensemble.

Globaler ML-State und Training für LightGBM + Stacking Ensemble
(LightGBM + XGBoost + CatBoost → Ridge Meta-Learner).
Migration über Compat-Shim — direkte Migration folgt schrittweise.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("push-balancer")

# ── Globaler ML State (module-level, thread-safe) ─────────────────────────
_ml_state: dict = {
    "model": None,
    "stats": None,
    "feature_names": [],
    "metrics": {},
    "shap_importance": [],
    "train_count": 0,
    "last_train_ts": 0,
    "next_retrain_ts": 0,
    "training": False,
    "residual_model": None,
    "calibrator": None,
    "conformal_radius": 1.0,
    "gbrt_lgbm_alpha": 0.6,
    "ml_heuristic_alpha": 0.55,
}
_ml_lock = threading.Lock()

# ── Unified (Stacking Ensemble) State ─────────────────────────────────────
_unified_state: dict = {
    "model": None,
    "feature_names": [],
    "stats": None,
    "calibrator": None,
    "conformal_radius": 1.0,
    "metrics": {},
    "train_count": 0,
    "last_train_ts": 0,
    "training": False,
    "base_models": {},
    "meta_model": None,
    "stacking_active": False,
}
_unified_lock = threading.Lock()


def ml_train_model() -> None:
    """Trainiert das LightGBM-Modell auf der aktuellen Push-Historie.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _ml_train_model()
        (Zeile 6533) hierher migrieren.
    """
    try:
        from push_balancer_server_compat import _ml_train_model  # type: ignore
        _ml_train_model()
    except ImportError:
        log.warning("[lightgbm] ml_train_model: Legacy-Import fehlgeschlagen")


def unified_train() -> None:
    """Trainiert das Stacking Ensemble (LightGBM + XGBoost + CatBoost → Ridge).

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _unified_train()
    """
    try:
        from push_balancer_server_compat import _unified_train  # type: ignore
        _unified_train()
    except ImportError:
        log.warning("[lightgbm] unified_train: Legacy-Import fehlgeschlagen")


def train_stacking_model(research_state: dict) -> None:
    """Trainiert den Meta-Learner des Stacking Ensembles.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Implementierung aus push-balancer-server.py: _train_stacking_model()
    """
    try:
        from push_balancer_server_compat import _train_stacking_model  # type: ignore
        _train_stacking_model(research_state)
    except ImportError:
        log.warning("[lightgbm] train_stacking_model: Legacy-Import fehlgeschlagen")


def monitoring_tick() -> None:
    """Monitoring-Tick: Drift, MAE-Spikes, A/B-Ergebnisse etc."""
    try:
        from push_balancer_server_compat import _monitoring_tick  # type: ignore
        _monitoring_tick()
    except ImportError:
        log.warning("[lightgbm] monitoring_tick: Legacy-Import fehlgeschlagen")
