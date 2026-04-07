"""app/state.py — Zentraler Shared State für alle Module.

Alle globalen State-Variablen aus push-balancer-server.py sind hier
als module-level globals definiert. Module importieren von hier statt
untereinander (verhindert circular imports).
"""
from __future__ import annotations

import threading

# ── Safety (NIEMALS ändern) ───────────────────────────────────────────────
SAFETY_MODE = "ADVISORY_ONLY"
_SAFETY_ADVISORY_ONLY = True


def safety_check():
    if SAFETY_MODE != "ADVISORY_ONLY" or not _SAFETY_ADVISORY_ONLY:
        raise RuntimeError("SAFETY VIOLATION: System muss im ADVISORY_ONLY Modus laufen!")


def safety_envelope(result: dict) -> dict:
    if isinstance(result, dict):
        result["advisory_only"] = True
        result["safety_mode"] = SAFETY_MODE
    return result


# ── LightGBM / ML State ───────────────────────────────────────────────────
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
    "ml_heuristic_alpha": 0.55,
}
_ml_lock = threading.Lock()

# ── Unified ML State (Stacking Ensemble) ─────────────────────────────────
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

# ── GBRT State ────────────────────────────────────────────────────────────
_gbrt_model = None
_gbrt_challenger = None
_gbrt_feature_names: list = []
_gbrt_train_ts: int = 0
_gbrt_history_stats: dict = {}
_gbrt_lock = threading.Lock()

# ── Stacking Meta-Model ───────────────────────────────────────────────────
_stacking_model: dict = {
    "weights": None,
    "bias": 0.0,
    "trained_at": 0,
    "n_samples": 0,
    "mae": 0.0,
}

# ── Residual Corrector ────────────────────────────────────────────────────
_residual_corrector: dict = {
    "global_bias": 0.0,
    "cat_bias": {},
    "hourgroup_bias": {},
    "n_samples": 0,
    "last_update_ts": 0,
    "recent_residuals": [],
}
_residual_corrector_lock = threading.Lock()

# ── Push Sync Cache ───────────────────────────────────────────────────────
_push_sync_cache: dict = {"messages": [], "ts": 0, "channels": []}
_push_sync_lock = threading.Lock()

# ── Tagesplan Cache ───────────────────────────────────────────────────────
def _TP_CACHE_EMPTY() -> dict:
    return {"result": None, "hour": -1, "ts": 0, "building": False, "model_id": None}

_tagesplan_cache: dict = {
    "redaktion": _TP_CACHE_EMPTY(),
    "sport": _TP_CACHE_EMPTY(),
}
_tagesplan_cache_lock = threading.Lock()

# ── Research State ────────────────────────────────────────────────────────
_research_state: dict = {
    "last_fetch": 0,
    "push_data": [],
    "prev_push_count": 0,
    "findings": {},
    "ticker_entries": [],
    "last_analysis": 0,
    "analysis_generation": 0,
    "cumulative_insights": 0,
    "analysis_lock": None,
    "accuracy_history": [],
    "rolling_accuracy": 0.0,
    "accuracy_by_cat": {},
    "accuracy_trend": [],
    "schwab_decisions": [],
    "schwab_current": "",
    "bild_adaptations": [],
    "live_rules": [],
    "live_rules_version": 0,
    "research_projects": [],
    "research_milestones": [],
    "research_memory": {},
    "research_log": [],
    "prev_accuracy": 0.0,
    "prev_findings_hash": "",
    "pending_approvals": [],
    "approval_counter": 0,
    "decided_topics": set(),
    "prediction_feedback": [],
    "tuning_history": [],
    "tuning_params": {},
    "tuning_params_version": 0,
    "_last_tuning_call": 0,
    "basis_mae": 0.0,
    "ensemble_mae": 0.0,
    "mae_trend": [],
    "mae_by_cat": {},
    "mae_by_hour": {},
    "ensemble_accuracy": 0.0,
    "ensemble_accuracy_trend": [],
    "ensemble_accuracy_delta": 0.0,
    "fresh_pushes": [],
    "mature_count": 0,
    "fresh_count": 0,
    "cutoff_24h": 0,
    "week_comparison": {},
    "live_pulse": [],
    "research_modifiers": {},
    "external_context": {},
    "algo_score_analysis": {},
    "phd_insights": {},
    "or_challenges": {},
    "_stacking_counter": 0,
    "_worker_first_log": True,
    "_sport_data": [],
    "_nonsport_data": [],
    "_sport_n": 0,
    "_nonsport_n": 0,
}
_research_state["analysis_lock"] = threading.RLock()

# ── Topic Tracker ─────────────────────────────────────────────────────────
_topic_tracker: dict = {"clusters": [], "ts": 0}
_topic_tracker_lock = threading.Lock()

# ── Online Bias ───────────────────────────────────────────────────────────
_gbrt_online_bias: float = 0.0

# ── Model Selector ────────────────────────────────────────────────────────
_model_selector_state: dict = {
    "active_model": "ml_ensemble",
    "unified_mae_24h": None,
    "ensemble_mae_24h": None,
    "consecutive_worse": 0,
    "evaluated_count": 0,
    "last_check_ts": 0,
}

# ── Competitor Feed Cache ─────────────────────────────────────────────────
_feed_cache: dict = {}
_feed_cache_lock = threading.Lock()

# ── Retro Cache ───────────────────────────────────────────────────────────
_retro_cache: dict = {"result": None, "ts": 0, "day": ""}
_retro_cache_lock = threading.Lock()
