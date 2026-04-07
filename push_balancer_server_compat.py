"""push_balancer_server_compat.py — Compatibility Shim.

Lädt push-balancer-server.py (Dateiname mit Bindestrich, kein direkter Import)
via importlib und re-exportiert alle benötigten Symbole.

Zweck: Stubs in app/ml/, app/tagesplan/, app/research/ können von hier
importieren solange die vollständige Migration noch läuft.
"""
import importlib.util
import sys
import os

_MODULE_NAME = "pbserver_legacy"
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "push-balancer-server.py")

def _load():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod

_legacy = _load()

# ── Re-Exports ────────────────────────────────────────────────────────────
# ML / GBRT
GBRTModel                    = _legacy.GBRTModel
_gbrt_predict                = _legacy._gbrt_predict
_gbrt_train                  = _legacy._gbrt_train
_gbrt_train_inner            = _legacy._gbrt_train_inner
_gbrt_check_drift            = _legacy._gbrt_check_drift
_gbrt_online_update          = _legacy._gbrt_online_update
_gbrt_extract_features       = _legacy._gbrt_extract_features
_gbrt_build_history_stats    = _legacy._gbrt_build_history_stats

# LightGBM / Unified
_ml_train_model              = _legacy._ml_train_model
_ml_predict                  = _legacy._ml_predict
_unified_train               = _legacy._unified_train
_train_stacking_model        = _legacy._train_stacking_model
_stacking_predict            = _legacy._stacking_predict
_monitoring_tick             = _legacy._monitoring_tick
_update_rolling_accuracy     = _legacy._update_rolling_accuracy

# predictOR pipeline
try:
    _predictOR               = _legacy._predictOR  # type: ignore
except AttributeError:
    pass

# Tagesplan
_ml_build_tagesplan          = _legacy._ml_build_tagesplan
_ml_build_tagesplan_inner    = _legacy._ml_build_tagesplan_inner
_build_tagesplan_retro       = _legacy._build_tagesplan_retro
_tagesplan_background_refresh= _legacy._tagesplan_background_refresh

# Research
_run_autonomous_analysis     = _legacy._run_autonomous_analysis
_run_autonomous_analysis_inner = _legacy._run_autonomous_analysis_inner
_fetch_push_data             = _legacy._fetch_push_data
_fetch_external_context      = _legacy._fetch_external_context
_generate_live_rules         = _legacy._generate_live_rules
_build_progress_ticker       = _legacy._build_progress_ticker

# Auto-Suggestion
try:
    _auto_save_suggestions   = _legacy._auto_save_suggestions  # type: ignore
except AttributeError:
    pass

# Score / Safety
_safety_check                = _legacy._safety_check
_safety_envelope             = _legacy._safety_envelope
_keyword_magnitude_heuristic = _legacy._keyword_magnitude_heuristic
_compute_topic_saturation_penalty = _legacy._compute_topic_saturation_penalty

# DB helpers (für Stubs die noch nicht app.database nutzen)
_push_db_load_all            = _legacy._push_db_load_all
_push_db_upsert              = _legacy._push_db_upsert
_push_db_count               = _legacy._push_db_count
_push_db_max_ts              = _legacy._push_db_max_ts
_push_db_log_prediction      = _legacy._push_db_log_prediction

# State-Objekte (Referenzen — Mutationen sehen alle Module)
_ml_state                    = _legacy._ml_state
_ml_lock                     = _legacy._ml_lock
_gbrt_model                  = _legacy._gbrt_model
_gbrt_feature_names          = _legacy._gbrt_feature_names
_gbrt_history_stats          = _legacy._gbrt_history_stats
_research_state              = _legacy._research_state
_tagesplan_cache             = _legacy._tagesplan_cache
_tagesplan_cache_lock        = _legacy._tagesplan_cache_lock
_push_sync_cache             = _legacy._push_sync_cache
_push_sync_lock              = _legacy._push_sync_lock
_residual_corrector          = _legacy._residual_corrector
_stacking_model              = _legacy._stacking_model

# Konstanten
SAFETY_MODE                  = _legacy.SAFETY_MODE
DEFAULT_TUNING_PARAMS        = _legacy.DEFAULT_TUNING_PARAMS
_GBRT_CATEGORIES             = _legacy._GBRT_CATEGORIES
_GBRT_EMOTION_WORDS          = _legacy._GBRT_EMOTION_WORDS
_GBRT_TOPIC_CLUSTERS         = _legacy._GBRT_TOPIC_CLUSTERS
