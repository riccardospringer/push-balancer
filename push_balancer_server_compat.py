"""push_balancer_server_compat.py — Compatibility Shim.

Lädt push-balancer-server.py (Dateiname mit Bindestrich, kein direkter Import)
via importlib und re-exportiert alle benötigten Symbole.

Zweck: Stubs in app/ml/, app/tagesplan/, app/research/ können von hier
importieren solange die vollständige Migration noch läuft.
"""
import importlib.util
import logging
import sys
import os

log = logging.getLogger("push-balancer")

_MODULE_NAME = "pbserver_legacy"
_SERVER_PATH = os.path.join(os.path.dirname(__file__), "push-balancer-server.py")

# Monolith deaktivieren wenn DISABLE_LEGACY_WORKER=true gesetzt.
# Auf Render ist der Monolith standardmäßig AKTIV — catboost+xgboost wurden aus
# requirements.txt entfernt, sodass nur lightgbm lädt (~150 MB, passt in 512 MB).
_LEGACY_DISABLED = os.environ.get("DISABLE_LEGACY_WORKER", "").lower() in ("1", "true", "yes")


class _MissingSymbol:
    """Platzhalter für Symbole die aus dem Legacy-Modul entfernt wurden.
    Wirft ImportError beim Aufruf, damit alle 'except ImportError' in App-Modulen greifen.
    """
    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(f"[compat] Symbol '{self._name}' nicht im Legacy-Modul — wurde entfernt")

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<MissingSymbol: {self._name}>"


def _load():
    if _LEGACY_DISABLED:
        log.info("[compat] Legacy-Monolith deaktiviert (Render/DISABLE_LEGACY_WORKER) — spart ~200 MB RAM")
        return None
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _re(name: str, default=None):
    """Sicherer Re-Export: gibt _MissingSymbol zurück wenn Symbol fehlt oder Monolith disabled."""
    if _legacy is None:
        if default is not None:
            return default
        return _MissingSymbol(name)
    if hasattr(_legacy, name):
        return getattr(_legacy, name)
    if default is not None:
        return default
    log.debug("[compat] Symbol '%s' nicht in Legacy-Modul — MissingSymbol gesetzt", name)
    return _MissingSymbol(name)


_legacy = _load()

# ── Re-Exports ────────────────────────────────────────────────────────────
# ML / GBRT
GBRTModel                    = _re("GBRTModel")
_gbrt_predict                = _re("_gbrt_predict")
_gbrt_train                  = _re("_gbrt_train")
_gbrt_train_inner            = _re("_gbrt_train_inner")
_gbrt_check_drift            = _re("_gbrt_check_drift")
_gbrt_online_update          = _re("_gbrt_online_update")
_gbrt_load_model             = _re("_gbrt_load_model")
_gbrt_extract_features       = _re("_gbrt_extract_features")
_gbrt_build_history_stats    = _re("_gbrt_build_history_stats")

# LightGBM / Unified
_ml_train_model              = _re("_ml_train_model")
_ml_predict                  = _re("_ml_predict")
_unified_train               = _re("_unified_train")
_train_stacking_model        = _re("_train_stacking_model")
_stacking_predict            = _re("_stacking_predict")
_monitoring_tick             = _re("_monitoring_tick")
_update_rolling_accuracy     = _re("_update_rolling_accuracy")

# predictOR pipeline
_predictOR                   = _re("_predictOR")

# Tagesplan
_ml_build_tagesplan          = _re("_ml_build_tagesplan")
_ml_build_tagesplan_inner    = _re("_ml_build_tagesplan_inner")
_build_tagesplan_retro       = _re("_build_tagesplan_retro")
_tagesplan_background_refresh= _re("_tagesplan_background_refresh")

# Research
_run_autonomous_analysis     = _re("_run_autonomous_analysis")
_run_autonomous_analysis_inner = _re("_run_autonomous_analysis_inner")
_fetch_push_data             = _re("_fetch_push_data")
_fetch_external_context      = _re("_fetch_external_context")
_generate_live_rules         = _re("_generate_live_rules")
_build_progress_ticker       = _re("_build_progress_ticker")
_build_xor_perf_cache        = _re("_build_xor_perf_cache")
_update_residual_corrector   = _re("_update_residual_corrector")

# Auto-Suggestion
_auto_save_suggestions       = _re("_auto_save_suggestions")

# Score / Safety
_safety_check                = _re("_safety_check")
_safety_envelope             = _re("_safety_envelope")
_keyword_magnitude_heuristic = _re("_keyword_magnitude_heuristic")
_compute_topic_saturation_penalty = _re("_compute_topic_saturation_penalty")

# DB helpers
_push_db_load_all            = _re("_push_db_load_all")
_push_db_upsert              = _re("_push_db_upsert")
_push_db_count               = _re("_push_db_count")
_push_db_max_ts              = _re("_push_db_max_ts")
_push_db_log_prediction      = _re("_push_db_log_prediction")

# State-Objekte (Referenzen — Mutationen sehen alle Module)
_ml_state                    = _re("_ml_state", {})
_ml_lock                     = _re("_ml_lock")
_gbrt_model                  = _re("_gbrt_model")
_gbrt_feature_names          = _re("_gbrt_feature_names", [])
_gbrt_history_stats          = _re("_gbrt_history_stats", {})
_research_state              = _re("_research_state", {})
_tagesplan_cache             = _re("_tagesplan_cache", {})
_tagesplan_cache_lock        = _re("_tagesplan_cache_lock")
_push_sync_cache             = _re("_push_sync_cache", {})
_push_sync_lock              = _re("_push_sync_lock")
_residual_corrector          = _re("_residual_corrector", {})
_stacking_model              = _re("_stacking_model", {})

# Konstanten
SAFETY_MODE                  = _re("SAFETY_MODE", "ADVISORY_ONLY")
DEFAULT_TUNING_PARAMS        = _re("DEFAULT_TUNING_PARAMS", {})
_GBRT_CATEGORIES             = _re("_GBRT_CATEGORIES", [])
_GBRT_EMOTION_WORDS          = _re("_GBRT_EMOTION_WORDS", [])
_GBRT_TOPIC_CLUSTERS         = _re("_GBRT_TOPIC_CLUSTERS", {})
