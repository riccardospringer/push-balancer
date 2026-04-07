"""app/ml/lightgbm_model.py — LightGBM-Training + Stacking Ensemble.

Globaler ML-State und Training für LightGBM + Stacking Ensemble
(LightGBM + XGBoost + CatBoost → Ridge Meta-Learner).
Migration über Compat-Shim — direkte Migration folgt schrittweise.
"""
from __future__ import annotations

import logging
import os
import threading
import time

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
    """Trainiert das LightGBM-Modell auf der gesamten Push-Historie."""
    try:
        _ml_train_model_impl()
    except Exception as exc:
        log.error("[lightgbm] ml_train_model fehlgeschlagen: %s", exc)


def _ml_train_model_impl() -> None:
    """Interne Implementierung des LightGBM-Trainings."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("[lightgbm] LightGBM nicht installiert — Training übersprungen")
        return

    try:
        import joblib
    except ImportError:
        log.warning("[lightgbm] joblib nicht installiert — Training übersprungen")
        return

    # 1. Push-Daten laden
    try:
        from app.database import push_db_load_all
        pushes = push_db_load_all()
    except Exception as exc:
        log.error("[lightgbm] push_db_load_all fehlgeschlagen: %s", exc)
        return

    # 2. Mindestanzahl prüfen (nur Pushes mit valider OR)
    valid_pushes = [p for p in pushes if p.get("or", 0) > 0]
    if len(valid_pushes) < 50:
        log.info("[lightgbm] Zu wenige Pushes mit OR > 0 (%d < 50) — Training übersprungen", len(valid_pushes))
        return

    # 3. History-Stats berechnen
    try:
        from app.ml.stats import _gbrt_build_history_stats
        stats = _gbrt_build_history_stats(valid_pushes)
    except Exception as exc:
        log.error("[lightgbm] _gbrt_build_history_stats fehlgeschlagen: %s", exc)
        return

    # 4+5. Features extrahieren und Feature-Matrix aufbauen
    try:
        from app.ml.features import gbrt_extract_features
    except ImportError:
        try:
            from app.ml.features import _gbrt_extract_features as gbrt_extract_features
        except ImportError as exc:
            log.error("[lightgbm] Feature-Import fehlgeschlagen: %s", exc)
            return

    X_rows = []
    y_vals = []
    feature_names = None

    for push in valid_pushes:
        try:
            feat_dict = gbrt_extract_features(push, stats, state=None, fast_mode=True)
        except Exception:
            continue
        if not feat_dict:
            continue
        if feature_names is None:
            feature_names = list(feat_dict.keys())
        row = [float(feat_dict.get(k, 0.0)) for k in feature_names]
        X_rows.append(row)
        y_vals.append(float(push["or"]))

    if len(X_rows) < 50:
        log.warning("[lightgbm] Zu wenige Feature-Rows nach Extraktion (%d) — Training übersprungen", len(X_rows))
        return

    # 6. LightGBM trainieren
    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 10,
        "verbose": -1,
        "n_jobs": 1,
    }

    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_rows, y_vals, feature_name=feature_names)
    except Exception as exc:
        log.error("[lightgbm] LGBMRegressor.fit fehlgeschlagen: %s", exc)
        return

    # In-sample MAE berechnen
    try:
        preds_train = model.predict(X_rows)
        mae = sum(abs(p - a) for p, a in zip(preds_train, y_vals)) / len(y_vals)
    except Exception:
        preds_train = []
        mae = 0.0

    # 7. Isotonische Kalibrierung
    calibrator = None
    try:
        from app.ml.core_classes import IsotonicCalibrator
        calibrator = IsotonicCalibrator()
        if len(preds_train) >= 10:
            calibrator.fit(list(preds_train), y_vals)
    except Exception as exc:
        log.warning("[lightgbm] IsotonicCalibrator fehlgeschlagen: %s", exc)
        calibrator = None

    # 8. Ergebnis in _ml_state speichern (thread-safe)
    now_ts = int(time.time())
    with _ml_lock:
        _ml_state["model"] = model
        _ml_state["stats"] = stats
        _ml_state["feature_names"] = feature_names or []
        _ml_state["calibrator"] = calibrator
        _ml_state["metrics"] = {"mae": round(mae, 4), "n_train": len(y_vals)}
        _ml_state["train_count"] = _ml_state.get("train_count", 0) + 1
        _ml_state["last_train_ts"] = now_ts
        _ml_state["next_retrain_ts"] = now_ts + 3600
        _ml_state["training"] = False

    log.info(
        "[lightgbm] Training abgeschlossen: %d Samples, MAE=%.4f, Features=%d",
        len(y_vals), mae, len(feature_names or []),
    )

    # 9. Modell auf Disk speichern
    try:
        from app.config import SERVE_DIR
        model_path = os.path.join(SERVE_DIR, ".ml_lgbm_model.pkl")
        joblib.dump({"model": model, "feature_names": feature_names, "calibrator": calibrator,
                     "stats_global_avg": stats.get("global_avg", 4.77)}, model_path)
        log.info("[lightgbm] Modell gespeichert: %s", model_path)
    except Exception as exc:
        log.warning("[lightgbm] Modell-Speicherung fehlgeschlagen: %s", exc)


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
    """Monitoring-Tick: Loggt aktuelle MAE aus _ml_state."""
    try:
        with _ml_lock:
            metrics = _ml_state.get("metrics", {})
            model_loaded = _ml_state.get("model") is not None
            train_count = _ml_state.get("train_count", 0)

        if model_loaded:
            mae = metrics.get("mae", None)
            n_train = metrics.get("n_train", 0)
            if mae is not None:
                log.info("[monitoring] LightGBM aktiv: MAE=%.4f, n_train=%d, train_count=%d",
                         mae, n_train, train_count)
            else:
                log.info("[monitoring] LightGBM geladen, keine MAE-Metriken verfügbar")
        else:
            log.debug("[monitoring] LightGBM-Modell noch nicht trainiert (train_count=%d)", train_count)
    except Exception as exc:
        log.warning("[monitoring] monitoring_tick Fehler: %s", exc)
