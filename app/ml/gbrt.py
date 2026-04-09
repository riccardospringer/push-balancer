"""app/ml/gbrt.py — GBRT-Modell (Gradient Boosted Regression Trees, reines Python).

Globaler State und Trainings-/Predict-Logik.
Direktimplementierung ohne Monolith-Abhängigkeit.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

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

# Safety Envelope: Predictions werden auf dieses Intervall geclippt
_SAFETY_MIN_OR: float = 0.5
_SAFETY_MAX_OR: float = 25.0


def gbrt_load_model() -> bool:
    """Lädt gespeichertes GBRT/LightGBM-Modell von Disk.

    Versucht in dieser Reihenfolge:
    1. .gbrt_lgbm_model.pkl (LightGBM, gespeichert von gbrt_train())
    2. .gbrt_model.json (pure-Python GBRTModel)

    Returns:
        True wenn Modell erfolgreich geladen, False wenn Datei fehlt oder Fehler.
    """
    global _gbrt_model, _gbrt_model_direct, _gbrt_feature_names, _gbrt_model_type
    global _gbrt_global_train_avg, _gbrt_cat_hour_baselines

    try:
        from app.config import SERVE_DIR
        from app.ml.core_classes import GBRTModel
    except ImportError as e:
        log.warning("[gbrt] gbrt_load_model: Import-Fehler: %s", e)
        return False

    # ── Versuch 1: LightGBM joblib ────────────────────────────────────────
    lgbm_path = os.path.join(SERVE_DIR, ".gbrt_lgbm_model.pkl")
    if os.path.exists(lgbm_path):
        try:
            import joblib
            from app.ml.core_classes import _LGBMModelWrapper
            data = joblib.load(lgbm_path)
            lgbm_model = data["model"]
            feature_names = data["feature_names"]
            wrapper = _LGBMModelWrapper(lgbm_model, feature_names)
            wrapper.train_metrics = data.get("metrics", {})
            with _gbrt_lock:
                _gbrt_model = wrapper
                _gbrt_model_direct = wrapper
                _gbrt_feature_names = list(feature_names)
                _gbrt_model_type = "direct"
                if data.get("global_train_avg", 0) > 0:
                    _gbrt_global_train_avg = float(data["global_train_avg"])
                if "cat_hour_baselines" in data:
                    _gbrt_cat_hour_baselines = data["cat_hour_baselines"]
            n_trees = len(wrapper.trees)
            log.info("[gbrt] LightGBM-Modell geladen: %d Bäume, %d Features",
                     n_trees, len(feature_names))
            return True
        except Exception as e:
            log.warning("[gbrt] LightGBM-Load fehlgeschlagen (%s), versuche JSON...", e)

    # ── Versuch 2: Pure-Python GBRTModel JSON ─────────────────────────────
    model_path = os.path.join(SERVE_DIR, ".gbrt_model.json")
    if not os.path.exists(model_path):
        log.debug("[gbrt] Kein gespeichertes Modell gefunden: %s", model_path)
        return False

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with _gbrt_lock:
            model = GBRTModel.from_json(data)
            _gbrt_model = model
            _gbrt_model_direct = model
            _gbrt_feature_names = list(model.feature_names)
            _gbrt_model_type = data.get("model_type", "direct")

            metrics = data.get("metrics", {})
            if metrics.get("train_mean"):
                _gbrt_global_train_avg = float(metrics["train_mean"])
            elif metrics.get("global_avg"):
                _gbrt_global_train_avg = float(metrics["global_avg"])
            elif model.initial_prediction > 0:
                _gbrt_global_train_avg = float(model.initial_prediction)

            if "cat_hour_baselines" in data:
                _gbrt_cat_hour_baselines = data["cat_hour_baselines"]

        n_trees = len(model.trees)
        n_feats = len(model.feature_names)
        log.info("[gbrt] GBRTModel geladen: %d Bäume, %d Features, Typ=%s",
                 n_trees, n_feats, _gbrt_model_type)
        return True

    except Exception as e:
        log.warning("[gbrt] gbrt_load_model fehlgeschlagen: %s", e)
        return False


def gbrt_predict(push: dict, stats: dict | None = None) -> dict | None:
    """GBRT-Prediction für einen einzelnen Push.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num etc.
        stats: Optionaler history_stats-Dict (von _gbrt_build_history_stats).
               Wenn None: leeres Dict wird verwendet (alle historischen Features = Fallbacks).

    Returns:
        Dict mit predicted_or, basis_method, confidence, q10, q90, features, importance
        oder None wenn kein Modell geladen.
    """
    with _gbrt_lock:
        model = _gbrt_model
        feature_names = _gbrt_feature_names[:]
        online_bias = _gbrt_online_bias
        global_avg = _gbrt_global_train_avg

    if model is None:
        return None

    history_stats = stats or {}

    try:
        from app.ml.features import gbrt_extract_features
    except ImportError as e:
        log.warning("[gbrt] gbrt_predict: Import-Fehler features: %s", e)
        return None

    try:
        feat_dict = gbrt_extract_features(push, history_stats, state=push)
    except Exception as e:
        log.warning("[gbrt] gbrt_predict: Feature-Extraktion fehlgeschlagen: %s", e)
        return None

    if not feature_names:
        feature_names = model.feature_names

    # Feature-Vektor in der gespeicherten Reihenfolge aufbauen
    try:
        feature_vec = [float(feat_dict.get(name, 0.0)) for name in feature_names]
    except Exception as e:
        log.warning("[gbrt] gbrt_predict: Feature-Vektor-Aufbau fehlgeschlagen: %s", e)
        return None

    # Prediction + Unsicherheitsschätzung
    try:
        result = model.predict_with_uncertainty(feature_vec)
        raw_pred = result.get("predicted", global_avg)
        confidence = result.get("confidence", 0.5)
        std = result.get("std", 1.0)
    except Exception as e:
        log.warning("[gbrt] gbrt_predict: Modell-Prediction fehlgeschlagen: %s", e)
        return None

    # Online-Bias addieren (aus gbrt_online_update)
    raw_pred = raw_pred + online_bias

    # Safety Envelope: Predictions auf realistisches Intervall clippen
    predicted_or = max(_SAFETY_MIN_OR, min(_SAFETY_MAX_OR, raw_pred))

    # Konfidenz-Intervall (Q10/Q90) aus Std-Schätzung
    z_80 = 1.28  # 80%-Intervall ≈ ±1.28σ
    q10 = max(_SAFETY_MIN_OR, predicted_or - z_80 * std)
    q90 = min(_SAFETY_MAX_OR, predicted_or + z_80 * std)

    # SHAP-artige Feature Importances für die Top-Features dieses Pushes
    top_importance: dict = {}
    try:
        shap_result = model.shap_values(feature_vec)
        shap_vals = shap_result.get("shap_values", {})
        # Top-10 nach absolutem Wert
        top_shap = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        top_importance = {k: round(v, 4) for k, v in top_shap}
    except Exception:
        pass

    return {
        "predicted_or": round(predicted_or, 4),
        "basis_method": "gbrt",
        "confidence": round(confidence, 3),
        "q10": round(q10, 4),
        "q90": round(q90, 4),
        "std": round(std, 4),
        "n_trees": len(model.trees),
        "online_bias": round(online_bias, 4),
        "features": feat_dict,
        "importance": top_importance,
    }


_gbrt_training_active = threading.Event()


def gbrt_train() -> bool:
    """Trainiert das GBRT/LightGBM-Modell auf der aktuellen Push-Historie.

    Bevorzugt LightGBM (speichert .gbrt_lgbm_model.pkl), fallback auf
    pure-Python GBRTModel (speichert .gbrt_model.json).

    Returns:
        True wenn Training erfolgreich, False sonst.
    """
    if _gbrt_training_active.is_set():
        log.info("[gbrt] Training bereits aktiv — übersprungen")
        return False
    _gbrt_training_active.set()
    try:
        return _gbrt_train_impl()
    except Exception as exc:
        log.error("[gbrt] gbrt_train fehlgeschlagen: %s", exc, exc_info=True)
        return False
    finally:
        _gbrt_training_active.clear()


def _gbrt_train_impl() -> bool:
    """Interne Trainings-Implementierung (LightGBM bevorzugt, GBRTModel fallback)."""
    global _gbrt_model, _gbrt_model_direct, _gbrt_feature_names
    global _gbrt_global_train_avg, _gbrt_cat_hour_baselines, _gbrt_history_stats

    t0 = time.time()

    try:
        from app.database import push_db_load_all
        from app.ml.stats import _gbrt_build_history_stats
        from app.ml.features import gbrt_extract_features
        from app.ml.core_classes import IsotonicCalibrator
        from app.config import SERVE_DIR
    except ImportError as exc:
        log.error("[gbrt] Import-Fehler: %s", exc)
        return False

    # 1. Push-Daten laden
    try:
        pushes = push_db_load_all()
    except Exception as exc:
        log.error("[gbrt] push_db_load_all fehlgeschlagen: %s", exc)
        return False

    # 2. Nur reife Pushes mit valider OR
    now_ts = int(time.time())
    valid = [
        p for p in pushes
        if 0 < (p.get("or") or 0) <= 20
        and p.get("ts_num", 0) > 0
        and p["ts_num"] < now_ts - 86400  # >24h alt (reif)
    ]

    if len(valid) < 100:
        log.warning("[gbrt] Zu wenige valide Pushes (%d < 100) — Training übersprungen", len(valid))
        return False

    valid.sort(key=lambda x: x["ts_num"])
    n = len(valid)
    log.info("[gbrt] Training gestartet: %d reife Pushes", n)

    # 3. Train/Val/Test Split (75 / 12.5 / 12.5)
    train_end = int(n * 0.75)
    val_end = int(n * 0.875)
    train_data = valid[:train_end]
    val_data = valid[train_end:val_end]
    test_data = valid[val_end:]

    # 4. History-Stats aus Trainings-Daten (kein Data Leakage)
    history_stats = _gbrt_build_history_stats(
        train_data, target_ts=train_data[-1]["ts_num"] + 1
    )

    # 5. Feature-Whitelist (generalisiert gut, kein Overfitting auf Lookup-Features)
    _FEATURE_WHITELIST = {
        "hour_sin", "hour_cos", "is_prime_time", "is_morning_commute",
        "is_late_night", "is_lunch", "is_weekend", "weekday",
        "title_len", "word_count", "has_question", "has_colon", "has_pipe",
        "has_exclamation", "has_numbers", "upper_ratio", "avg_word_len",
        "death_signal", "exclusivity_signal", "is_breaking_style",
        "emotional_word_count", "emotional_categories", "intensity_score",
        "is_eilmeldung", "n_channels",
        "cat_sport", "cat_politik", "cat_news", "cat_unterhaltung",
        "cat_regional", "cat_geld", "cat_digital", "cat_leben",
        "cat_avg_or_7d", "cat_avg_or_30d", "hour_avg_or_7d", "hour_avg_or_30d",
        "cat_hour_avg_or", "weekday_avg_or",
        "cat_or_std_7d", "cat_or_std_30d",
        "push_count_today", "mins_since_last_same_cat",
        "stacking_cat_hour_baseline", "stacking_cat_hour_n",
        "topic_crime", "topic_royals", "topic_kosten", "topic_gesundheit",
        "topic_auto", "topic_sex_beziehung", "topic_wetter_extrem",
        "has_age_parens", "has_das_so_pattern", "has_direct_address", "has_number_emphasis",
        "or_volatility_7d", "title_sentiment",
    }

    def _extract_matrix(data):
        rows = []
        targets = []
        feat_names = None
        for p in data:
            try:
                fd = gbrt_extract_features(p, history_stats, state=None, fast_mode=True)
            except Exception:
                continue
            if not fd:
                continue
            if feat_names is None:
                all_keys = sorted(fd.keys())
                feat_names = [k for k in all_keys if k in _FEATURE_WHITELIST]
                if len(feat_names) < 15:
                    feat_names = all_keys
            rows.append([float(fd.get(k, 0.0)) for k in feat_names])
            targets.append(float(p["or"]))
        return rows, targets, feat_names

    X_train, y_train, feature_names = _extract_matrix(train_data)
    if not X_train or not feature_names:
        log.warning("[gbrt] Feature-Extraktion fehlgeschlagen — Training abgebrochen")
        return False

    X_val, y_val, _ = _extract_matrix(val_data)
    X_test, y_test, _ = _extract_matrix(test_data)

    # 6. Cat×Hour Baselines mit Shrinkage
    global_avg = sum(y_train) / len(y_train)
    _SHRINKAGE_K = 20
    from collections import defaultdict
    _cat_hour_counts: dict = defaultdict(list)
    for p in train_data:
        key = f"{(p.get('cat', '') or 'news').lower().strip()}_{p.get('hour', 12)}"
        _cat_hour_counts[key].append(p.get("or", 0))
    cat_hour_baselines = {
        k: (len(v) * (sum(v) / len(v)) + _SHRINKAGE_K * global_avg) / (len(v) + _SHRINKAGE_K)
        for k, v in _cat_hour_counts.items()
    }

    # 7. Modell trainieren (LightGBM bevorzugt, GBRTModel fallback)
    model = None
    use_lgbm = False
    try:
        import lightgbm as lgb
        import numpy as np
        use_lgbm = True
    except ImportError:
        np = None

    if use_lgbm and np is not None:
        from app.ml.core_classes import _LGBMModelWrapper
        params = {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
            "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
            "num_leaves": 16, "reg_alpha": 1.0, "reg_lambda": 2.0,
            "colsample_bytree": 0.5, "objective": "regression_l1",
            "n_jobs": 1, "random_state": 42, "verbose": -1,
        }
        np_X_train = np.array(X_train, dtype=np.float64)
        np_y_train = np.array(y_train, dtype=np.float64)
        lgbm = lgb.LGBMRegressor(**params)
        if X_val:
            np_X_val = np.array(X_val, dtype=np.float64)
            np_y_val = np.array(y_val, dtype=np.float64)
            lgbm.fit(
                np_X_train, np_y_train,
                eval_set=[(np_X_val, np_y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            lgbm.fit(np_X_train, np_y_train)
        model = _LGBMModelWrapper(lgbm, feature_names)
        log.info("[gbrt] LightGBM trainiert (%d Bäume, %d Features)",
                 len(model.trees), len(feature_names))
    else:
        from app.ml.core_classes import GBRTModel
        gbrt = GBRTModel(
            n_trees=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=40, subsample=0.8, n_bins=128,
            loss="huber", huber_delta=1.5, log_target=False,
        )
        gbrt.fit(X_train, y_train, feature_names=feature_names)
        model = gbrt
        log.info("[gbrt] Pure-Python GBRT trainiert (%d Bäume, %d Features)",
                 len(model.trees), len(feature_names))

    # 8. Test-Metriken
    test_mae = 0.0
    test_r2 = 0.0
    if X_test and y_test:
        try:
            if use_lgbm and np is not None:
                test_preds = list(model.lgbm_model.predict(np.array(X_test, dtype=np.float64)))
            else:
                test_preds = model.predict(X_test)
            n_t = len(y_test)
            test_mae = sum(abs(test_preds[i] - y_test[i]) for i in range(n_t)) / n_t
            ymean_t = sum(y_test) / n_t
            ss_res = sum((y_test[i] - test_preds[i]) ** 2 for i in range(n_t))
            ss_tot = sum((y_test[i] - ymean_t) ** 2 for i in range(n_t))
            test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            pass

    # 9. Isotonische Kalibrierung
    calibrator = None
    try:
        if use_lgbm and np is not None and X_train:
            train_preds_cal = list(model.lgbm_model.predict(np.array(X_train, dtype=np.float64)))
        else:
            train_preds_cal = model.predict(X_train) if X_train else []
        if len(train_preds_cal) >= 20:
            calibrator = IsotonicCalibrator()
            calibrator.fit(list(train_preds_cal), y_train)
    except Exception as exc:
        log.debug("[gbrt] Kalibrierung fehlgeschlagen: %s", exc)

    model.train_metrics = {
        "test_mae": round(test_mae, 4),
        "test_r2": round(test_r2, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "global_avg": round(global_avg, 4),
        "trained_at": now_ts,
    }

    # 10. Globals aktualisieren (thread-safe)
    with _gbrt_lock:
        _gbrt_model = model
        _gbrt_model_direct = model
        _gbrt_feature_names = list(feature_names)
        _gbrt_global_train_avg = global_avg
        _gbrt_cat_hour_baselines = cat_hour_baselines
        _gbrt_history_stats = history_stats

    elapsed = time.time() - t0
    log.info(
        "[gbrt] Training abgeschlossen in %.1fs: test_MAE=%.4f, test_R²=%.4f, "
        "n_train=%d, n_test=%d, backend=%s",
        elapsed, test_mae, test_r2, len(y_train), len(y_test),
        "lightgbm" if use_lgbm else "pure_python",
    )

    # 11. Modell auf Disk speichern
    try:
        if use_lgbm:
            import joblib
            lgbm_path = os.path.join(SERVE_DIR, ".gbrt_lgbm_model.pkl")
            joblib.dump({
                "model": model.lgbm_model,
                "feature_names": feature_names,
                "cat_hour_baselines": cat_hour_baselines,
                "global_train_avg": global_avg,
                "trained_at": now_ts,
                "metrics": model.train_metrics,
            }, lgbm_path, compress=3)
            log.info("[gbrt] LightGBM-Modell gespeichert: %s", lgbm_path)
        else:
            gbrt_path = os.path.join(SERVE_DIR, ".gbrt_model.json")
            model_json = model.to_json()
            model_json["metrics"] = model.train_metrics
            model_json["cat_hour_baselines"] = cat_hour_baselines
            model_json["global_train_avg"] = global_avg
            model_json["model_type"] = "direct"
            if calibrator is not None:
                try:
                    model_json["calibrator"] = calibrator.to_dict()
                except Exception:
                    pass
            with open(gbrt_path, "w", encoding="utf-8") as f:
                json.dump(model_json, f)
            log.info("[gbrt] GBRTModel gespeichert: %s", gbrt_path)
    except Exception as exc:
        log.warning("[gbrt] Modell-Speicherung fehlgeschlagen: %s", exc)

    return True


def gbrt_online_update() -> None:
    """Online-Learning: aktualisiert GBRT-Bias anhand neuer Feedback-Daten.

    Liest die jüngsten Prediction-Log-Einträge (predicted_or vs actual_or),
    berechnet den rolling Mean-Error und korrigiert _gbrt_online_bias.

    Kein frisches Feedback verfügbar → No-op.
    """
    global _gbrt_online_bias

    try:
        import sqlite3 as _sqlite3
        from app.config import PUSH_DB_PATH as _PUSH_DB_PATH
    except ImportError:
        return  # Konfiguration nicht verfügbar — kein Update

    try:
        conn = _sqlite3.connect(_PUSH_DB_PATH, timeout=5)
        cursor = conn.execute("""
            SELECT predicted_or, actual_or
            FROM prediction_log
            WHERE actual_or > 0
              AND predicted_at > (strftime('%s','now') - 7*86400)
            ORDER BY predicted_at DESC
            LIMIT 200
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        log.debug("[gbrt] gbrt_online_update: DB-Fehler: %s", e)
        return

    if not rows or len(rows) < 10:
        return  # Zu wenig Feedback für sinnvolle Bias-Schätzung

    # Mean-Error (predicted − actual): positiv = Modell überschätzt
    errors = [float(r[0]) - float(r[1]) for r in rows]
    mean_error = sum(errors) / len(errors)

    # Sanity-Check: Bias nur anpassen wenn signifikant (> 0.1 OR-Punkte)
    if abs(mean_error) < 0.1:
        return

    # Bias schrittweise anpassen (Learning Rate 0.1, max ±2 OR-Punkte)
    with _gbrt_lock:
        new_bias = _gbrt_online_bias - 0.1 * mean_error
        new_bias = max(-2.0, min(2.0, new_bias))
        _gbrt_online_bias = new_bias

    log.info("[gbrt] Online-Bias aktualisiert: %.4f → %.4f (mean_error=%.4f, n=%d)",
             _gbrt_online_bias + 0.1 * mean_error, _gbrt_online_bias, mean_error, len(rows))


def gbrt_check_drift(state: dict) -> None:
    """Prüft auf Concept Drift anhand der Accuracy-History im Research-State.

    Accuracy-History: Liste von {"accuracy": float, "n": int} aus dem Research-Worker.
    Wenn die rolling Accuracy unter 40% fällt und genug Samples vorhanden sind,
    wird ein Warning geloggt (kein automatisches Retraining — das obliegt dem Operator).

    Args:
        state: Research-State-Dict mit optionalem "accuracy_history"-Key.
    """
    if not state or not isinstance(state, dict):
        return

    accuracy_history = state.get("accuracy_history", [])
    if not accuracy_history or len(accuracy_history) < 5:
        return  # Zu wenig History für zuverlässigen Drift-Check

    # Rolling Accuracy: letzte 10 Einträge
    recent = accuracy_history[-10:]
    total_n = sum(float(e.get("n", 0)) for e in recent)
    if total_n < 20:
        return  # Zu wenige Samples (Warmup-Phase)

    # Gewichteter Durchschnitt (nach Anzahl Samples gewichtet)
    weighted_sum = sum(float(e.get("accuracy", 0)) * float(e.get("n", 1)) for e in recent)
    rolling_accuracy = weighted_sum / total_n

    _DRIFT_THRESHOLD = 0.40  # unter 40% = Drift-Signal

    if rolling_accuracy < _DRIFT_THRESHOLD:
        log.warning(
            "[gbrt] DRIFT-WARNUNG: Rolling Accuracy %.1f%% < %.0f%% "
            "(n=%d Samples, letzte %d Messungen). "
            "Manuelles Retraining empfohlen.",
            rolling_accuracy * 100,
            _DRIFT_THRESHOLD * 100,
            int(total_n),
            len(recent),
        )
    else:
        log.debug("[gbrt] Drift-Check OK: Rolling Accuracy %.1f%% (n=%d)",
                  rolling_accuracy * 100, int(total_n))
