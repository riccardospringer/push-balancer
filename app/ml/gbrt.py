"""app/ml/gbrt.py — GBRT-Modell (Gradient Boosted Regression Trees, reines Python).

Globaler State und Trainings-/Predict-Logik.
Direktimplementierung ohne Monolith-Abhängigkeit.
"""
from __future__ import annotations

import json
import logging
import os
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

# Safety Envelope: Predictions werden auf dieses Intervall geclippt
_SAFETY_MIN_OR: float = 0.5
_SAFETY_MAX_OR: float = 25.0


def gbrt_load_model() -> bool:
    """Lädt gespeichertes GBRT-Modell von Disk.

    Pfad: os.path.join(SERVE_DIR, '.gbrt_model.json').
    Nutzt GBRTModel.from_json() aus app.ml.core_classes.

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
            _gbrt_model_direct = model  # Alias für direktes Modell
            _gbrt_feature_names = list(model.feature_names)
            _gbrt_model_type = data.get("model_type", "direct")

            # Globaler Trainings-Mittelwert aus Modell-Metriken (falls vorhanden)
            metrics = data.get("metrics", {})
            if metrics.get("train_mean"):
                _gbrt_global_train_avg = float(metrics["train_mean"])
            elif model.initial_prediction > 0:
                _gbrt_global_train_avg = float(model.initial_prediction)

            # Cat×Hour Baselines aus gespeichertem Modell (falls vorhanden)
            if "cat_hour_baselines" in data:
                _gbrt_cat_hour_baselines = data["cat_hour_baselines"]

        n_trees = len(model.trees)
        n_feats = len(model.feature_names)
        log.info("[gbrt] Modell geladen: %d Bäume, %d Features, Typ=%s",
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


def gbrt_train() -> None:
    """Trainiert das GBRT-Modell auf der aktuellen Push-Historie.

    HINWEIS: Vollständige Implementierung aus push-balancer-server.py (_gbrt_train)
    wurde bewusst nicht migriert, da Training:
      - Datenbankzugriff auf die gesamte Push-Historie erfordert (DB-Schicht)
      - Stacking-Ensemble-Logik mit TimeSeriesSplit und Ridge Meta-Learner nutzt
      - ~60–120 Sekunden CPU-Zeit benötigt (wird async im Research-Worker ausgeführt)
      - Abhängig von gbrt_build_history_stats() ist (ebenfalls nicht migriert)
    Das Modell wird stattdessen offline trainiert und via gbrt_load_model() geladen.
    Graceful Degradation: kein Fehler, nur ein Log-Eintrag.
    """
    log.info("[gbrt] gbrt_train: Training ist im Research-Worker implementiert. "
             "Stub — kein Action Required.")


def gbrt_online_update() -> None:
    """Online-Learning: aktualisiert GBRT-Bias anhand neuer Feedback-Daten.

    Liest die jüngsten Prediction-Log-Einträge (predicted_or vs actual_or),
    berechnet den rolling Mean-Error und korrigiert _gbrt_online_bias.

    Kein frisches Feedback verfügbar → No-op.
    """
    global _gbrt_online_bias

    try:
        from app.db.database import get_db_connection
    except ImportError:
        return  # DB-Modul nicht verfügbar — kein Update

    try:
        conn = get_db_connection()
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
