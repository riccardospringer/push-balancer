"""app/ml/predict.py — predictOR-Pipeline (9 Methoden).

Vollständige Migration aus push-balancer-server.py (_server_predict_or).
Fallback-Kette:
  1. LightGBM Stacking (wenn Modell geladen)
  2. GBRT (wenn GBRT-Modell geladen)
  3. ML-Heuristik-Blend: Einzelnes ML-Modell × Heuristik (falls push_data vorhanden)
  4. Vollständige M1-M9-Heuristik (falls push_data vorhanden, kein ML)
  5. Cat×Hour-Heuristik aus research_state
  6. Global Average (4.77%)
"""
from __future__ import annotations

import logging

from app.config import SAFETY_MODE

log = logging.getLogger("push-balancer")

# ── Safety Constants ───────────────────────────────────────────────────────
_SAFETY_ADVISORY_ONLY: bool = True  # Redundanter Guard

_GLOBAL_AVG_FALLBACK: float = 4.77


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


def predict_or(
    push: dict,
    research_state: dict | None = None,
    push_data: list | None = None,
) -> dict | None:
    """Hauptfunktion der predictOR-Pipeline.

    Fallback-Kette:
      1. LightGBM (wenn Modell geladen)
      2. GBRT (wenn GBRT-Modell geladen)
      3. ML-Blend mit Heuristik (einzelnes ML-Modell + push_data)
      4. Vollständige M1-M9-Heuristik (wenn push_data >= 10 Pushes)
      5. Cat×Hour Heuristik aus research_state
      6. Global Average (4.77%)

    Die Funktion gibt immer einen Wert zurück und crasht nie.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        research_state: Aktueller Research-Worker-State für Kontext
        push_data: Historische Push-Liste für M1-M9-Heuristik (optional).
                   Wenn None: aus research_state["push_data"] gelesen.

    Returns:
        Dict mit predicted_or, basis_method, confidence, q10, q90,
        advisory_only=True, action_allowed=False
    """
    # push_data aus research_state lesen wenn nicht explizit übergeben
    if push_data is None and research_state:
        push_data = research_state.get("push_data") or []

    try:
        result = _predict_or_impl(push, research_state, push_data or [])
        return safety_envelope(result)
    except Exception as exc:
        log.error("[predict] predict_or unerwarteter Fehler: %s", exc)
        return safety_envelope({
            "predicted_or": _GLOBAL_AVG_FALLBACK,
            "basis_method": "error_fallback",
            "confidence": 0.0,
            "q10": 2.0,
            "q90": 8.0,
        })


def _predict_or_impl(
    push: dict,
    research_state: dict | None,
    push_data: list,
) -> dict:
    """Interne Implementierung der Fallback-Kette."""

    # ── 1. LightGBM-Prediction ─────────────────────────────────────────────
    lgbm_result = None
    try:
        lgbm_result = _predict_lightgbm(push, research_state)
        if lgbm_result is not None:
            # Wenn push_data vorhanden: ML-Blend mit Heuristik
            if len(push_data) >= 10:
                return _ml_heuristic_blend(
                    push, research_state, push_data,
                    ml_name="lightgbm", ml_result=lgbm_result,
                )
            return lgbm_result
    except Exception as exc:
        log.warning("[predict] LightGBM-Prediction fehlgeschlagen: %s", exc)

    # ── 2. GBRT-Prediction ─────────────────────────────────────────────────
    gbrt_result = None
    try:
        gbrt_result = _predict_gbrt(push, research_state)
        if gbrt_result is not None:
            if len(push_data) >= 10:
                return _ml_heuristic_blend(
                    push, research_state, push_data,
                    ml_name="gbrt", ml_result=gbrt_result,
                )
            return gbrt_result
    except Exception as exc:
        log.warning("[predict] GBRT-Prediction fehlgeschlagen: %s", exc)

    # ── 3. Vollständige M1-M9-Heuristik ───────────────────────────────────
    if len(push_data) >= 10:
        try:
            result = _predict_full_heuristic(push, research_state, push_data)
            if result is not None:
                return result
        except Exception as exc:
            log.warning("[predict] Vollständige Heuristik fehlgeschlagen: %s", exc)

    # ── 4. Cat×Hour Heuristik ──────────────────────────────────────────────
    try:
        result = _predict_cat_hour_heuristic(push, research_state)
        if result is not None:
            return result
    except Exception as exc:
        log.warning("[predict] Cat×Hour-Heuristik fehlgeschlagen: %s", exc)

    # ── 5. Global Average Fallback ─────────────────────────────────────────
    return _predict_global_avg(push, research_state)


def _ml_heuristic_blend(
    push: dict,
    research_state: dict | None,
    push_data: list,
    ml_name: str,
    ml_result: dict,
) -> dict:
    """Blendet ein einzelnes ML-Modell mit der M1-M9-Heuristik.

    Gewichte aus _ml_state["ml_heuristic_alpha"] (Fallback: 0.55).
    """
    try:
        from app.ml.lightgbm_model import _ml_state, _ml_lock
        with _ml_lock:
            blend_w = _ml_state.get("ml_heuristic_alpha", 0.55)
    except Exception:
        blend_w = 0.55

    ml_pred = ml_result.get("predicted_or", _GLOBAL_AVG_FALLBACK)

    # Heuristik berechnen (wenn möglich)
    heur_pred = None
    try:
        heur_result = _predict_full_heuristic(push, research_state, push_data)
        if heur_result is not None:
            heur_pred = heur_result.get("predicted_or")
    except Exception:
        pass

    if heur_pred is None:
        # Kein Blend ohne Heuristik — ML-Ergebnis direkt zurückgeben
        return ml_result

    blended = ml_pred * blend_w + heur_pred * (1 - blend_w)
    blended = max(0.5, min(30.0, blended))

    # Konfidenz: Mittelwert der beiden Konfidenzwerte
    ml_conf = ml_result.get("confidence", 0.4)
    heur_conf = heur_result.get("confidence", 0.3) if heur_result else 0.3
    confidence = ml_conf * blend_w + heur_conf * (1 - blend_w)

    q10 = max(0.1, blended - 1.28 * 1.5)
    q90 = min(20.0, blended + 1.28 * 1.5)

    return {
        "predicted_or": round(blended, 4),
        "basis_method": f"{ml_name}_heuristic_blend({blend_w:.0%}+{1-blend_w:.0%})",
        "confidence": round(confidence, 3),
        "q10": round(q10, 4),
        "q90": round(q90, 4),
        f"{ml_name}_predicted": round(ml_pred, 4),
        "heuristic_predicted": round(heur_pred, 4) if heur_pred else None,
        "ml_heuristic_alpha": round(blend_w, 3),
    }


def _predict_lightgbm(push: dict, research_state: dict | None) -> dict | None:
    """LightGBM-Prediction, wenn Modell geladen."""
    from app.ml.lightgbm_model import _ml_state, _ml_lock

    with _ml_lock:
        model = _ml_state.get("model")
        feature_names = _ml_state.get("feature_names", [])
        stats = _ml_state.get("stats")
        calibrator = _ml_state.get("calibrator")

    if model is None:
        return None
    if not feature_names or stats is None:
        return None

    # Features extrahieren
    try:
        from app.ml.features import gbrt_extract_features
    except ImportError:
        try:
            from app.ml.features import _gbrt_extract_features as gbrt_extract_features
        except ImportError:
            return None

    feat_dict = gbrt_extract_features(push, stats, state=research_state, fast_mode=False)
    if not feat_dict:
        return None

    row = [[float(feat_dict.get(k, 0.0)) for k in feature_names]]

    raw_pred = float(model.predict(row)[0])

    # Kalibrierung anwenden
    if calibrator is not None:
        try:
            predicted = float(calibrator.calibrate(raw_pred))
        except Exception:
            predicted = raw_pred
    else:
        predicted = raw_pred

    # Konfidenz aus Modell-Metriken ableiten
    metrics = _ml_state.get("metrics", {})
    mae = metrics.get("mae", 1.5)
    n_train = metrics.get("n_train", 0)
    confidence = min(0.95, max(0.3, 1.0 - mae / 5.0)) if n_train >= 100 else 0.4

    # q10/q90 aus historischer Std-Abweichung
    cat = (push.get("cat", "") or "news").lower().strip()
    cat_vol = stats.get("cat_volatility", {}).get(cat, {})
    std = cat_vol.get("std_30d", 0.0) or cat_vol.get("std_7d", 0.0) or 1.5
    q10 = max(0.1, predicted - 1.28 * std)
    q90 = min(20.0, predicted + 1.28 * std)

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "lightgbm",
        "confidence": round(confidence, 3),
        "q10": round(q10, 4),
        "q90": round(q90, 4),
    }


def _predict_gbrt(push: dict, research_state: dict | None) -> dict | None:
    """GBRT-Prediction, wenn GBRT-Modell geladen."""
    try:
        from app.ml.gbrt import gbrt_predict
        result = gbrt_predict(push, research_state)
        if result is None:
            return None

        predicted = float(result.get("predicted", result.get("predicted_or", _GLOBAL_AVG_FALLBACK)))
        confidence = float(result.get("confidence", 0.4))
        q10 = float(result.get("q10", max(0.1, predicted - 1.5)))
        q90 = float(result.get("q90", min(20.0, predicted + 1.5)))

        return {
            "predicted_or": round(predicted, 4),
            "basis_method": "gbrt",
            "confidence": round(confidence, 3),
            "q10": round(q10, 4),
            "q90": round(q90, 4),
        }
    except Exception:
        return None


def _predict_full_heuristic(
    push: dict,
    research_state: dict | None,
    push_data: list,
) -> dict | None:
    """Vollständige M1-M9-Heuristik mit Post-Fusion-Korrektoren."""
    try:
        from app.ml.heuristic import predict_heuristic
    except ImportError as exc:
        log.warning("[predict] heuristic-Import fehlgeschlagen: %s", exc)
        return None

    # Residual-Corrector aus worker laden (lazy, kein circular import)
    residual_corrector = None
    try:
        from app.research.worker import _residual_corrector, _residual_corrector_lock
        with _residual_corrector_lock:
            residual_corrector = dict(_residual_corrector)
    except Exception:
        pass

    return predict_heuristic(
        push=push,
        push_data=push_data,
        state=research_state or {},
        residual_corrector=residual_corrector,
        tuning_params=(research_state or {}).get("tuning_params"),
    )


def _predict_cat_hour_heuristic(push: dict, research_state: dict | None) -> dict | None:
    """Cat×Hour Baseline aus research_state oder lokalen Stats."""
    cat = (push.get("cat", "") or "news").lower().strip()
    hour = push.get("hour")
    if hour is None:
        import datetime
        ts = push.get("ts_num", 0)
        if ts > 0:
            hour = datetime.datetime.fromtimestamp(ts).hour
        else:
            hour = datetime.datetime.now().hour

    # Versuche Cat×Hour aus research_state
    baseline = None
    if research_state:
        cat_hour_stats = research_state.get("cat_hour_stats", {})
        key = f"{cat}_{hour}"
        ch = cat_hour_stats.get(key, {})
        if isinstance(ch, dict) and ch.get("n", 0) >= 3:
            baseline = ch.get("avg", None)

        # Fallback: Cat-Stats aus research_state
        if baseline is None:
            cat_stats = research_state.get("cat_stats", {}).get(cat, {})
            n = cat_stats.get("n_30d", 0)
            if n >= 5:
                baseline = cat_stats.get("avg_30d", None)

        # Fallback: global_avg aus research_state
        if baseline is None:
            baseline = research_state.get("global_avg", None)

    if baseline is None:
        return None

    predicted = float(baseline)
    std = 1.5  # konservative Standard-Schätzung
    q10 = max(0.1, predicted - 1.28 * std)
    q90 = min(20.0, predicted + 1.28 * std)

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "cat_hour_heuristic",
        "confidence": 0.3,
        "q10": round(q10, 4),
        "q90": round(q90, 4),
    }


def _predict_global_avg(push: dict, research_state: dict | None) -> dict:
    """Letzter Fallback: Global Average 4.77%."""
    predicted = _GLOBAL_AVG_FALLBACK
    if research_state:
        ga = research_state.get("global_avg")
        if ga and isinstance(ga, (int, float)) and ga > 0:
            predicted = float(ga)

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "global_avg",
        "confidence": 0.1,
        "q10": round(max(0.1, predicted - 2.0), 4),
        "q90": round(min(20.0, predicted + 2.0), 4),
    }
