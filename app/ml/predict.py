"""app/ml/predict.py — predictOR-Pipeline (9 Methoden).

Vollständige, monolithfreie Runtime-Orchestrierung:
  1. Stacking/Unified-Prediction
  2. LightGBM
  3. GBRT
  4. ML-Ensemble (Stacking + LightGBM + GBRT)
  5. ML+Heuristik-Blend
  6. Vollständige Heuristik M1-M9 (inkl. optionalem LLM-Score)
  7. Residual-Corrector
  8. Cat×Hour-Heuristik
  9. Safety-Envelope (ADVISORY_ONLY)
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

    Die Funktion gibt immer einen Wert zurück und crasht nie.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        research_state: Aktueller Research-Worker-State für Kontext.
        push_data: Historische Push-Liste für Heuristik/Blend (optional).
                   Wenn None: aus research_state["push_data"] gelesen.

    Returns:
        Dict mit predicted_or, basis_method, confidence, q10, q90,
        methods_detail und Safety-Flags.
    """
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
            "methods_detail": {"error": str(exc)},
        })


def _predict_or_impl(
    push: dict,
    research_state: dict | None,
    push_data: list,
) -> dict:
    """Interne Implementierung der 9-Methoden-Pipeline."""
    methods_detail: dict = {}
    has_history = len(push_data) >= 10

    # ── 1-3. Primäre ML-Signale sammeln ───────────────────────────────────
    stacking_result = None
    lightgbm_result = None
    gbrt_result = None

    try:
        stacking_result = _predict_stacking(push, research_state)
        if stacking_result is not None:
            methods_detail["stacking_predicted"] = stacking_result["predicted_or"]
    except Exception as exc:
        methods_detail["stacking_error"] = str(exc)

    try:
        lightgbm_result = _predict_lightgbm(push, research_state)
        if lightgbm_result is not None:
            methods_detail["lightgbm_predicted"] = lightgbm_result["predicted_or"]
    except Exception as exc:
        methods_detail["lightgbm_error"] = str(exc)

    try:
        gbrt_result = _predict_gbrt(push, research_state)
        if gbrt_result is not None:
            methods_detail["gbrt_predicted"] = gbrt_result["predicted_or"]
            methods_detail["shadow_gbrt"] = gbrt_result["predicted_or"]
    except Exception as exc:
        methods_detail["gbrt_error"] = str(exc)

    ml_candidates = []
    if stacking_result is not None:
        ml_candidates.append(("stacking", stacking_result))
    if lightgbm_result is not None:
        ml_candidates.append(("lightgbm", lightgbm_result))
    if gbrt_result is not None:
        ml_candidates.append(("gbrt", gbrt_result))

    # ── 4. ML-Ensemble ─────────────────────────────────────────────────────
    chosen_ml = None
    if len(ml_candidates) >= 2:
        chosen_ml = _ensemble_models(push, ml_candidates)
        methods_detail["ml_ensemble"] = chosen_ml["predicted_or"]
    elif len(ml_candidates) == 1:
        chosen_ml = ml_candidates[0][1]

    # Für Monitoring/Selector immer unified_predicted mitschreiben
    if stacking_result is not None:
        methods_detail["unified_predicted"] = stacking_result["predicted_or"]

    # ── 5. ML+Heuristik-Blend ──────────────────────────────────────────────
    if chosen_ml is not None and has_history:
        blended = _ml_heuristic_blend(
            push=push,
            research_state=research_state,
            push_data=push_data,
            ml_name=chosen_ml.get("basis_method", "ml"),
            ml_result=chosen_ml,
        )
        _merge_methods_detail(blended, methods_detail)
        return _apply_residual_and_finalize(push, blended, research_state)

    if chosen_ml is not None:
        chosen_ml["methods_detail"] = methods_detail
        return _apply_residual_and_finalize(push, chosen_ml, research_state)

    # ── 6. Vollständige M1-M9-Heuristik ────────────────────────────────────
    if has_history:
        try:
            heur_result = _predict_full_heuristic(push, research_state, push_data)
            if heur_result is not None:
                _merge_methods_detail(heur_result, methods_detail)
                return _apply_residual_and_finalize(push, heur_result, research_state)
        except Exception as exc:
            methods_detail["heuristic_error"] = str(exc)

    # ── 8. Cat×Hour Heuristik ──────────────────────────────────────────────
    try:
        cat_hour = _predict_cat_hour_heuristic(push, research_state)
        if cat_hour is not None:
            cat_hour["methods_detail"] = methods_detail
            return _apply_residual_and_finalize(push, cat_hour, research_state)
    except Exception as exc:
        methods_detail["cat_hour_error"] = str(exc)

    # ── 9. Global Average Fallback ─────────────────────────────────────────
    fallback = _predict_global_avg(push, research_state)
    fallback["methods_detail"] = methods_detail
    return _apply_residual_and_finalize(push, fallback, research_state)


def _merge_methods_detail(result: dict, methods_detail: dict) -> None:
    """Merged bestehende methods_detail-Infos in result."""
    existing = result.get("methods_detail")
    if not isinstance(existing, dict):
        existing = {}
    existing.update(methods_detail)
    result["methods_detail"] = existing


def _get_residual_corrector_snapshot() -> dict | None:
    """Lädt Residual-Corrector atomar aus dem Research-Worker."""
    try:
        from app.research.worker import _residual_corrector, _residual_corrector_lock
        with _residual_corrector_lock:
            return dict(_residual_corrector)
    except Exception:
        return None


def _apply_residual_correction(
    predicted_or: float,
    cat: str,
    hour: int,
    residual_corrector: dict | None,
) -> tuple[float, float]:
    """Wendet den globalen Residual-Corrector modellübergreifend an."""
    if not residual_corrector or residual_corrector.get("n_samples", 0) < 10:
        return predicted_or, 0.0

    hour_groups = {
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 23),
    }
    hg = "night"
    for name, rng in hour_groups.items():
        if hour in rng:
            hg = name
            break

    gb = float(residual_corrector.get("global_bias", 0.0) or 0.0)
    cb = float(residual_corrector.get("cat_bias", {}).get(cat, gb) or gb)
    hb = float(residual_corrector.get("hourgroup_bias", {}).get(hg, gb) or gb)

    raw = 0.5 * gb + 0.3 * cb + 0.2 * hb
    if abs(raw) < 0.2:
        return predicted_or, 0.0

    correction = max(-2.0, min(2.0, raw * 0.5))
    corrected = max(0.5, min(30.0, predicted_or - correction))
    return corrected, round(correction, 3)


def _apply_residual_and_finalize(push: dict, result: dict, research_state: dict | None) -> dict:
    """Wendet Residual-Korrektur + Output-Normalisierung auf Endergebnis an."""
    if result is None:
        return result

    cat = str(push.get("cat", "News") or "News")
    hour = int(push.get("hour", 12) or 12)

    residual_corrector = _get_residual_corrector_snapshot()
    pred_before = float(result.get("predicted_or", _GLOBAL_AVG_FALLBACK) or _GLOBAL_AVG_FALLBACK)
    pred_after, correction = _apply_residual_correction(pred_before, cat, hour, residual_corrector)

    if abs(correction) > 0.0:
        result["predicted_or"] = round(pred_after, 4)
        methods_detail = result.get("methods_detail")
        if not isinstance(methods_detail, dict):
            methods_detail = {}
        methods_detail["residual_correction"] = correction
        methods_detail["pre_residual_predicted"] = round(pred_before, 4)
        result["methods_detail"] = methods_detail

    # q10/q90 konsistent halten
    q10 = float(result.get("q10", max(0.1, pred_after - 1.5)) or max(0.1, pred_after - 1.5))
    q90 = float(result.get("q90", min(20.0, pred_after + 1.5)) or min(20.0, pred_after + 1.5))
    if q10 > pred_after:
        q10 = max(0.1, pred_after - 0.3)
    if q90 < pred_after:
        q90 = min(20.0, pred_after + 0.3)
    result["q10"] = round(q10, 4)
    result["q90"] = round(q90, 4)

    return result


def _ensemble_models(push: dict, ml_candidates: list[tuple[str, dict]]) -> dict:
    """Gewichtetes Ensemble aus verfügbaren ML-Predictions."""
    try:
        from app.ml.lightgbm_model import _ml_state, _ml_lock
        with _ml_lock:
            lgbm_alpha = float(_ml_state.get("gbrt_lgbm_alpha", 0.6) or 0.6)
    except Exception:
        lgbm_alpha = 0.6

    weighted = []
    for name, pred in ml_candidates:
        conf = float(pred.get("confidence", 0.35) or 0.35)
        if name == "lightgbm":
            conf *= lgbm_alpha
        elif name == "gbrt":
            conf *= (1.0 - lgbm_alpha)
        elif name == "stacking":
            conf *= 1.10  # leicht bevorzugt, wenn verfügbar
        weighted.append((name, pred, max(0.05, conf)))

    w_sum = sum(w for _, _, w in weighted)
    if w_sum <= 0:
        return ml_candidates[0][1]

    predicted = sum(float(pred["predicted_or"]) * w for _, pred, w in weighted) / w_sum
    confidence = min(0.95, sum(float(pred.get("confidence", 0.35)) * w for _, pred, w in weighted) / w_sum)

    q10 = min(float(pred.get("q10", predicted - 1.5)) for _, pred, _ in weighted)
    q90 = max(float(pred.get("q90", predicted + 1.5)) for _, pred, _ in weighted)

    methods_detail = {
        "ensemble_sources": [n for n, _, _ in weighted],
        "ensemble_weights": {n: round(w / w_sum, 4) for n, _, w in weighted},
    }

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "ml_ensemble",
        "confidence": round(confidence, 3),
        "q10": round(max(0.1, q10), 4),
        "q90": round(min(20.0, q90), 4),
        "methods_detail": methods_detail,
    }


def _ml_heuristic_blend(
    push: dict,
    research_state: dict | None,
    push_data: list,
    ml_name: str,
    ml_result: dict,
) -> dict:
    """Blendet ein ML-Signal mit der M1-M9-Heuristik."""
    try:
        from app.ml.lightgbm_model import _ml_state, _ml_lock
        with _ml_lock:
            blend_w = float(_ml_state.get("ml_heuristic_alpha", 0.55) or 0.55)
    except Exception:
        blend_w = 0.55

    ml_pred = float(ml_result.get("predicted_or", _GLOBAL_AVG_FALLBACK) or _GLOBAL_AVG_FALLBACK)

    heur_result = None
    heur_pred = None
    try:
        heur_result = _predict_full_heuristic(push, research_state, push_data)
        if heur_result is not None:
            heur_pred = float(heur_result.get("predicted_or", _GLOBAL_AVG_FALLBACK) or _GLOBAL_AVG_FALLBACK)
    except Exception:
        pass

    if heur_pred is None:
        # Kein Blend ohne Heuristik — ML-Ergebnis direkt zurückgeben
        return ml_result

    blended = max(0.5, min(30.0, ml_pred * blend_w + heur_pred * (1 - blend_w)))

    ml_conf = float(ml_result.get("confidence", 0.4) or 0.4)
    heur_conf = float(heur_result.get("confidence", 0.3) or 0.3) if heur_result else 0.3
    confidence = ml_conf * blend_w + heur_conf * (1 - blend_w)

    q10 = max(0.1, blended - 1.28 * 1.5)
    q90 = min(20.0, blended + 1.28 * 1.5)

    methods_detail = {}
    if isinstance(ml_result.get("methods_detail"), dict):
        methods_detail.update(ml_result.get("methods_detail", {}))
    if isinstance(heur_result, dict) and isinstance(heur_result.get("methods"), dict):
        methods_detail["heuristic_methods"] = heur_result.get("methods", {})
    methods_detail.update({
        "ml_predicted": round(ml_pred, 4),
        "heuristic_predicted": round(heur_pred, 4),
        "ml_heuristic_alpha": round(blend_w, 3),
    })

    return {
        "predicted_or": round(blended, 4),
        "basis_method": f"{ml_name}_heuristic_blend({blend_w:.0%}+{1-blend_w:.0%})",
        "confidence": round(confidence, 3),
        "q10": round(q10, 4),
        "q90": round(q90, 4),
        "ml_predicted": round(ml_pred, 4),
        "heuristic_predicted": round(heur_pred, 4),
        "ml_heuristic_alpha": round(blend_w, 3),
        "methods_detail": methods_detail,
    }


def _predict_stacking(push: dict, research_state: dict | None) -> dict | None:
    """Stacking/Unified-Prediction, falls im unified_state verfügbar."""
    try:
        from app.ml.lightgbm_model import _unified_state, _unified_lock
        from app.ml.features import gbrt_extract_features
    except ImportError:
        return None

    with _unified_lock:
        model = _unified_state.get("model")
        feature_names = list(_unified_state.get("feature_names") or [])
        stats = _unified_state.get("stats")
        calibrator = _unified_state.get("calibrator")
        stacking_active = bool(_unified_state.get("stacking_active"))
        metrics = dict(_unified_state.get("metrics") or {})

    if model is None or not stacking_active:
        return None
    if not feature_names or stats is None:
        return None

    feat_dict = gbrt_extract_features(push, stats, state=research_state, fast_mode=False)
    if not feat_dict:
        return None

    row = [[float(feat_dict.get(k, 0.0)) for k in feature_names]]
    raw_pred = float(model.predict(row)[0])

    if calibrator is not None:
        try:
            predicted = float(calibrator.calibrate(raw_pred))
        except Exception:
            predicted = raw_pred
    else:
        predicted = raw_pred

    mae = float(metrics.get("mae", metrics.get("test_mae", 1.5)) or 1.5)
    confidence = min(0.95, max(0.35, 1.0 - mae / 5.0))

    cat = (push.get("cat", "") or "news").lower().strip()
    cat_vol = (stats.get("cat_volatility", {}) or {}).get(cat, {})
    std = float(cat_vol.get("std_30d", 0.0) or cat_vol.get("std_7d", 0.0) or 1.4)

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "stacking",
        "confidence": round(confidence, 3),
        "q10": round(max(0.1, predicted - 1.28 * std), 4),
        "q90": round(min(20.0, predicted + 1.28 * std), 4),
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

    if calibrator is not None:
        try:
            predicted = float(calibrator.calibrate(raw_pred))
        except Exception:
            predicted = raw_pred
    else:
        predicted = raw_pred

    metrics = _ml_state.get("metrics", {})
    mae = metrics.get("mae", 1.5)
    n_train = metrics.get("n_train", 0)
    confidence = min(0.95, max(0.3, 1.0 - mae / 5.0)) if n_train >= 100 else 0.4

    cat = (push.get("cat", "") or "news").lower().strip()
    cat_vol = stats.get("cat_volatility", {}).get(cat, {})
    std = cat_vol.get("std_30d", 0.0) or cat_vol.get("std_7d", 0.0) or 1.5

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "lightgbm",
        "confidence": round(confidence, 3),
        "q10": round(max(0.1, predicted - 1.28 * std), 4),
        "q90": round(min(20.0, predicted + 1.28 * std), 4),
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

    residual_corrector = _get_residual_corrector_snapshot()

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

    baseline = None
    if research_state:
        cat_hour_stats = research_state.get("cat_hour_stats", {})
        key = f"{cat}_{hour}"
        ch = cat_hour_stats.get(key, {})
        if isinstance(ch, dict) and ch.get("n", 0) >= 3:
            baseline = ch.get("avg", None)

        if baseline is None:
            cat_stats = research_state.get("cat_stats", {}).get(cat, {})
            n = cat_stats.get("n_30d", 0)
            if n >= 5:
                baseline = cat_stats.get("avg_30d", None)

        if baseline is None:
            baseline = research_state.get("global_avg", None)

    if baseline is None:
        return None

    predicted = float(baseline)
    std = 1.5

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": "cat_hour_heuristic",
        "confidence": 0.3,
        "q10": round(max(0.1, predicted - 1.28 * std), 4),
        "q90": round(min(20.0, predicted + 1.28 * std), 4),
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
