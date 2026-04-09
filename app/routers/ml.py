"""app/routers/ml.py — LightGBM/ML-Endpunkte.

GET  /api/ml/safety-status          — Safety-Status-Prüfung
GET  /api/ml/status                 — ML-Modell-Status
GET  /api/ml/predict                — Einzelne ML-Prediction
GET  /api/ml/experiments            — Experiment-Liste
GET  /api/ml/experiments/compare    — Experiment-Vergleich
GET  /api/ml/ab-status              — A/B-Test-Status
GET  /api/ml/monitoring             — Monitoring-Events
POST /api/ml/retrain                — Manuelles Retraining
POST /api/ml/monitoring/tick        — Manueller Monitoring-Tick
POST /api/ml/predict-batch          — Batch-Prediction (auch /api/predict-batch)
"""
import logging
import time
import hashlib
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import require_admin_key
from app.config import SAFETY_MODE
from app.database import (
    count_experiments,
    count_monitoring_events,
    load_experiments,
    load_monitoring_events,
    push_db_log_prediction,
)
from app.ml.lightgbm_model import _ml_state, _ml_lock, ml_train_model
from app.ml.predict import safety_envelope
from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


# ── Prediction-Cache (class-level, 5 Min TTL) ─────────────────────────────
_predict_cache: dict = {}


class PredictBatchRequest(BaseModel):
    articles: list[dict[str, Any]]


@router.get("/api/ml/safety-status")
def get_ml_safety_status() -> JSONResponse:
    """Gibt Safety-Status zurück — ADVISORY_ONLY Guard."""
    return JSONResponse(content={
        "safetyMode": SAFETY_MODE,
        "advisoryOnly": True,
        "actionAllowed": False,
        "message": "Alle Predictions sind ausschliesslich beratend. Das System darf niemals autonom Push-Benachrichtigungen senden.",
    })


@router.get("/api/ml/status")
def get_ml_status() -> JSONResponse:
    """Liefert den aktuellen ML-Modell-Status (Metriken, Feature Importance, etc.)."""
    with _ml_lock:
        s = dict(_ml_state)

    return JSONResponse(content={
        "trained": s["model"] is not None,
        "modelLoaded": s["model"] is not None,
        "trainCount": s["train_count"],
        "lastTrainTs": s["last_train_ts"],
        "nextRetrainTs": s["next_retrain_ts"],
        "training": s["training"],
        "metrics": s["metrics"],
        "shapImportance": s["shap_importance"],
        "featureCount": len(s["feature_names"]),
        "featureNames": s["feature_names"],
        "stackingActive": False,
        "safetyMode": SAFETY_MODE,
    })


@router.get("/api/ml/predict")
def get_ml_predict(
    title: str = Query(default=""),
    cat: str = Query(default="News"),
    hour: int = Query(default=-1),
    is_eilmeldung: int = Query(default=0),
    link: str = Query(default=""),
    push_id: str = Query(default=""),
) -> JSONResponse:
    """Einzelne ML-Prediction für einen Artikel.

    Query-Parameter:
        title: Artikel-Titel
        cat: Ressort (News, Sport, Politik, etc.)
        hour: Stunde (0–23), -1 = aktuelle Stunde
        is_eilmeldung: 0 oder 1
        link: Artikel-URL
        push_id: Optional — zum Loggen der Prediction
    """
    import datetime

    now = datetime.datetime.now()
    if hour < 0:
        hour = now.hour

    push = {
        "title": title,
        "headline": title,
        "cat": cat,
        "hour": hour,
        "ts_num": int(now.timestamp()),
        "is_eilmeldung": bool(is_eilmeldung),
        "link": link,
        "channels": [],
    }

    try:
        from app.ml.predict import predict_or
        result = predict_or(push, _research_state)
        if result is None:
            result = safety_envelope({
                "predicted_or": 5.0,
                "basis_method": "fallback",
                "confidence": 0.0,
                "q10": 2.0,
                "q90": 8.0,
            })

        # Prediction loggen wenn push_id angegeben
        if push_id and result:
            push_db_log_prediction(
                push_id=push_id,
                predicted_or=result.get("predicted_or", 0),
                actual_or=0,
                basis_method=result.get("basis_method", ""),
                confidence=result.get("confidence", 0),
                q10=result.get("q10", 0),
                q90=result.get("q90", 0),
                title=title,
            )

        # Transform interne snake_case Keys zu camelCase an der API-Grenze
        predicted_or_val = result.get("predicted_or") if result else None
        camel_result = {
            "predictedOr": predicted_or_val,
            "or": predicted_or_val,
            "basis": result.get("basis_method") if result else None,
            "confidence": result.get("confidence") if result else None,
            "q10": result.get("q10") if result else None,
            "q90": result.get("q90") if result else None,
            "advisoryOnly": result.get("advisory_only") if result else True,
            "actionAllowed": result.get("action_allowed") if result else False,
            "safetyMode": result.get("safety_mode") if result else SAFETY_MODE,
        } if result else {}
        return JSONResponse(content=camel_result)
    except Exception:
        log.exception("[ml] Fehler in get_ml_predict")
        return JSONResponse(status_code=500, content={"error": "Prediction fehlgeschlagen"})


@router.get("/api/ml/experiments")
def get_ml_experiments(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> JSONResponse:
    """Liefert Liste der ML-Experimente (paginiert).

    Query-Parameter:
        offset: Startindex (Standard: 0)
        limit:  Max. Anzahl Einträge (Standard: 20, max. 200)
    """
    total = count_experiments()
    items = load_experiments(limit=limit, offset=offset)
    return JSONResponse(content={"items": items, "total": total, "offset": offset, "limit": limit})


@router.get("/api/ml/experiments/compare")
def get_ml_experiments_compare(
    a: str = Query(default=""),
    b: str = Query(default=""),
) -> JSONResponse:
    """Vergleicht zwei ML-Experimente anhand ihrer IDs."""
    experiments = load_experiments(limit=100)
    exp_by_id = {e["experiment_id"]: e for e in experiments}
    exp_a = exp_by_id.get(a)
    exp_b = exp_by_id.get(b)

    delta = {}
    if exp_a and exp_b:
        for key in ("mae", "r2", "n_train"):
            va = (exp_a.get("metrics") or {}).get(key)
            vb = (exp_b.get("metrics") or {}).get(key)
            if va is not None and vb is not None:
                try:
                    delta[key] = round(float(vb) - float(va), 4)
                except Exception:
                    pass

    return JSONResponse(content={
        "experimentA": exp_a,
        "experimentB": exp_b,
        "comparable": exp_a is not None and exp_b is not None,
        "delta": delta,
    })


_ab_state: dict = {
    "active": False,
    "championMae": None,
    "challengerMae": None,
    "evaluated": 0,
}


@router.get("/api/ml/ab-status")
def get_ml_ab_status() -> JSONResponse:
    """Liefert A/B-Test-Status des GBRT Challenger-Modells.

    A/B-Testing noch nicht migriert — gibt lokalen Stub-State zurück.
    """
    return JSONResponse(content=_ab_state)


@router.get("/api/ml/monitoring")
def get_ml_monitoring(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> JSONResponse:
    """Liefert Monitoring-Events (Drift, MAE-Spikes, A/B-Ergebnisse), paginiert.

    Query-Parameter:
        offset: Startindex (Standard: 0)
        limit:  Max. Anzahl Einträge (Standard: 20, max. 200)
    """
    total = count_monitoring_events()
    items = load_monitoring_events(limit=limit, offset=offset)
    return JSONResponse(content={
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "safetyMode": SAFETY_MODE,
    })


@router.post("/api/ml/retrain", dependencies=[Depends(require_admin_key)])
def post_ml_retrain() -> JSONResponse:
    """Löst manuelles LightGBM-Retraining aus (asynchron via Thread)."""
    import threading

    with _ml_lock:
        if _ml_state["training"]:
            return JSONResponse(content={"ok": False, "reason": "Training läuft bereits"})

    def _do_retrain():
        try:
            ml_train_model()
        except Exception as e:
            log.warning("[ml] Manuelles Retraining fehlgeschlagen: %s", e)

    threading.Thread(target=_do_retrain, daemon=True).start()
    return JSONResponse(content={"ok": True, "message": "LightGBM-Retraining gestartet"})


@router.post("/api/ml/monitoring/tick", dependencies=[Depends(require_admin_key)])
def post_monitoring_tick() -> JSONResponse:
    """Manueller Monitoring-Tick (Drift-Check, MAE-Check etc.)."""
    try:
        from app.research.worker import monitoring_tick
        monitoring_tick()
        return JSONResponse(content={"ok": True})
    except Exception:
        log.exception("[ml] monitoring_tick Fehler")
        return JSONResponse(status_code=500, content={"error": "Monitoring-Tick fehlgeschlagen"})


@router.post("/api/ml/predict-batch")
@router.post("/api/predict-batch")
def post_predict_batch(body: PredictBatchRequest) -> JSONResponse:
    """Batch-Prediction mit Cache + Micro-Variation für Live-Ticker.

    Liefert predicted OR für bis zu 300 Artikel in einem Request.
    Gecachte Predictions (<5 Min alt) werden mit leichter Micro-Variation
    zurückgegeben um Live-Updates zu simulieren.
    """
    import datetime as _dt

    articles = body.articles
    if len(articles) > 300:
        return JSONResponse(
            status_code=400,
            content={"error": "articles must be a list with max 300 entries"},
        )

    t0 = time.monotonic()
    now = _dt.datetime.now()
    time_slot = int(time.time()) // 10
    cache = _predict_cache
    out: dict = {}
    need_predict: list = []

    # 1. Cache-Lookup
    for a in articles:
        art_id = a.get("id", a.get("message_id", ""))
        if not art_id:
            continue
        cached = cache.get(art_id)
        if cached and (time.time() - cached["ts"]) < 300:
            base_or = cached["base_or"]
            h = hashlib.md5(f"{art_id}:{time_slot}".encode()).digest()
            seed = int.from_bytes(h[:4], "little")
            var = ((seed % 1000) - 500) / 5000.0
            live_or = max(0.5, round(base_or * (1 + var), 2))
            out[art_id] = {
                "or": live_or, "predictedOr": live_or,
                "basis": cached["basis"], "confidence": cached["confidence"],
                "q10": cached["q10"], "q90": cached["q90"],
                "modelType": cached["basis"],
            }
        else:
            need_predict.append((art_id, a))

    # 2. Fehlende Predictions
    for art_id, a in need_predict:
        push = {
            "title": a.get("title", ""),
            "headline": a.get("title", ""),
            "cat": a.get("cat", "News"),
            "hour": a.get("hour", now.hour),
            "ts_num": now.timestamp(),
            "is_eilmeldung": a.get("is_eilmeldung", a.get("isBreaking", False)),
            "is_bild_plus": a.get("is_bild_plus", False),
            "channels": a.get("channels", []),
        }
        try:
            from app.ml.predict import predict_or
            res = predict_or(push, _research_state)
            base_or = res.get("predicted_or", 5.0) if res else 5.0
            basis = res.get("basis_method", "fallback") if res else "fallback"
            confidence = res.get("confidence", 0.3) if res else 0.3
            q10 = res.get("q10", max(0.1, base_or - 1.5)) if res else max(0.1, base_or - 1.5)
            q90 = res.get("q90", base_or + 1.5) if res else base_or + 1.5
        except Exception as e:
            log.warning("[predict-batch] Fehler für %s: %s", art_id, e)
            base_or, basis, confidence, q10, q90 = 5.0, "fallback", 0.3, 2.0, 8.0

        base_or = max(0.5, min(25.0, base_or))
        cache[art_id] = {
            "base_or": base_or, "basis": basis,
            "confidence": confidence, "q10": q10, "q90": q90,
            "ts": time.time(),
        }
        h = hashlib.md5(f"{art_id}:{time_slot}".encode()).digest()
        seed = int.from_bytes(h[:4], "little")
        var = ((seed % 1000) - 500) / 5000.0
        live_or = max(0.5, round(base_or * (1 + var), 2))
        out[art_id] = {
            "or": live_or, "predictedOr": live_or,
            "basis": basis, "confidence": confidence,
            "q10": q10, "q90": q90, "modelType": basis,
        }

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    model_type = next((v.get("modelType", "unknown") for v in out.values()), "none")

    return JSONResponse(content={
        "predictions": out,
        "modelType": model_type,
        "count": len(out),
        "elapsedMs": elapsed_ms,
        "cached": len(articles) - len(need_predict),
        "computed": len(need_predict),
    })
