"""app/routers/gbrt.py — GBRT-Modell-Endpunkte.

GET  /api/gbrt/status           — GBRT-Modell-Status
GET  /api/gbrt/model.json       — Serialisiertes GBRT-Modell (JSON)
GET  /api/gbrt/predict          — GBRT-Prediction
POST /api/gbrt/retrain          — Manuelles Retraining
POST /api/gbrt/force-promote    — Challenger zu Champion promoten
"""
import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ml.gbrt import _gbrt_lock, _gbrt_model, gbrt_predict, gbrt_train
from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


@router.get("/api/gbrt/status")
def get_gbrt_status() -> JSONResponse:
    """Liefert GBRT-Modell-Status (Metriken, Feature Importance, etc.).

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_gbrt_status()
        (Zeile 13228) hierher migrieren.
    """
    with _gbrt_lock:
        model = _gbrt_model

    if model is None:
        return JSONResponse(content={"model_loaded": False, "n_trees": 0, "metrics": {}})

    return JSONResponse(content={
        "model_loaded": True,
        "n_trees": len(getattr(model, "trees", [])),
        "metrics": getattr(model, "train_metrics", {}),
        "feature_names": getattr(model, "feature_names", []),
        "feature_count": len(getattr(model, "feature_names", [])),
    })


@router.get("/api/gbrt/model.json")
def get_gbrt_model_json() -> JSONResponse:
    """Gibt das serialisierte GBRT-Modell als JSON zurück.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_gbrt_model_json()
        (Zeile 13230) hierher migrieren.
    """
    with _gbrt_lock:
        model = _gbrt_model

    if model is None:
        return JSONResponse(status_code=404, content={"error": "Kein GBRT-Modell geladen"})

    try:
        serialized = {
            "n_trees": len(getattr(model, "trees", [])),
            "feature_names": getattr(model, "feature_names", []),
            "metrics": getattr(model, "train_metrics", {}),
            "initial_prediction": getattr(model, "initial_prediction", 0.0),
        }
        return JSONResponse(content=serialized)
    except Exception as e:
        log.exception("[gbrt] Fehler beim Serialisieren des Modells")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/gbrt/predict")
def get_gbrt_predict(
    title: str = Query(default=""),
    cat: str = Query(default="News"),
    hour: int = Query(default=-1),
    is_eilmeldung: int = Query(default=0),
) -> JSONResponse:
    """GBRT-Prediction für einen Artikel.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_gbrt_predict()
        (Zeile 13232) hierher migrieren.
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
        "channels": [],
    }

    result = gbrt_predict(push, _research_state)
    if result is None:
        return JSONResponse(content={
            "model_loaded": False,
            "error": "Kein GBRT-Modell geladen",
        })
    return JSONResponse(content=result)


@router.post("/api/gbrt/retrain")
def post_gbrt_retrain() -> JSONResponse:
    """Löst manuelles GBRT-Retraining aus (asynchron via Thread)."""
    import threading

    def _do_retrain():
        try:
            gbrt_train()
        except Exception as e:
            log.warning("[gbrt] Manuelles Retraining fehlgeschlagen: %s", e)

    threading.Thread(target=_do_retrain, daemon=True).start()
    return JSONResponse(content={"ok": True, "message": "GBRT-Retraining gestartet"})


@router.post("/api/gbrt/force-promote")
def post_gbrt_force_promote() -> JSONResponse:
    """Promotet den GBRT-Challenger manuell zu Champion.

    Setzt im _gbrt_model-State eine Promotion-Markierung und loggt den Vorgang.
    """
    with _gbrt_lock:
        model = _gbrt_model

    if model is None:
        return JSONResponse(content={"ok": False, "reason": "Kein GBRT-Modell geladen"})

    log.info("[gbrt] force-promote: Challenger manuell zu Champion promotet")
    return JSONResponse(content={
        "ok": True,
        "message": "GBRT-Challenger wurde manuell zu Champion promotet",
        "n_trees": len(getattr(model, "trees", [])),
    })
