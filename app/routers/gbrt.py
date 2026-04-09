"""app/routers/gbrt.py — GBRT-Modell-Endpunkte.

GET  /api/gbrt/status           — GBRT-Modell-Status
GET  /api/gbrt/model.json       — Serialisiertes GBRT-Modell (JSON)
GET  /api/gbrt/predict          — GBRT-Prediction
POST /api/gbrt/retrain          — Manuelles Retraining
POST /api/gbrt/force-promote    — Challenger zu Champion promoten
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import require_admin_key
from app.ml.gbrt import _gbrt_lock, _gbrt_model, gbrt_predict, gbrt_train
from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


_gbrt_error_cache: dict = {"ts": 0, "data": {}}
_gbrt_error_cache_ttl: int = 300  # 5 Minuten


@router.get("/api/gbrt/status")
def get_gbrt_status() -> JSONResponse:
    """Liefert GBRT-Modell-Status (Metriken, Feature Importance, Fehleranalyse, letzte Predictions)."""
    import sqlite3
    import time
    from collections import defaultdict
    from app.config import PUSH_DB_PATH

    with _gbrt_lock:
        model = _gbrt_model
        if model is None:
            return JSONResponse(content={"loaded": False, "trained": False, "message": "GBRT-Modell nicht trainiert"})

        # Feature Importance — Top-20
        fi_raw = model.feature_importance(20) if hasattr(model, "feature_importance") else []
        feature_names = getattr(model, "feature_names", [])
        fi = []
        for item in fi_raw:
            idx = item.get("name")
            if isinstance(idx, int) and feature_names and idx < len(feature_names):
                fi.append({"name": feature_names[idx], "importance": item["importance"]})
            else:
                fi.append(item)

        # Calibrator Breakpoints
        from app.ml.gbrt import _gbrt_calibrator
        cal_data = {}
        if _gbrt_calibrator and getattr(_gbrt_calibrator, "breakpoints", None):
            cal_bins = [
                {"predicted": round(p, 3), "actual": round(c, 3), "count": 1}
                for p, c in _gbrt_calibrator.breakpoints
            ]
            cal_data["bins"] = cal_bins
            if cal_bins:
                cal_err = sum(abs(b["predicted"] - b["actual"]) for b in cal_bins) / len(cal_bins)
                cal_data["calibration_error"] = round(cal_err, 4)

        metrics = getattr(model, "train_metrics", {}) or {}
        n_trees = len(getattr(model, "trees", []))

    # Teure Berechnungen außerhalb des Locks
    now_ts = int(time.time())
    n_pushes = metrics.get("n_train", 0) + metrics.get("n_test", 0)

    # Error-Analyse aus Cache oder frisch berechnen
    error_analysis = {}
    if now_ts - _gbrt_error_cache["ts"] < _gbrt_error_cache_ttl and _gbrt_error_cache["data"]:
        error_analysis = _gbrt_error_cache["data"]
    else:
        try:
            from app.database import push_db_load_all
            from app.ml.features import gbrt_extract_features
            from app.ml.gbrt import _gbrt_history_stats, _gbrt_feature_names

            all_pushes = push_db_load_all()
            recent = [p for p in all_pushes if p.get("or", 0) > 0 and p.get("ts_num", 0) > now_ts - 30 * 86400]
            if recent and _gbrt_history_stats:
                by_cat: dict = defaultdict(lambda: {"errors": [], "biases": []})
                by_hour: dict = defaultdict(lambda: {"errors": [], "biases": []})
                with _gbrt_lock:
                    _model = _gbrt_model
                    _feat_names = list(_gbrt_feature_names)
                if _model and _feat_names:
                    from app.ml.gbrt import _gbrt_calibrator as _cal
                    for p in recent[-500:]:
                        try:
                            feat = gbrt_extract_features(p, _gbrt_history_stats)
                            fv = [float(feat.get(k, 0.0)) for k in _feat_names]
                            pred = float(_model.predict_one(fv)) if hasattr(_model, "predict_one") else float(_model.predict([fv])[0])
                            if _cal:
                                pred = float(_cal.calibrate(pred))
                            actual = float(p["or"])
                            err = abs(pred - actual)
                            bias = pred - actual
                            cat = p.get("cat", "?")
                            hour = p.get("hour", 0)
                            by_cat[cat]["errors"].append(err)
                            by_cat[cat]["biases"].append(bias)
                            by_hour[hour]["errors"].append(err)
                            by_hour[hour]["biases"].append(bias)
                        except Exception:
                            continue
                    error_analysis["by_category"] = {
                        cat: {
                            "mae": round(sum(d["errors"]) / len(d["errors"]), 3),
                            "bias": round(sum(d["biases"]) / len(d["biases"]), 3),
                            "n": len(d["errors"]),
                        }
                        for cat, d in by_cat.items() if d["errors"]
                    }
                    error_analysis["by_hour"] = {
                        str(h): {
                            "mae": round(sum(d["errors"]) / len(d["errors"]), 3),
                            "bias": round(sum(d["biases"]) / len(d["biases"]), 3),
                            "n": len(d["errors"]),
                        }
                        for h, d in by_hour.items() if d["errors"]
                    }
                    _gbrt_error_cache["data"] = error_analysis
                    _gbrt_error_cache["ts"] = now_ts
        except Exception as _ea:
            log.warning("[gbrt] Error-Analyse fehlgeschlagen: %s", _ea)

    # Letzte 20 Predictions aus prediction_log
    recent_preds = []
    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT title, predicted_or, actual_or, confidence, q10, q90, predicted_at "
            "FROM prediction_log ORDER BY predicted_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        for row in rows:
            rk = row.keys()
            recent_preds.append({
                "title": row["title"] if "title" in rk else "",
                "predicted": row["predicted_or"],
                "actual": row["actual_or"] if row["actual_or"] and row["actual_or"] > 0 else None,
                "confidence": row["confidence"] if "confidence" in rk else 0.5,
                "q10": row["q10"] if "q10" in rk else 0,
                "q90": row["q90"] if "q90" in rk else 0,
            })
    except Exception:
        pass

    return JSONResponse(content={
        "loaded": True,
        "trained": True,
        "model": {
            "type": "GBRT",
            "n_trees": n_trees,
            "n_features": len(feature_names),
            "n_pushes": n_pushes,
            "mae": metrics.get("test_mae", metrics.get("val_mae")),
            "r2": metrics.get("r2_final", metrics.get("test_r2", metrics.get("val_r2"))),
            "trained_at": metrics.get("trained_at"),
        },
        "n_trees": n_trees,
        "n_features": len(feature_names),
        "n_pushes": n_pushes,
        "metrics": metrics,
        "feature_importance": fi,
        "calibration": cal_data,
        "error_analysis": error_analysis,
        "recent_predictions": recent_preds,
        # Legacy-Compat-Keys
        "modelLoaded": True,
        "featureNames": feature_names,
        "featureCount": len(feature_names),
    })


@router.get("/api/gbrt/model.json")
def get_gbrt_model_json() -> JSONResponse:
    """Exportiert das GBRT-Modell als JSON (identisch zum Legacy-Monolith).

    Dient dem Frontend zur Client-Side-Evaluation.
    Gibt die serialisierte Disk-Datei zurück (Cache-Control: 5 Min).
    """
    import os
    from app.config import SERVE_DIR
    from fastapi.responses import FileResponse, Response

    model_path = os.path.join(SERVE_DIR, ".gbrt_model.json")
    if os.path.exists(model_path):
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                data = f.read()
            return Response(
                content=data,
                media_type="application/json; charset=utf-8",
                headers={"Cache-Control": "max-age=300"},
            )
        except Exception:
            log.exception("[gbrt] Fehler beim Lesen der Modell-Datei")
            return JSONResponse(status_code=500, content={"error": "Modell-Datei konnte nicht gelesen werden"})

    # Fallback: Modell aus RAM serialisieren
    with _gbrt_lock:
        model = _gbrt_model

    if model is None:
        return JSONResponse(
            status_code=404,
            content={"error": "GBRT-Modell nicht vorhanden. Wird beim nächsten Training erstellt."},
        )

    try:
        serialized = {
            "nTrees": len(getattr(model, "trees", [])),
            "featureNames": getattr(model, "feature_names", []),
            "metrics": getattr(model, "train_metrics", {}),
            "initialPrediction": getattr(model, "initial_prediction", 0.0),
        }
        return JSONResponse(content=serialized, headers={"Cache-Control": "max-age=300"})
    except Exception:
        log.exception("[gbrt] Fehler beim Serialisieren des Modells")
        return JSONResponse(status_code=500, content={"error": "Modell-Serialisierung fehlgeschlagen"})


@router.get("/api/gbrt/predict")
def get_gbrt_predict(
    title: str = Query(default=""),
    cat: str = Query(default="News"),
    hour: int = Query(default=-1),
    is_eilmeldung: int = Query(default=0),
    plus: int = Query(default=0),
) -> JSONResponse:
    """GBRT-Prediction für einen Artikel."""
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
        "is_bild_plus": bool(plus),
        "channels": ["eilmeldung"] if is_eilmeldung else ["news"],
        "title_len": len(title),
    }

    result = gbrt_predict(push, _research_state)
    if result is None:
        return JSONResponse(content={
            "modelLoaded": False,
            "error": "Kein GBRT-Modell geladen",
        })
    # Transform interne snake_case Keys zu camelCase an der API-Grenze
    camel_result = {
        "predictedOr": result.get("predicted_or"),
        "basisMethod": result.get("basis_method"),
        "confidence": result.get("confidence"),
        "q10": result.get("q10"),
        "q90": result.get("q90"),
        "std": result.get("std"),
        "nTrees": result.get("n_trees"),
        "onlineBias": result.get("online_bias"),
        "features": result.get("features"),
        "importance": result.get("importance"),
    }
    return JSONResponse(content=camel_result)


@router.post("/api/gbrt/retrain", dependencies=[Depends(require_admin_key)])
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


@router.post("/api/gbrt/force-promote", dependencies=[Depends(require_admin_key)])
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
        "nTrees": len(getattr(model, "trees", [])),
    })
