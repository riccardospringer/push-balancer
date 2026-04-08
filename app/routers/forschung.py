"""app/routers/forschung.py — Forschungs-Endpunkte.

GET /api/forschung         — Research-Institut-Daten (autonome Analyse)
GET /api/learnings         — ML-Learnings
GET /api/research-rules    — Aktive Forschungsregeln
"""
import logging
import math
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


@router.get("/api/forschung")
def get_forschung() -> JSONResponse:
    """Liefert Research-Institut-Daten mit autonomer Push-Analyse.

    Der Research-Worker analysiert Daten im Hintergrund alle 20s.
    Dieser Endpoint gibt den vollständigen aktuellen State zurück.
    """
    if not _research_state.get("push_data"):
        return JSONResponse(content={
            "accuracy": 0, "accuracyTrend": 0, "accuracyTarget": 99.5,
            "insightsToday": 0, "insightsTrend": 0, "pagesToday": 0, "pagesTrend": 0,
            "researchers": [], "guestResearchers": [], "guestExchanges": [],
            "orchestrator": {
                "id": "ml-system", "name": "ML Pipeline",
                "role": "Autonomes System", "status": "loading",
                "currentDirective": "Lade Push-Daten...",
                "teamsActive": 1, "decisionsToday": 0, "schwabDecisions": [],
            },
            "bildTeam": [], "ticker": [], "learning": [],
            "dissertations": [], "diskurs": [],
            "weekComparison": {}, "liveRules": [], "liveRulesCount": 0,
            "nPushes": 0, "lastPushTs": 0, "loading": True,
            "matureCount": 0, "freshCount": 0, "freshPushes": [],
            "researchMemory": {}, "researchMemoryTotal": 0, "researchLog": [],
            "researchProjects": [], "researchMilestones": [], "instituteReview": {},
        })

    s = _research_state
    findings = s.get("findings", {})
    push_data = s.get("push_data", [])
    n_pushes = s.get("mature_count", len(push_data))
    ticker_entries = s.get("ticker_entries", [])
    schwab_decisions = s.get("schwab_decisions", [])
    live_rules = s.get("live_rules", [])
    n_live_rules = len([r for r in live_rules if r.get("active")])
    rolling_acc = s.get("rolling_accuracy", 0.0)

    # Accuracy-Trend: Differenz letzter zwei Einträge der Accuracy-History
    acc_history = s.get("accuracy_history", [])
    acc_trend = 0.0
    if len(acc_history) >= 2:
        acc_trend = round(
            acc_history[-1].get("accuracy", 0.0) - acc_history[-2].get("accuracy", 0.0), 3
        )

    # Orchestrator-Direktive aus MAE-Score ableiten
    primary_mae = s.get("ensemble_mae", 0) or s.get("basis_mae", 0)
    all_ors = sorted([p["or"] for p in push_data if 0 < p.get("or", 0) <= 30.0])
    mean_or = all_ors[len(all_ors) // 2] if all_ors else 4.0
    mean_or_directive = max(mean_or, 0.5)
    treff_score = max(0.0, min(100.0, (1 - primary_mae / mean_or_directive) * 100)) if primary_mae > 0 else 0.0
    if treff_score < 50:
        directive = (f"Priorität: Treffsicherheit bei {treff_score:.0f}%"
                     f" (Ø {primary_mae:.1f}pp daneben) — Modell verbessern, Ziel >90%")
    elif treff_score < 75:
        directive = (f"Treffsicherheit {treff_score:.0f}%"
                     f" (Ø {primary_mae:.1f}pp) — auf gutem Weg. {n_live_rules} Live-Regeln aktiv")
    else:
        directive = (f"Treffsicherheit {treff_score:.0f}%"
                     f" (Ø {primary_mae:.1f}pp) — stark! Fokus auf Feintuning")

    orchestrator = {
        "id": "ml-system",
        "name": "ML Pipeline",
        "role": "Autonomes System",
        "status": "active",
        "currentDirective": directive,
        "teamsActive": 1,
        "decisionsToday": len(schwab_decisions),
        "schwabDecisions": schwab_decisions[-10:],
    }

    # ML-Modell-Info aus aktuellen States
    gbrt_info: dict = {}
    try:
        from app.ml.gbrt import _gbrt_lock, _gbrt_model
        with _gbrt_lock:
            _gm = _gbrt_model
        if _gm is not None:
            gbrt_info = {
                "type": "GBRT",
                "nTrees": len(getattr(_gm, "trees", [])),
                "metrics": getattr(_gm, "train_metrics", {}),
                "featureImportance": _gm.feature_importance(10) if hasattr(_gm, "feature_importance") else [],
            }
    except Exception:
        pass

    lgbm_info: dict = {}
    try:
        from app.ml.lightgbm_model import _ml_state, _ml_lock
        with _ml_lock:
            lgbm_loaded = _ml_state.get("model") is not None
            lgbm_metrics = _ml_state.get("metrics", {})
            lgbm_features = _ml_state.get("feature_names", [])
        if lgbm_loaded:
            lgbm_info = {
                "type": "LightGBM",
                "loaded": True,
                "metrics": lgbm_metrics,
                "featureCount": len(lgbm_features),
            }
    except Exception:
        pass

    # Pending Approvals (nur offene)
    pending_approvals = [
        a for a in s.get("pending_approvals", [])[-100:]
        if a.get("status") == "pending"
    ]

    # Research Log (letzte 20 Einträge)
    research_log = s.get("research_log", [])[-20:]

    last_push_ts = max((p.get("ts_num", 0) for p in push_data), default=0)

    return JSONResponse(content={
        # Accuracy
        "accuracy": round(rolling_acc * 100, 1),
        "accuracyTrend": round(acc_trend * 100, 2),
        "accuracyTarget": 99.5,
        "accuracyByCat": s.get("accuracy_by_cat", {}),
        "accuracyTrend7d": s.get("accuracy_trend", []),
        "maeTrend": s.get("mae_trend", []),
        "maeByCat": s.get("mae_by_cat", {}),
        "maeByHour": s.get("mae_by_hour", {}),
        "ensembleAccuracy": s.get("ensemble_accuracy", 0.0),
        "ensembleMae": s.get("ensemble_mae", 0.0),
        "ensembleAccuracyDelta": s.get("ensemble_accuracy_delta", 0.0),
        "basisMae": s.get("basis_mae", 0.0),
        # Insights
        "insightsToday": len(ticker_entries),
        "insightsTrend": 0,
        "pagesToday": n_pushes * 8,
        "pagesTrend": 0,
        # Researchers (leer — keine fiktiven Profile)
        "researchers": [],
        "guestResearchers": [],
        "guestExchanges": [],
        "bildTeam": [],
        "dissertations": [],
        "diskurs": [],
        # Orchestrator
        "orchestrator": orchestrator,
        # Research State
        "findings": findings,
        "ticker": ticker_entries[-50:],
        "learning": research_log,
        "weekComparison": s.get("week_comparison", {}),
        "liveRules": live_rules,
        "liveRulesCount": len([r for r in live_rules if r.get("active")]),
        "researchModifiers": s.get("research_modifiers", {}),
        "externalContext": s.get("external_context", {}),
        "algoScoreAnalysis": s.get("algo_score_analysis", {}),
        # Push-Counts
        "nPushes": n_pushes,
        "lastPushTs": last_push_ts,
        "matureCount": s.get("mature_count", 0),
        "freshCount": s.get("fresh_count", 0),
        "freshPushes": s.get("fresh_pushes", [])[:20],
        # Research Memory
        "researchMemory": s.get("research_memory", {}),
        "researchMemoryTotal": len(s.get("research_memory", {})),
        "researchLog": research_log,
        "researchProjects": [],
        "researchMilestones": [],
        "instituteReview": {},
        # Schwab
        "schwabDecisions": schwab_decisions[-20:],
        "pendingApprovals": pending_approvals,
        # ML-Modell-Info
        "mlAnalytics": {
            "gbrt": gbrt_info,
            "lightgbm": lgbm_info,
        },
        # Analysis Meta
        "analysisGeneration": s.get("analysis_generation", 0),
        "lastAnalysis": s.get("last_analysis", 0),
        "loading": False,
    })


@router.get("/api/learnings")
def get_learnings() -> JSONResponse:
    """Liefert ML-Learnings aus dem Research-State.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_learnings()
        hierher migrieren.
    """
    findings = _research_state.get("findings", {})
    return JSONResponse(content={
        "findings": findings,
        "researchMemory": _research_state.get("research_memory", {}),
        "nPushes": len(_research_state.get("push_data", [])),
        "lastAnalysis": _research_state.get("last_analysis", 0),
    })


@router.get("/api/research-rules")
def get_research_rules(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> JSONResponse:
    """Liefert aktive Forschungsregeln (paginiert).

    Query-Parameter:
        offset: Startindex (Standard: 0)
        limit:  Max. Anzahl Regeln (Standard: 20, max. 200)
    """
    rules = _research_state.get("live_rules", [])
    active = [r for r in rules if r.get("active")]
    total = len(active)
    items = active[offset: offset + limit]
    accuracy = _research_state.get("rolling_accuracy", 0.0)
    return JSONResponse(content={
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "version": _research_state.get("live_rules_version", 0),
        "accuracy": round(accuracy, 1),
        "nPushesAnalyzed": len(_research_state.get("push_data", [])),
        "lastUpdate": _research_state.get("last_analysis", 0),
    })
