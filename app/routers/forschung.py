"""app/routers/forschung.py — Forschungs-Endpunkte.

GET /api/forschung         — Research-Institut-Daten (autonome Analyse)
GET /api/learnings         — ML-Learnings
GET /api/research-rules    — Aktive Forschungsregeln
"""
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


@router.get("/api/forschung")
def get_forschung() -> JSONResponse:
    """Liefert Research-Institut-Daten mit autonomer Push-Analyse.

    Der Research-Worker analysiert Daten im Hintergrund alle 20s.
    Dieser Endpoint gibt nur den aktuellen State zurück.

    IMPLEMENTIERUNGSHINWEIS:
        Vollständige Handler-Logik aus push-balancer-server.py: _serve_forschung()
        (Zeile 13400) hierher migrieren. Enthält:
        - Fallback-Berechnung wenn findings leer
        - temporal_trends Fallback
        - week_comparison Fallback
        - research_modifiers Fallback
        - ML Analytics (GBRT, LightGBM Info)
        - Researcher-Profile Generierung
        - Ticker-Einträge
        - Live-Rules
    """
    try:
        from push_balancer_server_compat import PushBalancerHandler  # type: ignore
        # Die Forschungs-Logik ist sehr komplex — delegiere an Legacy-Handler
        # bis zur vollständigen Migration
        pass
    except ImportError:
        pass

    if not _research_state.get("push_data"):
        return JSONResponse(content={
            "accuracy": 0, "accuracy_trend": 0, "accuracy_target": 99.5,
            "insights_today": 0, "insights_trend": 0, "pages_today": 0, "pages_trend": 0,
            "researchers": [], "guest_researchers": [], "guest_exchanges": [],
            "orchestrator": {
                "id": "ml-system", "name": "ML Pipeline",
                "role": "Autonomes System", "status": "loading",
                "current_directive": "Lade Push-Daten...",
                "teams_active": 1, "decisions_today": 0, "schwab_decisions": [],
            },
            "bild_team": [], "ticker": [], "learning": [],
            "dissertations": [], "diskurs": [],
            "week_comparison": {}, "live_rules": [], "live_rules_count": 0,
            "n_pushes": 0, "last_push_ts": 0, "loading": True,
            "mature_count": 0, "fresh_count": 0, "fresh_pushes": [],
            "research_memory": {}, "research_memory_total": 0, "research_log": [],
            "research_projects": [], "research_milestones": [], "institute_review": {},
        })

    return JSONResponse(content={
        "n_pushes": len(_research_state.get("push_data", [])),
        "live_rules": _research_state.get("live_rules", []),
        "live_rules_count": len(_research_state.get("live_rules", [])),
        "week_comparison": _research_state.get("week_comparison", {}),
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
        "research_memory": _research_state.get("research_memory", {}),
        "n_pushes": len(_research_state.get("push_data", [])),
        "last_analysis": _research_state.get("last_analysis", 0),
    })


@router.get("/api/research-rules")
def get_research_rules() -> JSONResponse:
    """Liefert aktive Forschungsregeln für den Push-Kandidaten-Ablauf."""
    rules = _research_state.get("live_rules", [])
    active = [r for r in rules if r.get("active")]
    accuracy = _research_state.get("rolling_accuracy", 0.0)
    return JSONResponse(content={
        "rules": active,
        "version": _research_state.get("live_rules_version", 0),
        "accuracy": round(accuracy, 1),
        "n_pushes_analyzed": len(_research_state.get("push_data", [])),
        "last_update": _research_state.get("last_analysis", 0),
    })
