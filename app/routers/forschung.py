"""app/routers/forschung.py — Forschungs-Endpunkte.

GET /api/forschung         — Research-Institut-Daten (autonome Analyse)
GET /api/learnings         — ML-Learnings
GET /api/research-rules    — Aktive Forschungsregeln
"""
import logging
import time

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
    """Ehrliche, datenbasierte Learnings aus den letzten Monaten.

    Berechnet Stunden-, Kategorie-, Titellängen-, Wochentag- und Vorhersage-Analysen
    direkt aus der DB — identisch zur Legacy-Implementierung _serve_learnings().
    """
    import sqlite3
    import datetime
    from collections import defaultdict
    from app.database import push_db_load_all
    from app.config import PUSH_DB_PATH

    now_ts = time.time()
    cutoff_24h = now_ts - 24 * 3600
    cutoff_90d = now_ts - 90 * 86400
    cutoff_180d = now_ts - 180 * 86400

    try:
        raw = push_db_load_all()
    except Exception as exc:
        log.warning("[learnings] push_db_load_all fehlgeschlagen: %s", exc)
        return JSONResponse(content={"ready": False, "n": 0, "error": str(exc)})

    mature = [p for p in raw if p.get("ts_num", 0) < cutoff_24h and 0 < p.get("or", 0) <= 30]

    if len(mature) < 50:
        return JSONResponse(content={"ready": False, "n": len(mature)})

    # ── Datenbasis: neuere vs ältere Periode ──
    recent = [p for p in mature if p["ts_num"] >= cutoff_90d]
    older = [p for p in mature if cutoff_180d <= p["ts_num"] < cutoff_90d]

    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def _median(lst: list) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]

    mean_or_recent = _mean([p["or"] for p in recent])
    mean_or_older = _mean([p["or"] for p in older])
    or_trend_pp = round(mean_or_recent - mean_or_older, 2) if older else None

    # ── Stunden-Analyse ──
    by_hour: dict = defaultdict(list)
    for p in mature:
        h = p.get("hour", -1)
        if 0 <= h <= 23:
            by_hour[h].append(p["or"])
    hour_stats = {h: {"mean": round(_mean(v), 2), "n": len(v)} for h, v in by_hour.items() if len(v) >= 5}
    best_hour = max(hour_stats, key=lambda h: hour_stats[h]["mean"], default=None)
    worst_hour = min(hour_stats, key=lambda h: hour_stats[h]["mean"], default=None)

    # ── Kategorie-Analyse ──
    by_cat: dict = defaultdict(list)
    for p in mature:
        c = p.get("cat", "Unbekannt") or "Unbekannt"
        by_cat[c].append(p["or"])
    cat_stats = {c: {"mean": round(_mean(v), 2), "n": len(v)} for c, v in by_cat.items() if len(v) >= 10}
    cat_ranked = sorted(cat_stats.items(), key=lambda x: x[1]["mean"], reverse=True)

    # ── Titellänge-Analyse ──
    buckets: dict = {"kurz (1-40)": [], "mittel (41-65)": [], "lang (66+)": []}
    for p in mature:
        tl = p.get("title_len", 0) or 0
        if tl <= 0:
            continue
        if tl <= 40:
            buckets["kurz (1-40)"].append(p["or"])
        elif tl <= 65:
            buckets["mittel (41-65)"].append(p["or"])
        else:
            buckets["lang (66+)"].append(p["or"])
    len_stats = {k: {"mean": round(_mean(v), 2), "n": len(v)} for k, v in buckets.items() if v}

    # ── Wochentag-Analyse ──
    by_dow: dict = defaultdict(list)
    dow_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    for p in mature:
        ts = p.get("ts_num", 0)
        if ts > 0:
            dow = datetime.datetime.fromtimestamp(ts).weekday()
            by_dow[dow].append(p["or"])
    dow_stats = {dow_names[d]: {"mean": round(_mean(v), 2), "n": len(v)} for d, v in by_dow.items() if len(v) >= 5}
    best_dow = max(dow_stats, key=lambda d: dow_stats[d]["mean"], default=None)
    worst_dow = min(dow_stats, key=lambda d: dow_stats[d]["mean"], default=None)

    # ── Vorhersage-Genauigkeit aus prediction_log ──
    pred_accuracy = None
    pred_n = 0
    pred_within_2pp = 0
    pred_off_more_than_3pp = 0
    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
        pred_rows = conn.execute(
            """SELECT predicted_or, actual_or FROM prediction_log
               WHERE actual_or > 0 AND actual_or < 30 AND predicted_or > 0
               AND predicted_at > ?""",
            (int(now_ts) - 90 * 86400,),
        ).fetchall()
        conn.close()
        if pred_rows:
            errors = [abs(r[0] - r[1]) for r in pred_rows]
            pred_n = len(errors)
            pred_accuracy = round(_mean(errors), 2)
            pred_within_2pp = round(100 * sum(1 for e in errors if e <= 2) / pred_n, 1)
            pred_off_more_than_3pp = round(100 * sum(1 for e in errors if e > 3) / pred_n, 1)
    except Exception:
        pass

    # ── Eilmeldungs-Effekt ──
    breaking_or = _mean([p["or"] for p in mature if p.get("is_eilmeldung")])
    nonbreaking_or = _mean([p["or"] for p in mature if not p.get("is_eilmeldung")])
    n_breaking = sum(1 for p in mature if p.get("is_eilmeldung"))

    # ── Monats-Trend (letzte 6 Monate) ──
    monthly: dict = defaultdict(list)
    for p in mature:
        ts = p.get("ts_num", 0)
        if ts > cutoff_180d:
            m = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m")
            monthly[m].append(p["or"])
    monthly_trend = [
        {"month": m, "mean_or": round(_mean(v), 2), "n": len(v)}
        for m, v in sorted(monthly.items()) if len(v) >= 5
    ]

    return JSONResponse(content={
        "ready": True,
        "n_total": len(mature),
        "n_recent_90d": len(recent),
        "mean_or": round(_mean([p["or"] for p in mature]), 2),
        "mean_or_recent": round(mean_or_recent, 2),
        "mean_or_older": round(mean_or_older, 2),
        "or_trend_pp": or_trend_pp,
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "hour_stats": hour_stats,
        "cat_ranked": [[c, s] for c, s in cat_ranked],
        "len_stats": len_stats,
        "dow_stats": dow_stats,
        "best_dow": best_dow,
        "worst_dow": worst_dow,
        "pred_accuracy_mae": pred_accuracy,
        "pred_n": pred_n,
        "pred_within_2pp": pred_within_2pp,
        "pred_off_more_than_3pp": pred_off_more_than_3pp,
        "breaking_or": round(breaking_or, 2) if breaking_or else None,
        "nonbreaking_or": round(nonbreaking_or, 2) if nonbreaking_or else None,
        "n_breaking": n_breaking,
        "monthly_trend": monthly_trend,
        # Zusätzlich: Research-State-Daten für Kompatibilität
        "findings": _research_state.get("findings", {}),
        "researchMemory": _research_state.get("research_memory", {}),
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
