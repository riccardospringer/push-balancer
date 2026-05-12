"""app/routers/forschung.py — Forschungs-Endpunkte.

GET /api/research-insights — Verdichtete Research-Insights
GET /api/research-rules    — Aktive Forschungsregeln
"""
import json
import logging
import time

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.research.worker import _research_state

log = logging.getLogger("push-balancer")
router = APIRouter()


def _warm_research_state_if_needed(max_age_s: int = 600) -> None:
    """Berechnet Research-Daten nur auf Anfrage, wenn kein frischer State vorliegt."""
    if _research_state.get("push_data") and time.time() - _research_state.get("last_analysis", 0) < max_age_s:
        return
    try:
        from app.research.worker import run_autonomous_analysis

        run_autonomous_analysis()
    except Exception as exc:
        log.warning("[forschung] on-demand research warmup failed: %s", exc)


def get_forschung() -> JSONResponse:
    """Liefert Research-Institut-Daten mit autonomer Push-Analyse.

    Der Research-Worker analysiert Daten im Hintergrund alle 20s.
    Dieser Endpoint gibt den vollständigen aktuellen State zurück.
    Keys sind snake_case — identisch zum Legacy-Monolithen (push-balancer.html erwartet das).
    """
    _warm_research_state_if_needed()

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
            "mean_or": 0, "best_hour": None, "best_hour_or": 0,
            "top_category": None, "top_category_or": 0, "accuracy_mae": 0,
            "temporal_trends": {}, "or_distribution": {}, "findings": {},
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
        "current_directive": directive,
        "teams_active": 1,
        "decisions_today": len(schwab_decisions),
        "schwab_decisions": schwab_decisions[-10:],
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

    # Fehlende Felder aus push_data berechnen (erwartet von push-balancer.html)
    all_ors_sorted = sorted([p["or"] for p in push_data if 0 < p.get("or", 0) <= 30.0])
    _median_or = all_ors_sorted[len(all_ors_sorted) // 2] if all_ors_sorted else 0.0
    # Stunden-Analyse für best_hour
    _hour_agg: dict = {}
    for p in push_data:
        h = p.get("hour", -1)
        if 0 <= h <= 23 and 0 < p.get("or", 0) <= 30:
            _hour_agg.setdefault(h, []).append(p["or"])
    _best_hour = max(_hour_agg, key=lambda h: sum(_hour_agg[h]) / len(_hour_agg[h]), default=None) if _hour_agg else None
    _best_hour_or = round(sum(_hour_agg[_best_hour]) / len(_hour_agg[_best_hour]), 2) if _best_hour is not None else 0
    # Kategorie-Analyse für top_category
    _cat_agg: dict = {}
    for p in push_data:
        c = p.get("cat", "News") or "News"
        if 0 < p.get("or", 0) <= 30:
            _cat_agg.setdefault(c, []).append(p["or"])
    _top_cat = max(_cat_agg, key=lambda c: sum(_cat_agg[c]) / len(_cat_agg[c]), default=None) if _cat_agg else None
    _top_cat_or = round(sum(_cat_agg[_top_cat]) / len(_cat_agg[_top_cat]), 2) if _top_cat else 0
    # OR-Verteilung
    _or_dist: dict = {}
    if all_ors_sorted:
        n = len(all_ors_sorted)
        _or_dist = {
            "min": round(all_ors_sorted[0], 2),
            "q1": round(all_ors_sorted[n // 4], 2),
            "median": round(all_ors_sorted[n // 2], 2),
            "q3": round(all_ors_sorted[3 * n // 4], 2),
            "max": round(all_ors_sorted[-1], 2),
            "mean": round(sum(all_ors_sorted) / n, 2),
            "n": n,
        }

    return JSONResponse(content={
        # Accuracy — rolling_acc ist bereits in % (0-100), KEIN *100 mehr!
        "accuracy": round(rolling_acc, 1),
        "accuracy_trend": round(acc_trend, 2),
        "accuracy_target": 99.5,
        "accuracy_by_cat": s.get("accuracy_by_cat", {}),
        "accuracy_mae": round(s.get("basis_mae", 0.0) or s.get("ensemble_mae", 0.0), 3),
        "mae_trend": s.get("mae_trend", []),
        "mae_by_cat": s.get("mae_by_cat", {}),
        "mae_by_hour": s.get("mae_by_hour", {}),
        "ensemble_accuracy": s.get("ensemble_accuracy", 0.0),
        "ensemble_mae": s.get("ensemble_mae", 0.0),
        "ensemble_accuracy_delta": s.get("ensemble_accuracy_delta", 0.0),
        "basis_mae": s.get("basis_mae", 0.0),
        # Insights
        "insights_today": len(ticker_entries),
        "insights_trend": 0,
        "pages_today": n_pushes * 8,
        "pages_trend": 0,
        # Researchers (leer — keine fiktiven Profile)
        "researchers": [],
        "guest_researchers": [],
        "guest_exchanges": [],
        "bild_team": [],
        "dissertations": [],
        "diskurs": [],
        # Orchestrator
        "orchestrator": orchestrator,
        # Research State
        "findings": findings,
        "ticker": ticker_entries[-50:],
        "learning": research_log,
        "week_comparison": s.get("week_comparison", {}),
        "live_rules": [r for r in live_rules if r.get("active")],
        "live_rules_count": n_live_rules,
        "research_modifiers": s.get("research_modifiers", {}),
        "external_context": s.get("external_context", {}),
        "algo_score_analysis": s.get("algo_score_analysis", {}),
        # Push-Counts (snake_case — erwartet von push-balancer.html)
        "n_pushes": n_pushes,
        "last_push_ts": last_push_ts,
        "mature_count": s.get("mature_count", 0),
        "fresh_count": s.get("fresh_count", 0),
        "fresh_pushes": s.get("fresh_pushes", [])[:20],
        # Berechnete OR-Felder (fehlten bisher)
        "mean_or": round(_median_or, 2),
        "best_hour": _best_hour,
        "best_hour_or": _best_hour_or,
        "top_category": _top_cat,
        "top_category_or": _top_cat_or,
        "or_distribution": _or_dist,
        "temporal_trends": s.get("temporal_trends", {}),
        # Research Memory
        "research_memory": s.get("research_memory", {}),
        "research_memory_total": len(s.get("research_memory", {})),
        "research_log": research_log,
        "research_projects": [],
        "research_milestones": [],
        "institute_review": {},
        # Schwab
        "schwab_decisions": schwab_decisions[-20:],
        "pending_approvals": pending_approvals,
        # ML-Modell-Info
        "ml_analytics": {
            "gbrt": gbrt_info,
            "lightgbm": lgbm_info,
        },
        # Analysis Meta
        "analysis_generation": s.get("analysis_generation", 0),
        "last_analysis": s.get("last_analysis", 0),
        "loading": False,
    })


@router.get("/api/research-insights")
def get_research_insights() -> JSONResponse:
    """Stable research insights contract for the frontend and OpenAPI clients."""
    response = get_forschung()
    payload = json.loads(response.body.decode("utf-8"))
    learnings = payload.get("learning", [])[-10:]
    return JSONResponse(
        content={
            "learnings": [
                {
                    "id": str(index),
                    "text": str(item.get("message") or item.get("title") or item),
                    "impact": "medium",
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                for index, item in enumerate(learnings, start=1)
            ],
            "experiments": [],
            "abTest": None,
        }
    )


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
    generated_at = _research_state.get("last_analysis", 0)
    return JSONResponse(content={
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "version": _research_state.get("live_rules_version", 0),
        "accuracy": round(accuracy, 1),
        "nPushesAnalyzed": len(_research_state.get("push_data", [])),
        "lastUpdate": generated_at,
        "rules": [
            {
                "id": str(rule.get("id", index)),
                "category": str(rule.get("category") or rule.get("cat") or "news"),
                "rule": str(
                    rule.get("title")
                    or rule.get("rule")
                    or rule.get("message")
                    or "Research rule"
                ),
                "confidence": float(rule.get("confidence") or 0),
                "supportCount": int(rule.get("supportCount") or rule.get("n") or 0),
                "createdAt": (
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(generated_at))
                    if generated_at
                    else ""
                ),
            }
            for index, rule in enumerate(items, start=offset + 1)
        ],
        "rollingAccuracy": round(float(accuracy), 1),
        "generatedAt": (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(generated_at))
            if generated_at
            else ""
        ),
    })


@router.get("/api/forschung")
def get_forschung_alias() -> JSONResponse:
    """Legacy-Alias für push-balancer.html."""
    return get_forschung()


@router.get("/api/learnings")
def get_learnings_alias() -> JSONResponse:
    """Legacy-Alias für push-balancer.html."""
    return get_forschung()
