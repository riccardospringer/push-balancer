"""app/research/worker.py — Autonomer Research Worker Thread.

Bündelt den Research-Worker und alle Hilfsfunktionen aus push-balancer-server.py.
"""
from __future__ import annotations

import datetime
import logging
import math
import re
import sqlite3
import threading
import time
from collections import defaultdict

log = logging.getLogger("push-balancer")

# ── Globaler Research State (module-level) ─────────────────────────────────
_research_state: dict = {
    "last_fetch": 0,
    "push_data": [],
    "prev_push_count": 0,
    "findings": {},
    "ticker_entries": [],
    "last_analysis": 0,
    "analysis_generation": 0,
    "cumulative_insights": 0,
    "analysis_lock": None,
    "accuracy_history": [],
    "rolling_accuracy": 0.0,
    "accuracy_by_cat": {},
    "accuracy_trend": [],
    "mae_trend": [],
    "mae_by_cat": {},
    "mae_by_hour": {},
    "basis_mae": 0.0,
    "ensemble_accuracy": 0.0,
    "ensemble_mae": 0.0,
    "ensemble_accuracy_trend": [],
    "ensemble_accuracy_delta": 0.0,
    "schwab_decisions": [],
    "schwab_current": "",
    "live_rules": [],
    "live_rules_version": 0,
    "research_memory": {},
    "research_log": [],
    "prev_accuracy": 0.0,
    "prev_findings_hash": "",
    "pending_approvals": [],
    "approval_counter": 0,
    "decided_topics": set(),
    "prediction_feedback": [],
    "tuning_history": [],
    "tuning_params": {},
    "tuning_params_version": 0,
    "_last_tuning_call": 0,
    "fresh_pushes": [],
    "mature_count": 0,
    "fresh_count": 0,
    "cutoff_24h": 0,
    "week_comparison": {},
    "live_pulse": [],
    "research_modifiers": {},
    "external_context": {},
    "algo_score_analysis": {},
    "_stacking_counter": 0,
    "_worker_first_log": True,
    "_sport_data": [],
    "_nonsport_data": [],
    "_sport_n": 0,
    "_nonsport_n": 0,
}
_research_state["analysis_lock"] = threading.RLock()
_research_state_lock = threading.Lock()

# ── Health State ───────────────────────────────────────────────────────────
_health_state: dict = {
    "status": "starting",
    "uptime_start": time.time(),
    "last_check": 0,
    "checks_ok": 0,
    "checks_fail": 0,
    "endpoints": {},
}

# ── Feed Cache ─────────────────────────────────────────────────────────────
_feed_cache: dict = {
    "competitors": {"data": None, "ts": 0},
    "international": {"data": None, "ts": 0},
    "sport_competitors": {"data": None, "ts": 0},
    "sport_europa": {"data": None, "ts": 0},
    "sport_global": {"data": None, "ts": 0},
}
_feed_cache_lock = threading.Lock()
_FEED_CACHE_TTL: int = 300  # 5 Minuten

# ── XOR Perf Cache ─────────────────────────────────────────────────────────
_xor_perf_cache: dict = {
    "word_perf": {},
    "cat_hour_perf": {},
    "eil_perf": {},
    "global_avg": 4.77,
    "built_at": 0,
}
_xor_perf_lock = threading.Lock()

# ── Residual Corrector ────────────────────────────────────────────────────
_residual_corrector: dict = {
    "global_bias": 0.0,
    "cat_bias": {},
    "hourgroup_bias": {},
    "n_samples": 0,
    "last_update_ts": 0,
    "recent_residuals": [],
}
_residual_corrector_lock = threading.Lock()

# ── Tageszeit-Gruppen für Residual Corrector ──────────────────────────────
_RESIDUAL_HOURGROUPS: dict = {
    "morning": range(6, 12),
    "afternoon": range(12, 18),
    "evening": range(18, 23),
    "night": list(range(23, 24)) + list(range(0, 6)),
}


def _hour_to_group(h: int) -> str:
    """Stunde (0-23) → Tageszeit-Gruppe."""
    for name, hours in _RESIDUAL_HOURGROUPS.items():
        if h in hours:
            return name
    return "afternoon"


# ── Rolling Accuracy ───────────────────────────────────────────────────────

def _update_rolling_accuracy(push_data: list, state: dict) -> None:
    """Berechne Rolling Prediction Accuracy mit echtem temporalem Walk-Forward.

    Kein Data Leakage: jeder Push wird nur mit zeitlich vorherigen Daten bewertet.
    """
    valid = [p for p in push_data if 0 < p.get("or", 0) <= 30.0 and p.get("ts_num", 0) > 0]
    if len(valid) < 10:
        return
    valid.sort(key=lambda x: x["ts_num"])

    emo_words = {"krieg", "terror", "tod", "sterben", "schock", "skandal",
                 "drama", "horror", "mord", "crash", "warnung", "razzia", "exklusiv"}
    breaking_markers = {"+++", "eilmeldung", "breaking", "sondermeldung"}

    for p in valid:
        p["_weekday"] = datetime.datetime.fromtimestamp(p["ts_num"]).weekday()
        _tl = p.get("title", "").lower()
        p["_is_emo"] = any(w in _tl for w in emo_words)
        p["_is_breaking"] = any(m in _tl for m in breaking_markers)
        p["_title_len"] = "short" if len(_tl) < 50 else "long" if len(_tl) > 100 else "medium"
        p["_cat_hour"] = f"{p.get('cat', 'News')}_{p.get('hour', 0)}"

    cat_sums: dict = defaultdict(float)
    cat_counts: dict = defaultdict(int)
    hour_sums: dict = defaultdict(float)
    hour_counts: dict = defaultdict(int)
    day_sums: dict = defaultdict(float)
    day_counts_d: dict = defaultdict(int)
    emo_sums: dict = {"emo": 0.0, "neutral": 0.0}
    emo_counts_d: dict = {"emo": 0, "neutral": 0}
    brk_sums: dict = {"brk": 0.0, "normal": 0.0}
    brk_counts: dict = {"brk": 0, "normal": 0}
    tlen_sums: dict = defaultdict(float)
    tlen_counts: dict = defaultdict(int)
    cat_hour_sums: dict = defaultdict(float)
    cat_hour_counts: dict = defaultdict(int)
    total_or = 0.0
    total_count = 0

    def _add(p: dict) -> None:
        cat_sums[p.get("cat", "News")] += p["or"]
        cat_counts[p.get("cat", "News")] += 1
        hour_sums[p.get("hour", 0)] += p["or"]
        hour_counts[p.get("hour", 0)] += 1
        day_sums[p["_weekday"]] += p["or"]
        day_counts_d[p["_weekday"]] += 1
        ek = "emo" if p["_is_emo"] else "neutral"
        emo_sums[ek] += p["or"]
        emo_counts_d[ek] += 1
        bk = "brk" if p["_is_breaking"] else "normal"
        brk_sums[bk] += p["or"]
        brk_counts[bk] += 1
        tlen_sums[p["_title_len"]] += p["or"]
        tlen_counts[p["_title_len"]] += 1
        cat_hour_sums[p["_cat_hour"]] += p["or"]
        cat_hour_counts[p["_cat_hour"]] += 1

    warmup = min(50, len(valid) // 2)
    for p in valid[:warmup]:
        _add(p)
        total_or += p["or"]
        total_count += 1

    cat_residuals: dict = defaultdict(list)
    prelim_predictions: list = []
    baseline_global_errors: list = []
    baseline_cat_errors: list = []

    for i in range(warmup, len(valid)):
        p = valid[i]
        actual = p["or"]
        cat = p.get("cat", "News")
        hr = p.get("hour", 0)
        weekday = p["_weekday"]
        is_emo = p["_is_emo"]

        global_mean = total_or / total_count if total_count > 0 else actual
        baseline_global_errors.append(abs(global_mean - actual))

        cat_mean = cat_sums[cat] / cat_counts[cat] if cat_counts[cat] > 0 else global_mean
        baseline_cat_errors.append(abs(cat_mean - actual))

        _ch_key = p["_cat_hour"]
        if cat_hour_counts[_ch_key] >= 3:
            cat_pred = cat_hour_sums[_ch_key] / cat_hour_counts[_ch_key]
            hour_factor = 1.0
        else:
            cat_pred = cat_mean
            hour_mean = hour_sums[hr] / hour_counts[hr] if hour_counts[hr] > 0 else global_mean
            hour_factor = hour_mean / global_mean if global_mean > 0 else 1.0

        if day_counts_d[weekday] > 0 and global_mean > 0:
            day_mean = day_sums[weekday] / day_counts_d[weekday]
            day_factor = 0.85 + (day_mean / global_mean) * 0.15
        else:
            day_factor = 1.0

        ek = "emo" if is_emo else "neutral"
        if emo_counts_d[ek] > 0 and global_mean > 0:
            emo_mean = emo_sums[ek] / emo_counts_d[ek]
            emo_factor = 0.9 + (emo_mean / global_mean) * 0.1
        else:
            emo_factor = 1.0

        bk = "brk" if p["_is_breaking"] else "normal"
        if brk_counts[bk] > 0 and global_mean > 0:
            brk_mean = brk_sums[bk] / brk_counts[bk]
            brk_factor = 0.9 + (brk_mean / global_mean) * 0.1
        else:
            brk_factor = 1.0

        tl_key = p["_title_len"]
        if tlen_counts[tl_key] >= 3 and global_mean > 0:
            tlen_mean = tlen_sums[tl_key] / tlen_counts[tl_key]
            len_factor = 0.95 + (tlen_mean / global_mean) * 0.05
        else:
            len_factor = 1.0

        predicted = cat_pred * hour_factor * day_factor * emo_factor * brk_factor * len_factor
        prelim_predictions.append(predicted)
        cat_residuals[cat].append(abs(predicted - actual))
        _add(p)
        total_or += actual
        total_count += 1

    eval_valid = valid[warmup:]

    cat_std: dict = {}
    for cat, residuals in cat_residuals.items():
        if len(residuals) >= 5:
            mean_r = sum(residuals) / len(residuals)
            cat_std[cat] = math.sqrt(
                sum((r - mean_r) ** 2 for r in residuals) / (len(residuals) - 1)
            )
        else:
            cat_std[cat] = None  # type: ignore[assignment]

    hits = 0
    n_eval = len(eval_valid)
    history: list = []
    for i, p in enumerate(eval_valid):
        actual = p["or"]
        predicted = prelim_predictions[i]
        error = predicted - actual
        cat = p.get("cat", "News")
        if cat_std.get(cat) is not None:
            tolerance = max(0.5, cat_std[cat] * 1.0)
        else:
            tolerance = max(0.5, actual * 0.25)
        effective_tolerance = tolerance * 0.85 if error > 0 else tolerance
        if abs(error) <= effective_tolerance:
            hits += 1
        history.append({
            "predicted": round(predicted, 2),
            "actual": round(actual, 2),
            "title": p.get("title", "")[:50],
            "error": round(abs(error), 2),
            "cat": cat,
            "tolerance": round(effective_tolerance, 2),
        })

    accuracy = (hits / n_eval * 100) if n_eval > 0 else 0.0
    total_abs_error = sum(h["error"] for h in history)
    basis_mae = round(total_abs_error / n_eval, 3) if n_eval > 0 else 0.0

    baseline_global_mae = (
        round(sum(baseline_global_errors) / len(baseline_global_errors), 3)
        if baseline_global_errors else 0.0
    )
    baseline_cat_mae = (
        round(sum(baseline_cat_errors) / len(baseline_cat_errors), 3)
        if baseline_cat_errors else 0.0
    )

    state["rolling_accuracy"] = round(accuracy, 1)
    state["basis_mae"] = basis_mae
    state["accuracy_history"] = history[-200:]
    state["baseline_global_mae"] = baseline_global_mae
    state["baseline_cat_mae"] = baseline_cat_mae
    state["basis_vs_baseline"] = {
        "model_mae": basis_mae,
        "baseline_global_mean_mae": baseline_global_mae,
        "baseline_category_mean_mae": baseline_cat_mae,
        "improvement_vs_global": round((1 - basis_mae / baseline_global_mae) * 100, 1)
            if baseline_global_mae > 0 else 0,
        "improvement_vs_cat": round((1 - basis_mae / baseline_cat_mae) * 100, 1)
            if baseline_cat_mae > 0 else 0,
        "eval_method": "walk-forward (kein Data Leakage)",
        "n_evaluated": n_eval,
        "n_warmup": warmup,
    }

    state.setdefault("accuracy_trend", []).append(accuracy)
    if len(state["accuracy_trend"]) > 20:
        state["accuracy_trend"] = state["accuracy_trend"][-20:]
    state.setdefault("mae_trend", []).append(basis_mae)
    if len(state["mae_trend"]) > 20:
        state["mae_trend"] = state["mae_trend"][-20:]
    state["cat_error_std"] = {c: round(v, 3) for c, v in cat_std.items() if v is not None}

    cat_errors: dict = defaultdict(list)
    for h in history:
        cat_errors[h["cat"]].append(h["error"])
    state["mae_by_cat"] = {c: round(sum(e) / len(e), 3) for c, e in cat_errors.items() if e}

    hour_errors: dict = defaultdict(list)
    for i, p in enumerate(eval_valid):
        if i < len(history):
            hour_errors[p.get("hour", 0)].append(history[i]["error"])
    state["mae_by_hour"] = {h: round(sum(e) / len(e), 3) for h, e in hour_errors.items() if e}

    cat_acc: dict = defaultdict(lambda: [0, 0])
    for i, p in enumerate(eval_valid):
        if i < len(history):
            cat_acc[p.get("cat", "News")][1] += 1
            if history[i]["error"] <= history[i].get("tolerance", max(0.5, p["or"] * 0.25)):
                cat_acc[p.get("cat", "News")][0] += 1
    state["accuracy_by_cat"] = {
        c: round(v[0] / v[1] * 100, 1) if v[1] > 0 else 0 for c, v in cat_acc.items()
    }

    hour_acc: dict = defaultdict(lambda: [0, 0])
    for i, p in enumerate(eval_valid):
        if i < len(history):
            hour_acc[p.get("hour", 0)][1] += 1
            if history[i]["error"] <= history[i].get("tolerance", max(0.5, p["or"] * 0.25)):
                hour_acc[p.get("hour", 0)][0] += 1
    state["accuracy_by_hour"] = {
        h: round(v[0] / v[1] * 100, 1) if v[1] > 0 else 0 for h, v in hour_acc.items()
    }


def _generate_live_rules(findings: dict, state: dict) -> None:
    """Generiert Live-Regeln aus den aktuellen Findings."""
    rules: list = []
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    rule_id = 0

    # 1. Timing-Regel
    hour_data = findings.get("hour_analysis", {})
    if isinstance(hour_data, dict) and hour_data.get("best_hour") is not None:
        best_h = hour_data["best_hour"]
        best_or = hour_data.get("best_or", 0)
        worst_h = hour_data.get("worst_hour", 0)
        worst_or = hour_data.get("worst_or", 0)
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": (
                f"Primaer-Slot: {best_h}:00 Uhr bevorzugen (OR {best_or:.1f}%), "
                f"Schwach-Slot {worst_h}:00 Uhr meiden (OR {worst_or:.1f}%)"
            ),
            "source": f"Weber/Liu: Timing-Analyse aus {len(state.get('push_data', []))} BILD-Pushes",
            "impact": f"+{best_or - worst_or:.1f}% OR-Differenz",
            "approved_by": "Prof. Schwab",
            "approved_at": now_str,
            "category": "timing",
        })

    # 2. Kategorie-Regel
    cat_data = findings.get("cat_analysis", [])
    if cat_data and len(cat_data) >= 2:
        best_cat = max(cat_data, key=lambda c: c.get("avg_or", 0))
        worst_cat = min(cat_data, key=lambda c: c.get("avg_or", 0))
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": (
                f"{best_cat['category']} priorisieren (OR {best_cat['avg_or']:.1f}%), "
                f"{worst_cat['category']} nur bei hoher Relevanz (OR {worst_cat['avg_or']:.1f}%)"
            ),
            "source": f"Nash/Chen: Kategorie-Analyse, n={sum(c.get('count', 0) for c in cat_data)}",
            "impact": f"+{best_cat['avg_or'] - worst_cat['avg_or']:.1f}% OR-Differenz",
            "approved_by": "Prof. Schwab",
            "approved_at": now_str,
            "category": "kategorie",
        })

    # 3. Titel-Länge
    len_data = findings.get("title_length", {})
    if isinstance(len_data, dict) and len_data.get("best_range"):
        best_range = len_data["best_range"]
        best_len_or = len_data.get("best_or", 0)
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": f"Titel-Laenge {best_range} bevorzugen (OR {best_len_or:.1f}%)",
            "source": "Shannon/Lakoff: Titel-Laengen-Analyse",
            "impact": f"Optimale Scanbarkeit bei {best_range}",
            "approved_by": "Prof. Schwab",
            "approved_at": now_str,
            "category": "titel",
        })

    # 4. Framing-Regel
    framing = findings.get("framing_analysis", {})
    if isinstance(framing, dict):
        emo_or = framing.get("emotional_or", 0)
        neutral_or = framing.get("neutral_or", 0)
        if emo_or > 0 and neutral_or > 0:
            diff = emo_or - neutral_or
            rule_id += 1
            rules.append({
                "id": rule_id, "active": True,
                "rule": (
                    f"Emotionales Framing: {'bevorzugen' if diff > 0 else 'zurueckhaltend einsetzen'} "
                    f"({emo_or:.1f}% vs. {neutral_or:.1f}% neutral)"
                ),
                "source": "Kahneman/Cialdini: Framing-Analyse + Ethics Review Zuboff",
                "impact": f"{'+' if diff > 0 else ''}{diff:.1f}% OR-Differenz",
                "approved_by": "Prof. Schwab",
                "approved_at": now_str,
                "category": "framing",
            })

    # 5. Frequenz-Regel
    freq = findings.get("frequency_correlation", {})
    if isinstance(freq, dict) and freq.get("optimal_daily"):
        opt = freq["optimal_daily"]
        rule_id += 1
        rules.append({
            "id": rule_id, "active": True,
            "rule": (
                f"Max. {opt} Pushes/Tag — darueber sinkt OR "
                f"(Frequenz-Korrelation r={freq.get('correlation', 0):.2f})"
            ),
            "source": "Bertalanffy: Systemtheorie + Shirazi (2014)",
            "impact": "Push-Fatigue vermeiden",
            "approved_by": "Prof. Schwab",
            "approved_at": now_str,
            "category": "frequenz",
        })

    # 6. Linguistik-Regel
    ling = findings.get("linguistic_analysis", {})
    if isinstance(ling, dict):
        colon_or = ling.get("colon_or", 0)
        no_colon_or = ling.get("no_colon_or", 0)
        if colon_or > 0 and no_colon_or > 0:
            better = "Doppelpunkt" if colon_or > no_colon_or else "Ohne Doppelpunkt"
            rule_id += 1
            rules.append({
                "id": rule_id, "active": True,
                "rule": f"Titel-Separator: {better} bevorzugen ({colon_or:.1f}% vs. {no_colon_or:.1f}%)",
                "source": "Lakoff: Linguistik-Analyse",
                "impact": f"{abs(colon_or - no_colon_or):.1f}% OR-Differenz",
                "approved_by": "Prof. Schwab",
                "approved_at": now_str,
                "category": "linguistik",
            })

    state["live_rules"] = rules


# ── Haupt-Analyse ──────────────────────────────────────────────────────────

def run_autonomous_analysis() -> None:
    """Analysiert Push-Daten autonom — wird vom Research-Worker alle 20s aufgerufen."""
    analysis_lock = _research_state.get("analysis_lock")
    if analysis_lock and not analysis_lock.acquire(blocking=False):
        return
    try:
        _run_analysis_inner()
    except Exception as exc:
        log.error("[research] run_autonomous_analysis Fehler: %s", exc, exc_info=True)
    finally:
        if analysis_lock:
            try:
                analysis_lock.release()
            except RuntimeError:
                pass


def _run_analysis_inner() -> None:
    """Kernlogik der autonomen Analyse — läuft im Lock."""
    state = _research_state
    now = time.time()

    # Daten laden (max. alle 120s neu laden)
    if now - state.get("last_fetch", 0) < 120 and state.get("push_data"):
        push_data = state["push_data"]
    else:
        try:
            from app.database import push_db_load_all
            push_data = push_db_load_all(max_days=90)
        except Exception as exc:
            log.warning("[research] DB-Ladefehler: %s", exc)
            push_data = state.get("push_data", [])

        if push_data:
            state["push_data"] = push_data
            state["last_fetch"] = now
        else:
            push_data = state.get("push_data", [])

    if not push_data:
        return

    # 24h-Reifephase
    cutoff_24h = now - 24 * 3600
    mature_pushes = [p for p in push_data if p.get("ts_num", 0) > 0 and p["ts_num"] < cutoff_24h]
    fresh_pushes = [p for p in push_data if p.get("ts_num", 0) > 0 and p["ts_num"] >= cutoff_24h]

    state["fresh_pushes"] = fresh_pushes
    state["mature_count"] = len(mature_pushes)
    state["fresh_count"] = len(fresh_pushes)
    state["cutoff_24h"] = cutoff_24h

    if not mature_pushes:
        log.warning(
            "[research] Keine reifen Pushes (>24h) verfuegbar (%d frische). Analyse ausgesetzt.",
            len(fresh_pushes),
        )
        return

    push_data = mature_pushes
    n = len(push_data)

    # Sport / Non-Sport Split
    sport_data = [p for p in push_data if p.get("cat") == "Sport"]
    nonsport_data = [p for p in push_data if p.get("cat") != "Sport"]
    state["_sport_data"] = sport_data
    state["_nonsport_data"] = nonsport_data
    state["_sport_n"] = len(sport_data)
    state["_nonsport_n"] = len(nonsport_data)

    # Neue reife Pushes seit letzter Analyse?
    new_pushes = n - state.get("prev_push_count", 0)
    is_new_data = new_pushes > 0
    state["prev_push_count"] = n

    if is_new_data:
        _update_rolling_accuracy(push_data, state)
        state["live_rules"] = []  # Force regeneration

    # Volle Re-Analyse: bei neuen Daten oder alle 60s
    if not is_new_data and state.get("findings") and now - state.get("last_analysis", 0) < 60:
        return

    state["last_analysis"] = now
    state["analysis_generation"] = state.get("analysis_generation", 0) + 1
    gen = state["analysis_generation"]
    log.info(
        "[research] Volle Analyse gestartet: Gen #%d, %d Pushes, %d Sport, %d NonSport",
        gen, n, len(sport_data), len(nonsport_data),
    )

    # Rolling Accuracy
    _update_rolling_accuracy(push_data, state)

    # ── Basis-Analysen ────────────────────────────────────────────────
    _or_max_sane = 30.0
    or_pushes = [p for p in push_data if 0 < p.get("or", 0) <= _or_max_sane]
    or_values = [p["or"] for p in or_pushes]
    sorted_or = sorted(or_values) if or_values else [0]
    median_or = sorted_or[len(sorted_or) // 2]
    mean_or = median_or
    std_or = (
        math.sqrt(sum((x - median_or) ** 2 for x in or_values) / max(1, len(or_values) - 1))
        if len(or_values) > 1 else 0
    )

    # Stunden-Analyse
    hours: dict = defaultdict(list)
    for p in push_data:
        if 0 <= p.get("hour", -1) <= 23 and p.get("or", 0) > 0:
            hours[p["hour"]].append(p["or"])
    hour_avgs = {h: sum(v) / len(v) for h, v in hours.items() if v}
    state["_hour_avgs_cache"] = hour_avgs
    best_hour = max(hour_avgs, key=hour_avgs.get) if hour_avgs else 18
    worst_hour = min(hour_avgs, key=hour_avgs.get) if hour_avgs else 3
    best_or = hour_avgs.get(best_hour, 0)
    worst_or = hour_avgs.get(worst_hour, 0)

    # Kategorie-Analyse
    cat_or: dict = defaultdict(list)
    for p in push_data:
        if p.get("or", 0) > 0:
            cat_or[p.get("cat") or "News"].append(p["or"])
    cat_avgs = {c: sum(v) / len(v) for c, v in cat_or.items() if v}
    state["_cat_avgs_cache"] = cat_avgs
    best_cat = max(cat_avgs, key=cat_avgs.get) if cat_avgs else "News"
    worst_cat = min(cat_avgs, key=cat_avgs.get) if cat_avgs else "News"

    # Framing-Analyse
    emo_words = {
        "schock", "drama", "skandal", "angst", "tod", "sterben", "krieg", "panik",
        "horror", "warnung", "gefahr", "krise", "irre", "wahnsinn", "hammer", "brutal", "bitter",
    }
    emo_pushes: list = []
    q_pushes: list = []
    neutral_pushes: list = []
    for p in push_data:
        if p.get("or", 0) <= 0:
            continue
        _tl = p.get("title", "").lower()
        if any(w in _tl for w in emo_words):
            emo_pushes.append(p)
        elif "?" in p.get("title", ""):
            q_pushes.append(p)
        else:
            neutral_pushes.append(p)
    emo_or = sum(p["or"] for p in emo_pushes) / len(emo_pushes) if emo_pushes else 0.0
    q_or = sum(p["or"] for p in q_pushes) / len(q_pushes) if q_pushes else 0.0
    neutral_or = sum(p["or"] for p in neutral_pushes) / len(neutral_pushes) if neutral_pushes else 0.0

    # Titel-Länge
    len_data: dict = {"kurz": [], "mittel": [], "lang": []}
    for p in push_data:
        if p.get("or", 0) > 0:
            tl = p.get("title_len", len(p.get("title", "")))
            if tl < 50:
                len_data["kurz"].append(p["or"])
            elif tl <= 80:
                len_data["mittel"].append(p["or"])
            else:
                len_data["lang"].append(p["or"])
    len_avgs = {k: sum(v) / len(v) if v else 0 for k, v in len_data.items()}
    best_len = max(len_avgs, key=len_avgs.get) if len_avgs else "mittel"

    # Tages-Korrelation
    day_counts: dict = defaultdict(int)
    day_or: dict = defaultdict(list)
    for p in push_data:
        try:
            ts = int(p.get("ts", p.get("ts_num", 0)))
            if ts > 1e12:
                ts //= 1000
            dk = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            dk = "unknown"
        day_counts[dk] += 1
        if p.get("or", 0) > 0:
            day_or[dk].append(p["or"])
    day_stats = [
        (day_counts[d], sum(day_or[d]) / len(day_or[d]))
        for d in day_counts
        if d in day_or and day_or[d]
    ]
    if len(day_stats) > 1:
        xs = [s[0] for s in day_stats]
        ys = [s[1] for s in day_stats]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        freq_corr = cov / (sx * sy) if sx * sy > 0 else 0.0
    else:
        freq_corr = 0.0

    # Keyword-Analyse
    stops = {
        "der", "die", "das", "und", "in", "von", "fuer", "mit", "auf", "den", "ist",
        "ein", "eine", "es", "im", "zu", "an", "nach", "vor", "ueber", "bei", "wie",
        "nicht", "auch", "er", "sie", "sich", "so", "als", "aber", "dem", "zum", "hat",
        "aus", "noch", "am", "nur", "einen", "dass", "jetzt", "bild", "news", "alle",
        "neue", "neuer", "neues", "schon", "ab", "wird", "wurde",
    }
    word_or: dict = defaultdict(list)
    for p in push_data:
        if p.get("or", 0) > 0:
            for w in re.findall(r'[A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]{4,}', p.get("title", "")):
                wl = w.lower()
                if wl not in stops:
                    word_or[wl].append(p["or"])
    kw_avgs = {w: sum(v) / len(v) for w, v in word_or.items() if len(v) >= 2}
    top_kw = sorted(kw_avgs, key=kw_avgs.get, reverse=True)[:10]

    # Linguistik
    colon_pushes = [p for p in push_data if (":" in p.get("title", "") or "|" in p.get("title", "")) and p.get("or", 0) > 0]
    no_colon = [p for p in push_data if ":" not in p.get("title", "") and "|" not in p.get("title", "") and p.get("or", 0) > 0]
    colon_or = sum(p["or"] for p in colon_pushes) / len(colon_pushes) if colon_pushes else 0.0
    no_colon_or = sum(p["or"] for p in no_colon) / len(no_colon) if no_colon else 0.0

    # Top & Flop
    top_push = max(push_data, key=lambda p: p.get("or", 0)) if push_data else None
    flop_push = (
        min([p for p in push_data if p.get("or", 0) > 0], key=lambda p: p["or"])
        if [p for p in push_data if p.get("or", 0) > 0] else None
    )

    # ── Findings zusammenstellen ──────────────────────────────────────
    findings: dict = {}
    findings["hour_analysis"] = {
        "best_hour": best_hour,
        "best_or": best_or,
        "worst_hour": worst_hour,
        "worst_or": worst_or,
        "hour_avgs": dict(hour_avgs),
    }
    findings["cat_analysis"] = [
        {"category": c, "avg_or": v, "count": len(cat_or.get(c, []))}
        for c, v in sorted(cat_avgs.items(), key=lambda x: -x[1])
    ]
    findings["framing_analysis"] = {
        "emotional_or": emo_or,
        "neutral_or": neutral_or,
        "emotional_count": len(emo_pushes),
        "neutral_count": len(neutral_pushes),
        "question_or": q_or,
        "question_count": len(q_pushes),
    }
    findings["title_length"] = {
        "best_range": best_len,
        "best_or": len_avgs.get(best_len, 0),
        "kurz_or": len_avgs.get("kurz", 0),
        "mittel_or": len_avgs.get("mittel", 0),
        "lang_or": len_avgs.get("lang", 0),
    }
    findings["frequency_correlation"] = {
        "correlation": freq_corr,
        "optimal_daily": int(sum(s[0] for s in day_stats) / max(1, len(day_stats))) if day_stats else 0,
        "days_analyzed": len(day_stats),
    }
    findings["linguistic_analysis"] = {
        "colon_or": colon_or,
        "no_colon_or": no_colon_or,
        "colon_count": len(colon_pushes),
        "no_colon_count": len(no_colon),
    }
    findings["keyword_analysis"] = {
        "top_keywords": top_kw,
        "keyword_count": len(kw_avgs),
    }
    findings["day_stats"] = day_stats

    # State atomar aktualisieren
    lock = state.get("analysis_lock")
    if lock:
        with lock:
            state["findings"] = findings
            state["cumulative_insights"] = state.get("cumulative_insights", 0) + 1
    else:
        state["findings"] = findings
        state["cumulative_insights"] = state.get("cumulative_insights", 0) + 1

    # Live-Regeln generieren
    if findings:
        _generate_live_rules(findings, state)

    log.info("[research] Volle Analyse FERTIG: Gen #%d, n=%d, acc=%.1f%%", gen, n, state["rolling_accuracy"])

    # Residual Corrector aktualisieren
    update_residual_corrector()


# ── Residual Corrector ─────────────────────────────────────────────────────

def update_residual_corrector() -> None:
    """Aktualisiert den Online-Bias-Korrektor aus der Prediction-Log-DB."""
    WINDOW = 20
    MAX_CORRECTION = 2.0
    MIN_SAMPLES = 10

    try:
        from app.database import PUSH_DB_PATH
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT p.predicted_or, p.actual_or, p.predicted_at, "
            "       pu.cat, pu.hour "
            "FROM prediction_log p "
            "LEFT JOIN pushes pu ON p.push_id = pu.message_id "
            "WHERE p.actual_or > 0 AND p.predicted_or > 0 "
            "ORDER BY p.predicted_at DESC LIMIT 100"
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.warning("[ResidualCorrector] DB-Fehler: %s", exc)
        return

    if not rows or len(rows) < MIN_SAMPLES:
        return

    try:
        sorted_rows = sorted(rows, key=lambda r: r["predicted_at"])

        all_residuals = [r["predicted_or"] - r["actual_or"] for r in sorted_rows]
        recent_global = all_residuals[-WINDOW:]
        global_bias = sum(recent_global) / len(recent_global)
        global_bias = max(-MAX_CORRECTION, min(MAX_CORRECTION, global_bias))

        cat_residuals: dict = defaultdict(list)
        for r in sorted_rows:
            cat = r["cat"] or "News"
            cat_residuals[cat].append(r["predicted_or"] - r["actual_or"])
        cat_bias: dict = {}
        for c, resids in cat_residuals.items():
            if len(resids) >= MIN_SAMPLES:
                recent = resids[-WINDOW:]
                bias = sum(recent) / len(recent)
                cat_bias[c] = max(-MAX_CORRECTION, min(MAX_CORRECTION, bias))

        hg_residuals: dict = defaultdict(list)
        for r in sorted_rows:
            h = int(r["hour"]) if r["hour"] is not None else 12
            hg = _hour_to_group(h)
            hg_residuals[hg].append(r["predicted_or"] - r["actual_or"])
        hourgroup_bias: dict = {}
        for g, resids in hg_residuals.items():
            if len(resids) >= MIN_SAMPLES:
                recent = resids[-WINDOW:]
                bias = sum(recent) / len(recent)
                hourgroup_bias[g] = max(-MAX_CORRECTION, min(MAX_CORRECTION, bias))

        recent_debug: list = []
        for r in sorted_rows[-50:]:
            cat = r["cat"] or ""
            h = int(r["hour"]) if r["hour"] is not None else -1
            recent_debug.append({
                "predicted": round(r["predicted_or"], 2),
                "actual": round(r["actual_or"], 2),
                "residual": round(r["predicted_or"] - r["actual_or"], 2),
                "cat": cat,
                "hour": h,
            })

        with _residual_corrector_lock:
            _residual_corrector["global_bias"] = round(global_bias, 4)
            _residual_corrector["cat_bias"] = {k: round(v, 4) for k, v in cat_bias.items()}
            _residual_corrector["hourgroup_bias"] = {k: round(v, 4) for k, v in hourgroup_bias.items()}
            _residual_corrector["n_samples"] = len(sorted_rows)
            _residual_corrector["last_update_ts"] = int(time.time())
            _residual_corrector["recent_residuals"] = recent_debug

        log.info(
            "[ResidualCorrector] Update: global_bias=%+.3f, cats=%s, n=%d",
            global_bias, list(cat_bias.keys()), len(sorted_rows),
        )
    except Exception as exc:
        log.warning("[ResidualCorrector] Berechnungsfehler: %s", exc)


def monitoring_tick() -> None:
    """Monitoring-Tick: prüft Drift, MAE-Spikes, A/B-Ergebnisse etc."""
    try:
        from push_balancer_server_compat import _monitoring_tick  # type: ignore
        _monitoring_tick()
    except ImportError:
        pass
    except Exception as exc:
        log.warning("[research] monitoring_tick Fehler: %s", exc)


def build_xor_perf_cache() -> None:
    """Baut/aktualisiert den XOR-Performance-Cache für /api/competitor-xor."""
    try:
        from push_balancer_server_compat import _build_xor_perf_cache  # type: ignore
        _build_xor_perf_cache()
    except ImportError:
        pass
    except Exception as exc:
        log.warning("[research] build_xor_perf_cache Fehler: %s", exc)


def get_cached_feeds(feed_type: str) -> dict | list:
    """Liefert gecachte Feeds aus dem Background-Cache."""
    with _feed_cache_lock:
        entry = _feed_cache.get(feed_type, {})
        if entry.get("data") and (time.time() - entry.get("ts", 0)) < _FEED_CACHE_TTL * 3:
            return entry["data"]
    return {}
