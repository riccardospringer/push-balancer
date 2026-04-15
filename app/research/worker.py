"""app/research/worker.py — Autonomer Research Worker Thread.

Buendelt den Research-Worker und alle Hilfsfunktionen aus dem frueheren Monolithen.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import re
import sqlite3
import ssl
import threading
import time
import urllib.request
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

# ── Monitoring State ──────────────────────────────────────────────────────
_monitoring_state: dict = {
    "last_tick": 0,
    "mae_24h": 0.0,
    "mae_7d": 0.0,
    "mae_trend": [],
    "calibration_bias": 0.0,
    "calibration_trend": [],
    "feature_drift": {},
    "residual_corrector": {},
}
_monitoring_state_lock = threading.Lock()

# ── Auto-Retrain State ─────────────────────────────────────────────────────
_auto_retrain_state: dict = {
    "consecutive_degraded_ticks": 0,
    "last_retrain_trigger_ts": 0,
    "total_retrains": 0,
}

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

# ── Deutsche Feiertage 2025-2027 ──────────────────────────────────────────
_GERMAN_HOLIDAYS: dict = {
    "2025-01-01": "Neujahr", "2025-04-18": "Karfreitag", "2025-04-21": "Ostermontag",
    "2025-05-01": "Tag der Arbeit", "2025-05-29": "Christi Himmelfahrt",
    "2025-06-09": "Pfingstmontag", "2025-10-03": "Tag der dt. Einheit",
    "2025-12-25": "1. Weihnachtstag", "2025-12-26": "2. Weihnachtstag",
    "2026-01-01": "Neujahr", "2026-04-03": "Karfreitag", "2026-04-06": "Ostermontag",
    "2026-05-01": "Tag der Arbeit", "2026-05-14": "Christi Himmelfahrt",
    "2026-05-25": "Pfingstmontag", "2026-10-03": "Tag der dt. Einheit",
    "2026-12-25": "1. Weihnachtstag", "2026-12-26": "2. Weihnachtstag",
    "2027-01-01": "Neujahr", "2027-03-26": "Karfreitag", "2027-03-29": "Ostermontag",
    "2027-05-01": "Tag der Arbeit", "2027-05-06": "Christi Himmelfahrt",
    "2027-05-17": "Pfingstmontag", "2027-10-03": "Tag der dt. Einheit",
    "2027-12-25": "1. Weihnachtstag", "2027-12-26": "2. Weihnachtstag",
}

# ── Externer Kontext Cache ────────────────────────────────────────────────
_external_context_cache: dict = {
    "weather": {},
    "trends": [],
    "holiday": "",
    "last_fetch": 0,
}

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
    state["accuracy_history"] = history[-100:]
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

    # ── Ensemble-Accuracy: ML-Pipeline (predict_or) ───────────────────────
    # PERFORMANCE: Nur ausführen wenn ML-Modell geladen ist.
    # Ohne ML fällt predict_or auf Heuristik zurück (O(n) pro Push) und blockiert den Worker.
    _has_ml_model = False
    try:
        from app.ml.gbrt import _gbrt_model as _gbrt_m, _gbrt_lock as _gbrt_lk
        with _gbrt_lk:
            _has_ml_model = _gbrt_m is not None
    except Exception:
        pass
    if not _has_ml_model:
        try:
            from app.ml.lightgbm_model import _ml_state as _ml_st, _ml_lock as _ml_lk
            with _ml_lk:
                _has_ml_model = _ml_st.get("model") is not None
        except Exception:
            pass

    if _has_ml_model:
        try:
            from app.ml.predict import predict_or as _predict_or
            sample = valid[-50:] if len(valid) > 50 else valid
            ens_hits, ens_total, ens_mae_sum = 0, 0, 0.0
            for p in sample:
                result = _predict_or(p, research_state=state, push_data=valid)
                if result is None:
                    continue
                pred = result.get("predicted_or", 0.0)
                actual = p["or"]
                err = abs(pred - actual)
                ens_mae_sum += err
                ens_total += 1
                cat = p.get("cat", "News")
                tol = max(0.5, cat_std[cat] * 1.0) if cat_std.get(cat) is not None else max(0.5, actual * 0.25)
                if err <= tol:
                    ens_hits += 1
            if ens_total > 0:
                ens_accuracy = round(ens_hits / ens_total * 100, 1)
                ens_mae = round(ens_mae_sum / ens_total, 3)
                prev_ens = state.get("ensemble_accuracy", 0)
                state["ensemble_accuracy"] = ens_accuracy
                state["ensemble_mae"] = ens_mae
                state.setdefault("ensemble_accuracy_trend", []).append(ens_accuracy)
                if len(state["ensemble_accuracy_trend"]) > 20:
                    state["ensemble_accuracy_trend"] = state["ensemble_accuracy_trend"][-20:]
                if prev_ens > 0:
                    state["ensemble_accuracy_delta"] = round(ens_accuracy - prev_ens, 2)
        except Exception as _ens_err:
            log.debug("[research] Ensemble-Accuracy-Berechnung fehlgeschlagen: %s", _ens_err)

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


# ── Temporale Trend-Analyse ───────────────────────────────────────────────

def _compute_temporal_trends(push_data: list) -> dict:
    """Temporale Segmentierung: Monats-, Wochen-, Wochentag-, Stunden-Trends."""
    now = time.time()
    result: dict = {}

    def _ts(p: dict) -> int:
        ts = p.get("ts_num", 0)
        if not ts:
            try:
                ts = int(p.get("ts", 0))
                if ts > 1e12:
                    ts //= 1000
            except (ValueError, TypeError):
                ts = 0
        return ts

    def _stats(group: list) -> dict:
        ors = [p["or"] for p in group if p.get("or", 0) > 0]
        if not ors:
            return {"avg_or": 0, "median_or": 0, "push_count": len(group), "or_count": 0}
        s = sorted(ors)
        return {
            "avg_or": round(sum(ors) / len(ors), 2),
            "median_or": round(s[len(s) // 2], 2),
            "push_count": len(group),
            "or_count": len(ors),
        }

    def _best_cat(group: list) -> str:
        cat_or: dict = defaultdict(list)
        for p in group:
            if p.get("or", 0) > 0:
                cat_or[p.get("cat") or "News"].append(p["or"])
        if not cat_or:
            return ""
        return max(cat_or, key=lambda c: sum(cat_or[c]) / len(cat_or[c]))

    def _best_hour(group: list) -> int:
        h_or: dict = defaultdict(list)
        for p in group:
            if p.get("or", 0) > 0 and 0 <= p.get("hour", -1) <= 23:
                h_or[p["hour"]].append(p["or"])
        if not h_or:
            return 0
        return max(h_or, key=lambda h: sum(h_or[h]) / len(h_or[h]))

    # Monats-Vergleich (bis 12 Monate)
    monthly: dict = {}
    for p in push_data:
        ts = _ts(p)
        if ts > 0:
            key = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m")
            monthly.setdefault(key, []).append(p)
    monthly_stats = []
    for month_key in sorted(monthly.keys()):
        group = monthly[month_key]
        s = _stats(group)
        s["month"] = month_key
        s["best_cat"] = _best_cat(group)
        s["best_hour"] = _best_hour(group)
        monthly_stats.append(s)
    result["monthly"] = monthly_stats

    recent_months = monthly_stats[-6:]
    if len(recent_months) >= 3:
        xs = list(range(len(recent_months)))
        ys = [m["avg_or"] for m in recent_months]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else 0
        if slope > 0.05:
            result["trend_direction"] = "steigend"
        elif slope < -0.05:
            result["trend_direction"] = "fallend"
        else:
            result["trend_direction"] = "stabil"
        result["trend_slope"] = round(slope, 4)
    else:
        result["trend_direction"] = "zu wenig Daten"
        result["trend_slope"] = 0

    # Tages-Vergleich (letzte 30 Tage)
    cutoff_30d = now - 30 * 86400
    daily: dict = {}
    for p in push_data:
        ts = _ts(p)
        if ts > cutoff_30d and ts > 0:
            key = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            daily.setdefault(key, []).append(p)
    daily_stats = []
    for day_key in sorted(daily.keys()):
        group = daily[day_key]
        s = _stats(group)
        s["date"] = day_key
        s["best_cat"] = _best_cat(group)
        daily_stats.append(s)
    result["daily_30"] = daily_stats

    # Wochen-Evolution (letzte 12 Wochen)
    weekly: dict = {}
    for p in push_data:
        ts = _ts(p)
        if ts > 0:
            kw = datetime.datetime.fromtimestamp(ts).strftime("%Y-W%W")
            weekly.setdefault(kw, []).append(p)
    weekly_stats = []
    for wk_key in sorted(weekly.keys()):
        group = weekly[wk_key]
        s = _stats(group)
        s["week"] = wk_key
        s["top_cat"] = _best_cat(group)
        weekly_stats.append(s)
    weekly_stats = weekly_stats[-12:]
    result["weekly"] = weekly_stats

    moving_avg = []
    for i in range(len(weekly_stats)):
        window = weekly_stats[max(0, i - 3):i + 1]
        avg = sum(w["avg_or"] for w in window) / len(window) if window else 0
        moving_avg.append({
            "week": weekly_stats[i]["week"],
            "moving_avg_4w": round(avg, 2),
        })
    result["moving_avg_4w"] = moving_avg

    return result


# ── Sport/NonSport Subset-Analysen ────────────────────────────────────────

def _update_rolling_accuracy_subset(subset_data: list, state: dict, key: str) -> None:
    """Berechne Rolling Accuracy für ein Subset (sport/nonsport) und speichere unter state[key]."""
    if len([p for p in subset_data if p.get("or", 0) > 0]) < 10:
        return
    tmp_state: dict = {
        "rolling_accuracy": 0, "basis_mae": 0, "accuracy_history": [],
        "accuracy_trend": [], "mae_trend": [], "mae_by_cat": {}, "mae_by_hour": {},
        "accuracy_by_cat": {}, "ensemble_accuracy": 0, "ensemble_mae": 0,
        "ensemble_accuracy_trend": [], "ensemble_accuracy_delta": 0,
        "tuning_params": state.get("tuning_params"),
    }
    _update_rolling_accuracy(subset_data, tmp_state)
    state[key] = {
        "rolling_accuracy": tmp_state.get("rolling_accuracy", 0),
        "basis_mae": tmp_state.get("basis_mae", 0),
        "mae_by_cat": tmp_state.get("mae_by_cat", {}),
        "mae_by_hour": tmp_state.get("mae_by_hour", {}),
        "ensemble_mae": tmp_state.get("ensemble_mae", 0),
        "accuracy_by_cat": tmp_state.get("accuracy_by_cat", {}),
        "n": len(subset_data),
        "n_with_or": len([p for p in subset_data if p.get("or", 0) > 0]),
    }


_EMOTION_GROUPS: dict = {
    "Angst/Bedrohung":    {"words": ["krieg","terror","angriff","bombe","tod","sterben","opfer","gefahr","warnung","alarm","attacke","explosion","gewalt","mord","bedroh","toedlich","anschlag","crash","absturz","katastrophe"], "icon": "warn"},
    "Empoerung/Skandal":  {"words": ["skandal","betrug","luege","schock","unglaublich","dreist","frechheit","enthuellung","vorwurf","ermittl","anklage","razzia","affaere","korrupt","verdacht","versagen","versagt","beschuldigt"], "icon": "anger"},
    "Neugier/Geheimnis":  {"words": ["geheimnis","wahrheit","ueberraschung","raetsel","enthuellt","exklusiv","kurios","irre","unfassbar","verraet","insider","daran liegt","dahinter","warum","wieso","was steckt"], "icon": "search"},
    "Freude/Erfolg":      {"words": ["gewinn","sieg","triumph","rekord","sensation","held","glueck","traum","jubel","feier","meister","gold","beste","weltmeister","gewinnt","siegt","tor","titel","champion"], "icon": "trophy"},
    "Mitgefuehl/Drama":   {"words": ["trauer","abschied","schicksal","drama","tragoedie","bewegend","ruehrend","verlust","weint","traenen","tot","gestorben","verstorben","letzter wille","beerdigung","nachruf"], "icon": "sad"},
    "Dringlichkeit":      {"words": ["jetzt","sofort","dringend","warnung","alarm","achtung","notfall","letzte chance","nur noch","deadline","eilmeldung","breaking","+++","aktuell","gerade","live"], "icon": "urgent"},
    "Personalisierung":   {"words": ["so lebt","privat","zuhause","geheime","liebes","hochzeit","baby","schwanger","trennung","ehe","familie","verlobt","kind","star","promi","vip","millionaer"], "icon": "person"},
}


def _compute_findings_for_subset(subset_data: list) -> dict:
    """Berechne strukturierte Analyse-Findings für ein Subset (Sport/NonSport)."""
    findings: dict = {}
    _or_max_sane = 30.0
    or_pushes = [p for p in subset_data if 0 < p.get("or", 0) <= _or_max_sane]
    or_values = [p["or"] for p in or_pushes]
    sorted_or = sorted(or_values) if or_values else [0]
    median_or = sorted_or[len(sorted_or) // 2]
    mean_or = sum(or_values) / len(or_values) if or_values else 0.0
    std_or = (
        math.sqrt(sum((x - mean_or) ** 2 for x in or_values) / max(1, len(or_values) - 1))
        if len(or_values) > 1
        else 0.0
    )

    hours: dict = defaultdict(list)
    for p in subset_data:
        if 0 <= p.get("hour", -1) <= 23 and p.get("or", 0) > 0:
            hours[p["hour"]].append(p["or"])
    hour_avgs = {h: sum(v) / len(v) for h, v in hours.items() if v}
    best_hour = max(hour_avgs, key=hour_avgs.get) if hour_avgs else 18
    worst_hour = min(hour_avgs, key=hour_avgs.get) if hour_avgs else 3
    findings["hour_analysis"] = {
        "best_hour": best_hour, "best_or": hour_avgs.get(best_hour, 0),
        "worst_hour": worst_hour, "worst_or": hour_avgs.get(worst_hour, 0),
        "hour_avgs": dict(hour_avgs),
    }

    cat_or: dict = defaultdict(list)
    for p in subset_data:
        if p.get("or", 0) > 0:
            cat_or[p.get("cat") or "News"].append(p["or"])
    cat_avgs = {c: sum(v) / len(v) for c, v in cat_or.items() if v}
    findings["cat_analysis"] = [
        {"category": c, "avg_or": v, "count": len(cat_or.get(c, []))}
        for c, v in sorted(cat_avgs.items(), key=lambda x: -x[1])
    ]

    emo_words = {"schock","drama","skandal","angst","tod","sterben","krieg","panik",
                 "horror","warnung","gefahr","krise","irre","wahnsinn","hammer","brutal","bitter"}
    emo_pushes: list = []
    q_pushes: list = []
    neutral_pushes: list = []
    for p in subset_data:
        if p.get("or", 0) <= 0:
            continue
        _tl = p.get("title", "").lower()
        if any(w in _tl for w in emo_words):
            emo_pushes.append(p)
        elif "?" in p.get("title", ""):
            q_pushes.append(p)
        else:
            neutral_pushes.append(p)
    findings["framing_analysis"] = {
        "emotional_or": sum(p["or"] for p in emo_pushes) / len(emo_pushes) if emo_pushes else 0,
        "neutral_or": sum(p["or"] for p in neutral_pushes) / len(neutral_pushes) if neutral_pushes else 0,
        "emotional_count": len(emo_pushes), "neutral_count": len(neutral_pushes),
        "question_or": sum(p["or"] for p in q_pushes) / len(q_pushes) if q_pushes else 0,
        "question_count": len(q_pushes),
    }

    len_data: dict = {"kurz": [], "mittel": [], "lang": []}
    for p in subset_data:
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
    findings["title_length"] = {
        "best_range": best_len, "best_or": len_avgs.get(best_len, 0),
        "kurz_or": len_avgs.get("kurz", 0), "mittel_or": len_avgs.get("mittel", 0),
        "lang_or": len_avgs.get("lang", 0),
    }

    day_counts: dict = defaultdict(int)
    day_or_map: dict = defaultdict(list)
    for p in subset_data:
        try:
            ts = int(p.get("ts", p.get("ts_num", 0)))
            if ts > 1e12:
                ts //= 1000
            dk = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            dk = "unknown"
        day_counts[dk] += 1
        if p.get("or", 0) > 0:
            day_or_map[dk].append(p["or"])
    day_stats = [
        (day_counts[d], sum(day_or_map[d]) / len(day_or_map[d]))
        for d in day_counts if d in day_or_map and day_or_map[d]
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
    findings["frequency_correlation"] = {
        "correlation": freq_corr,
        "optimal_daily": int(sum(s[0] for s in day_stats) / max(1, len(day_stats))) if day_stats else 0,
        "days_analyzed": len(day_stats),
    }

    colon_pushes = [p for p in subset_data if (":" in p.get("title", "") or "|" in p.get("title", "")) and p.get("or", 0) > 0]
    no_colon = [p for p in subset_data if ":" not in p.get("title", "") and "|" not in p.get("title", "") and p.get("or", 0) > 0]
    findings["linguistic_analysis"] = {
        "colon_or": sum(p["or"] for p in colon_pushes) / len(colon_pushes) if colon_pushes else 0,
        "no_colon_or": sum(p["or"] for p in no_colon) / len(no_colon) if no_colon else 0,
        "colon_count": len(colon_pushes), "no_colon_count": len(no_colon),
    }

    stops = {
        "der","die","das","und","in","von","fuer","mit","auf","den","ist","ein","eine",
        "es","im","zu","an","nach","vor","ueber","bei","wie","nicht","auch","er","sie",
        "sich","so","als","aber","dem","zum","hat","aus","noch","am","nur","einen","dass",
        "jetzt","bild","news","alle","neue","neuer","neues","schon","ab","wird","wurde",
    }
    word_or: dict = defaultdict(list)
    for p in subset_data:
        if p.get("or", 0) > 0:
            for w in re.findall(r'[A-Za-z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]{4,}', p.get("title", "")):
                wl = w.lower()
                if wl not in stops:
                    word_or[wl].append(p["or"])
    kw_avgs = {w: sum(v) / len(v) for w, v in word_or.items() if len(v) >= 2}
    findings["keyword_analysis"] = {
        "top_keywords": sorted(kw_avgs, key=kw_avgs.get, reverse=True)[:10],
        "keyword_count": len(kw_avgs),
    }

    total_with_or = len(or_pushes)
    _sub_emo_sums: dict = {g: 0.0 for g in _EMOTION_GROUPS}
    _sub_emo_counts: dict = {g: 0 for g in _EMOTION_GROUPS}
    for p in subset_data:
        if p.get("or", 0) <= 0:
            continue
        _tl = p.get("title", "").lower()
        for gn, cfg in _EMOTION_GROUPS.items():
            if any(w in _tl for w in cfg["words"]):
                _sub_emo_sums[gn] += p["or"]
                _sub_emo_counts[gn] += 1
    emotion_results = []
    for group_name, cfg in _EMOTION_GROUPS.items():
        ec = _sub_emo_counts[group_name]
        avg_or = _sub_emo_sums[group_name] / ec if ec > 0 else 0
        emotion_results.append({
            "group": group_name, "icon": cfg["icon"],
            "avg_or": round(avg_or, 2), "diff": round(avg_or - mean_or if ec > 0 else 0, 2),
            "count": ec, "pct": round(ec / max(1, total_with_or) * 100, 1),
        })
    emotion_results.sort(key=lambda e: e["avg_or"], reverse=True)
    findings["emotion_radar"] = emotion_results
    findings["_summary"] = {
        "n": len(subset_data), "n_with_or": total_with_or,
        "mean_or": round(mean_or, 3), "median_or": round(median_or, 3), "std_or": round(std_or, 3),
    }
    return findings


def _generate_live_rules_for_subset(findings: dict, rules_out: list) -> None:
    """Generiere Live-Rules für ein Subset (Sport oder NonSport) — lightweight Version."""
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    rule_id = 0
    hour_data = findings.get("hour_analysis", {})
    if isinstance(hour_data, dict) and hour_data.get("best_hour") is not None:
        rule_id += 1
        rules_out.append({
            "id": rule_id, "active": True,
            "rule": f"Primaer-Slot: {hour_data['best_hour']}:00 ({hour_data.get('best_or', 0):.1f}%), Meiden: {hour_data.get('worst_hour', 0)}:00 ({hour_data.get('worst_or', 0):.1f}%)",
            "source": "Subset-Timing-Analyse", "category": "timing",
            "approved_by": "Auto", "approved_at": now_str,
        })
    cat_data = findings.get("cat_analysis", [])
    if cat_data and len(cat_data) >= 2:
        best_cat = max(cat_data, key=lambda c: c.get("avg_or", 0))
        rule_id += 1
        rules_out.append({
            "id": rule_id, "active": True,
            "rule": f"Top-Kategorie: {best_cat['category']} ({best_cat['avg_or']:.1f}%)",
            "source": "Subset-Kategorie-Analyse", "category": "kategorie",
            "approved_by": "Auto", "approved_at": now_str,
        })


# ── Research Modifiers ────────────────────────────────────────────────────

def _compute_research_modifiers(push_data: list, findings: dict, state: dict) -> None:
    """Berechnet konkrete Scoring-Modifier aus Forschungserkenntnissen.

    Jeder Modifier ist ein multiplikativer Faktor (1.0 = neutral).
    """
    modifiers: dict = {
        "version": state.get("live_rules_version", 0),
        "n_rules": len([r for r in state.get("live_rules", []) if r.get("active")]),
        "timing": {}, "category": {}, "framing": {},
        "length": {}, "frequency": {}, "linguistic": {}, "emotion": {},
    }

    valid = [p for p in push_data if p.get("or", 0) > 0]
    if not valid:
        state["research_modifiers"] = modifiers
        return

    global_avg = sum(p["or"] for p in valid) / len(valid)
    if global_avg <= 0:
        state["research_modifiers"] = modifiers
        return

    def _clamp(val: float) -> float:
        return max(0.3, min(3.0, round(val, 3)))

    hour_avgs = findings.get("hour_analysis", {}).get("hour_avgs", {})
    for h, avg in hour_avgs.items():
        modifiers["timing"][str(h)] = _clamp(avg / global_avg)

    for c in findings.get("cat_analysis", []):
        if c.get("avg_or", 0) > 0 and c.get("count", 0) >= 3:
            modifiers["category"][c["category"]] = _clamp(c["avg_or"] / global_avg)

    framing = findings.get("framing_analysis", {})
    for key in ("emotional", "neutral", "question"):
        val = framing.get(f"{key}_or", 0)
        if val > 0:
            modifiers["framing"][key] = _clamp(val / global_avg)

    for key in ("kurz", "mittel", "lang"):
        val = findings.get("title_length", {}).get(f"{key}_or", 0)
        if val > 0:
            modifiers["length"][key] = _clamp(val / global_avg)

    freq = findings.get("frequency_correlation", {})
    if freq:
        modifiers["frequency"]["max_daily"] = freq.get("optimal_daily", 20)
        modifiers["frequency"]["fatigue_r"] = round(freq.get("correlation", 0), 3)

    ling = findings.get("linguistic_analysis", {})
    for key, field in (("with_colon", "colon_or"), ("no_colon", "no_colon_or")):
        val = ling.get(field, 0)
        if val > 0:
            modifiers["linguistic"][key] = _clamp(val / global_avg)

    for e in findings.get("emotion_radar", []):
        if e.get("count", 0) >= 3 and e.get("avg_or", 0) > 0:
            modifiers["emotion"][e["group"]] = _clamp(e["avg_or"] / global_avg)

    eil_ors = [p["or"] for p in valid if p.get("is_eilmeldung")]
    normal_ors = [p["or"] for p in valid if not p.get("is_eilmeldung")]
    if eil_ors and normal_ors:
        modifiers["channel"] = {
            "eilmeldung": _clamp(sum(eil_ors) / len(eil_ors) / global_avg),
            "normal": _clamp(sum(normal_ors) / len(normal_ors) / global_avg),
            "n_eilmeldung": len(eil_ors),
        }

    accuracy = state.get("rolling_accuracy", 0)
    n_pushes = len(valid)
    confidence = min(0.85, (n_pushes / 2000) * 0.4 + (accuracy / 100) * 0.45)
    modifiers["confidence"] = round(confidence, 3)
    modifiers["global_avg"] = round(global_avg, 3)
    modifiers["n_pushes"] = n_pushes
    state["research_modifiers"] = modifiers


# ── Score-Komponenten-Analyse ─────────────────────────────────────────────

def _analyze_score_components(push_data: list, findings: dict, state: dict) -> None:
    """Algo-Team: Berechnet Feature-Importance und Score-Dekomposition."""
    valid = [p for p in push_data if p.get("or", 0) > 0]
    if len(valid) < 10:
        return

    global_avg = sum(p["or"] for p in valid) / len(valid)
    n = len(valid)
    total_var = sum((p["or"] - global_avg) ** 2 for p in valid) / max(1, n - 1)
    if total_var <= 0:
        total_var = 0.01

    hour_groups: dict = defaultdict(list)
    for p in valid:
        hour_groups[p.get("hour", 0)].append(p["or"])
    timing_var = sum(len(vs) * (sum(vs) / len(vs) - global_avg) ** 2 for vs in hour_groups.values() if vs) / max(1, n - 1)

    cat_groups: dict = defaultdict(list)
    for p in valid:
        cat_groups[p.get("cat", "Sonstige")].append(p["or"])
    cat_var = sum(len(vs) * (sum(vs) / len(vs) - global_avg) ** 2 for vs in cat_groups.values() if vs) / max(1, n - 1)

    emo_words = ["schock","drama","skandal","angst","tod","krieg","panik","horror","warnung","krise","alarm","unfall","mord","terror"]
    emo_ors = [p["or"] for p in valid if any(w in p.get("title", "").lower() for w in emo_words)]
    neutral_ors = [p["or"] for p in valid if not any(w in p.get("title", "").lower() for w in emo_words)]
    framing_var = 0.0
    if emo_ors and neutral_ors:
        emo_avg = sum(emo_ors) / len(emo_ors)
        neu_avg = sum(neutral_ors) / len(neutral_ors)
        framing_var = (len(emo_ors) * (emo_avg - global_avg) ** 2 + len(neutral_ors) * (neu_avg - global_avg) ** 2) / max(1, n - 1)

    len_groups: dict = {"kurz": [], "mittel": [], "lang": []}
    for p in valid:
        tl = len(p.get("title", ""))
        if tl < 50:
            len_groups["kurz"].append(p["or"])
        elif tl > 80:
            len_groups["lang"].append(p["or"])
        else:
            len_groups["mittel"].append(p["or"])
    length_var = sum(len(vs) * (sum(vs) / len(vs) - global_avg) ** 2 for vs in len_groups.values() if vs) / max(1, n - 1)

    colon_ors = [p["or"] for p in valid if ":" in p.get("title", "") or "|" in p.get("title", "")]
    no_colon_ors = [p["or"] for p in valid if ":" not in p.get("title", "") and "|" not in p.get("title", "")]
    ling_var = 0.0
    if colon_ors and no_colon_ors:
        c_avg = sum(colon_ors) / len(colon_ors)
        nc_avg = sum(no_colon_ors) / len(no_colon_ors)
        ling_var = (len(colon_ors) * (c_avg - global_avg) ** 2 + len(no_colon_ors) * (nc_avg - global_avg) ** 2) / max(1, n - 1)

    explained_total = timing_var + cat_var + framing_var + length_var + ling_var
    residual_var = max(0.0, total_var - explained_total)

    feature_importance = {
        "timing": round(timing_var / total_var * 100, 1),
        "kategorie": round(cat_var / total_var * 100, 1),
        "framing": round(framing_var / total_var * 100, 1),
        "titel_laenge": round(length_var / total_var * 100, 1),
        "linguistik": round(ling_var / total_var * 100, 1),
        "residual": round(residual_var / total_var * 100, 1),
    }

    modifiers = state.get("research_modifiers", {})
    score_decomposition = {
        "global_avg": round(global_avg, 2),
        "timing_effect": round(modifiers.get("timing", {}).get(str(findings.get("hour_analysis", {}).get("best_hour", 18)), 1.0) - 1.0, 3),
        "category_effect": round(max(modifiers.get("category", {}).values(), default=1.0) - 1.0, 3) if modifiers.get("category") else 0,
        "framing_effect": round(modifiers.get("framing", {}).get("emotional", 1.0) - modifiers.get("framing", {}).get("neutral", 1.0), 3),
        "length_effect": round(max(modifiers.get("length", {}).values(), default=1.0) - min(modifiers.get("length", {}).values(), default=1.0), 3) if modifiers.get("length") else 0,
        "linguistic_effect": round(modifiers.get("linguistic", {}).get("with_colon", 1.0) - modifiers.get("linguistic", {}).get("no_colon", 1.0), 3) if modifiers.get("linguistic") else 0,
    }

    xor_suggestions: list = []
    if feature_importance["timing"] > 20:
        current_w = 0.3
        suggested_w = round(min(0.6, feature_importance["timing"] / 100 * 1.2), 2)
        if suggested_w != current_w:
            xor_suggestions.append({
                "type": "timing_weight", "current": current_w, "suggested": suggested_w,
                "reason": f"Timing erklaert {feature_importance['timing']:.1f}% der OR-Varianz",
                "expected_impact": f"+{(suggested_w - current_w) * feature_importance['timing']:.1f}% Score-Praezision",
            })
    if feature_importance["kategorie"] > 15:
        cat_data = findings.get("cat_analysis", [])
        if cat_data and len(cat_data) > 1:
            best = cat_data[0]
            worst = cat_data[-1]
            if best["avg_or"] - worst["avg_or"] > 1.0:
                xor_suggestions.append({
                    "type": "category_boost",
                    "current": 1.0, "suggested": round(best["avg_or"] / global_avg, 2),
                    "reason": f"{best['category']} ({best['avg_or']:.1f}%) vs. {worst['category']} ({worst['avg_or']:.1f}%)",
                    "expected_impact": "Bessere Kategorie-Differenzierung im Score",
                })
    if feature_importance["framing"] > 10 and emo_ors and neutral_ors:
        emo_avg = sum(emo_ors) / len(emo_ors)
        neu_avg = sum(neutral_ors) / len(neutral_ors)
        if abs(emo_avg - neu_avg) > 0.5:
            xor_suggestions.append({
                "type": "framing_factor",
                "current": 1.0, "suggested": round(emo_avg / neu_avg, 2),
                "reason": f"Emotional {emo_avg:.1f}% vs. Neutral {neu_avg:.1f}%",
                "expected_impact": "Emotionales Framing korrekt einpreisen",
            })

    state["algo_score_analysis"] = {
        "ts": datetime.datetime.now().strftime("%d.%m. %H:%M"),
        "n_pushes": n,
        "feature_importance": feature_importance,
        "score_decomposition": score_decomposition,
        "xor_suggestions": xor_suggestions,
        "total_variance": round(total_var, 4),
        "explained_variance": round(explained_total / total_var * 100, 1) if total_var > 0 else 0,
    }


# ── Externer Kontext ──────────────────────────────────────────────────────

def _fetch_external_context(state: dict) -> dict:
    """Holt Wetter, Google Trends und Feiertag-Info. Alle 30min."""
    global _external_context_cache
    try:
        from app.config import RESEARCH_EXTERNAL_CONTEXT_ENABLED
    except Exception:
        RESEARCH_EXTERNAL_CONTEXT_ENABLED = False

    now = time.time()
    if now - _external_context_cache["last_fetch"] < 1800:
        return _external_context_cache

    if not RESEARCH_EXTERNAL_CONTEXT_ENABLED:
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        holiday = _GERMAN_HOLIDAYS.get(today_str, "")
        weekday = datetime.datetime.now().weekday()
        day_type = "holiday" if holiday else ("weekend" if weekday >= 5 else "weekday")
        hour = datetime.datetime.now().hour
        if 6 <= hour <= 8:
            time_context = "pendler_morgen"
        elif 11 <= hour <= 13:
            time_context = "mittagspause"
        elif 16 <= hour <= 18:
            time_context = "feierabend"
        elif 20 <= hour <= 23:
            time_context = "prime_time"
        elif 0 <= hour <= 5:
            time_context = "nacht"
        else:
            time_context = "normal"

        _external_context_cache = {
            "weather": {"bad_weather_score": 0.3, "temp_c": 15, "weather_desc": "disabled"},
            "trends": [],
            "holiday": holiday,
            "last_fetch": now,
        }
        state["external_context"] = {
            "weather": _external_context_cache["weather"],
            "trends_count": 0,
            "trends_top5": [],
            "holiday": holiday,
            "day_type": day_type,
            "time_context": time_context,
            "last_update": datetime.datetime.now().strftime("%H:%M"),
            "mode": "local-defaults",
        }
        return _external_context_cache

    log.info("[Kontext] Fetche externe Datenquellen...")

    # Wetter Berlin via wttr.in
    weather: dict = {"bad_weather_score": 0.3, "temp_c": 15, "weather_desc": "unbekannt"}
    try:
        ssl_ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://wttr.in/Berlin?format=j1",
            headers={"User-Agent": "PushBalancer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            cur = data.get("current_condition", [{}])[0]
            temp_c = int(cur.get("temp_C", 15))
            precip_mm = float(cur.get("precipMM", 0))
            cloud_cover = int(cur.get("cloudcover", 50))
            wind_kmph = int(cur.get("windspeedKmph", 0))
            bad_score = 0.0
            if precip_mm > 0.5:
                bad_score += 0.3
            if temp_c < 5:
                bad_score += 0.2
            elif temp_c > 30:
                bad_score += 0.1
            if cloud_cover > 80:
                bad_score += 0.15
            if wind_kmph > 30:
                bad_score += 0.1
            weather = {
                "temp_c": temp_c, "precip_mm": precip_mm, "cloud_cover": cloud_cover,
                "wind_kmph": wind_kmph, "humidity": int(cur.get("humidity", 50)),
                "weather_desc": cur.get("weatherDesc", [{}])[0].get("value", ""),
                "bad_weather_score": round(min(1.0, bad_score), 2),
            }
    except Exception as exc:
        log.warning("[Kontext] Wetter-Fetch fehlgeschlagen: %s", exc)

    # Google Trends Deutschland
    trends: list = []
    try:
        ssl_ctx = ssl.create_default_context()
        req = urllib.request.Request(
            "https://trends.google.com/trending/rss?geo=DE",
            headers={"User-Agent": "PushBalancer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
            titles = re.findall(r"<title>([^<]+)</title>", xml)
            for t in titles[1:21]:
                t = t.strip()
                if t and len(t) > 1:
                    trends.append(t.lower())
    except Exception as exc:
        log.warning("[Kontext] Google Trends fehlgeschlagen: %s", exc)

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    holiday = _GERMAN_HOLIDAYS.get(today_str, "")
    weekday = datetime.datetime.now().weekday()
    day_type = "holiday" if holiday else ("weekend" if weekday >= 5 else "weekday")

    hour = datetime.datetime.now().hour
    if 6 <= hour <= 8:
        time_context = "pendler_morgen"
    elif 11 <= hour <= 13:
        time_context = "mittagspause"
    elif 16 <= hour <= 18:
        time_context = "feierabend"
    elif 20 <= hour <= 23:
        time_context = "prime_time"
    elif 0 <= hour <= 5:
        time_context = "nacht"
    else:
        time_context = "normal"

    _external_context_cache = {
        "weather": weather, "trends": trends,
        "holiday": holiday, "last_fetch": now,
    }
    state["external_context"] = {
        "weather": weather, "trends_count": len(trends),
        "trends_top5": trends[:5], "holiday": holiday,
        "day_type": day_type, "time_context": time_context,
        "last_update": datetime.datetime.now().strftime("%H:%M"),
    }
    return _external_context_cache


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
        _update_rolling_accuracy_subset(sport_data, state, "_sport_accuracy")
        _update_rolling_accuracy_subset(nonsport_data, state, "_nonsport_accuracy")
        state["live_rules"] = []  # Force regeneration

    # Volle Re-Analyse: bei neuen Daten oder alle 60s
    if not is_new_data and state.get("findings") and now - state.get("last_analysis", 0) < 60:
        _fetch_external_context(state)
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
    _update_rolling_accuracy_subset(sport_data, state, "_sport_accuracy")
    _update_rolling_accuracy_subset(nonsport_data, state, "_nonsport_accuracy")

    # ── Basis-Analysen ────────────────────────────────────────────────
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
        for r in sorted_rows[-25:]:
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


# ── RAM-Cleanup ────────────────────────────────────────────────────────────

# Maximale Pufferlängen (halbiert gegenüber ursprünglichen Werten)
_BUFFER_LIMITS: dict = {
    "ticker_entries":     100,
    "research_log":        50,
    "schwab_decisions":    50,
    "prediction_feedback": 50,
    "tuning_history":      50,
    "live_pulse":          50,
    "bild_adaptations":    50,
    "accuracy_history":   100,
}

# Cleanup-Tracking (wird von /api/memory-stats ausgelesen)
_cleanup_stats: dict = {
    "last_cleanup_ts": 0,
    "items_freed_total": 0,
    "items_freed_last": 0,
    "cleanup_runs": 0,
    "done_runs_pending_cleanup": 0,
}
_cleanup_stats_lock = threading.Lock()


def trim_state_buffers() -> int:
    """Kürzt alle unbegrenzt wachsenden Listen in _research_state auf ihre Maximalwerte.

    Gibt die Anzahl der entfernten Einträge zurück.
    Wird alle 2 Minuten vom Memory-Cleanup-Worker aufgerufen.
    """
    freed = 0
    lock = _research_state.get("analysis_lock")
    acquire = lock and lock.acquire(blocking=False)
    try:
        for key, limit in _BUFFER_LIMITS.items():
            lst = _research_state.get(key)
            if isinstance(lst, list) and len(lst) > limit:
                excess = len(lst) - limit
                _research_state[key] = lst[-limit:]
                freed += excess
    finally:
        if acquire:
            try:
                lock.release()
            except RuntimeError:
                pass

    now = time.time()
    with _cleanup_stats_lock:
        _cleanup_stats["items_freed_total"] += freed
        _cleanup_stats["items_freed_last"] = freed
        _cleanup_stats["last_cleanup_ts"] = now
        _cleanup_stats["cleanup_runs"] += 1
        # "done_runs_pending_cleanup" = Einträge in accuracy_history, die durch
        # den nächsten Cleanup-Lauf noch entfernt werden könnten
        ah = _research_state.get("accuracy_history", [])
        limit_ah = _BUFFER_LIMITS["accuracy_history"]
        _cleanup_stats["done_runs_pending_cleanup"] = max(0, len(ah) - limit_ah)

    if freed > 0:
        log.debug("[MemCleanup] %d Einträge aus State-Puffern entfernt", freed)
    return freed


_XOR_STOP_WORDS = {
    "der", "die", "das", "und", "oder", "ist", "war", "hat", "ein", "eine",
    "von", "für", "mit", "auf", "an", "im", "zu", "am",
}


# ── Model-Selector ─────────────────────────────────────────────────────────

def _model_selector_update(rows_24h: list | None = None) -> None:
    """Aktualisiert den Model-Selector basierend auf letzten 100 Predictions.

    Entscheidet ob Unified oder ML-Ensemble als primärer Predictor genutzt wird.
    Aufgerufen von monitoring_tick().
    """
    from app.state import _model_selector_state

    now = time.time()
    if now - _model_selector_state.get("last_check_ts", 0) < 600:
        return
    _model_selector_state["last_check_ts"] = now

    try:
        from app.ml.lightgbm_model import _unified_state, _unified_lock
        with _unified_lock:
            unified_available = _unified_state.get("model") is not None
    except Exception:
        unified_available = False

    if not unified_available:
        _model_selector_state["active_model"] = "ml_ensemble"
        return

    try:
        from app.config import PUSH_DB_PATH
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        cutoff = int(now - 86400)
        rows = conn.execute(
            "SELECT predicted_or, actual_or, methods_detail "
            "FROM prediction_log "
            "WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at > ? "
            "ORDER BY predicted_at DESC LIMIT 100",
            (cutoff,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.debug("[ModelSelector] DB-Fehler: %s", exc)
        return

    if len(rows) < 10:
        return

    import json as _json
    unified_errors: list = []
    ensemble_errors: list = []
    for r in rows:
        try:
            detail = _json.loads(r["methods_detail"] or "{}")
            actual = float(r["actual_or"])
            u_pred = float(detail.get("unified_predicted", 0) or 0)
            if u_pred > 0:
                unified_errors.append(abs(u_pred - actual))
            e_pred = float(
                detail.get("ml_ensemble", 0) or
                detail.get("shadow_gbrt", 0) or
                detail.get("gbrt_predicted", 0) or 0
            )
            if e_pred > 0:
                ensemble_errors.append(abs(e_pred - actual))
        except (ValueError, TypeError):
            continue

    _model_selector_state["evaluated_count"] = len(unified_errors)

    if len(unified_errors) < 30:
        _model_selector_state["active_model"] = "ml_ensemble"
        log.info("[ModelSelector] Cold-Start: ml_ensemble (nur %d Unified-Predictions)", len(unified_errors))
        return

    unified_mae = sum(unified_errors) / len(unified_errors)
    _model_selector_state["unified_mae_24h"] = round(unified_mae, 4)

    if ensemble_errors:
        ensemble_mae = sum(ensemble_errors) / len(ensemble_errors)
        _model_selector_state["ensemble_mae_24h"] = round(ensemble_mae, 4)
    else:
        ensemble_mae = float("inf")

    if unified_mae < ensemble_mae * 1.05:
        if _model_selector_state.get("active_model") != "unified":
            log.info(
                "[ModelSelector] Wechsel zu unified: MAE=%.4f < Ensemble=%.4f×1.05",
                unified_mae, ensemble_mae,
            )
        _model_selector_state["active_model"] = "unified"
        _model_selector_state["consecutive_worse"] = 0
    else:
        _model_selector_state["consecutive_worse"] = _model_selector_state.get("consecutive_worse", 0) + 1
        if _model_selector_state["consecutive_worse"] >= 3 and unified_mae > ensemble_mae * 1.15:
            _model_selector_state["active_model"] = "ml_ensemble"
            log.warning(
                "[ModelSelector] Fallback zu ml_ensemble: Unified MAE=%.4f > Ensemble=%.4f×1.15, "
                "%d× schlechter → Retrain",
                unified_mae, ensemble_mae, _model_selector_state["consecutive_worse"],
            )
            _model_selector_state["consecutive_worse"] = 0
            try:
                from app.ml.lightgbm_model import unified_train
                threading.Thread(target=unified_train, daemon=True).start()
            except Exception:
                pass
        else:
            _model_selector_state["active_model"] = "ml_ensemble"

    log.info(
        "[ModelSelector] active=%s, unified_MAE=%.4f, ensemble_MAE=%.4f, evaluated=%d",
        _model_selector_state["active_model"], unified_mae, ensemble_mae, len(unified_errors),
    )


def monitoring_tick() -> None:
    """Monitoring-Tick: MAE-Statistiken, Calibration, Residual-Corrector, Auto-Retrain."""
    try:
        from app.config import PUSH_DB_PATH
        now_ts = int(time.time())
        ts_24h = now_ts - 86400
        ts_7d = now_ts - 7 * 86400

        conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row

        rows_24h = conn.execute(
            "SELECT predicted_or, actual_or, predicted_at FROM prediction_log "
            "WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at >= ? "
            "ORDER BY predicted_at DESC",
            (ts_24h,),
        ).fetchall()

        rows_7d = conn.execute(
            "SELECT predicted_or, actual_or FROM prediction_log "
            "WHERE actual_or > 0 AND predicted_or > 0 AND predicted_at >= ?",
            (ts_7d,),
        ).fetchall()

        conn.close()

        mae_24h = 0.0
        if rows_24h:
            mae_24h = round(
                sum(abs(r["predicted_or"] - r["actual_or"]) for r in rows_24h) / len(rows_24h), 4
            )

        mae_7d = 0.0
        if rows_7d:
            mae_7d = round(
                sum(abs(r["predicted_or"] - r["actual_or"]) for r in rows_7d) / len(rows_7d), 4
            )

        # Stündlicher MAE-Trend (24h, als sortierte Liste wie im Legacy)
        hourly_buckets: dict = {}
        for r in rows_24h:
            h_ts = (r["predicted_at"] // 3600) * 3600
            hourly_buckets.setdefault(h_ts, []).append(abs(r["predicted_or"] - r["actual_or"]))
        mae_trend = [
            {
                "ts": h_ts,
                "mae": round(sum(hourly_buckets[h_ts]) / len(hourly_buckets[h_ts]), 4),
                "n": len(hourly_buckets[h_ts]),
            }
            for h_ts in sorted(hourly_buckets)
            if hourly_buckets[h_ts]
        ][-24:]

        # Calibration-Bias + Trend (letzte 100, in 20er-Blöcken)
        calibration_bias = 0.0
        calibration_trend: list = []
        recent_100 = list(rows_24h[:100])
        if recent_100:
            signed = [r["predicted_or"] - r["actual_or"] for r in recent_100]
            calibration_bias = round(sum(signed) / len(signed), 4)
            for i in range(0, len(signed), 20):
                block = signed[i:i + 20]
                if block:
                    calibration_trend.append(round(sum(block) / len(block), 4))

        # Residual Corrector aktualisieren + Snapshot erfassen
        update_residual_corrector()
        with _residual_corrector_lock:
            rc_snapshot = {
                "global_bias": _residual_corrector["global_bias"],
                "cat_bias": dict(_residual_corrector["cat_bias"]),
                "hourgroup_bias": dict(_residual_corrector["hourgroup_bias"]),
                "n_samples": _residual_corrector["n_samples"],
                "last_update_ts": _residual_corrector["last_update_ts"],
            }

        # Ergebnisse atomar in beide State-Dicts schreiben
        with _monitoring_state_lock:
            _monitoring_state["last_tick"] = now_ts
            _monitoring_state["mae_24h"] = mae_24h
            _monitoring_state["mae_7d"] = mae_7d
            _monitoring_state["mae_trend"] = mae_trend
            _monitoring_state["calibration_bias"] = calibration_bias
            _monitoring_state["calibration_trend"] = calibration_trend
            _monitoring_state["residual_corrector"] = rc_snapshot

        with _research_state_lock:
            _research_state["mae_24h"] = mae_24h
            _research_state["mae_7d"] = mae_7d
            _research_state["mae_trend"] = mae_trend
            _research_state["calibration_bias"] = calibration_bias
            _research_state["monitoring_last_ts"] = now_ts

        # MAE-Spike Warnung + DB-Event
        if mae_7d > 0 and mae_24h > mae_7d * 1.15:
            log.warning(
                "[monitoring_tick] MAE-Spike: mae_24h=%.4f ist %.1f%% über 7d-Baseline=%.4f",
                mae_24h, (mae_24h / mae_7d - 1) * 100, mae_7d,
            )
            try:
                from app.database import log_monitoring_event
                log_monitoring_event(
                    "mae_spike", "warning",
                    f"MAE 24h ({mae_24h:.4f}) ist {(mae_24h / mae_7d - 1) * 100:.1f}% über 7d-Baseline ({mae_7d:.4f})",
                    {"mae_24h": mae_24h, "mae_7d": mae_7d, "ratio": round(mae_24h / mae_7d, 3)},
                )
            except Exception:
                pass
        elif mae_24h > 2.0:
            log.warning(
                "[monitoring_tick] MAE verschlechtert: mae_24h=%.4f (Schwellwert 2.0), mae_7d=%.4f",
                mae_24h, mae_7d,
            )
        else:
            log.info(
                "[monitoring_tick] mae_24h=%.4f, mae_7d=%.4f, bias=%+.4f, n_24h=%d",
                mae_24h, mae_7d, calibration_bias, len(rows_24h),
            )

        # Calibration-Shift Warnung + DB-Event
        if abs(calibration_bias) > 0.5:
            log.warning(
                "[monitoring_tick] Calibration-Shift: bias=%+.4f (Modell %sschätzt systematisch)",
                calibration_bias, "über" if calibration_bias > 0 else "unter",
            )
            try:
                from app.database import log_monitoring_event
                log_monitoring_event(
                    "calibration_shift", "warning",
                    f"Calibration Bias = {calibration_bias:+.4f} "
                    f"(Modell {'über' if calibration_bias > 0 else 'unter'}schätzt systematisch)",
                    {"bias": calibration_bias},
                )
            except Exception:
                pass

        # Auto-Retrain Trigger: MAE_24h > MAE_7d × 1.3 für 3 aufeinanderfolgende Ticks
        if mae_7d > 0 and mae_24h > mae_7d * 1.3:
            _auto_retrain_state["consecutive_degraded_ticks"] += 1
            if (
                _auto_retrain_state["consecutive_degraded_ticks"] >= 3
                and now_ts - _auto_retrain_state["last_retrain_trigger_ts"] > 3600
            ):
                _auto_retrain_state["last_retrain_trigger_ts"] = now_ts
                _auto_retrain_state["consecutive_degraded_ticks"] = 0
                _auto_retrain_state["total_retrains"] += 1
                log.warning(
                    "[AutoRetrain] Trigger: mae_24h=%.4f > mae_7d×1.3=%.4f → Retrain wird gestartet",
                    mae_24h, mae_7d * 1.3,
                )
                try:
                    from app.database import log_monitoring_event
                    log_monitoring_event(
                        "auto_retrain", "warning",
                        f"Auto-Retrain getriggert: MAE_24h={mae_24h:.4f} > MAE_7d×1.3",
                        {"mae_24h": mae_24h, "mae_7d": mae_7d},
                    )
                except Exception:
                    pass
                try:
                    from app.ml.lightgbm_model import ml_train_model
                    threading.Thread(target=ml_train_model, daemon=True).start()
                except Exception as _e:
                    log.debug("[AutoRetrain] ml_train_model nicht verfügbar: %s", _e)
                try:
                    from app.ml.gbrt import gbrt_train
                    threading.Thread(target=gbrt_train, daemon=True).start()
                except Exception as _e:
                    log.debug("[AutoRetrain] gbrt_train nicht verfügbar: %s", _e)
        else:
            _auto_retrain_state["consecutive_degraded_ticks"] = 0

        # Model-Selector aktualisieren
        try:
            _model_selector_update(rows_24h)
        except Exception as _mse:
            log.debug("[ModelSelector] Update-Fehler: %s", _mse)

    except Exception as exc:
        log.warning("[research] monitoring_tick Fehler: %s", exc)


def build_xor_perf_cache() -> None:
    """Baut XOR-Performance-Cache aus historischer Push-Daten."""
    try:
        from app.database import push_db_load_all

        # Daten laden: aus State wenn frisch, sonst direkt aus DB
        push_data = _research_state.get("push_data") or []
        if len(push_data) < 100:
            push_data = push_db_load_all()

        # Filtere: OR > 0 und OR <= 25
        valid = [p for p in push_data if 0 < p.get("or", 0) <= 25]

        if len(valid) < 100:
            log.info("[build_xor_perf_cache] Zu wenig Daten (%d), Cache nicht gebaut", len(valid))
            return

        # ── word_perf ──────────────────────────────────────────────────────
        word_ors: dict = defaultdict(list)
        for p in valid:
            title_lower = (p.get("title") or "").lower()
            tokens = re.findall(r"[a-zäöüß]{3,}", title_lower)
            for tok in tokens:
                if tok not in _XOR_STOP_WORDS:
                    word_ors[tok].append(p["or"])

        word_perf: dict = {}
        for word, ors in word_ors.items():
            if len(ors) < 5:
                continue
            ors_sorted = sorted(ors)
            n = len(ors_sorted)
            avg = sum(ors_sorted) / n
            p25 = ors_sorted[int(n * 0.25)]
            p75 = ors_sorted[int(n * 0.75)]
            p90 = ors_sorted[int(n * 0.90)]
            word_perf[word] = {
                "avg": round(avg, 3),
                "count": n,
                "p25": round(p25, 3),
                "p75": round(p75, 3),
                "p90": round(p90, 3),
            }

        # ── cat_hour_perf ─────────────────────────────────────────────────
        cat_hour_ors: dict = defaultdict(list)
        for p in valid:
            cat = p.get("cat") or "News"
            hour = p.get("hour", 0)
            key = f"{cat}_{hour}"
            cat_hour_ors[key].append(p["or"])

        cat_hour_perf: dict = {}
        for key, ors in cat_hour_ors.items():
            if len(ors) < 3:
                continue
            ors_sorted = sorted(ors)
            n = len(ors_sorted)
            avg = sum(ors_sorted) / n
            p25 = ors_sorted[int(n * 0.25)]
            p50 = ors_sorted[int(n * 0.50)]
            p75 = ors_sorted[int(n * 0.75)]
            cat_hour_perf[key] = {
                "avg": round(avg, 3),
                "p25": round(p25, 3),
                "p50": round(p50, 3),
                "p75": round(p75, 3),
                "count": n,
            }

        # ── eil_perf ───────────────────────────────────────────────────────
        eil_ors = [p["or"] for p in valid if p.get("is_eilmeldung") or
                   any(m in (p.get("title") or "").lower() for m in ("eilmeldung", "+++", "breaking"))]
        eil_perf: dict = {}
        if eil_ors:
            eil_sorted = sorted(eil_ors)
            ne = len(eil_sorted)
            eil_perf = {
                "avg": round(sum(eil_sorted) / ne, 3),
                "count": ne,
                "p25": round(eil_sorted[int(ne * 0.25)], 3),
                "p75": round(eil_sorted[int(ne * 0.75)], 3),
                "p90": round(eil_sorted[int(ne * 0.90)], 3),
            }

        # ── global_avg ─────────────────────────────────────────────────────
        global_avg = round(sum(p["or"] for p in valid) / len(valid), 3)

        # Thread-safe schreiben
        with _xor_perf_lock:
            _xor_perf_cache["word_perf"] = word_perf
            _xor_perf_cache["cat_hour_perf"] = cat_hour_perf
            _xor_perf_cache["eil_perf"] = eil_perf
            _xor_perf_cache["global_avg"] = global_avg
            _xor_perf_cache["built_at"] = time.time()

        log.info(
            "[build_xor_perf_cache] Fertig: %d Wörter, %d Cat-Hour-Slots, global_avg=%.2f%%, n=%d",
            len(word_perf), len(cat_hour_perf), global_avg, len(valid),
        )
    except Exception as exc:
        log.warning("[research] build_xor_perf_cache Fehler: %s", exc)


def get_cached_feeds(feed_type: str) -> dict | list:
    """Liefert gecachte Feeds aus dem Background-Cache."""
    with _feed_cache_lock:
        entry = _feed_cache.get(feed_type, {})
        if entry.get("data") and (time.time() - entry.get("ts", 0)) < _FEED_CACHE_TTL * 3:
            return entry["data"]
    return {}
