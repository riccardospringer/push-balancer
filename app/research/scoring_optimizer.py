"""app/research/scoring_optimizer.py — Agentenarmee für Scoring-Optimierung.

Fünf spezialisierte Agenten analysieren Live-Push-Performance und verbessern
sukzessive die Vorhersagequalität und das Scoring:

  Agent 1 — Feedback-Bridge:   prediction_log → research_state
  Agent 2 — Bias-Calibrator:   systematische Fehler pro Kategorie/Stunde korrigieren
  Agent 3 — Score-Validator:   Push-Score-Komponenten gegen echte OR validieren
  Agent 4 — Retraining-Gate:   ML-Modell neu trainieren wenn nötig
  Agent 5 — Slot-Optimizer:    beste Push-Fenster (Kategorie × Stunde) aktualisieren

Wird von _run_analysis_inner() am Ende jedes Analyse-Zyklus aufgerufen.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections import defaultdict

log = logging.getLogger("push-balancer")

# Mindestwerte für statistisch zuverlässige Aussagen
_MIN_SAMPLES = 8
_MIN_SAMPLES_HOUR = 5
_WINDOW_RECENT = 50      # Letzte N Predictions für Bias-Berechnung
_RETRAIN_COOLDOWN = 3600  # Mindestens 1h zwischen zwei Retrainings
_RETRAIN_NEW_MIN = 30     # Mindestens 30 neue Feedback-Einträge seit letztem Training
_DRIFT_MAE_FACTOR = 1.35  # Retraining wenn MAE um 35% über Rolling-Durchschnitt


def run_scoring_optimizer(state: dict, push_data: list) -> None:
    """Orchestriert alle 5 Agenten — wird nach jeder vollen Analyse aufgerufen."""
    t0 = time.monotonic()
    results: dict[str, str] = {}

    for name, fn in [
        ("feedback_bridge", _agent_feedback_bridge),
        ("bias_calibrator", _agent_bias_calibrator),
        ("score_validator", _agent_score_validator),
        ("retraining_gate", _agent_retraining_gate),
        ("slot_optimizer", _agent_slot_optimizer),
    ]:
        try:
            fn(state, push_data)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"err: {exc}"
            log.warning("[optimizer/%s] Fehler: %s", name, exc)

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    ok_count = sum(1 for v in results.values() if v == "ok")
    log.info(
        "[optimizer] %d/5 Agenten OK in %sms — %s",
        ok_count, elapsed,
        ", ".join(f"{k}={v}" for k, v in results.items()),
    )
    state["optimizer_last_run"] = time.time()
    state["optimizer_results"] = results


# ── Agent 1: Feedback-Bridge ───────────────────────────────────────────────

def _agent_feedback_bridge(state: dict, push_data: list) -> None:
    """Lädt Prediction-Log-Daten und hängt sie an push_data an.

    Ergänzt push_data um predicted_or aus prediction_log — damit sind
    bei jedem Push sowohl der Artikel-Kontext als auch predicted/actual OR verfügbar.
    """
    try:
        from app.database import PUSH_DB_PATH
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT push_id, predicted_or, actual_or, basis_method, predicted_at
               FROM prediction_log
               WHERE actual_or > 0 AND predicted_or > 0
               ORDER BY predicted_at DESC LIMIT 500"""
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.debug("[optimizer/feedback_bridge] DB-Fehler: %s", exc)
        return

    if not rows:
        return

    # Index prediction_log nach push_id
    pred_map: dict[str, dict] = {
        row["push_id"]: {
            "predicted_or": float(row["predicted_or"]),
            "actual_or": float(row["actual_or"]),
            "basis_method": row["basis_method"] or "",
            "predicted_at": int(row["predicted_at"]),
        }
        for row in rows
    }

    # Push-Daten mit prediction_log-Daten anreichern
    enriched = 0
    for p in push_data:
        mid = p.get("message_id") or p.get("id", "")
        if mid in pred_map:
            entry = pred_map[mid]
            p["_predicted_or"] = entry["predicted_or"]
            p["_prediction_error"] = entry["predicted_or"] - (p.get("or") or entry["actual_or"])
            p["_basis_method"] = entry["basis_method"]
            enriched += 1

    # Statistiken speichern
    n = len(rows)
    recent = list(rows[:_WINDOW_RECENT])
    if recent:
        errors = [float(r["predicted_or"]) - float(r["actual_or"]) for r in recent]
        abs_errors = [abs(e) for e in errors]
        mae = sum(abs_errors) / len(abs_errors)
        bias = sum(errors) / len(errors)
    else:
        mae = 0.0
        bias = 0.0

    state["prediction_log_n"] = n
    state["prediction_log_enriched"] = enriched
    state["prediction_log_mae"] = round(mae, 4)
    state["prediction_log_bias"] = round(bias, 4)
    log.debug("[optimizer/feedback_bridge] %d Predictions geladen, %d angereichert, MAE=%.3f", n, enriched, mae)


# ── Agent 2: Bias-Calibrator ───────────────────────────────────────────────

def _agent_bias_calibrator(state: dict, push_data: list) -> None:
    """Berechnet systematische Fehler pro Kategorie, Stunde und Wochentag.

    Ergebnis: state["ml_calibration"] mit granularen Bias-Korrekturen.
    Diese werden von predict_or() über den bestehenden Residual-Corrector genutzt.
    """
    # Pushes mit Prediction-Anreicherung (von Agent 1)
    enriched = [p for p in push_data if "_predicted_or" in p and p.get("or", 0) > 0]
    if len(enriched) < _MIN_SAMPLES:
        return

    now = time.time()
    cutoff_30d = now - 30 * 86400

    recent = [p for p in enriched if p.get("ts_num", 0) > cutoff_30d]
    if not recent:
        recent = enriched

    # Fehler = predicted - actual (positiv: over-prediction, negativ: under-prediction)
    def _bias(items: list) -> float:
        if not items:
            return 0.0
        errors = [p["_predicted_or"] - p.get("or", p["_predicted_or"]) for p in items]
        return sum(errors) / len(errors)

    # Globaler Bias
    global_bias = _bias(recent)

    # Kategorie-spezifisch
    cat_groups: dict[str, list] = defaultdict(list)
    for p in recent:
        cat_groups[p.get("cat") or "News"].append(p)

    cat_bias: dict[str, float] = {}
    for cat, items in cat_groups.items():
        if len(items) >= _MIN_SAMPLES:
            cat_bias[cat] = round(_bias(items), 3)

    # Stunden-spezifisch
    hour_groups: dict[int, list] = defaultdict(list)
    for p in recent:
        h = p.get("hour")
        if h is not None:
            hour_groups[int(h)].append(p)

    hour_bias: dict[int, float] = {}
    for h, items in hour_groups.items():
        if len(items) >= _MIN_SAMPLES_HOUR:
            hour_bias[h] = round(_bias(items), 3)

    # Wochentag-spezifisch
    weekday_groups: dict[int, list] = defaultdict(list)
    for p in recent:
        ts = p.get("ts_num", 0)
        if ts > 0:
            import datetime
            wd = datetime.datetime.fromtimestamp(ts).weekday()
            weekday_groups[wd].append(p)

    weekday_bias: dict[int, float] = {}
    for wd, items in weekday_groups.items():
        if len(items) >= _MIN_SAMPLES:
            weekday_bias[wd] = round(_bias(items), 3)

    calibration = {
        "global_bias": round(global_bias, 3),
        "cat_bias": cat_bias,
        "hour_bias": hour_bias,
        "weekday_bias": weekday_bias,
        "n_samples": len(recent),
        "updated_at": now,
    }

    state["ml_calibration"] = calibration

    # Auch in Residual-Corrector schreiben (kompatibel mit predict.py)
    if len(recent) >= _MIN_SAMPLES:
        hour_groups_str = {str(k): v for k, v in hour_bias.items()}
        existing = state.get("_residual_corrector", {})
        existing["global_bias"] = round(global_bias, 4)
        existing["cat_bias"] = {c: round(b, 4) for c, b in cat_bias.items()}
        existing["hourgroup_bias"] = {
            "morning": _weighted_hour_bias(hour_bias, range(6, 12)),
            "afternoon": _weighted_hour_bias(hour_bias, range(12, 18)),
            "evening": _weighted_hour_bias(hour_bias, range(18, 23)),
            "night": _weighted_hour_bias(hour_bias, list(range(0, 6)) + [23]),
        }
        existing["n_samples"] = len(recent)
        state["_residual_corrector"] = existing

    log.debug(
        "[optimizer/bias_calibrator] global_bias=%.3f, %d cat, %d hour biases",
        global_bias, len(cat_bias), len(hour_bias),
    )


def _weighted_hour_bias(hour_bias: dict[int, float], hours: range | list) -> float:
    vals = [hour_bias[h] for h in hours if h in hour_bias]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


# ── Agent 3: Score-Validator ───────────────────────────────────────────────

def _agent_score_validator(state: dict, push_data: list) -> None:
    """Misst ob der Push-Score (0–100) tatsächlich mit hoher OR korreliert.

    Teilt Push-Daten in Score-Quartile auf und misst die OR pro Quartil.
    Ergebnis in state["score_validation"] — zeigt ob das Scoring Signal hat.
    """
    valid = [
        p for p in push_data
        if p.get("or", 0) > 0 and p.get("score") is not None
    ]
    if len(valid) < 20:
        return

    # Push-Score-Quartile
    sorted_by_score = sorted(valid, key=lambda p: p.get("score", 0))
    n = len(sorted_by_score)
    q = n // 4

    quartiles = {
        "Q1_low_score": sorted_by_score[:q],
        "Q2": sorted_by_score[q:2*q],
        "Q3": sorted_by_score[2*q:3*q],
        "Q4_high_score": sorted_by_score[3*q:],
    }

    quartile_stats: dict[str, dict] = {}
    for name, items in quartiles.items():
        if items:
            ors = [p["or"] for p in items]
            scores = [p.get("score", 0) for p in items]
            quartile_stats[name] = {
                "n": len(items),
                "avg_score": round(sum(scores) / len(scores), 1),
                "avg_or": round(sum(ors) / len(ors), 2),
                "or_std": round(_std(ors), 2),
            }

    # Pearson-Korrelation Score ↔ OR
    corr = _pearson_corr(
        [p.get("score", 0) for p in valid],
        [p["or"] for p in valid],
    )

    # Score-Signal pro Kategorie
    cat_corr: dict[str, float] = {}
    cat_groups: dict[str, list] = defaultdict(list)
    for p in valid:
        cat_groups[p.get("cat") or "News"].append(p)
    for cat, items in cat_groups.items():
        if len(items) >= 15:
            cat_corr[cat] = round(_pearson_corr(
                [p.get("score", 0) for p in items],
                [p["or"] for p in items],
            ), 3)

    state["score_validation"] = {
        "quartile_stats": quartile_stats,
        "score_or_correlation": round(corr, 3),
        "cat_correlations": cat_corr,
        "n_samples": n,
        "updated_at": time.time(),
        "signal_strength": "strong" if abs(corr) > 0.3 else ("moderate" if abs(corr) > 0.15 else "weak"),
    }

    log.debug(
        "[optimizer/score_validator] n=%d, score↔OR Korrelation=%.3f (%s)",
        n, corr, state["score_validation"]["signal_strength"],
    )


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _pearson_corr(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx * dy < 1e-10:
        return 0.0
    return num / (dx * dy)


# ── Agent 4: Retraining-Gate ───────────────────────────────────────────────

def _agent_retraining_gate(state: dict, push_data: list) -> None:
    """Entscheidet ob das ML-Modell neu trainiert werden soll.

    Kriterien für Retraining:
    1. Genug neue Feedback-Daten seit letztem Training (_RETRAIN_NEW_MIN)
    2. MAE ist signifikant schlechter als Rolling-Durchschnitt (_DRIFT_MAE_FACTOR)
    3. Cooldown (_RETRAIN_COOLDOWN) abgelaufen
    """
    now = time.time()

    # Cooldown prüfen
    last_train = state.get("last_ml_retrain_ts", 0)
    if now - last_train < _RETRAIN_COOLDOWN:
        return

    # Neue Feedback-Einträge seit letztem Training zählen
    try:
        from app.database import PUSH_DB_PATH
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        new_count = conn.execute(
            "SELECT COUNT(*) as n FROM prediction_log WHERE predicted_or > 0 AND actual_or > 0 AND predicted_at > ?",
            (int(last_train),),
        ).fetchone()["n"]
        conn.close()
    except Exception:
        return

    # Aktuelle MAE aus Feedback-Bridge
    current_mae = state.get("prediction_log_mae", 0.0)
    mae_history = state.get("mae_history", [])

    should_retrain = False
    reason = ""

    # Retraining wenn genug neue Daten
    if new_count >= _RETRAIN_NEW_MIN:
        should_retrain = True
        reason = f"{new_count} neue Feedback-Einträge seit letztem Training"

    # Retraining bei Genauigkeitsverlust (Drift)
    elif len(mae_history) >= 5 and current_mae > 0:
        rolling_mae = sum(mae_history[-5:]) / 5
        if current_mae > rolling_mae * _DRIFT_MAE_FACTOR:
            should_retrain = True
            reason = f"MAE-Drift: {current_mae:.3f} > {rolling_mae:.3f}*{_DRIFT_MAE_FACTOR} (rolling)"

    # MAE-History aktualisieren
    if current_mae > 0:
        mae_history.append(current_mae)
        if len(mae_history) > 20:
            mae_history = mae_history[-20:]
        state["mae_history"] = mae_history

    if not should_retrain:
        return

    log.info("[optimizer/retraining_gate] Retraining ausgelöst: %s", reason)
    state["last_ml_retrain_ts"] = now
    state["last_retrain_reason"] = reason
    state["retrain_count"] = state.get("retrain_count", 0) + 1

    # Retraining in Hintergrund-Thread
    try:
        import threading
        from app.ml.lightgbm_model import unified_train
        t = threading.Thread(target=_run_retrain_safe, args=(unified_train, state), daemon=True)
        t.start()
        log.info("[optimizer/retraining_gate] Retraining-Thread gestartet")
    except Exception as exc:
        log.warning("[optimizer/retraining_gate] Konnte Retraining nicht starten: %s", exc)


def _run_retrain_safe(train_fn, state: dict) -> None:
    """Führt Retraining aus und loggt das Ergebnis."""
    try:
        t0 = time.time()
        train_fn()
        elapsed = round(time.time() - t0, 1)
        state["last_retrain_duration_s"] = elapsed
        log.info("[optimizer/retraining_gate] Retraining abgeschlossen in %.1fs", elapsed)
    except Exception as exc:
        log.warning("[optimizer/retraining_gate] Retraining fehlgeschlagen: %s", exc)


# ── Agent 5: Slot-Optimizer ────────────────────────────────────────────────

def _agent_slot_optimizer(state: dict, push_data: list) -> None:
    """Berechnet optimale Push-Fenster (Kategorie × Stunde) aus den letzten 30 Tagen.

    Ergebnis: state["optimal_slots"] mit OR-Potential pro Slot.
    Nutzt Decay-Gewichtung: neuere Daten zählen mehr.
    """
    now = time.time()
    cutoff = now - 30 * 86400

    recent = [
        p for p in push_data
        if p.get("or", 0) > 0
        and p.get("ts_num", 0) > cutoff
        and p.get("hour") is not None
    ]

    if len(recent) < 20:
        return

    # Decay-Gewichtung: exponentiell mit Halbwertszeit 7 Tage
    half_life = 7 * 86400
    for p in recent:
        age_s = now - p.get("ts_num", now)
        p["_decay_weight"] = math.exp(-age_s * math.log(2) / half_life)

    # Kategorie × Stunde Slots
    slots: dict[str, dict] = {}
    global_avg = sum(p["or"] for p in recent) / len(recent)

    cat_hour: dict[str, list] = defaultdict(list)
    for p in recent:
        cat = (p.get("cat") or "News").lower()
        h = int(p.get("hour", 12))
        key = f"{cat}_{h}"
        cat_hour[key].append((p["or"], p["_decay_weight"]))

    for key, pairs in cat_hour.items():
        if len(pairs) < _MIN_SAMPLES_HOUR:
            continue
        total_w = sum(w for _, w in pairs)
        weighted_or = sum(or_val * w for or_val, w in pairs) / total_w
        cat_part, hour_part = key.rsplit("_", 1)
        slots[key] = {
            "cat": cat_part,
            "hour": int(hour_part),
            "avg_or": round(weighted_or, 2),
            "n": len(pairs),
            "vs_global": round((weighted_or - global_avg) / global_avg * 100, 1),
        }

    # Top-Slots identifizieren
    top_slots = sorted(slots.values(), key=lambda s: -s["avg_or"])[:20]

    # Kategorie-spezifische Best-Hours
    cat_best_hours: dict[str, list] = defaultdict(list)
    for slot in top_slots:
        cat_best_hours[slot["cat"]].append((slot["hour"], slot["avg_or"]))

    # Cat×Hour-Stats für predict_or() aktualisieren
    cat_hour_stats: dict[str, dict] = {}
    for key, data in slots.items():
        cat_hour_stats[key] = {
            "avg": data["avg_or"],
            "n": data["n"],
        }
    state["cat_hour_stats"] = cat_hour_stats

    # Stunden-Durchschnitte für Heuristik aktualisieren
    hour_stats: dict[str, dict] = defaultdict(lambda: {"sum": 0.0, "n": 0, "w": 0.0})
    for p in recent:
        h = str(int(p.get("hour", 12)))
        hour_stats[h]["sum"] += p["or"] * p["_decay_weight"]
        hour_stats[h]["w"] += p["_decay_weight"]
        hour_stats[h]["n"] += 1
    state["hour_stats"] = {
        h: {"avg": round(v["sum"] / v["w"], 3), "n": v["n"]}
        for h, v in hour_stats.items() if v["w"] > 0
    }

    state["optimal_slots"] = {
        "top_slots": top_slots[:10],
        "cat_best_hours": {c: sorted(hours, key=lambda x: -x[1])[:3] for c, hours in cat_best_hours.items()},
        "global_avg_30d": round(global_avg, 2),
        "n_samples": len(recent),
        "updated_at": now,
    }

    log.debug(
        "[optimizer/slot_optimizer] %d Slots berechnet, global_avg_30d=%.2f%%",
        len(slots), global_avg,
    )
