"""app/tagesplan/builder.py — Tagesplan-Aufbau (ML-gestützt).

Migriert aus push-balancer-server.py (Zeilen 7995–8532, 8533–8755).
Enthält _ml_build_tagesplan, _ml_build_tagesplan_inner, _build_tagesplan_retro,
_tagesplan_background_refresh sowie _TP_CACHE_EMPTY und _auto_sug_worker.
"""
from __future__ import annotations

import datetime
import logging
import math
import re
import sqlite3
import threading
import time

log = logging.getLogger("push-balancer")

from app.state import (
    _tagesplan_cache,
    _tagesplan_cache_lock,
    _research_state,
    _ml_state,
    _ml_lock,
    _gbrt_model,
    _gbrt_feature_names,
    _gbrt_history_stats,
    _push_sync_lock,
    _push_sync_cache,
    _retro_cache,
    _retro_cache_lock,
)
from app.ml.features import gbrt_extract_features as _gbrt_extract_features
from app.ml.stats import gbrt_build_history_stats as _gbrt_build_history_stats
from app.ml.gbrt import gbrt_predict as _gbrt_predict, gbrt_train as _gbrt_train, gbrt_check_drift as _gbrt_check_drift, gbrt_online_update as _gbrt_online_update
from app.ml.lightgbm_model import ml_train_model as _ml_train_model, unified_train as _unified_train, train_stacking_model as _train_stacking_model, monitoring_tick as _monitoring_tick
from app.database import push_db_load_all, push_db_upsert, push_db_count, push_db_max_ts
from app.config import PUSH_API_BASE, PUSH_DB_PATH, RENDER_SYNC_URL, SYNC_SECRET, BILD_SITEMAP, IS_RENDER

# ── Convenience aliases (Monolith-kompatibel) ─────────────────────────────────
# Auf Render max. 5000 Rows laden (statt 15000) — spart RAM bei jedem Tagesplan-Build
_TP_MAX_ROWS = 5000 if IS_RENDER else 15000


def _push_db_load_all(min_ts: int = 0, max_days: int = 90) -> list:
    """Wrapper um push_db_load_all mit Render-spezifischem Zeilenlimit."""
    return push_db_load_all(min_ts=min_ts, max_days=max_days, max_rows=_TP_MAX_ROWS)
_push_db_upsert = push_db_upsert
_push_db_count = push_db_count
_push_db_max_ts = push_db_max_ts

# Regex-Patterns (identisch zum Monolith, Zeile 6517/6518)
_ML_BREAKING_KW = re.compile(r"(?i)\b(eilmeldung|breaking|exklusiv|liveticker|alarm|schock|sensation)\b")
_ML_EMOTION_KW = re.compile(r"(?i)\b(drama|tragödie|skandal|schock|horror|wahnsinn|irre|unfassbar|krass|hammer)\b")

# DB-Lock (Monolith: _push_db_lock in server.py)
from app.database import _push_db_lock


def _ml_build_stats(pushes):
    """Baut ML-Stats aus Push-Liste (Fallback wenn _gbrt_history_stats leer)."""
    from app.ml.stats import _gbrt_build_history_stats as _bhs
    return _bhs(pushes)


def _ml_extract_features(row, stats):
    """Extrahiert LightGBM-Features (delegiert an app.ml.features)."""
    try:
        from app.ml.features import _gbrt_extract_features
        return _gbrt_extract_features(row, stats)
    except Exception:
        return {}


# ── _TP_CACHE_EMPTY ──────────────────────────────────────────────────────────
_TP_CACHE_EMPTY = lambda: {"result": None, "hour": -1, "ts": 0, "building": False, "model_id": None}


# ── _tagesplan_background_refresh ────────────────────────────────────────────
def _tagesplan_background_refresh(mode="redaktion"):
    """Berechnet den Tagesplan im Hintergrund und aktualisiert den Cache."""
    try:
        now = datetime.datetime.now()
        _ml_build_tagesplan_inner(now, now.hour, mode=mode)
    except Exception as e:
        log.warning(f"[Tagesplan] Background-Refresh Fehler ({mode}): {e}")
        with _tagesplan_cache_lock:
            _tagesplan_cache[mode]["building"] = False


# ── _ml_build_tagesplan ───────────────────────────────────────────────────────
def _ml_build_tagesplan(background=False, mode="redaktion"):
    """Baut den Tagesplan: 18 Stunden-Slots (06-23) mit Empfehlungen.

    Args:
        background: True = aus Background-Worker (berechnet neu wenn stale).
                    False = aus API-Request (liefert IMMER sofort aus Cache).
        mode: "redaktion" (alle Kategorien) oder "sport" (nur Sport-Pushes).
    """
    now = datetime.datetime.now()
    current_hour = now.hour

    # Modell-ID: erkennt ob sich das ML- oder GBRT-Modell geaendert hat
    _current_model_id = id(_ml_state.get("model")) if _ml_state.get("model") else id(_gbrt_model)

    with _tagesplan_cache_lock:
        c = _tagesplan_cache[mode]
        age = time.time() - c["ts"]
        model_changed = c.get("model_id") != _current_model_id
        is_stale = c["hour"] != current_hour or age >= 300 or model_changed

        # ── API-Request: IMMER sofort antworten, nie blockieren ──
        if not background:
            if c["result"] and not is_stale:
                return c["result"]
            # Veraltet oder leer → Background-Refresh triggern
            if not c["building"]:
                c["building"] = True
                threading.Thread(target=_tagesplan_background_refresh, args=(mode,), daemon=True).start()
            # Sofort antworten: altes Ergebnis oder Loading-Skelett
            if c["result"]:
                return c["result"]
            return {"slots": [], "date": now.strftime("%d.%m.%Y"), "n_pushed_today": 0,
                    "golden_hour": None, "total_pushes_db": 0, "ml_trained": False,
                    "already_pushed_today": [], "must_have_hours": [], "ml_metrics": {},
                    "loading": True, "mode": mode}

        # ── Background-Aufruf: neu berechnen wenn stale ──
        if not is_stale and c["result"]:
            return c["result"]
        if c["building"]:
            return c["result"] or {"slots": [], "loading": True, "mode": mode}
        c["building"] = True

    try:
        return _ml_build_tagesplan_inner(now, current_hour, mode=mode)
    except Exception:
        with _tagesplan_cache_lock:
            _tagesplan_cache[mode]["building"] = False
        raise


# ── _ml_build_tagesplan_inner ─────────────────────────────────────────────────
def _ml_build_tagesplan_inner(now, current_hour, mode="redaktion"):
    """Innere Tagesplan-Berechnung (gecacht durch _ml_build_tagesplan)."""
    current_weekday = now.weekday()
    _WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    today_start = now.replace(hour=0, minute=0, second=0).timestamp()

    with _ml_lock:
        model = _ml_state.get("model")
        stats = _ml_state.get("stats")
        feature_names = _ml_state.get("feature_names")
        metrics = _ml_state.get("metrics", {})

    # ── GBRT-Fallback: wenn kein ML-Modell geladen, GBRT nutzen (immer von Disk verfuegbar) ──
    _using_gbrt_fallback = False
    if model is None and _gbrt_model is not None:
        model = _gbrt_model
        feature_names = _gbrt_feature_names
        _using_gbrt_fallback = True
        # GBRT-Metriken als Fallback fuer ml_metrics
        if not metrics and _gbrt_model is not None:
            _gbrt_tm = getattr(_gbrt_model, 'train_metrics', {})
            metrics = {
                "mae": _gbrt_tm.get("test_mae", _gbrt_tm.get("mae", 0)),
                "r2": _gbrt_tm.get("test_r2", _gbrt_tm.get("r2", 0)),
                "n_features": len(_gbrt_feature_names),
                "source": "GBRT-Fallback",
            }
        log.info("[Tagesplan] ML-Modell nicht verfuegbar, nutze GBRT als Fallback")

    push_data = _research_state.get("push_data", [])
    # Fallback: wenn research_state noch leer, Pushes aus DB laden
    if not push_data:
        try:
            _tp_db_pushes = _push_db_load_all()
            push_data = _tp_db_pushes
            log.info(f"[Tagesplan] push_data aus DB geladen: {len(push_data)} Pushes")
        except Exception as _pd_e:
            log.warning(f"[Tagesplan] DB-Fallback fuer push_data fehlgeschlagen: {_pd_e}")
    # Optimierung: GBRT-History-Stats wiederverwenden statt alles neu zu berechnen
    if not stats and _gbrt_history_stats:
        stats = _gbrt_history_stats
    # pushes nur laden wenn stats fehlt (Fallback) — SQL-Aggregation ersetzt den großen Loop
    pushes = []
    if not stats:
        pushes = _push_db_load_all()
        if not pushes:
            # Fallback: In-Memory Daten vom Research Worker (nach Server-Neustart)
            pushes = _research_state.get("push_data", [])
            if pushes:
                log.info(f"[Tagesplan] DB leer, nutze {len(pushes)} In-Memory Pushes als Fallback")
        stats = _ml_build_stats(pushes)

    # ── Sport-Filter (wird an alle SQL-Queries angehaengt) ──
    _SPORT_CATS_SQL = "('sport','fussball','bundesliga','formel1','formel-1','tennis','boxen','motorsport')"
    _cat_filter = f" AND LOWER(TRIM(cat)) IN {_SPORT_CATS_SQL}" if mode == "sport" else ""

    # ── Bereits heute gepushte Artikel (direkt aus DB, nie veralteter Cache) ──
    already_pushed_today = []
    pushed_ids = set()
    try:
        with _push_db_lock:
            _tp_today_conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
            _tp_today_conn.row_factory = sqlite3.Row
            _tp_today_rows = _tp_today_conn.execute(f"""
                SELECT hour, title, LOWER(TRIM(cat)) AS cat, or_val, is_eilmeldung, link, message_id
                FROM pushes
                WHERE ts_num >= ?
                  AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                  {_cat_filter}
                ORDER BY ts_num
            """, (int(today_start),)).fetchall()
            _tp_today_conn.close()
        for r in _tp_today_rows:
            title = r["title"] or ""
            mid = r["message_id"] or ""
            pushed_ids.add(mid)
            pushed_ids.add(title)
            already_pushed_today.append({
                "title": title, "cat": r["cat"] or "news",
                "or": round(r["or_val"] or 0, 2),
                "hour": r["hour"] if r["hour"] is not None else -1,
                "is_eilmeldung": bool(r["is_eilmeldung"]),
                "link": r["link"] or "",
            })
    except Exception as _tp_today_e:
        log.warning(f"[Tagesplan] DB-Fallback fuer heutige Pushes: {_tp_today_e}")
        # Fallback auf In-Memory
        for p in push_data:
            ts = p.get("ts_num", 0)
            if ts >= today_start:
                title = p.get("title") or p.get("headline") or ""
                cat = (p.get("cat") or "news").lower().strip()
                if mode == "sport" and cat not in ("sport", "fussball", "bundesliga"):
                    continue
                orv = p.get("or") or 0
                mid = p.get("message_id") or ""
                pushed_ids.add(mid)
                pushed_ids.add(title)
                already_pushed_today.append({
                    "title": title, "cat": cat, "or": round(orv, 2),
                    "hour": p.get("hour", -1), "is_eilmeldung": p.get("is_eilmeldung", False),
                    "link": p.get("link") or "",
                })

    # ── Historische Analyse pro Stunde (SQL-optimiert) ──
    from collections import defaultdict
    hour_cat_pushes = defaultdict(lambda: defaultdict(list))
    hour_title_patterns = defaultdict(lambda: {"question": 0, "exclamation": 0, "number": 0, "breaking": 0, "emotion": 0, "total": 0})
    hour_best_titles = defaultdict(list)
    _tp_total_db = 0

    try:
        with _push_db_lock:
            _tp_conn = sqlite3.connect(PUSH_DB_PATH)
            _tp_conn.row_factory = sqlite3.Row

            # 1a) Stunde×Kategorie Aggregation für gleichen Wochentag — komplett in SQL
            _sql_wd = (current_weekday + 1) % 7  # Python→SQLite Weekday-Mapping
            _hc_agg = _tp_conn.execute(f"""
                SELECT hour, LOWER(TRIM(cat)) as cat,
                       AVG(or_val) as avg_or, COUNT(*) as cnt
                FROM pushes
                WHERE or_val > 0 AND or_val <= 20 AND ts_num > 0
                  AND CAST(strftime('%w', ts_num, 'unixepoch') AS INTEGER) = ?
                  AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                  {_cat_filter}
                GROUP BY hour, LOWER(TRIM(cat))
            """, (_sql_wd,)).fetchall()

            for r in _hc_agg:
                h = r["hour"]
                cat = r["cat"] or "news"
                # top_cats_for_hour erwartet Einzelwerte — simuliere mit avg×count
                hour_cat_pushes[h][cat] = [r["avg_or"]] * r["cnt"]

            # 1b) Best-Titles: Top 2 pro Stunde via Window-Function (statt alle laden + Python-Sort)
            _bt_rows = _tp_conn.execute(f"""
                SELECT hour, or_val, title, LOWER(TRIM(cat)) as cat, link
                FROM (
                    SELECT hour, or_val, title, cat, link,
                           ROW_NUMBER() OVER (PARTITION BY hour ORDER BY or_val DESC) as rn
                    FROM pushes
                    WHERE or_val >= 4.0 AND or_val <= 20 AND ts_num > 0
                      AND received >= 10000
                      AND CAST(strftime('%w', ts_num, 'unixepoch') AS INTEGER) = ?
                      AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%'
                      {_cat_filter}
                ) WHERE rn <= 2
            """, (_sql_wd,)).fetchall()

            for r in _bt_rows:
                hour_best_titles[r["hour"]].append(
                    (r["or_val"], r["title"] or "", r["cat"] or "news", r["link"] or ""))

            # 2) Titel-Patterns: ?, !, Ziffern komplett in SQL aggregieren
            #    Breaking/Emotion-KW per LIKE (häufigste Keywords abdecken)
            _patt_rows = _tp_conn.execute(f"""
                SELECT hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN title LIKE '%%?%%' THEN 1 ELSE 0 END) as question,
                    SUM(CASE WHEN title LIKE '%%!%%' THEN 1 ELSE 0 END) as exclamation,
                    SUM(CASE WHEN title GLOB '*[0-9]*' THEN 1 ELSE 0 END) as has_number,
                    SUM(CASE WHEN LOWER(title) LIKE '%%eilmeldung%%'
                              OR LOWER(title) LIKE '%%breaking%%'
                              OR LOWER(title) LIKE '%%exklusiv%%'
                              OR LOWER(title) LIKE '%%liveticker%%'
                              OR LOWER(title) LIKE '%%alarm%%'
                              OR LOWER(title) LIKE '%%schock%%'
                              OR LOWER(title) LIKE '%%sensation%%'
                        THEN 1 ELSE 0 END) as breaking,
                    SUM(CASE WHEN LOWER(title) LIKE '%%drama%%'
                              OR LOWER(title) LIKE '%%skandal%%'
                              OR LOWER(title) LIKE '%%horror%%'
                              OR LOWER(title) LIKE '%%wahnsinn%%'
                              OR LOWER(title) LIKE '%%irre%%'
                              OR LOWER(title) LIKE '%%unfassbar%%'
                              OR LOWER(title) LIKE '%%krass%%'
                              OR LOWER(title) LIKE '%%hammer%%'
                        THEN 1 ELSE 0 END) as emotion
                FROM pushes
                WHERE or_val > 0 AND ts_num > 0
                  {_cat_filter}
                GROUP BY hour
            """).fetchall()

            for r in _patt_rows:
                h = r["hour"]
                hour_title_patterns[h] = {
                    "total": r["total"], "question": r["question"],
                    "exclamation": r["exclamation"], "number": r["has_number"],
                    "breaking": r["breaking"], "emotion": r["emotion"],
                }

            _tp_total_db = _tp_conn.execute(f"SELECT COUNT(*) FROM pushes WHERE 1=1 {_cat_filter}").fetchone()[0]
            _tp_conn.close()

    except Exception as _sql_e:
        log.warning(f"[Tagesplan] SQL-Aggregation Fehler, Fallback: {_sql_e}")
        for p in pushes:
            orv = p.get("or") or 0
            if orv <= 0:
                continue
            ts = p.get("ts_num", 0)
            if ts <= 0:
                continue
            dt = datetime.datetime.fromtimestamp(ts)
            h = dt.hour
            wd = dt.weekday()
            cat = (p.get("cat") or "news").lower().strip()
            title = p.get("title") or p.get("headline") or ""
            if wd == current_weekday:
                hour_cat_pushes[h][cat].append(orv)
            patt = hour_title_patterns[h]
            patt["total"] += 1
            if "?" in title: patt["question"] += 1
            if "!" in title: patt["exclamation"] += 1
            if re.search(r"\d", title): patt["number"] += 1
            if _ML_BREAKING_KW.search(title): patt["breaking"] += 1
            if _ML_EMOTION_KW.search(title): patt["emotion"] += 1
            link = p.get("link") or ""
            # Nur bild.de — keine SportBild/AutoBild
            if "sportbild." in link or "autobild." in link:
                continue
            recv = p.get("received") or 0
            if wd == current_weekday and 4.0 <= orv <= 20 and recv >= 10000:
                hour_best_titles[h].append((orv, title, cat, link))
        _tp_total_db = len(pushes)

    for h in hour_best_titles:
        hour_best_titles[h].sort(key=lambda x: -x[0])

    def top_cats_for_hour(h):
        cats = hour_cat_pushes.get(h, {})
        ranked = []
        for cat, vals in cats.items():
            avg_or = sum(vals) / len(vals) if vals else 0
            ranked.append({"cat": cat, "avg_or": round(avg_or, 2), "count": len(vals)})
        ranked.sort(key=lambda x: -x["avg_or"])
        return ranked[:3]

    def mood_reasoning(h, patt):
        total = max(patt.get("total", 1), 1)
        q_pct = patt.get("question", 0) / total * 100
        e_pct = patt.get("exclamation", 0) / total * 100
        em_pct = patt.get("emotion", 0) / total * 100
        b_pct = patt.get("breaking", 0) / total * 100
        n_pct = patt.get("number", 0) / total * 100
        best_mood, best_score, reasons = "Informativ", 0, []
        if em_pct > 15 and em_pct > best_score:
            best_mood, best_score = "Emotional", em_pct
            reasons.append(f"{em_pct:.0f}% nutzen emotionale Sprache")
        if b_pct > 10 and b_pct > best_score:
            best_mood, best_score = "Breaking", b_pct
            reasons.append(f"{b_pct:.0f}% Eilmeldungen")
        if q_pct > 20 and q_pct > best_score:
            best_mood, best_score = "Neugier", q_pct
            reasons.append(f"{q_pct:.0f}% Frage-Titel")
        if e_pct > 30 and e_pct > best_score:
            best_mood, best_score = "Dringend", e_pct
            reasons.append(f"{e_pct:.0f}% mit Ausrufezeichen")
        if n_pct > 40:
            reasons.append(f"{n_pct:.0f}% mit Zahlen")
        if not reasons:
            if 6 <= h <= 9: best_mood, reasons = "Informativ", ["Morgen: sachlich-informativ"]
            elif 12 <= h <= 14: best_mood, reasons = "Neugier", ["Mittag: Klick-Neugier"]
            elif 18 <= h <= 21: best_mood, reasons = "Emotional", ["Primetime: Emotion holt Top-OR"]
            elif h >= 22: best_mood, reasons = "Ergebnis", ["Spaetabend: Ergebnis-Pushes"]
            else: reasons = ["Sachliche Titel empfohlen"]
        return best_mood, reasons

    _SHAP_LABELS = {
        "hour_weekday_avg_or": "Wochentag-Timing", "hour_cat_avg_or": "Ressort-Timing",
        "hour_avg_or": "Tageszeit", "cat_avg_or": "Ressort", "weekday_avg_or": "Wochentag",
        "is_eilmeldung": "Eilmeldung", "word_count": "Wortanzahl", "title_len": "Zeichenzahl",
        "upper_ratio": "Grossbuchstaben", "has_exclamation": "Ausrufezeichen",
        "has_question": "Fragezeichen", "has_numbers": "Zahlen", "has_breaking_kw": "Breaking-KW",
        "has_emotion_kw": "Emotions-KW", "is_prime_time": "Primetime", "is_morning": "Morgen",
        "is_weekend": "Wochenende", "n_channels": "Kanalanzahl",
    }

    import numpy as np

    # ── Slot-Metadaten sammeln (ohne ML, schnell) ──
    slot_meta = []
    for h in range(6, 24):
        top_cats = top_cats_for_hour(h)
        primary_cat = top_cats[0]["cat"] if top_cats else "news"
        hist_or = top_cats[0]["avg_or"] if top_cats else 0
        n_hist = top_cats[0]["count"] if top_cats else 0
        # hour_avg: aus hour_stats (7d-Avg) ableiten, Fallback auf global_avg
        _h_stats = stats.get("hour_stats", {}).get(h, {})
        hour_avg = _h_stats.get("avg_7d", _h_stats.get("avg_30d", stats.get("global_avg", 0)))
        patt = hour_title_patterns.get(h, {})
        mood, mood_reasons = mood_reasoning(h, patt)
        best_titles = hour_best_titles.get(h, [])[:2]
        pushed_this_hour = [a for a in already_pushed_today if a.get("hour") == h]
        slot_meta.append({
            "h": h, "primary_cat": primary_cat, "top_cats": top_cats,
            "hist_or": hist_or, "n_hist": n_hist, "hour_avg": hour_avg,
            "mood": mood, "mood_reasons": mood_reasons,
            "best_titles": best_titles, "pushed_this_hour": pushed_this_hour,
        })

    # ── Batch-ML: alle 18 Slots auf einmal predicten + SHAP (1× Explainer statt 18×) ──
    ml_predictions = {}  # h → predicted_or
    ml_shap_dicts = {}   # h → shap_dict
    ml_shap_texts = {}   # h → shap_explanation

    if model is not None and feature_names:
        rows_by_h = {}
        X_rows = []
        h_order = []
        for sm in slot_meta:
            h = sm["h"]
            primary_cat = sm["primary_cat"]
            row = {"title": f"Typischer {primary_cat.title()}-Push", "cat": primary_cat, "hour": h,
                   "ts_num": int(now.timestamp()), "is_eilmeldung": primary_cat == "news" and h >= 18,
                   "channels": ["news"]}
            if _using_gbrt_fallback:
                feat = _gbrt_extract_features(row, stats, state=None, fast_mode=True)
            else:
                feat = _ml_extract_features(row, stats)
            X_rows.append([feat.get(k, 0.0) for k in feature_names])
            h_order.append(h)

        X_all = np.array(X_rows)
        try:
            preds = model.predict(X_all)
        except Exception:
            # Fallback: predict_one pro Slot
            preds = []
            for row in X_rows:
                try:
                    if hasattr(model, 'predict_one'):
                        preds.append(model.predict_one(row))
                    else:
                        preds.append(float(model.predict([row])[0]))
                except Exception:
                    preds.append(0.0)

        for i, h in enumerate(h_order):
            pred_val = float(preds[i])
            # ML-Modell nutzt Log-Transform, GBRT nicht
            if not _using_gbrt_fallback:
                pred_val = math.expm1(pred_val)
            ml_predictions[h] = round(max(0.01, min(20.0, pred_val)), 2)

        # SHAP: 1x Explainer, 1x Batch statt 18x Einzeln (~50s -> ~3s)
        try:
            _shap_model = model
            if _using_gbrt_fallback and hasattr(model, 'sklearn_model'):
                _shap_model = model.sklearn_model
            import shap as _shap
            explainer = _shap.TreeExplainer(_shap_model)
            sv_all = explainer.shap_values(X_all)
            for i, h in enumerate(h_order):
                shap_dict = {}
                for j, fn in enumerate(feature_names):
                    if abs(sv_all[i][j]) > 0.05:
                        shap_dict[fn] = round(float(sv_all[i][j]), 3)
                shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                ml_shap_dicts[h] = shap_dict
                pos = [(k, v) for k, v in shap_dict.items() if v > 0][:2]
                neg = [(k, v) for k, v in shap_dict.items() if v < 0][:1]
                parts = []
                for k, v in pos: parts.append(f"{_SHAP_LABELS.get(k, k)} +{v:.2f}pp")
                for k, v in neg: parts.append(f"{_SHAP_LABELS.get(k, k)} {v:.2f}pp")
                ml_shap_texts[h] = ", ".join(parts)
        except Exception:
            pass

    # ── Slots zusammenbauen ──
    slots = []
    for sm in slot_meta:
        h = sm["h"]
        predicted_or = ml_predictions.get(h)
        shap_dict = ml_shap_dicts.get(h, {})
        shap_explanation = ml_shap_texts.get(h, "")
        hist_or = sm["hist_or"]
        n_hist = sm["n_hist"]

        expected_or = predicted_or if predicted_or is not None else round(hist_or or sm["hour_avg"], 2)
        confidence = "hoch" if n_hist >= 30 else ("mittel" if n_hist >= 10 else "niedrig")
        color = "green" if expected_or >= 5.5 else ("yellow" if expected_or >= 4.0 else "gray")

        slots.append({
            "hour": h, "best_cat": sm["primary_cat"], "top_cats": sm["top_cats"],
            "expected_or": expected_or, "hist_or": round(hist_or, 2) if hist_or else None,
            "n_historical": n_hist, "confidence": confidence, "mood": sm["mood"],
            "mood_reasons": sm["mood_reasons"], "color": color,
            "is_now": h == current_hour, "is_past": h < current_hour,
            "shap": shap_dict, "shap_explanation": shap_explanation,
            "has_ml": predicted_or is not None,
            "best_historical": [{"title": t[1][:70], "cat": t[2], "or": round(t[0], 1), "link": t[3] if len(t) > 3 else ""} for t in sm["best_titles"]],
            "pushed_this_hour": sm["pushed_this_hour"],
        })

    # ── Must-Have Stunden markieren (Top 3 nach expected_or, nur Zukunft) ──
    future_slots = [s for s in slots if not s["is_past"]]
    future_by_or = sorted(future_slots, key=lambda s: -s["expected_or"])
    must_have_hours = set()
    for s in future_by_or[:3]:
        s["must_have"] = True
        must_have_hours.add(s["hour"])
    for s in slots:
        if "must_have" not in s:
            s["must_have"] = False

    # ── Sport-Kalender-Kontext (nur im Sport-Modus) ──
    if mode == "sport":
        _is_season = now.month >= 8 or now.month <= 5
        for s in slots:
            ctx = []
            h = s["hour"]
            if _is_season:
                if current_weekday == 5 and 15 <= h <= 18:
                    ctx.append("Bundesliga")
                elif current_weekday == 6 and 15 <= h <= 19:
                    ctx.append("Bundesliga")
                elif current_weekday == 4 and 20 <= h <= 22:
                    ctx.append("Bundesliga Freitag")
                if current_weekday in (1, 2) and 20 <= h <= 23:
                    ctx.append("Champions League")
            if now.month in (1, 7, 8):
                ctx.append("Transferfenster")
            s["sport_context"] = ctx

    best_slot = max(slots, key=lambda s: s["expected_or"]) if slots else None
    golden = future_by_or[0] if future_by_or else best_slot
    strong_slots = [s for s in future_slots if s["expected_or"] >= 5.0]

    # Ø OR heute berechnen
    _today_ors = [a["or"] for a in already_pushed_today if a.get("or", 0) > 0]
    _avg_or_today = round(sum(_today_ors) / len(_today_ors), 2) if _today_ors else None

    _tp_result = {
        "date": now.strftime(f"{_WOCHENTAGE[current_weekday]}, %d.%m.%Y"),
        "weekday": current_weekday, "weekday_name": _WOCHENTAGE[current_weekday],
        "current_hour": current_hour, "n_future": len(future_slots),
        "n_strong": len(strong_slots),
        "golden_hour": golden["hour"] if golden else None,
        "golden_cat": golden["best_cat"] if golden else None,
        "golden_or": golden["expected_or"] if golden else None,
        "best_hour": best_slot["hour"] if best_slot else None,
        "best_cat": best_slot["best_cat"] if best_slot else None,
        "best_or": best_slot["expected_or"] if best_slot else None,
        "ml_metrics": metrics, "ml_trained": model is not None,
        "avg_or_today": _avg_or_today,
        "total_pushes_db": _tp_total_db or len(pushes), "slots": slots,
        "already_pushed_today": already_pushed_today,
        "n_pushed_today": len(already_pushed_today),
        "must_have_hours": sorted(must_have_hours),
    }

    _tp_result["mode"] = mode

    # Cache befuellen
    _cache_model_id = id(_ml_state.get("model")) if _ml_state.get("model") else id(_gbrt_model)
    with _tagesplan_cache_lock:
        _tagesplan_cache[mode]["result"] = _tp_result
        _tagesplan_cache[mode]["hour"] = current_hour
        _tagesplan_cache[mode]["ts"] = time.time()
        _tagesplan_cache[mode]["building"] = False
        _tagesplan_cache[mode]["model_id"] = _cache_model_id

    return _tp_result


# ── _build_tagesplan_retro ────────────────────────────────────────────────────
def _build_tagesplan_retro():
    """Baut die 7-Tage-Retrospektive: Was wurde gepusht, was hat das ML prognostiziert."""
    now = datetime.datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Cache prüfen (1h TTL, invalidiert bei Tageswechsel)
    with _retro_cache_lock:
        if (_retro_cache["result"] is not None
                and _retro_cache["day"] == today_str
                and time.time() - _retro_cache["ts"] < 3600):
            return _retro_cache["result"]

    # Zeitgrenzen: 7 Tage zurück (Mitternacht) bis heute Mitternacht
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int((today_midnight - datetime.timedelta(days=7)).timestamp())
    end_ts = int(today_midnight.timestamp())

    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    try:
        conn = sqlite3.connect(PUSH_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                DATE(p.ts_num, 'unixepoch', 'localtime') AS day,
                p.hour, p.message_id, p.title,
                LOWER(TRIM(p.cat)) AS cat,
                p.or_val AS actual_or,
                p.is_eilmeldung, p.link, p.received,
                pl.predicted_or, pl.basis_method
            FROM pushes p
            LEFT JOIN prediction_log pl ON p.message_id = pl.push_id
            WHERE p.ts_num >= ?
              AND p.ts_num < ?
              AND p.link NOT LIKE '%sportbild.%'
              AND p.link NOT LIKE '%autobild.%'
            ORDER BY p.ts_num
        """, (start_ts, end_ts)).fetchall()

        # Systemempfehlungen laden (was haette das System gepusht?)
        start_date = (today_midnight - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (today_midnight - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        sug_rows = conn.execute("""
            SELECT date_iso, slot_hour, suggestion_num, article_title, article_link,
                   article_category, article_score, expected_or, best_cat
            FROM tagesplan_suggestions
            WHERE date_iso >= ? AND date_iso <= ?
            ORDER BY date_iso, slot_hour, suggestion_num
        """, (start_date, end_date)).fetchall()
        conn.close()
    except Exception as e:
        log.warning(f"[Retro] DB-Fehler: {e}")
        return {"days": [], "summary": {}}

    # Suggestions nach Tag+Stunde gruppieren
    suggestions_by_day = {}
    for sr in sug_rows:
        day = sr["date_iso"]
        if day not in suggestions_by_day:
            suggestions_by_day[day] = {}
        h_str = str(sr["slot_hour"])
        if h_str not in suggestions_by_day[day]:
            suggestions_by_day[day][h_str] = []
        suggestions_by_day[day][h_str].append({
            "title": sr["article_title"] or "",
            "link": sr["article_link"] or "",
            "cat": sr["article_category"] or "",
            "score": round(sr["article_score"] or 0, 1),
            "expected_or": round(sr["expected_or"] or 0, 2),
        })

    # Nach Tagen gruppieren
    days_dict = {}
    for row in rows:
        day = row["day"]
        if day not in days_dict:
            days_dict[day] = []
        days_dict[day].append(dict(row))

    days_list = []
    total_pushes = 0
    all_ors = []
    all_deltas = []
    cat_stats = {}

    for day_str in sorted(days_dict.keys()):
        pushes = days_dict[day_str]
        dt = datetime.datetime.strptime(day_str, "%Y-%m-%d")
        weekday = weekday_names[dt.weekday()]
        date_display = dt.strftime("%d.%m.%Y")

        n_pushed = len(pushes)
        ors = [p["actual_or"] for p in pushes if p["actual_or"] and p["actual_or"] > 0]
        avg_or = round(sum(ors) / len(ors), 2) if ors else 0

        # Best/Worst Push
        best_push = max(pushes, key=lambda p: p["actual_or"] or 0) if pushes else None
        worst_push = min(pushes, key=lambda p: p["actual_or"] if p["actual_or"] and p["actual_or"] > 0 else 999) if pushes else None

        # Prognose-Analyse
        n_predicted = 0
        deltas = []
        for p in pushes:
            if p["predicted_or"] is not None and p["actual_or"] and p["actual_or"] > 0:
                n_predicted += 1
                delta = abs(p["predicted_or"] - p["actual_or"])
                deltas.append(delta)

        mae = round(sum(deltas) / len(deltas), 2) if deltas else None
        matches = sum(1 for d in deltas if d <= 1.0)
        match_quote = f"{matches}/{n_predicted}" if n_predicted > 0 else "0/0"

        # Stunden-Grid (nur Stunden mit Pushes)
        hours_dict = {}
        for p in pushes:
            h = p["hour"]
            if h < 0:
                continue
            h_str = str(h)
            if h_str not in hours_dict:
                hours_dict[h_str] = {"pushes": []}
            pred_or = p["predicted_or"]
            act_or = p["actual_or"] or 0
            delta_val = round(pred_or - act_or, 2) if pred_or is not None and act_or > 0 else None
            hours_dict[h_str]["pushes"].append({
                "title": p["title"] or "",
                "cat": p["cat"] or "news",
                "actual_or": round(act_or, 2),
                "predicted_or": round(pred_or, 2) if pred_or is not None else None,
                "delta": delta_val,
                "is_eilmeldung": bool(p["is_eilmeldung"]),
                "link": p["link"] or "",
            })

        # Kategorie-Statistiken
        for p in pushes:
            cat = p["cat"] or "news"
            if cat not in cat_stats:
                cat_stats[cat] = {"n": 0, "ors": []}
            cat_stats[cat]["n"] += 1
            if p["actual_or"] and p["actual_or"] > 0:
                cat_stats[cat]["ors"].append(p["actual_or"])

        total_pushes += n_pushed
        all_ors.extend(ors)
        all_deltas.extend(deltas)

        day_obj = {
            "date": date_display,
            "date_iso": day_str,
            "weekday": weekday,
            "n_pushed": n_pushed,
            "avg_or": avg_or,
            "best_push": {
                "title": best_push["title"] or "", "or": round(best_push["actual_or"] or 0, 2),
                "cat": best_push["cat"] or "news", "hour": best_push["hour"],
                "link": best_push["link"] or "",
            } if best_push else None,
            "worst_push": {
                "title": worst_push["title"] or "", "or": round(worst_push["actual_or"] or 0, 2),
                "cat": worst_push["cat"] or "news", "hour": worst_push["hour"],
            } if worst_push else None,
            "n_predicted": n_predicted,
            "prediction_mae": mae,
            "match_quote": match_quote,
            "hours": hours_dict,
            "suggestions": suggestions_by_day.get(day_str, {}),
        }
        days_list.append(day_obj)

    # Summary
    avg_or_7d = round(sum(all_ors) / len(all_ors), 2) if all_ors else 0
    mae_7d = round(sum(all_deltas) / len(all_deltas), 2) if all_deltas else None

    best_day = max(days_list, key=lambda d: d["avg_or"]) if days_list else None
    worst_day = min(days_list, key=lambda d: d["avg_or"] if d["avg_or"] > 0 else 999) if days_list else None

    # Top-Stunde über alle 7 Tage
    hour_ors = {}
    for day in days_list:
        for h_str, h_data in day["hours"].items():
            for p in h_data["pushes"]:
                h_int = int(h_str)
                if h_int not in hour_ors:
                    hour_ors[h_int] = []
                if p["actual_or"] > 0:
                    hour_ors[h_int].append(p["actual_or"])
    top_hour = None
    top_hour_avg = 0
    for h, ors_h in hour_ors.items():
        if ors_h:
            avg_h = sum(ors_h) / len(ors_h)
            if avg_h > top_hour_avg:
                top_hour = h
                top_hour_avg = avg_h

    cat_breakdown = {}
    for cat, data in cat_stats.items():
        cat_breakdown[cat] = {
            "n": data["n"],
            "avg_or": round(sum(data["ors"]) / len(data["ors"]), 2) if data["ors"] else 0,
        }

    result = {
        "days": days_list,
        "summary": {
            "total_pushes": total_pushes,
            "avg_or_7d": avg_or_7d,
            "best_day": {"date": best_day["date"][:5] if best_day else "", "weekday": best_day["weekday"] if best_day else "", "avg_or": best_day["avg_or"] if best_day else 0} if best_day else None,
            "worst_day": {"date": worst_day["date"][:5] if worst_day else "", "weekday": worst_day["weekday"] if worst_day else "", "avg_or": worst_day["avg_or"] if worst_day else 0} if worst_day else None,
            "prediction_mae_7d": mae_7d,
            "top_hour": top_hour,
            "top_hour_avg_or": round(top_hour_avg, 2) if top_hour else 0,
            "category_breakdown": cat_breakdown,
        },
    }

    with _retro_cache_lock:
        _retro_cache["result"] = result
        _retro_cache["ts"] = time.time()
        _retro_cache["day"] = today_str

    return result


# ── _auto_save_suggestions ───────────────────────────────────────────────────
_auto_sug_last_hour: int = -1

_ASG_CAT_SCORES: dict = {
    "politik": 22,
    "unterhaltung": 20,
    "panorama": 21,
    "sport": 18,
    "wirtschaft": 19,
    "ratgeber": 15,
    "regional": 14,
    "lifestyle": 13,
    "reise": 12,
    "auto": 11,
}
_ASG_HIGH_KW = {"TRUMP", "PUTIN", "UKRAINE", "KRIEG"}
_ASG_URG_KW = {"EILMELDUNG", "TERROR", "TOD"}

# Slots an denen Vorschläge erzeugt werden (ganztägig 06–23 Uhr)
_ASG_PUSH_SLOTS = list(range(6, 24))


def _auto_save_suggestions() -> None:
    """Scoret aktuelle BILD-Sitemap-Artikel und speichert Top-3 pro Slot in DB."""
    global _auto_sug_last_hour

    now = datetime.datetime.now()
    if now.hour < 6 or now.hour > 23:
        return

    # Duplikat-Guard: pro Stunde nur einmal
    if now.hour == _auto_sug_last_hour:
        return
    _auto_sug_last_hour = now.hour

    try:
        from app.routers.feed import _fetch_url
        from app.config import BILD_SITEMAP
        from app.database import save_tagesplan_suggestions

        xml_bytes = _fetch_url(BILD_SITEMAP)
        if not xml_bytes:
            log.warning("[AutoSug] BILD Sitemap nicht erreichbar")
            return

        # Sitemap XML parsen (news:news Format)
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))
        except ET.ParseError as pe:
            log.warning("[AutoSug] Sitemap XML-Fehler: %s", pe)
            return

        ns_sitemap = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        ns_news = {"news": "http://www.google.com/schemas/sitemap-news/0.9"}

        articles: list[dict] = []
        for url_el in root.findall("sm:url", ns_sitemap):
            loc = (url_el.findtext("sm:loc", "", ns_sitemap) or "").strip()
            news_el = url_el.find("news:news", ns_news)
            if news_el is None:
                continue
            title = (news_el.findtext("news:title", "", ns_news) or "").strip()
            pub_date = (news_el.findtext("news:publication_date", "", ns_news) or "").strip()
            if not title or not loc:
                continue

            # Kategorie aus URL ableiten
            cat = "news"
            loc_lower = loc.lower()
            for cat_key in _ASG_CAT_SCORES:
                if f"/{cat_key}/" in loc_lower or loc_lower.endswith(f"/{cat_key}"):
                    cat = cat_key
                    break

            # Heuristik-Score berechnen
            score = _ASG_CAT_SCORES.get(cat, 10)
            title_upper = title.upper()
            for kw in _ASG_HIGH_KW:
                if kw in title_upper:
                    score += 4
            for kw in _ASG_URG_KW:
                if kw in title_upper:
                    score += 5

            articles.append({
                "title": title,
                "link": loc,
                "cat": cat,
                "score": float(score),
                "expected_or": 0.0,
                "best_cat": cat,
                "pub_date": pub_date,
            })

        if not articles:
            log.info("[AutoSug] Keine Artikel in Sitemap gefunden")
            return

        date_iso = now.strftime("%Y-%m-%d")
        articles_by_score = sorted(articles, key=lambda a: a["score"], reverse=True)

        saved_slots = 0
        for slot_hour in _ASG_PUSH_SLOTS:
            top3 = articles_by_score[:3]
            if top3:
                save_tagesplan_suggestions(date_iso, slot_hour, top3)
                saved_slots += 1

        log.info(
            "[AutoSug] %d Artikel gescort, %d Slots gespeichert (Stunde %d)",
            len(articles), saved_slots, now.hour,
        )

    except Exception as exc:
        log.warning("[AutoSug] _auto_save_suggestions Fehler: %s", exc)


# ── _auto_sug_worker ──────────────────────────────────────────────────────────
def _auto_sug_worker():
    import time as _asw_t
    _asw_t.sleep(30)  # Warte bis ML-Modelle geladen
    log.info("[AutoSug] Worker gestartet (prüft alle 10 Min)")
    while True:
        try:
            _auto_save_suggestions()
        except Exception as _asw_e:
            log.warning(f"[AutoSug] Worker-Fehler: {_asw_e}")
        _asw_t.sleep(600)  # Alle 10 Minuten prüfen (Duplikat-Guard in Funktion)


def start_auto_sug_worker():
    """Startet den Auto-Suggestion Worker als Daemon-Thread."""
    threading.Thread(target=_auto_sug_worker, daemon=True).start()
    print("  [AutoSug] Worker gestartet (stündlich)")


# ── Public API ────────────────────────────────────────────────────────────────
def build_tagesplan(background: bool = False, mode: str = "redaktion") -> dict:
    """Öffentliche API: ML-Tagesplan bauen."""
    return _ml_build_tagesplan(background=background, mode=mode)


def build_tagesplan_retro() -> dict:
    """Öffentliche API: Tagesplan-Retrospektive bauen."""
    return _build_tagesplan_retro()
