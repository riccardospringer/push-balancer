"""app/ml/stats.py — History-Statistiken für GBRT- und LightGBM-Modelle.

Migriert aus dem frueheren Monolithen.

Enthält:
- _gbrt_build_history_stats(): Aggregat-Statistiken für Feature Engineering
"""
from __future__ import annotations

import math
import re
import time
import datetime
import logging
from collections import defaultdict

log = logging.getLogger("push-balancer")


def _gbrt_build_history_stats(pushes, target_ts=0):
    """Baut Aggregat-Statistiken fuer Feature Engineering.

    Args:
        pushes: Liste aller historischen Pushes
        target_ts: Timestamp des Ziel-Pushes (alles danach wird ignoriert fuer LOO)
    Returns:
        Dict mit vorberechneten Statistiken
    """
    now_ts = target_ts or int(time.time())
    cutoff_7d = now_ts - 7 * 86400
    cutoff_30d = now_ts - 30 * 86400

    valid = [p for p in pushes if 0 < p.get("or", 0) <= 20 and p.get("ts_num", 0) > 0 and p["ts_num"] < now_ts]
    if not valid:
        return {"global_avg": 4.77, "global_n": 0, "cat_stats": {}, "hour_stats": {},
                "cat_hour_stats": {}, "weekday_stats": {}, "recent_pushes": [],
                "entity_or": {}, "last_push_ts_by_cat": {},
                "push_timeline": [], "channel_stats": {}, "cat_momentum": {},
                "cat_weekday_stats": {}}

    all_or = [p["or"] for p in valid]
    global_avg = sum(all_or) / len(all_or)
    global_n = len(all_or)

    # Category stats (7d, 30d, all)
    cat_data = defaultdict(lambda: {"or_7d": [], "or_30d": [], "or_all": []})
    hour_data = defaultdict(lambda: {"or_7d": [], "or_30d": [], "or_all": []})
    cat_hour_data = defaultdict(list)
    weekday_data = defaultdict(list)
    cat_weekday_data = defaultdict(list)
    weekday_hour_data = defaultdict(list)
    last_push_ts_by_cat = {}
    channel_or_data = defaultdict(lambda: {"or_all": [], "n_all": 0})
    timeline_raw = []
    recipient_counts = []
    cat_recipient_data = defaultdict(list)

    for p in valid:
        ts = p["ts_num"]
        cat = (p.get("cat", "") or "news").lower().strip()
        h = p.get("hour", datetime.datetime.fromtimestamp(ts).hour)
        wd = datetime.datetime.fromtimestamp(ts).weekday()
        orv = p["or"]
        _recip = p.get("total_recipients", 0) or p.get("received", 0) or 0
        if _recip > 0:
            recipient_counts.append(_recip)
            cat_recipient_data[cat].append(_recip)

        cat_data[cat]["or_all"].append(orv)
        hour_data[h]["or_all"].append(orv)
        cat_hour_data[f"{cat}_{h}"].append(orv)
        weekday_data[wd].append(orv)
        cat_weekday_data[f"{cat}_{wd}"].append(orv)
        weekday_hour_data[f"{wd}_{h}"].append(orv)

        # Push timeline entry
        timeline_raw.append((ts, orv, cat))

        if ts >= cutoff_7d:
            cat_data[cat]["or_7d"].append(orv)
            hour_data[h]["or_7d"].append(orv)
        if ts >= cutoff_30d:
            cat_data[cat]["or_30d"].append(orv)
            hour_data[h]["or_30d"].append(orv)

        # Track last push per category
        if cat not in last_push_ts_by_cat or ts > last_push_ts_by_cat[cat]:
            last_push_ts_by_cat[cat] = ts

        # Channel stats
        channels = p.get("channels", [])
        if isinstance(channels, list):
            for ch in channels:
                ch_lower = str(ch).lower().strip()
                if ch_lower:
                    channel_or_data[ch_lower]["or_all"].append(orv)
                    channel_or_data[ch_lower]["n_all"] += 1

        # word_or_data / bigram_or_data entfernt — wurden nie zurückgegeben (totes RAM ~100MB)

    def _agg(lst):
        return {"avg": sum(lst) / len(lst), "n": len(lst)} if lst else {"avg": 0, "n": 0}

    cat_stats = {}
    for cat, d in cat_data.items():
        cat_stats[cat] = {
            "avg_7d": sum(d["or_7d"]) / len(d["or_7d"]) if d["or_7d"] else global_avg,
            "n_7d": len(d["or_7d"]),
            "avg_30d": sum(d["or_30d"]) / len(d["or_30d"]) if d["or_30d"] else global_avg,
            "n_30d": len(d["or_30d"]),
            "avg_all": sum(d["or_all"]) / len(d["or_all"]) if d["or_all"] else global_avg,
            "n_all": len(d["or_all"]),
        }

    hour_stats = {}
    for h, d in hour_data.items():
        hour_stats[h] = {
            "avg_7d": sum(d["or_7d"]) / len(d["or_7d"]) if d["or_7d"] else global_avg,
            "n_7d": len(d["or_7d"]),
            "avg_30d": sum(d["or_30d"]) / len(d["or_30d"]) if d["or_30d"] else global_avg,
            "n_30d": len(d["or_30d"]),
        }

    cat_hour_stats = {k: _agg(v) for k, v in cat_hour_data.items()}
    weekday_stats = {k: _agg(v) for k, v in weekday_data.items()}
    cat_weekday_stats = {k: _agg(v) for k, v in cat_weekday_data.items()}
    weekday_hour_stats = {k: _agg(v) for k, v in weekday_hour_data.items()}

    # Volatilitaet (Std) pro Kategorie und Hour (7d + 30d)
    def _std(lst):
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

    cat_volatility = {}
    for cat, d in cat_data.items():
        cat_volatility[cat] = {
            "std_7d": _std(d["or_7d"]),
            "std_30d": _std(d["or_30d"]),
        }
    hour_volatility = {}
    for h, d in hour_data.items():
        hour_volatility[h] = {
            "std_7d": _std(d["or_7d"]),
            "std_30d": _std(d["or_30d"]),
        }

    # Push timeline (sorted ascending by ts for bisect lookups)
    timeline_raw.sort(key=lambda x: x[0])
    push_timeline_ts = [t[0] for t in timeline_raw]  # timestamps only for bisect
    push_timeline = timeline_raw  # full (ts, or, cat) tuples

    # Channel stats (aggregated)
    channel_stats = {}
    for ch, d in channel_or_data.items():
        ors = d["or_all"]
        channel_stats[ch] = {
            "avg": sum(ors) / len(ors) if ors else global_avg,
            "n": len(ors),
        }

    # Category momentum (7d vs 30d trend)
    cat_momentum = {}
    for cat, cs in cat_stats.items():
        avg_7d = cs.get("avg_7d", global_avg)
        avg_30d = cs.get("avg_30d", global_avg)
        cat_momentum[cat] = {
            "momentum": (avg_7d - avg_30d) / max(avg_30d, 0.01),
            "ratio_7d_all": avg_7d / max(cs.get("avg_all", global_avg), 0.01),
        }

    # Recent pushes with pre-computed word sets (fuer Similarity)
    stops = {"der", "die", "das", "und", "von", "fuer", "mit", "auf", "den", "ist", "ein", "eine",
             "sich", "auch", "noch", "nur", "jetzt", "alle", "neue", "wird", "wurde", "nach", "ueber",
             "dass", "aber", "oder", "wenn", "dann", "mehr", "sein", "hat", "haben", "kann", "sind"}
    sorted_valid = sorted(valid, key=lambda x: x["ts_num"], reverse=True)
    recent_pushes = []
    for p in sorted_valid[:2000]:  # Letzte 2000 fuer Similarity
        words = set(re.findall(r'[a-zäöüß]{4,}', p.get("title", "").lower())) - stops
        recent_pushes.append({"words": words, "or": p["or"], "ts": p["ts_num"],
                              "title_raw": p.get("title", "")})

    # Entity OR mapping
    entity_or = defaultdict(list)
    entity_freq_7d = defaultdict(int)  # Wie oft taucht Entity in letzten 7d auf
    for p in sorted_valid[:3000]:
        entities = re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', p.get("title", ""))
        for ent in entities:
            ent_l = ent.lower()
            entity_or[ent_l].append(p["or"])
            if p["ts_num"] >= cutoff_7d:
                entity_freq_7d[ent_l] += 1

    # Recent Titles fuer Embedding-Vergleich (letzte 500)
    recent_titles = [(p.get("title", ""), p["or"]) for p in sorted_valid[:500]
                     if p.get("title")]

    return {
        "global_avg": global_avg,
        "global_n": global_n,
        "cat_stats": cat_stats,
        "hour_stats": hour_stats,
        "cat_hour_stats": cat_hour_stats,
        "weekday_stats": weekday_stats,
        "cat_weekday_stats": cat_weekday_stats,
        "weekday_hour_stats": weekday_hour_stats,
        "cat_volatility": cat_volatility,
        "hour_volatility": hour_volatility,
        "recent_pushes": recent_pushes,
        "entity_or": dict(entity_or),
        "entity_freq_7d": dict(entity_freq_7d),
        "last_push_ts_by_cat": last_push_ts_by_cat,
        "_recent_titles": recent_titles,
        "push_timeline": push_timeline,
        "push_timeline_ts": push_timeline_ts,
        "channel_stats": channel_stats,
        "cat_momentum": cat_momentum,
        "median_recipients": sorted(recipient_counts)[len(recipient_counts)//2] if recipient_counts else 0,
        "cat_median_recipients": {cat: sorted(vals)[len(vals)//2] for cat, vals in cat_recipient_data.items() if vals},
    }


# Public alias (tagesplan/builder.py importiert ohne Unterstrich)
gbrt_build_history_stats = _gbrt_build_history_stats
