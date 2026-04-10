"""app/ml/heuristic.py — Vollständige 9-Methoden-Heuristik für predictOR.

Migriert aus dem frueheren Monolithen (_server_predict_or).

Enthält:
- DEFAULT_TUNING_PARAMS
- _context_topic_match()
- compute_topic_saturation_penalty()
- predict_heuristic() — M1-M9 + Post-Fusion-Korrektoren + ML-Blend
"""
from __future__ import annotations

import datetime
import logging
import math
import re
from collections import defaultdict

log = logging.getLogger("push-balancer")

# ── Stop-Words (identisch mit features.py _TOPIC_STOPS) ──────────────────────
_HEURISTIC_STOPS = {
    "der", "die", "das", "und", "von", "fuer", "für", "mit", "auf", "den", "ist",
    "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
    "wird", "wurde", "nach", "ueber", "über", "dass", "oder", "aber", "wenn",
    "weil", "nicht", "hat", "haben", "sind", "sein", "kann", "aus", "wie",
    "vor", "bei", "zum", "zur", "vom", "dem", "des",
}

# ── Tuning-Parameter (aus dem frueheren Monolithen) ────────────────────────
DEFAULT_TUNING_PARAMS: dict = {
    "m1_conf_cap": 1.20,
    "m2_conf_cap": 1.10,
    "m3_conf_cap": 1.10,
    "m4_conf_cap": 0.90,
    "m5_conf_cap": 0.85,
    "m6_phd_cap":  0.75,
    "m7_conf_cap": 1.00,
    "fusion_prior_weight": 0.08,
    "cat_damp": 0.55,
    "timing_damp": 0.50,
    "framing_damp": 0.40,
    "length_damp": 0.30,
    "ling_damp": 0.30,
    "phd_bayes_damp": 0.20,
    "phd_interaction_damp": 0.25,
    "phd_entity_ctx_damp": 0.15,
    "phd_recency_damp": 0.20,
    "phd_bias_correction_damp": 0.55,
    "phd_fatigue_damp": 0.25,
    "phd_breaking_boost": 1.15,
}

# ── Sport-Entities für Korrektor 4 ────────────────────────────────────────────
_SPORT_HIGH_ENTITIES = frozenset({
    "bayern", "dortmund", "bvb", "kimmich", "musiala", "sane", "mueller",
    "real madrid", "barcelona", "champions league", "champions",
    "transfer", "wechsel", "abloesung", "vertragsverlaengerung",
})
_SPORT_BOOST_ENTITIES = frozenset({
    "bundesliga", "dfb", "nationalmannschaft", "nagelsmann", "tuchel", "flick",
    "lewandowski", "haaland", "bellingham", "mbappe", "mbappé",
})


def _context_topic_match(push_title: str, trends: list) -> float:
    """Prüft ob ein Push-Titel ein Trending-Topic trifft. Gibt 0-1 Score zurück."""
    if not trends or not push_title:
        return 0.0
    title_lower = push_title.lower()
    title_words = set(title_lower.split())
    matches = 0
    for trend in trends:
        trend_words = set(trend.split())
        if trend in title_lower:
            matches += 2
        elif len(title_words & trend_words) >= 1:
            matches += 1
    return min(1.0, matches / 3.0)


def compute_topic_saturation_penalty(
    push: dict,
    push_data: list,
    state: dict | None = None,
) -> dict:
    """Berechnet Themen-Sättigungs-Faktor für einen Push.

    Analysiert ob zum gleichen Thema bereits Pushes gesendet wurden.

    Returns:
        Dict mit penalty, topic_push_count_6h, topic_push_count_24h,
        highest_jaccard, hours_since_last, or_decay, reason.
    """
    result = {
        "penalty": 1.0,
        "topic_push_count_6h": 0,
        "topic_push_count_24h": 0,
        "highest_jaccard": 0.0,
        "hours_since_last": 999.0,
        "or_decay": 0.0,
        "reason": "",
    }

    try:
        import time as _time
        title = push.get("title", "") or ""
        title_lower = title.lower()
        push_ts = push.get("ts_num", 0) or int(_time.time())

        push_words = set(re.findall(r"[A-Za-zäöüÄÖÜß]{4,}", title_lower)) - _HEURISTIC_STOPS
        if not push_words or len(push_words) < 2:
            return result

        cutoff_6h = push_ts - 6 * 3600
        cutoff_24h = push_ts - 24 * 3600

        if push_ts > 0:
            candidates = [
                p for p in push_data
                if p.get("ts_num", 0) > 0
                and p["ts_num"] < push_ts
                and p["ts_num"] > cutoff_24h
                and (p.get("or", 0) or 0) > 0
            ]
        else:
            candidates = [
                p for p in push_data
                if (p.get("or", 0) or 0) > 0 and p.get("ts_num", 0) > cutoff_24h
            ]

        if not candidates:
            return result

        similar_pushes = []
        highest_jaccard = 0.0
        last_similar_ts = 0

        for p in candidates:
            p_title = (p.get("title", "") or "").lower()
            p_words = set(re.findall(r"[A-Za-zäöüÄÖÜß]{4,}", p_title)) - _HEURISTIC_STOPS
            if not p_words:
                continue
            intersection = push_words & p_words
            union = push_words | p_words
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            if jaccard > highest_jaccard:
                highest_jaccard = jaccard
            if jaccard > 0.25:
                similar_pushes.append({
                    "ts": p["ts_num"],
                    "or": p.get("or", 0),
                    "jaccard": jaccard,
                })
                if p["ts_num"] > last_similar_ts:
                    last_similar_ts = p["ts_num"]

        result["highest_jaccard"] = round(highest_jaccard, 3)

        count_6h = sum(1 for s in similar_pushes if s["ts"] > cutoff_6h)
        count_24h = len(similar_pushes)
        result["topic_push_count_6h"] = count_6h
        result["topic_push_count_24h"] = count_24h

        if last_similar_ts > 0 and push_ts > 0:
            hours_since = (push_ts - last_similar_ts) / 3600
            result["hours_since_last"] = round(hours_since, 2)
        else:
            hours_since = 999.0

        # OR-Trend aus ähnlichen Pushes
        or_decay = 0.0
        if len(similar_pushes) >= 3:
            recent_ors = sorted(similar_pushes, key=lambda s: s["ts"])
            n = len(recent_ors)
            first_half = recent_ors[:n // 2]
            second_half = recent_ors[n // 2:]
            avg_first = sum(s["or"] for s in first_half) / len(first_half) if first_half else 0
            avg_second = sum(s["or"] for s in second_half) / len(second_half) if second_half else 0
            if avg_first > 0:
                or_decay = (avg_second - avg_first) / avg_first
                result["or_decay"] = round(or_decay, 3)

        # Penalty berechnen
        penalty = 1.0
        reason_parts = []

        if count_6h >= 4:
            penalty = min(penalty, 0.60)
            reason_parts.append(f"{count_6h} gleiche Themen in 6h")
        elif count_6h >= 2:
            penalty = min(penalty, 0.80)
            reason_parts.append(f"{count_6h} gleiche Themen in 6h")
        elif count_6h >= 1 and hours_since < 2:
            penalty = min(penalty, 0.90)
            reason_parts.append(f"Letzter ähnlicher Push vor {hours_since:.1f}h")

        if highest_jaccard > 0.6 and hours_since < 4:
            penalty = min(penalty, 0.75)
            reason_parts.append(f"Hohe Jaccard-Ähnlichkeit ({highest_jaccard:.2f})")

        if or_decay < -0.3 and count_24h >= 2:
            extra_penalty = max(0.85, 1.0 + or_decay * 0.3)
            penalty = min(penalty, extra_penalty)
            reason_parts.append(f"OR-Trend sinkend ({or_decay:.1%})")

        result["penalty"] = round(max(0.50, penalty), 3)
        result["reason"] = ", ".join(reason_parts) if reason_parts else ""

    except Exception as exc:
        log.debug("[heuristic] compute_topic_saturation_penalty Fehler: %s", exc)

    return result


def _apply_residual_correction_local(
    predicted_or: float,
    cat: str,
    hour: int,
    residual_corrector: dict | None,
) -> tuple[float, float]:
    """Wendet Residual-Korrektur an. Gibt (corrected_or, correction) zurück."""
    if not residual_corrector or residual_corrector.get("n_samples", 0) < 10:
        return predicted_or, 0.0

    # Tageszeit-Gruppe bestimmen
    hour_groups = {
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 23),
    }
    hg = "night"
    for name, rng in hour_groups.items():
        if hour in rng:
            hg = name
            break

    gb = residual_corrector.get("global_bias", 0.0)
    cb = residual_corrector.get("cat_bias", {}).get(cat, gb)
    hb = residual_corrector.get("hourgroup_bias", {}).get(hg, gb)

    raw = 0.5 * gb + 0.3 * cb + 0.2 * hb
    if abs(raw) < 0.2:
        return predicted_or, 0.0
    correction = raw * 0.5
    correction = max(-2.0, min(2.0, correction))
    corrected = max(0.5, min(30.0, predicted_or - correction))
    return corrected, round(correction, 3)


def predict_heuristic(
    push: dict,
    push_data: list,
    state: dict | None,
    residual_corrector: dict | None = None,
    tuning_params: dict | None = None,
) -> dict | None:
    """Vollständige 9-Methoden Heuristik für Opening-Rate-Prediction.

    Methoden M1-M9 + Post-Fusion-Korrektoren.
    Benötigt mindestens 10 historische Pushes in push_data.

    Args:
        push: Push-Dict (title, cat, hour, ts_num, is_eilmeldung).
        push_data: Historische Push-Liste (mind. 10 mit OR > 0).
        state: Research-State (research_modifiers, external_context, etc.).
        residual_corrector: Residual-Corrector-State aus worker.py.
        tuning_params: Tuning-Parameter (oder DEFAULT_TUNING_PARAMS).

    Returns:
        Dict mit predicted_or, basis_method, confidence, q10, q90, methods
        oder None wenn push_data zu klein.
    """

    state = state or {}
    params = tuning_params or state.get("tuning_params") or {}
    # Defaults einsetzen wo nötig
    for k, v in DEFAULT_TUNING_PARAMS.items():
        params.setdefault(k, v)

    # Temporal Causal Filter: nur Pushes vor dem aktuellen
    push_ts = push.get("ts_num", 0)
    push_title = push.get("title", "")
    if push_ts > 0:
        valid = [
            p for p in push_data
            if 0 < p.get("or", 0) <= 100
            and p.get("ts_num", 0) > 0
            and p["ts_num"] < push_ts
        ]
    else:
        valid = [
            p for p in push_data
            if 0 < p.get("or", 0) <= 100
            and not (p.get("ts_num", -999) == push_ts and p.get("title", "") == push_title)
        ]

    if len(valid) < 10:
        return None

    global_avg = sum(p["or"] for p in valid) / len(valid)
    push_title_lower = push_title.lower()
    push_cat = push.get("cat", "News")
    push_hour = push.get("hour", 12)
    push_weekday = (
        datetime.datetime.fromtimestamp(push_ts).weekday()
        if push_ts > 0
        else datetime.datetime.now().weekday()
    )

    methods: dict = {}

    # ── Breaking-Signale ──────────────────────────────────────────────────────
    push_title_raw = push.get("title", "")
    breaking_signals = 0
    if "+++" in push_title_raw:
        breaking_signals += 2
    elif "++" in push_title_raw:
        breaking_signals += 1
    if "|" in push_title_raw:
        breaking_signals += 1
    if "EXKLUSIV" in push_title_raw.upper() or "BREAKING" in push_title_raw.upper():
        breaking_signals += 2
    if push.get("is_eilmeldung"):
        breaking_signals += 2
    if push_title_raw.strip().endswith("!"):
        breaking_signals += 1
    is_breaking_style = breaking_signals >= 3

    # ── Emotion-Intensität ────────────────────────────────────────────────────
    intensity_words = {
        "angst": {"tot", "tod", "sterben", "gestorben", "stirbt", "lebensgefahr", "leiche", "mord", "tote", "opfer"},
        "katastrophe": {"erdbeben", "tsunami", "explosion", "brand", "feuer", "absturz", "crash",
                        "ueberschwemmung", "hochwasser", "sturm", "orkan"},
        "sensation": {"sensation", "historisch", "erstmals", "rekord", "unfassbar", "unglaublich",
                      "wahnsinn", "hammer", "mega", "schock", "krass"},
        "bedrohung": {"warnung", "alarm", "gefahr", "notfall", "panik", "terror", "angriff",
                      "anschlag", "krieg", "drohung", "evakuierung"},
        "prominenz": {"kanzler", "praesident", "papst", "koenig", "merkel", "scholz", "trump", "putin"},
        "empoerung": {"skandal", "verrat", "betrug", "korrupt", "dreist", "frechheit"},
    }
    intensity_score = 0.0
    matched_categories: set = set()
    for cat_name, words in intensity_words.items():
        matches_n = sum(1 for w in words if w in push_title_lower)
        if matches_n > 0:
            matched_categories.add(cat_name)
            intensity_score += matches_n * 0.15
    if len(matched_categories) >= 2:
        intensity_score *= 1.0 + (len(matched_categories) - 1) * 0.3
    if is_breaking_style:
        intensity_score += 0.3
    intensity_score = min(1.0, intensity_score)
    methods["breaking_signals"] = breaking_signals
    methods["is_breaking_style"] = is_breaking_style

    # ── M1: Similarity — Keyword-Jaccard + Entity-Overlap ─────────────────────
    stops = _HEURISTIC_STOPS
    push_words = set(re.findall(r"[A-Za-zäöüÄÖÜßaeoeue]{4,}", push_title_lower)) - stops
    max_jaccard = 0.0
    sim_scores = []
    if push_words:
        for p in valid:
            p_words = (
                set(re.findall(r"[A-Za-zäöüÄÖÜßaeoeue]{4,}", p.get("title", "").lower()))
                - stops
            )
            if p_words:
                jaccard = len(push_words & p_words) / len(push_words | p_words)
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                if jaccard > 0.1:
                    sim_scores.append((jaccard, p["or"], p.get("title", "")))
        if sim_scores:
            sim_scores.sort(key=lambda x: -x[0])
            top_n = sim_scores[:min(10, len(sim_scores))]
            weights = [s[0] for s in top_n]
            w_sum = sum(weights)
            m1 = sum(s[0] * s[1] for s in top_n) / w_sum if w_sum > 0 else global_avg
            conf_m1 = min(params["m1_conf_cap"], len(top_n) / 10)
        else:
            m1 = global_avg
            conf_m1 = 0.1
    else:
        m1 = global_avg
        conf_m1 = 0.1

    novelty_boost = 1.0
    if max_jaccard < 0.15 and intensity_score > 0.2:
        novelty_boost = 1.0 + intensity_score * 0.5
        methods["novelty_boost"] = round(novelty_boost, 3)
    elif max_jaccard < 0.10:
        novelty_boost = 1.05
        methods["novelty_boost"] = round(novelty_boost, 3)
    methods["max_jaccard"] = round(max_jaccard, 3)
    methods["intensity_score"] = round(intensity_score, 3)
    methods["similarity"] = round(m1, 3)

    # ── M2: Keyword-OR — Inverse-Frequency-gewichtet ──────────────────────────
    word_or: dict = defaultdict(list)
    for p in valid:
        for w in re.findall(r"[A-Za-zäöüÄÖÜßaeoeue]{4,}", p.get("title", "").lower()):
            if w.lower() not in stops:
                word_or[w.lower()].append(p["or"])
    kw_scores = []
    kw_weights = []
    for w in push_words:
        if w in word_or and len(word_or[w]) >= 2:
            avg_w = sum(word_or[w]) / len(word_or[w])
            idf = math.log(len(valid) / len(word_or[w]))
            kw_scores.append(avg_w * idf)
            kw_weights.append(idf)
    if kw_scores:
        m2 = sum(kw_scores) / sum(kw_weights)
        m2 = max(0, min(m2, global_avg * 3))
        conf_m2 = min(params["m2_conf_cap"], len(kw_scores) / 4)
    else:
        m2 = global_avg
        conf_m2 = 0.1
    methods["keyword_or"] = round(m2, 3)

    # ── M3: Entity-OR — Personen/Orte aus Titel ───────────────────────────────
    push_entities = set(re.findall(r"[A-ZÄÖÜ][a-zäöüß]{3,}", push_title_raw))
    entity_ors = []
    for ent in push_entities:
        ent_l = ent.lower()
        for p in valid:
            if ent_l in p.get("title", "").lower():
                entity_ors.append(p["or"])
    if entity_ors:
        m3 = sum(entity_ors) / len(entity_ors)
        cat_entity_ors = [
            p["or"] for p in valid
            if p.get("cat") == push_cat
            and any(e.lower() in p.get("title", "").lower() for e in push_entities)
        ]
        if len(cat_entity_ors) >= 5:
            m3 = sum(cat_entity_ors) / len(cat_entity_ors)
        conf_m3 = min(params["m3_conf_cap"], len(entity_ors) / 6)
    else:
        m3 = global_avg
        conf_m3 = 0.1
    methods["entity_or"] = round(m3, 3)

    # ── M4: Kategorie × Stunde × Wochentag × Emotion ─────────────────────────
    cat_sums, cat_counts = 0.0, 0
    hour_sums, hour_counts = 0.0, 0
    day_sums, day_counts = 0.0, 0
    emo_words_set = {
        "schock", "drama", "skandal", "angst", "tod", "sterben", "krieg", "panik",
        "horror", "warnung", "gefahr", "krise", "irre", "wahnsinn", "hammer", "brutal", "bitter",
    }
    is_emo = any(w in push_title_lower for w in emo_words_set)
    emo_sums, emo_counts = 0.0, 0

    for p in valid:
        if p.get("cat") == push_cat:
            cat_sums += p["or"]
            cat_counts += 1
        if p.get("hour") == push_hour:
            hour_sums += p["or"]
            hour_counts += 1
        p_wd = (
            datetime.datetime.fromtimestamp(p.get("ts_num", 0)).weekday()
            if p.get("ts_num", 0) > 0
            else 0
        )
        if p_wd == push_weekday:
            day_sums += p["or"]
            day_counts += 1
        p_emo = any(w in p.get("title", "").lower() for w in emo_words_set)
        if p_emo == is_emo:
            emo_sums += p["or"]
            emo_counts += 1

    cat_avg = cat_sums / cat_counts if cat_counts > 0 else global_avg
    hour_avg = hour_sums / hour_counts if hour_counts > 0 else global_avg
    day_factor = (day_sums / day_counts / global_avg) if day_counts > 0 and global_avg > 0 else 1.0
    emo_factor = (emo_sums / emo_counts / global_avg) if emo_counts > 0 and global_avg > 0 else 1.0
    hour_factor = hour_avg / global_avg if global_avg > 0 else 1.0
    m4 = cat_avg * hour_factor * (0.85 + day_factor * 0.15) * (0.9 + emo_factor * 0.1)
    m4 = min(m4, global_avg * 3.0)
    conf_m4 = min(params["m4_conf_cap"], cat_counts / 10 if cat_counts > 0 else 0.1)
    methods["cat_hour_day_emo"] = round(m4, 3)

    # ── M5: Research-Modifier ─────────────────────────────────────────────────
    mods = state.get("research_modifiers", {})
    m5_factor = 1.0
    timing_mod = mods.get("timing", {}).get(str(push_hour), 1.0)
    m5_factor *= (1.0 - params["timing_damp"]) + params["timing_damp"] * timing_mod
    cat_mod = mods.get("category", {}).get(push_cat, 1.0)
    m5_factor *= (1.0 - params["cat_damp"]) + params["cat_damp"] * cat_mod
    if is_emo:
        framing_mod = mods.get("framing", {}).get("emotional", 1.0)
    elif "?" in push_title_raw:
        framing_mod = mods.get("framing", {}).get("question", 1.0)
    else:
        framing_mod = mods.get("framing", {}).get("neutral", 1.0)
    m5_factor *= (1.0 - params["framing_damp"]) + params["framing_damp"] * framing_mod
    tl = push.get("title_len", len(push_title_raw))
    len_key = "kurz" if tl < 50 else ("lang" if tl > 80 else "mittel")
    len_mod = mods.get("length", {}).get(len_key, 1.0)
    m5_factor *= (1.0 - params["length_damp"]) + params["length_damp"] * len_mod
    has_colon = ":" in push_title_raw or "|" in push_title_raw
    ling_key = "with_colon" if has_colon else "no_colon"
    ling_mod = mods.get("linguistic", {}).get(ling_key, 1.0)
    m5_factor *= (1.0 - params["ling_damp"]) + params["ling_damp"] * ling_mod
    channel_mods = mods.get("channel", {})
    ch_mod = channel_mods.get("eilmeldung", 1.0) if push.get("is_eilmeldung") else channel_mods.get("normal", 1.0)
    m5_factor *= 0.8 + 0.2 * ch_mod
    m5 = global_avg * m5_factor
    conf_m5 = params["m5_conf_cap"] if m5 != global_avg else 0.1
    methods["research_modifier"] = round(m5, 3)

    # ── M6: PhD-Ensemble ──────────────────────────────────────────────────────
    m6_factor = 1.0
    phd_details: dict = {}

    bayes_shrunk = mods.get("bayes_shrinkage", {}).get(push_cat, 1.0)
    if bayes_shrunk != 1.0:
        damp = params["phd_bayes_damp"]
        m6_factor *= (1.0 - damp) + damp * bayes_shrunk
        phd_details["bayes"] = round(bayes_shrunk, 3)

    interactions = mods.get("interactions", {})
    hour_interaction = interactions.get(push_cat, {}).get(str(push_hour), {})
    if hour_interaction:
        inter_factor = hour_interaction.get("factor", 1.0)
        damp = params["phd_interaction_damp"]
        m6_factor *= (1.0 - damp) + damp * max(0.8, min(1.2, inter_factor))
        phd_details["interaction"] = round(inter_factor, 3)

    spectral = mods.get("spectral_timing", {})
    spectral_mod = spectral.get(str(push_hour), 1.0) if isinstance(spectral, dict) else 1.0
    if spectral_mod != 1.0:
        m6_factor *= 0.85 + 0.15 * spectral_mod
        phd_details["spectral"] = round(spectral_mod, 3)

    # Markov: letzter Push vor diesem (temporal causal)
    last_push_cat = ""
    if push_ts > 0:
        prev_pushes = [p for p in valid if p.get("ts_num", 0) > 0 and p["ts_num"] < push_ts]
        if prev_pushes:
            prev_pushes.sort(key=lambda x: x["ts_num"])
            last_push_cat = prev_pushes[-1].get("cat", "")
    markov_seq = mods.get("markov_sequence", {})
    if last_push_cat and last_push_cat in markov_seq:
        best_next = markov_seq[last_push_cat].get("best_next", "")
        if best_next == push_cat:
            markov_boost = markov_seq[last_push_cat].get("boost", 1.0)
            m6_factor *= 0.90 + 0.10 * min(1.15, markov_boost)
            phd_details["markov_seq"] = round(markov_boost, 3)

    entity_ctx = mods.get("entity_context", {})
    if entity_ctx and push_words:
        push_word_list = sorted(push_words)
        ctx_boosts = []
        for i in range(len(push_word_list)):
            for j in range(i + 1, min(i + 4, len(push_word_list))):
                pair = f"{push_word_list[i]}+{push_word_list[j]}"
                if pair in entity_ctx:
                    ctx_boosts.append(entity_ctx[pair])
        if ctx_boosts:
            avg_ctx = sum(ctx_boosts) / len(ctx_boosts)
            damp = params["phd_entity_ctx_damp"]
            m6_factor *= (1.0 - damp) + damp * max(0.8, min(1.2, avg_ctx))
            phd_details["entity_ctx"] = round(avg_ctx, 3)

    recency = mods.get("recency", {})
    recency_factor = recency.get("recency_factor", 1.0) if isinstance(recency, dict) else 1.0
    if recency_factor != 1.0:
        damp = params["phd_recency_damp"]
        m6_factor *= (1.0 - damp) + damp * max(0.9, min(1.1, recency_factor))
        phd_details["recency"] = round(recency_factor, 3)

    entropy_mod = mods.get("entropy", {})
    if entropy_mod and isinstance(entropy_mod, dict):
        tl_len = len(push_title_raw)
        if tl_len < 50:
            ent_factor = entropy_mod.get("low_entropy", 1.0)
        elif tl_len > 80:
            ent_factor = entropy_mod.get("high_entropy", 1.0)
        else:
            ent_factor = entropy_mod.get("mid_entropy", 1.0)
        if ent_factor != 1.0:
            m6_factor *= 0.9 + 0.1 * ent_factor
            phd_details["entropy"] = round(ent_factor, 3)

    m6 = global_avg * m6_factor
    conf_m6 = min(params["m6_phd_cap"], len(phd_details) / 3 if phd_details else 0.1)
    methods["phd_ensemble"] = round(m6, 3)
    methods["phd_details"] = phd_details

    # ── M7: Kontext-Signal (Wetter, Trends, Tagestyp) ─────────────────────────
    # Liest aus state.get("external_context") statt _external_context_cache
    ctx = state.get("external_context", {}) or {}
    m7 = global_avg
    ctx_adjustments = []
    if ctx.get("last_fetch", 0) > 0:
        bad_w = ctx.get("weather", {}).get("bad_weather_score", 0.3)
        weather_boost = 1.0 + bad_w * 0.08
        m7 *= weather_boost
        if bad_w > 0.3:
            ctx_adjustments.append(f"wetter+{(weather_boost - 1) * 100:.1f}%")
        trend_score = _context_topic_match(push_title_raw, ctx.get("trends", []))
        if trend_score > 0:
            trend_boost = 1.0 + trend_score * 0.15
            m7 *= trend_boost
            ctx_adjustments.append(f"trend+{(trend_boost - 1) * 100:.1f}%")
        day_type = ctx.get("day_type", "weekday")
        if day_type == "holiday":
            m7 *= 1.05
            ctx_adjustments.append("feiertag+5%")
        elif day_type == "weekend":
            m7 *= 1.02
            ctx_adjustments.append("wochenende+2%")
        time_ctx = ctx.get("time_context", "normal")
        if time_ctx == "prime_time":
            m7 *= 1.04
            ctx_adjustments.append("primetime+4%")
        elif time_ctx == "nacht":
            m7 *= 0.92
            ctx_adjustments.append("nacht-8%")
        elif time_ctx == "pendler_morgen":
            m7 *= 1.03
            ctx_adjustments.append("pendler+3%")
    conf_m7 = params["m7_conf_cap"] if ctx_adjustments else 0.15
    methods["context_signal"] = round(m7, 3)
    if ctx_adjustments:
        methods["context_details"] = ", ".join(ctx_adjustments)

    # ── M8: GPT-Content-Scoring (optional) ───────────────────────────────────
    m8 = global_avg
    conf_m8 = 0.0
    try:
        from app.config import OPENAI_API_KEY
        if OPENAI_API_KEY and push_title_raw:
            import openai as _oai
            _client = _oai.OpenAI(api_key=OPENAI_API_KEY)
            _m8_examples = []
            if sim_scores:
                for _sim, _or, _title in sim_scores[:5]:
                    _m8_examples.append(f"  - \"{_title}\" → OR {_or:.1f}% (Ähnlichkeit {_sim:.0%})")
            _prompt = (
                f"Du bist ein Experte für Push-Benachrichtigungen der BILD-Zeitung (~8 Mio Abonnenten).\n"
                f"Analysiere diesen Push-Titel und prognostiziere die Opening-Rate (OR) in Prozent.\n\n"
                f"Push-Titel: \"{push_title_raw}\"\n"
                f"Kategorie: {push_cat}\n"
                f"Uhrzeit: {push_hour}:00 Uhr\n"
                f"Wochentag: {['Mo','Di','Mi','Do','Fr','Sa','So'][push_weekday]}\n\n"
                f"Historischer Durchschnitt: {round(m4, 1) if m4 != global_avg else round(global_avg, 1)}%\n"
                + (("\nÄhnliche historische Pushes:\n" + "\n".join(_m8_examples)) if _m8_examples else "")
                + "\n\nAntworte NUR in diesem JSON-Format (kein anderer Text):\n"
                + '{"or_prognose": <float>, "reasoning": "<1 Satz>"}'
            )
            _resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": _prompt}],
                max_tokens=100,
                temperature=0.3,
            )
            import json as _json
            _text = _resp.choices[0].message.content.strip()
            _jmatch = re.search(r"\{.*\}", _text, re.DOTALL)
            if _jmatch:
                _data = _json.loads(_jmatch.group())
                _m8_or = float(_data.get("or_prognose", 0))
                if 0.5 <= _m8_or <= 50:
                    m8 = _m8_or
                    conf_m8 = 1.0
                    methods["gpt_content_scoring"] = round(m8, 3)
                    methods["gpt_reasoning"] = _data.get("reasoning", "")
    except Exception as _m8_exc:
        log.debug("[heuristic] GPT-Scoring fehlgeschlagen: %s", _m8_exc)

    # ── M9: Competitor-Overlap (Exklusivität vs. Sättigung) ───────────────────
    m9 = global_avg
    conf_m9 = 0.0
    _comp_data = state.get("_competitor_cache", {})
    if _comp_data and push_title_lower:
        _comp_overlap_n = 0
        _comp_total_n = 0
        for _src, _items in _comp_data.items():
            if isinstance(_items, list):
                for _it in _items:
                    _comp_t = (
                        (_it.get("title", "") if isinstance(_it, dict) else str(_it)).lower()
                    )
                    if not _comp_t:
                        continue
                    _comp_total_n += 1
                    _comp_words = set(re.findall(r"[a-zäöüß]{4,}", _comp_t)) - stops
                    if push_words and _comp_words:
                        _overlap = len(push_words & _comp_words) / max(1, len(push_words | _comp_words))
                        if _overlap > 0.25:
                            _comp_overlap_n += 1
        if _comp_total_n > 0:
            overlap_ratio = _comp_overlap_n / _comp_total_n
            if overlap_ratio > 0.3:
                m9 = global_avg * (1.0 - overlap_ratio * 0.3)
                conf_m9 = 0.50
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "saturiert"
            elif overlap_ratio < 0.05 and intensity_score > 0.2:
                m9 = global_avg * 1.25
                conf_m9 = 0.50
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "exklusiv"
            else:
                m9 = global_avg * (1.05 - overlap_ratio * 0.15)
                conf_m9 = 0.25
                methods["competitor_overlap"] = round(overlap_ratio, 3)
                methods["competitor_signal"] = "normal"

    # ── Fusion: Gewichteter Durchschnitt mit adaptivem Dampening ─────────────
    method_list = [
        ("similarity",        m1, conf_m1 if m1 != global_avg else 0.1),
        ("keyword_or",        m2, conf_m2 if m2 != global_avg else 0.1),
        ("entity_or",         m3, conf_m3 if m3 != global_avg else 0.1),
        ("cat_hour_day_emo",  m4, conf_m4),
        ("research_modifier", m5, conf_m5 if m5 != global_avg else 0.1),
        ("phd_ensemble",      m6, conf_m6),
        ("context_signal",    m7, conf_m7),
        ("gpt_content",       m8, conf_m8),
        ("competitor_overlap",m9, conf_m9),
    ]

    prior_weight = params["fusion_prior_weight"]

    method_values = [val for _, val, _ in method_list if val > 0]
    if method_values:
        directions = [1 if v > global_avg else -1 for v in method_values]
        convergence = abs(sum(directions)) / len(directions)
        adaptive_prior = prior_weight * (1.0 - convergence * 0.5)
    else:
        adaptive_prior = prior_weight
        convergence = 0.0

    weighted_sum = global_avg * adaptive_prior
    weight_sum = adaptive_prior
    for _, val, cap in method_list:
        weighted_sum += val * cap
        weight_sum += cap
    heuristic_predicted = weighted_sum / weight_sum if weight_sum > 0 else global_avg
    predicted = heuristic_predicted

    methods["convergence"] = round(convergence, 3)
    methods["adaptive_prior"] = round(adaptive_prior, 3)

    # ── Novelty-Boost ─────────────────────────────────────────────────────────
    if novelty_boost > 1.0:
        predicted *= novelty_boost
        methods["pre_novelty"] = round(predicted / novelty_boost, 3)

    # ── Intensity-Boost ───────────────────────────────────────────────────────
    if intensity_score > 0.2 and novelty_boost <= 1.0:
        intensity_factor = 1.0 + intensity_score * 0.45
        predicted *= intensity_factor
        methods["intensity_factor"] = round(intensity_factor, 3)
        methods["intensity_cats"] = ",".join(matched_categories)

    predicted = max(0.01, min(99.99, predicted))

    # ── Post-Fusion-Korrektoren ───────────────────────────────────────────────
    corrections_applied = []

    # Korrektor 1: Fatigue-Penalty
    fatigue = mods.get("fatigue", {})
    if fatigue and isinstance(fatigue, dict) and fatigue.get("alpha", 0) > 0:
        push_day = (
            datetime.datetime.fromtimestamp(push_ts).strftime("%Y-%m-%d")
            if push_ts > 0 else ""
        )
        today_count = push.get("_push_number_today", 0)
        if today_count == 0:
            today_count = state.get("_today_push_count", {}).get(push_day, 0)
        if today_count > 2:
            alpha = fatigue["alpha"]
            penalty = max(0.75, 1.0 - alpha * 1.5 * math.log(max(1, today_count)))
            damp = params["phd_fatigue_damp"]
            fatigue_adj = (1.0 - damp) + damp * penalty
            predicted *= fatigue_adj
            corrections_applied.append(f"fatigue({today_count}th)={fatigue_adj:.3f}")

    # Korrektor 2: Breaking-Regime-Boost
    breaking = mods.get("breaking_regime", {})
    if breaking and isinstance(breaking, dict) and breaking.get("n_breaking", 0) >= 3:
        if is_emo and push_cat == breaking.get("top_cat", ""):
            boost = min(params["phd_breaking_boost"], breaking.get("regime_boost", 1.0))
            predicted *= boost
            corrections_applied.append(f"breaking={boost:.3f}")

    # Korrektor 3: Bias-Korrektur
    bias = mods.get("bias_corrections", {})
    if bias and isinstance(bias, dict):
        cat_bias = bias.get("category", {}).get(push_cat, 0)
        hour_bias = bias.get("hour", {}).get(str(push_hour), 0)
        weekday_bias = bias.get("weekday", {}).get(str(push_weekday), 0)
        total_bias = cat_bias + hour_bias + weekday_bias
        if abs(total_bias) > 0.1:
            damp = params["phd_bias_correction_damp"]
            predicted += total_bias * damp
            corrections_applied.append(f"bias={total_bias * damp:+.2f}")

    # Korrektor 4: Sport-Entity-Boost
    sport_entity_hits = sum(1 for e in _SPORT_HIGH_ENTITIES if e in push_title_lower)
    sport_boost_hits = sum(1 for e in _SPORT_BOOST_ENTITIES if e in push_title_lower)
    if sport_entity_hits > 0 or sport_boost_hits > 0:
        sport_entity_ors = [
            p["or"] for p in valid
            if any(e in p.get("title", "").lower() for e in _SPORT_HIGH_ENTITIES)
            or any(e in p.get("title", "").lower() for e in _SPORT_BOOST_ENTITIES)
        ]
        if len(sport_entity_ors) >= 3:
            sport_avg = sum(sport_entity_ors) / len(sport_entity_ors)
            if sport_avg > predicted * 1.1:
                entity_weight = 0.35 if sport_entity_hits > 0 else 0.20
                predicted = predicted * (1 - entity_weight) + sport_avg * entity_weight
                corrections_applied.append(
                    f"sport_entity(n={len(sport_entity_ors)},avg={sport_avg:.1f},w={entity_weight})"
                )

    # Korrektor 5: Topic-Saturation Penalty
    try:
        _tsp = compute_topic_saturation_penalty(push, push_data, state)
        if _tsp and _tsp.get("penalty", 1.0) < 1.0:
            _tsp_factor = _tsp["penalty"]
            predicted *= _tsp_factor
            corrections_applied.append(
                f"topic_sat(6h={_tsp.get('topic_push_count_6h', 0)},"
                f"j={_tsp.get('highest_jaccard', 0):.2f},"
                f"p={_tsp_factor:.3f})"
            )
            methods["topic_saturation_penalty"] = round(_tsp_factor, 3)
            methods["topic_saturation_reason"] = _tsp.get("reason", "")
    except Exception:
        pass

    # Korrektor 6: Quantil-basierte Clamp
    quantiles = mods.get("quantiles", {})
    cat_q = quantiles.get("category", {}).get(push_cat, {}) if isinstance(quantiles, dict) else {}
    if cat_q:
        q10_floor = cat_q.get("q10", 0)
        q90_cap = cat_q.get("q90", 99)
        if predicted < q10_floor * 0.8:
            predicted = q10_floor * 0.85
            corrections_applied.append(f"q10_floor={q10_floor:.1f}")
        elif predicted > q90_cap * 1.3:
            predicted = q90_cap * 1.15
            corrections_applied.append(f"q90_cap={q90_cap:.1f}")

    predicted = max(0.01, min(99.99, predicted))

    # ── Residual-Korrektur ────────────────────────────────────────────────────
    predicted, residual_corr = _apply_residual_correction_local(
        predicted, push_cat, push_hour, residual_corrector
    )
    methods["residual_correction"] = residual_corr
    if abs(residual_corr) > 0.01:
        corrections_applied.append(f"ResidualCorr={residual_corr:+.2f}")

    # ── Basis-String ──────────────────────────────────────────────────────────
    basis_parts = []
    if kw_scores:
        basis_parts.append(f"{len(kw_scores)} Keywords")
    if entity_ors:
        basis_parts.append(f"{len(push_entities)} Entities")
    basis_parts.append(f"Kat={push_cat}")
    basis_parts.append(f"H={push_hour}")
    if phd_details:
        basis_parts.append(f"PhD({len(phd_details)})")
    if corrections_applied:
        basis_parts.append(f"Korr({len(corrections_applied)})")

    # Konfidenz: basiert auf Datenlage (sim_scores + kw_scores)
    data_points = len(sim_scores) + len(kw_scores) + len(entity_ors)
    confidence = min(0.75, max(0.2, data_points / 30))

    # Q10/Q90 aus Kategorien-Quantilen oder konservativer Schätzung
    std_estimate = 1.5
    q10 = max(0.1, predicted - 1.28 * std_estimate)
    q90 = min(20.0, predicted + 1.28 * std_estimate)

    return {
        "predicted_or": round(predicted, 4),
        "basis_method": f"heuristic_m1m9({', '.join(basis_parts)})",
        "confidence": round(confidence, 3),
        "q10": round(q10, 4),
        "q90": round(q90, 4),
        "methods": methods,
        "phd_corrections": corrections_applied,
        "global_avg": round(global_avg, 4),
        "n_valid": len(valid),
    }
