"""app/ml/features.py — Feature-Extraktion für GBRT- und LightGBM-Modelle.

Migriert aus dem frueheren Monolithen.

Enthält:
- Alle Konstanten (_GBRT_CATEGORIES, _GBRT_EMOTION_WORDS, _GBRT_TOPIC_CLUSTERS, etc.)
- _gbrt_extract_features(): ~80 Features für das GBRT/LightGBM-Modell
- keyword_magnitude_heuristic() wird aus app.scoring.magnitude importiert
"""
from __future__ import annotations

import math
import re
import datetime
import logging
from bisect import bisect_left

from app.scoring.magnitude import keyword_magnitude_heuristic as _keyword_magnitude_heuristic

log = logging.getLogger("push-balancer")

# ── Module-level Singleton für CharNGramTFIDF ────────────────────────────────
# Wird beim ersten Aufruf von _gbrt_extract_features lazy initialisiert.
# Kein Monolith-Import erforderlich — lebt vollständig in features.py.
_char_ngram_tfidf = None  # type: ignore[assignment]

# ══════════════════════════════════════════════════════════════════════════════
# ══ GBRT: Gradient Boosted Regression Trees (pure Python, kein numpy) ═══════
# ══════════════════════════════════════════════════════════════════════════════

_GBRT_CATEGORIES = ["sport", "politik", "unterhaltung", "geld", "regional", "digital", "leben", "news"]

_TOPIC_STOPS = {"der", "die", "das", "und", "von", "für", "mit", "auf", "den", "ist",
                "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
                "wird", "wurde", "nach", "über", "dass", "oder", "aber", "wenn", "weil",
                "nicht", "hat", "haben", "sind", "sein", "kann", "aus", "wie", "vor",
                "bei", "zum", "zur", "vom", "dem", "des"}

_GBRT_DEATH_WORDS = {"tot", "tod", "sterben", "gestorben", "stirbt", "toetet", "getoetet",
                     "lebensgefahr", "leiche", "mord", "tote", "opfer", "ums leben"}
_GBRT_EXCLUSIVITY_WORDS = {"exklusiv", "nur bei bild", "bild erfuhr", "bild weiss",
                           "nach bild-informationen", "nach bild-info"}
_GBRT_EMOTION_WORDS = {
    "angst": {"tot", "tod", "sterben", "gestorben", "stirbt", "lebensgefahr", "mord", "tote", "opfer"},
    "katastrophe": {"erdbeben", "tsunami", "explosion", "brand", "feuer", "absturz", "crash",
                    "ueberschwemmung", "hochwasser", "sturm", "orkan"},
    "sensation": {"sensation", "historisch", "erstmals", "rekord", "unfassbar", "unglaublich",
                  "wahnsinn", "hammer", "mega", "schock", "krass"},
    "bedrohung": {"warnung", "alarm", "gefahr", "notfall", "panik", "terror", "angriff",
                  "anschlag", "krieg", "drohung", "evakuierung"},
    "prominenz": {"kanzler", "praesident", "papst", "koenig", "merkel", "scholz", "trump", "putin"},
    "empoerung": {"skandal", "verrat", "betrug", "korrupt", "dreist", "frechheit"},
}
_GBRT_BREAKING_RE = re.compile(r"(?i)\b(eilmeldung|breaking|exklusiv|liveticker|alarm|schock|sensation)\b")

# BILD-Kernthemen Topic-Cluster
_GBRT_TOPIC_CLUSTERS = {
    "crime": {"mord", "messer", "messerattacke", "vergewaltigung", "raub", "räuber", "einbruch",
              "verhaftet", "festnahme", "täter", "polizei", "überfall", "totschlag", "leiche",
              "verbrechen", "kriminalität", "fahndung", "festgenommen", "verdächtig", "tatort"},
    "royals": {"könig", "königin", "prinz", "prinzessin", "harry", "meghan", "william", "kate",
               "palace", "thronfolger", "royal", "monarchie", "windsor", "buckingham", "charles"},
    "kosten": {"inflation", "rente", "bürgergeld", "steuer", "preise", "teuer", "sparen", "miete",
               "energie", "strom", "gas", "heizung", "einkommen", "lohn", "gehalt", "zuschlag",
               "preissteigerung", "verbraucher", "kosten"},
    "gesundheit": {"krebs", "herzinfarkt", "symptome", "arzt", "krankenhaus", "studie", "warnung",
                   "rückruf", "medikament", "diagnose", "therapie", "virus", "infektion", "impfung",
                   "krankheit", "notaufnahme", "operation"},
    "auto": {"tesla", "bmw", "mercedes", "audi", "porsche", "blitzer", "stau", "führerschein",
             "tempolimit", "unfall", "rückruf", "verbrenner", "elektroauto", "verkehr", "autobahn"},
    "sex_beziehung": {"nackt", "affäre", "freundin", "trennung", "hochzeit", "ehe", "dating",
                      "flirt", "erotik", "untreu", "scheidung", "liebesleben", "verlobt", "paar"},
    "wetter_extrem": {"hitze", "kälte", "unwetter", "schnee", "gewitter", "hagel", "frost",
                      "hitzewelle", "sahara", "orkan", "tornado", "überschwemmung", "hochwasser",
                      "sturmflut", "rekordtemperatur", "eisregen", "glätte"},
}

# BILD-Titelstil Regex-Patterns
_GBRT_AGE_PATTERN = re.compile(r'\(\d{1,3}\)')  # "(34)", "(14)"
_GBRT_DAS_SO_PATTERN = re.compile(r'(?i)^(DAS|SO|HIER|DIESE[RS]?|JETZT)\s')
_GBRT_DIRECT_ADDRESS = re.compile(r'(?i)\b(ihnen|sie|ihr|ihre[mnrs]?|du|dein[em]?|man)\b')
_GBRT_NUMBER_EMPHASIS = re.compile(r'\d[\d\s.,]*\s*(euro|prozent|grad|meter|kilo|milliard|million|tausend|%|°|km)', re.IGNORECASE)

# Deutsche Labels für GBRT-Feature-SHAP-Erklärungen
_GBRT_SHAP_LABELS = {
    # Text-Features
    "title_len": "Titellänge", "word_count": "Wortanzahl", "avg_word_len": "Ø Wortlänge",
    "has_question": "Fragezeichen", "has_exclamation": "Ausrufezeichen", "has_colon": "Doppelpunkt",
    "has_pipe": "Pipe-Zeichen", "has_plus_plus": "++Ticker", "has_numbers": "Zahlen im Titel",
    "upper_ratio": "Großbuchstaben-Anteil", "name_density": "Namens-Dichte",
    "death_signal": "Todes-Signal", "exclusivity_signal": "Exklusiv-Signal",
    "kicker_pattern": "Kicker-Muster", "emotional_word_count": "Emotionale Wörter",
    "emotional_categories": "Emotions-Kategorien", "intensity_score": "Intensitäts-Score",
    "breaking_signals": "Breaking-Signale", "is_breaking_style": "Breaking-Stil",
    "is_bild_plus": "BILD Plus (Paywall)",
    # Temporal-Features
    "hour": "Stunde", "hour_sin": "Tageszeit (sin)", "hour_cos": "Tageszeit (cos)",
    "weekday": "Wochentag", "weekday_sin": "Wochentag (sin)", "weekday_cos": "Wochentag (cos)",
    "is_weekend": "Wochenende", "is_prime_time": "Primetime (18-22h)",
    "is_morning_commute": "Morgen-Pendler (6-9h)", "is_late_night": "Spätabend/Nacht",
    "is_lunch": "Mittagszeit (11-13h)", "mins_since_last_same_cat": "Min. seit letztem Push (Ressort)",
    "push_count_today": "Pushes heute bisher", "day_of_month": "Tag im Monat",
    # Category-Features
    "is_eilmeldung": "Eilmeldung", "n_channels": "Kanalanzahl",
    "cat_sport": "Ressort: Sport", "cat_politik": "Ressort: Politik",
    "cat_unterhaltung": "Ressort: Unterhaltung", "cat_geld": "Ressort: Geld",
    "cat_regional": "Ressort: Regional", "cat_digital": "Ressort: Digital",
    "cat_leben": "Ressort: Leben", "cat_news": "Ressort: News",
    # Historical-Features
    "cat_avg_or_7d": "Ressort-Ø OR (7d)", "cat_avg_or_30d": "Ressort-Ø OR (30d)",
    "cat_avg_or_all": "Ressort-Ø OR (gesamt)", "hour_avg_or_7d": "Stunden-Ø OR (7d)",
    "hour_avg_or_30d": "Stunden-Ø OR (30d)", "cat_hour_avg_or": "Ressort×Stunde Ø OR",
    "stacking_cat_hour_baseline": "Stacking: Cat×Hour Baseline", "stacking_cat_hour_n": "Stacking: Cat×Hour Anzahl",
    "stacking_baseline_diff": "Stacking: Bayesian vs Raw Diff",
    "weekday_avg_or": "Wochentag-Ø OR", "max_similarity": "Max. Titel-Ähnlichkeit",
    "top_similar_or": "OR ähnlichster Push", "n_similar_pushes": "Anz. ähnlicher Pushes",
    "avg_similar_or": "Ø OR ähnlicher Pushes", "entity_avg_or": "Entity-Ø OR",
    "entity_count": "Anzahl Entities", "global_avg_or": "Globaler Ø OR",
    # TF-IDF Features
    "tfidf_max_sim": "TF-IDF Max-Ähnlichkeit", "tfidf_avg_sim": "TF-IDF Ø Ähnlichkeit",
    "tfidf_n_similar": "TF-IDF ähnliche Pushes", "tfidf_similar_avg_or": "TF-IDF ähnl. Ø OR",
    # Kontext-Features
    "weather_score": "Wetter-Score", "is_holiday": "Feiertag",
    "is_ctx_weekend": "Wochenende (Kontext)", "trend_match": "Trend-Match",
    # Embedding-Features
    "emb_max_sim": "Embedding Max-Ähnlichkeit", "emb_avg_sim_top10": "Embedding Ø Ähnl. Top-10",
    "emb_n_similar_50": "Embedding ähnl. Pushes", "emb_similar_avg_or": "Embedding ähnl. Ø OR",
    # BILD Topic-Cluster
    "topic_crime": "Thema: Crime", "topic_royals": "Thema: Royals",
    "topic_kosten": "Thema: Kosten/Geld", "topic_gesundheit": "Thema: Gesundheit",
    "topic_auto": "Thema: Auto/Verkehr", "topic_sex_beziehung": "Thema: Beziehung",
    "topic_wetter_extrem": "Thema: Wetter-Extrem", "topic_score_total": "Topic-Score gesamt",
    # Sport-Kalender
    "is_bundesliga_time": "Bundesliga-Zeitfenster", "is_cl_evening": "Champions-League-Abend",
    "is_transfer_window": "Transfer-Fenster",
    # Person-Tier
    "top_entity_or": "Top-Entity OR", "entity_hype_7d": "Entity-Hype (7d)",
    # BILD-Titelstil
    "has_age_parens": "Alter in Klammern", "has_das_so_pattern": "DAS/SO-Muster",
    "has_direct_address": "Direkte Anrede", "has_number_emphasis": "Zahlen-Betonung",
    # Volatilität + Interaktionen
    "cat_or_std_7d": "Ressort OR-Volatilität (7d)", "cat_or_std_30d": "Ressort OR-Volatilität (30d)",
    "hour_or_std_7d": "Stunden OR-Volatilität (7d)", "hour_or_std_30d": "Stunden OR-Volatilität (30d)",
    "weekday_hour_avg_or": "Wochentag×Stunde Ø OR",
    # Neue Features
    "hour_squared": "Stunde² (quadrat. Effekt)", "title_sentiment": "Titel-Sentiment",
    "days_since_similar": "Tage seit ähnl. Push", "or_volatility_7d": "OR-Volatilität (7d)",
}


def _gbrt_extract_features(push, history_stats, state=None, fast_mode=False):
    """Extrahiert ~80 Features aus einem Push fuer das GBRT-Modell.

    Args:
        push: Push-Dict mit title, cat, hour, ts_num, etc.
        history_stats: Vorberechnete Aggregat-Statistiken (von _gbrt_build_history_stats)
        state: Optionaler Research-State fuer Kontext-Features
    Returns:
        Dict mit Feature-Name → Float-Wert
    """
    # CharNGramTFIDF: Singleton, wird beim ersten Aufruf aus core_classes initialisiert.
    # Das Objekt wird in _char_ngram_tfidf (module-level) gecacht — kein Monolith-Import nötig.
    global _char_ngram_tfidf
    if _char_ngram_tfidf is None:
        try:
            from app.ml.core_classes import CharNGramTFIDF as _CharNGramTFIDF
            _char_ngram_tfidf = _CharNGramTFIDF()
        except Exception:
            _char_ngram_tfidf = None  # Graceful Degradation: TF-IDF Features = 0

    # Embedding-Features: Optional — Monolith nicht verfügbar → 0-Fallback.
    # _embedding_model, _get_embedding, _cosine_similarity, _compute_embedding_features
    # und _embedding_pca werden NICHT mehr aus push_balancer_server importiert.
    # Alle Feature-Gruppen die davon abhängen liefern 0-Werte (graceful degradation).
    _embedding_model = None
    _embedding_pca = None
    _embedding_pca_mean = None
    _get_embedding = None
    _cosine_similarity = None
    _compute_embedding_features = None
    _external_context_cache: dict = {}
    _context_topic_match = lambda t, tr: 0.0  # noqa: E731

    try:
        import numpy as np
    except ImportError:
        np = None

    title = push.get("title", "") or ""
    title_lower = title.lower()
    cat = (push.get("cat", "") or "News").strip()
    cat_lower = cat.lower()
    ts = push.get("ts_num", 0)
    dt = datetime.datetime.fromtimestamp(ts) if ts > 0 else datetime.datetime.now()
    hour = push.get("hour", dt.hour)
    weekday = dt.weekday()

    words = title.split()
    word_count = len(words)
    title_len = len(title)

    feat = {}

    # ── Text-Features (~20) ──────────────────────────────────────────────
    feat["title_len"] = title_len
    feat["word_count"] = word_count
    feat["avg_word_len"] = sum(len(w) for w in words) / max(1, word_count)
    feat["has_question"] = 1.0 if "?" in title else 0.0
    feat["has_exclamation"] = 1.0 if "!" in title else 0.0
    feat["has_colon"] = 1.0 if ":" in title else 0.0
    feat["has_pipe"] = 1.0 if "|" in title else 0.0
    feat["has_plus_plus"] = 1.0 if "++" in title else 0.0
    feat["has_numbers"] = 1.0 if re.search(r"\d", title) else 0.0
    feat["upper_ratio"] = sum(1 for c in title if c.isupper()) / max(1, title_len)

    # Name-Density: Gross geschriebene Woerter (Entities)
    cap_words = re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', title)
    feat["name_density"] = len(cap_words) / max(1, word_count)

    # Death Signal
    feat["death_signal"] = 1.0 if any(w in title_lower for w in _GBRT_DEATH_WORDS) else 0.0

    # Exclusivity Signal
    feat["exclusivity_signal"] = 1.0 if any(w in title_lower for w in _GBRT_EXCLUSIVITY_WORDS) else 0.0

    # Kicker Pattern (Doppelpunkt am Anfang: "SPORT:" oder "Name:")
    feat["kicker_pattern"] = 1.0 if re.match(r'^[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß\s]{1,20}:', title) else 0.0

    # Emotional Word Count (multi-category)
    total_emo = 0
    emo_cats_hit = 0
    for cat_name, words_set in _GBRT_EMOTION_WORDS.items():
        hits = sum(1 for w in words_set if w in title_lower)
        if hits > 0:
            emo_cats_hit += 1
            total_emo += hits
    feat["emotional_word_count"] = float(total_emo)
    feat["emotional_categories"] = float(emo_cats_hit)
    feat["intensity_score"] = min(1.0, total_emo * 0.15 * (1.0 + max(0, emo_cats_hit - 1) * 0.3))

    # Title Sentiment: Ratio negativer vs positiver Signalwörter
    _neg_words = {"tot", "tod", "sterben", "mord", "crash", "absturz", "krieg", "terror",
                  "unfall", "opfer", "skandal", "gefahr", "alarm", "notfall", "warnung"}
    _pos_words = {"rekord", "sensation", "historisch", "erstmals", "gewonnen", "sieg",
                  "gerettet", "durchbruch", "freude", "feier", "gold", "triumph", "held"}
    neg_count = sum(1 for w in _neg_words if w in title_lower)
    pos_count = sum(1 for w in _pos_words if w in title_lower)
    total_sent = neg_count + pos_count
    feat["title_sentiment"] = (pos_count - neg_count) / max(1, total_sent)  # -1.0 bis +1.0

    # Breaking Signals
    breaking = 0
    if "++" in title: breaking += 1
    if "+++" in title: breaking += 2
    if "|" in title: breaking += 1
    if _GBRT_BREAKING_RE.search(title): breaking += 2
    if push.get("is_eilmeldung"): breaking += 2
    if title.strip().endswith("!"): breaking += 1
    feat["breaking_signals"] = float(breaking)
    feat["is_breaking_style"] = 1.0 if breaking >= 3 else 0.0

    # ── BILD Plus (Paywall) — aus Feld oder URL ableiten ──
    _is_plus = push.get("is_bild_plus") or push.get("isBildPlus")
    if not _is_plus:
        _link = push.get("link", "") or ""
        if re.search(r"/bild-?plus/|/bild_plus/|/bildplus/|bildplus-gewinnspiele|/premium-event/|\.bild_plus\.", _link):
            _is_plus = True
    feat["is_bild_plus"] = 1.0 if _is_plus else 0.0

    # ── BILD Topic-Cluster-Scores (~8) ────────────────────────────────────
    topic_total = 0
    for topic_name, topic_words in _GBRT_TOPIC_CLUSTERS.items():
        hits = sum(1 for w in topic_words if w in title_lower)
        feat[f"topic_{topic_name}"] = float(hits)
        topic_total += hits
    feat["topic_score_total"] = float(topic_total)

    # ── BILD-Titelstil (~4) ──────────────────────────────────────────────
    feat["has_age_parens"] = 1.0 if _GBRT_AGE_PATTERN.search(title) else 0.0
    feat["has_das_so_pattern"] = 1.0 if _GBRT_DAS_SO_PATTERN.search(title) else 0.0
    feat["has_direct_address"] = 1.0 if _GBRT_DIRECT_ADDRESS.search(title) else 0.0
    feat["has_number_emphasis"] = 1.0 if _GBRT_NUMBER_EMPHASIS.search(title) else 0.0

    # ── Temporal-Features (~15) ──────────────────────────────────────────
    feat["hour"] = float(hour)
    feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    feat["weekday"] = float(weekday)
    feat["weekday_sin"] = math.sin(2 * math.pi * weekday / 7)
    feat["weekday_cos"] = math.cos(2 * math.pi * weekday / 7)
    feat["is_weekend"] = 1.0 if weekday >= 5 else 0.0
    feat["is_prime_time"] = 1.0 if 18 <= hour <= 22 else 0.0
    feat["is_morning_commute"] = 1.0 if 6 <= hour <= 9 else 0.0
    feat["is_late_night"] = 1.0 if hour < 6 or hour >= 23 else 0.0
    feat["is_lunch"] = 1.0 if 11 <= hour <= 13 else 0.0
    feat["hour_squared"] = float(hour * hour)  # Quadratischer Tageszeit-Effekt

    # Minutes since last push same category (Fatigue-Signal)
    last_same_cat_ts = history_stats.get("last_push_ts_by_cat", {}).get(cat_lower, 0)
    if last_same_cat_ts > 0 and ts > last_same_cat_ts:
        feat["mins_since_last_same_cat"] = (ts - last_same_cat_ts) / 60.0
    else:
        feat["mins_since_last_same_cat"] = 1440.0  # Default: 24h

    # Push count today so far (Sättigung)
    feat["push_count_today"] = float(push.get("_push_number_today", 0))

    # Day of month (fuer Monatsend-/Monatsanfang-Effekte)
    feat["day_of_month"] = float(dt.day)

    # ── Sport-Kalender (~3) ──────────────────────────────────────────────
    # Bundesliga: Sa 15:30-18:30, So 15:30-19:30 (Sep-Mai)
    month = dt.month
    is_season = month >= 8 or month <= 5  # Aug-Mai = Saison
    if is_season and weekday == 5 and 15 <= hour <= 18:  # Samstag
        feat["is_bundesliga_time"] = 1.0
    elif is_season and weekday == 6 and 15 <= hour <= 19:  # Sonntag
        feat["is_bundesliga_time"] = 1.0
    elif is_season and weekday == 4 and 20 <= hour <= 22:  # Freitag Abend
        feat["is_bundesliga_time"] = 1.0
    else:
        feat["is_bundesliga_time"] = 0.0
    # Champions League: Di/Mi Abend (Sep-Mai)
    feat["is_cl_evening"] = 1.0 if (is_season and weekday in (1, 2) and 20 <= hour <= 23) else 0.0
    # Transfer-Fenster: Jan + Juli/Aug
    feat["is_transfer_window"] = 1.0 if month in (1, 7, 8) else 0.0

    # ── Category-Features (~10) ──────────────────────────────────────────
    feat["is_eilmeldung"] = 1.0 if push.get("is_eilmeldung") else 0.0
    channels = push.get("channels", [])
    feat["n_channels"] = float(len(channels) if isinstance(channels, list) else 1)

    # Channel-Features (global_avg needed early)
    global_avg = history_stats.get("global_avg", 4.77)
    ch_stats_all = history_stats.get("channel_stats", {})
    ch_names_lower = [str(c).lower().strip() for c in channels] if isinstance(channels, list) else []
    ch_avg_ors = []
    ch_reach = 0.0
    for ch_name in ch_names_lower:
        ch_s = ch_stats_all.get(ch_name, {})
        if ch_s.get("n", 0) > 0:
            ch_avg_ors.append(ch_s["avg"])
            ch_reach += ch_s["n"]
    feat["channel_avg_or"] = (sum(ch_avg_ors) / len(ch_avg_ors)) if ch_avg_ors else global_avg
    feat["channel_max_or"] = max(ch_avg_ors) if ch_avg_ors else global_avg
    feat["has_channel_eilmeldung"] = 1.0 if "eilmeldung" in ch_names_lower else 0.0
    feat["has_channel_news"] = 1.0 if "news" in ch_names_lower else 0.0
    feat["channel_reach_proxy"] = ch_reach

    # ── App-Mix & Reichweite Features (~8) ──────────────────────────────
    # ── Reichweite-Features (KEINE OR-basierten Features = Data Leakage!) ──
    # Nur strukturelle Infos die VOR dem Send bekannt sind:
    n_apps = push.get("n_apps", 0)
    total_recipients = push.get("total_recipients", 0) or push.get("received", 0) or 0
    feat["n_apps"] = float(n_apps) if n_apps else float(len(push.get("app_list", [])))
    feat["log_recipients"] = math.log1p(total_recipients) if total_recipients > 0 else 0.0
    # Per-App Empfänger-Anteile (recipientCount pro App, NICHT openingRate!)
    target_stats = push.get("target_stats", {})
    ios_recip_share = 0.0
    android_recip_share = 0.0
    sport_recip_share = 0.0
    if isinstance(target_stats, dict) and target_stats and total_recipients > 0:
        for app_name, stats in target_stats.items():
            if not isinstance(stats, dict):
                continue
            app_recip = float(stats.get("recipientCount", 0) or 0)
            app_lower = app_name.lower()
            if "ios" in app_lower and "sport" not in app_lower:
                ios_recip_share += app_recip
            if "android" in app_lower and "sport" not in app_lower:
                android_recip_share += app_recip
            if "sport" in app_lower:
                sport_recip_share += app_recip
        ios_recip_share /= max(total_recipients, 1)
        android_recip_share /= max(total_recipients, 1)
        sport_recip_share /= max(total_recipients, 1)
    feat["ios_recip_share"] = ios_recip_share
    feat["android_recip_share"] = android_recip_share
    feat["sport_recip_share"] = sport_recip_share
    feat["ios_android_ratio"] = ios_recip_share / max(android_recip_share, 0.01) if android_recip_share > 0 else 0.0
    # Reichweite relativ zum Median
    median_recip = history_stats.get("median_recipients", 0)
    cat_med = history_stats.get("cat_median_recipients", {}).get(cat_lower, median_recip)
    feat["recipients_vs_median"] = total_recipients / max(cat_med, 1) if cat_med > 0 and total_recipients > 0 else 1.0

    # Category One-Hot
    for c in _GBRT_CATEGORIES:
        feat[f"cat_{c}"] = 1.0 if cat_lower == c else 0.0

    # ── Historical-Features (~20, Bayesian-smoothed) ─────────────────────
    bayesian_prior_n = 10  # Shrinkage-Staerke (reduziert von 30: weniger Regression zum Mittelwert)

    def _bayesian_avg(group_avg, group_n, prior=global_avg, prior_n=bayesian_prior_n):
        """Bayesian Shrinkage: gewichteter Mix aus Gruppen-Durchschnitt und Prior."""
        if group_n <= 0:
            return prior
        return (group_avg * group_n + prior * prior_n) / (group_n + prior_n)

    # Category averages (7d, 30d, all)
    cat_stats = history_stats.get("cat_stats", {}).get(cat_lower, {})
    feat["cat_avg_or_7d"] = _bayesian_avg(cat_stats.get("avg_7d", global_avg), cat_stats.get("n_7d", 0))
    feat["cat_avg_or_30d"] = _bayesian_avg(cat_stats.get("avg_30d", global_avg), cat_stats.get("n_30d", 0))
    feat["cat_avg_or_all"] = _bayesian_avg(cat_stats.get("avg_all", global_avg), cat_stats.get("n_all", 0))

    # Category Momentum
    cat_mom = history_stats.get("cat_momentum", {}).get(cat_lower, {})
    feat["cat_momentum"] = cat_mom.get("momentum", 0.0)
    feat["cat_7d_vs_all_ratio"] = cat_mom.get("ratio_7d_all", 1.0)

    # Hour averages (7d, 30d)
    hour_stats = history_stats.get("hour_stats", {}).get(hour, {})
    feat["hour_avg_or_7d"] = _bayesian_avg(hour_stats.get("avg_7d", global_avg), hour_stats.get("n_7d", 0))
    feat["hour_avg_or_30d"] = _bayesian_avg(hour_stats.get("avg_30d", global_avg), hour_stats.get("n_30d", 0))

    # Category x Hour interaction
    cat_hour_key = f"{cat_lower}_{hour}"
    ch_stats = history_stats.get("cat_hour_stats", {}).get(cat_hour_key, {})
    feat["cat_hour_avg_or"] = _bayesian_avg(ch_stats.get("avg", global_avg), ch_stats.get("n", 0))

    # Stacking: Cat×Hour-Baseline als explizites Feature (= raw Mean ohne Bayesian-Smoothing)
    feat["stacking_cat_hour_baseline"] = ch_stats.get("avg", global_avg)
    feat["stacking_cat_hour_n"] = float(ch_stats.get("n", 0))
    # Differenz zwischen Bayesian-smoothed und raw Baseline
    feat["stacking_baseline_diff"] = feat["cat_hour_avg_or"] - feat["stacking_cat_hour_baseline"]

    # Weekday averages
    wd_stats = history_stats.get("weekday_stats", {}).get(weekday, {})
    feat["weekday_avg_or"] = _bayesian_avg(wd_stats.get("avg", global_avg), wd_stats.get("n", 0))

    # Category x Weekday interaction
    cat_wd_key = f"{cat_lower}_{weekday}"
    cat_wd_stats = history_stats.get("cat_weekday_stats", {}).get(cat_wd_key, {})
    feat["cat_weekday_avg_or"] = _bayesian_avg(cat_wd_stats.get("avg", global_avg), cat_wd_stats.get("n", 0))

    # Weekday x Hour interaction
    wd_hour_key = f"{weekday}_{hour}"
    wd_hour_stats = history_stats.get("weekday_hour_stats", {}).get(wd_hour_key, {})
    feat["weekday_hour_avg_or"] = _bayesian_avg(wd_hour_stats.get("avg", global_avg), wd_hour_stats.get("n", 0))

    # Volatilitaet (Std der OR) — wie vorhersagbar ist diese Kategorie/Stunde?
    cat_vol = history_stats.get("cat_volatility", {}).get(cat_lower, {})
    feat["cat_or_std_7d"] = cat_vol.get("std_7d", 0.0)
    feat["cat_or_std_30d"] = cat_vol.get("std_30d", 0.0)
    hour_vol = history_stats.get("hour_volatility", {}).get(hour, {})
    feat["hour_or_std_7d"] = hour_vol.get("std_7d", 0.0)
    feat["hour_or_std_30d"] = hour_vol.get("std_30d", 0.0)

    # Similarity to top-10 historical pushes (Jaccard) — skip in fast_mode (training)
    if not fast_mode:
        push_words = set(re.findall(r'[a-zäöüß]{4,}', title_lower))
        stops = {"der", "die", "das", "und", "von", "fuer", "mit", "auf", "den", "ist", "ein", "eine",
                 "sich", "auch", "noch", "nur", "jetzt", "alle", "neue", "wird", "wurde", "nach", "ueber",
                 "dass", "aber", "oder", "wenn", "dann", "mehr", "sein", "hat", "haben", "kann", "sind"}
        push_words -= stops
        max_jaccard = 0.0
        top_sim_or = 0.0
        sim_count = 0
        sim_or_sum = 0.0
        for hist_push in history_stats.get("recent_pushes", []):
            h_words = hist_push.get("words", set())
            if push_words and h_words:
                jaccard = len(push_words & h_words) / len(push_words | h_words)
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    top_sim_or = hist_push.get("or", global_avg)
                if jaccard > 0.15:
                    sim_count += 1
                    sim_or_sum += hist_push.get("or", global_avg)
        feat["max_similarity"] = max_jaccard
        feat["top_similar_or"] = top_sim_or if max_jaccard > 0.1 else global_avg
        feat["n_similar_pushes"] = float(sim_count)
        feat["avg_similar_or"] = (sim_or_sum / sim_count) if sim_count > 0 else global_avg
    else:
        feat["max_similarity"] = 0.0
        feat["top_similar_or"] = global_avg
        feat["n_similar_pushes"] = 0.0
        feat["avg_similar_or"] = global_avg

    # Entity-based historical OR
    entities = set(re.findall(r'[A-ZÄÖÜ][a-zäöüß]{2,}', title))
    entity_or_list = history_stats.get("entity_or", {})
    entity_ors = []
    for ent in entities:
        ent_l = ent.lower()
        if ent_l in entity_or_list and len(entity_or_list[ent_l]) >= 2:
            entity_ors.extend(entity_or_list[ent_l])
    feat["entity_avg_or"] = (sum(entity_ors) / len(entity_ors)) if entity_ors else global_avg
    feat["entity_count"] = float(len(entities))

    # Person-Tier: Top-Entity OR (stärkstes Entity im Titel) + Hype-Faktor
    entity_freq_7d = history_stats.get("entity_freq_7d", {})
    top_ent_or = global_avg
    max_ent_hype = 0.0
    for ent in entities:
        ent_l = ent.lower()
        ent_or_hist = entity_or_list.get(ent_l, [])
        if len(ent_or_hist) >= 2:
            ent_avg = sum(ent_or_hist) / len(ent_or_hist)
            if ent_avg > top_ent_or:
                top_ent_or = ent_avg
        freq = entity_freq_7d.get(ent_l, 0)
        if freq > max_ent_hype:
            max_ent_hype = freq
    feat["top_entity_or"] = top_ent_or
    feat["entity_hype_7d"] = float(max_ent_hype)

    # Global average as baseline reference
    feat["global_avg_or"] = global_avg

    # ── Character N-Gram TF-IDF Similarity Features — skip in fast_mode ──
    if not fast_mode and _char_ngram_tfidf and _char_ngram_tfidf.vocab:
        try:
            push_vec = _char_ngram_tfidf.transform_one(title)
            recent = history_stats.get("recent_pushes", [])
            tfidf_sims = []
            tfidf_sim_ors = []
            for hist_push in recent[:500]:
                htitle = hist_push.get("title_raw", "")
                if not htitle:
                    continue
                hvec = _char_ngram_tfidf.transform_one(htitle)
                sim = _char_ngram_tfidf.cosine_similarity(push_vec, hvec)
                if sim > 0.15:
                    tfidf_sims.append(sim)
                    tfidf_sim_ors.append(hist_push.get("or", global_avg))
            feat["tfidf_max_sim"] = max(tfidf_sims) if tfidf_sims else 0.0
            feat["tfidf_avg_sim"] = (sum(tfidf_sims) / len(tfidf_sims)) if tfidf_sims else 0.0
            feat["tfidf_n_similar"] = float(len(tfidf_sims))
            feat["tfidf_similar_avg_or"] = (sum(tfidf_sim_ors) / len(tfidf_sim_ors)) if tfidf_sim_ors else global_avg
        except Exception:
            feat["tfidf_max_sim"] = 0.0
            feat["tfidf_avg_sim"] = 0.0
            feat["tfidf_n_similar"] = 0.0
            feat["tfidf_similar_avg_or"] = global_avg
    else:
        feat["tfidf_max_sim"] = 0.0
        feat["tfidf_avg_sim"] = 0.0
        feat["tfidf_n_similar"] = 0.0
        feat["tfidf_similar_avg_or"] = global_avg

    # ── Kontext-Features (~5) ────────────────────────────────────────────
    ctx = _external_context_cache if '_external_context_cache' in dir() else {}
    if not ctx:
        ctx = globals().get("_external_context_cache", {})
    feat["weather_score"] = ctx.get("weather", {}).get("bad_weather_score", 0.3) if ctx.get("last_fetch", 0) > 0 else 0.3
    feat["is_holiday"] = 1.0 if ctx.get("day_type") == "holiday" else 0.0
    feat["is_ctx_weekend"] = 1.0 if ctx.get("day_type") == "weekend" else 0.0
    feat["trend_match"] = 0.0
    if ctx.get("trends") and title:
        try:
            feat["trend_match"] = _context_topic_match(title, ctx["trends"])
        except Exception:
            pass

    # ── Rolling/Lag-Features (~10) ──────────────────────────────────────
    push_timeline = history_stats.get("push_timeline", [])
    push_timeline_ts = history_stats.get("push_timeline_ts", [])
    if push_timeline_ts and ts > 0:
        # Binary search for current position
        pos = bisect_left(push_timeline_ts, ts)

        def _rolling_stats(window_secs):
            """Avg OR and count of pushes in [ts - window_secs, ts)."""
            start_ts = ts - window_secs
            start_idx = bisect_left(push_timeline_ts, start_ts)
            end_idx = pos  # exclusive (current push not included)
            if start_idx >= end_idx:
                return global_avg, 0
            window = push_timeline[start_idx:end_idx]
            ors = [w[1] for w in window]
            return sum(ors) / len(ors), len(ors)

        r1h_avg, r1h_n = _rolling_stats(3600)
        r3h_avg, r3h_n = _rolling_stats(10800)
        r6h_avg, r6h_n = _rolling_stats(21600)
        r24h_avg, r24h_n = _rolling_stats(86400)

        feat["rolling_or_1h"] = r1h_avg
        feat["rolling_or_3h"] = r3h_avg
        feat["rolling_or_6h"] = r6h_avg
        feat["rolling_or_24h"] = r24h_avg
        feat["rolling_n_1h"] = float(r1h_n)
        feat["rolling_n_3h"] = float(r3h_n)
        feat["rolling_n_6h"] = float(r6h_n)

        # Last-N pushes OR
        prev_pushes = push_timeline[max(0, pos - 10):pos]
        prev_ors = [w[1] for w in prev_pushes]
        feat["rolling_or_last3"] = (sum(prev_ors[-3:]) / len(prev_ors[-3:])) if len(prev_ors) >= 3 else global_avg
        feat["rolling_or_last5"] = (sum(prev_ors[-5:]) / len(prev_ors[-5:])) if len(prev_ors) >= 5 else global_avg

        # Momentum: avg_last3 - avg_last10
        avg_last3 = (sum(prev_ors[-3:]) / len(prev_ors[-3:])) if len(prev_ors) >= 3 else global_avg
        avg_last10 = (sum(prev_ors) / len(prev_ors)) if prev_ors else global_avg
        feat["rolling_momentum"] = avg_last3 - avg_last10

        # Sättigungs-Features
        if prev_pushes:
            last_push_ts_any = prev_pushes[-1][0]
            feat["mins_since_last_push"] = (ts - last_push_ts_any) / 60.0
        else:
            feat["mins_since_last_push"] = 1440.0
        feat["push_rate_3h"] = r3h_n / 3.0
        feat["saturation_score"] = min(1.0, r3h_n / 8.0)
    else:
        feat["rolling_or_1h"] = global_avg
        feat["rolling_or_3h"] = global_avg
        feat["rolling_or_6h"] = global_avg
        feat["rolling_or_24h"] = global_avg
        feat["rolling_n_1h"] = 0.0
        feat["rolling_n_3h"] = 0.0
        feat["rolling_n_6h"] = 0.0
        feat["rolling_or_last3"] = global_avg
        feat["rolling_or_last5"] = global_avg
        feat["rolling_momentum"] = 0.0
        feat["mins_since_last_push"] = 1440.0
        feat["push_rate_3h"] = 0.0
        feat["saturation_score"] = 0.0

    # ── Days since similar push (gleiche Kategorie als Proxy) ───────────
    # Nutze last_push_ts_by_cat aus history_stats (O(1) statt O(n) Timeline-Scan)
    last_cat_ts = history_stats.get("last_push_ts_by_cat", {}).get(cat_lower, 0)
    if last_cat_ts > 0 and ts > last_cat_ts:
        feat["days_since_similar"] = min(365.0, (ts - last_cat_ts) / 86400.0)
    else:
        feat["days_since_similar"] = 365.0

    # ── OR-Volatilität der letzten 7 Tage (Marktvolatilität) ──────────
    or_volatility_7d = 0.0
    push_timeline_ts = history_stats.get("push_timeline_ts", [])
    if push_timeline and push_timeline_ts and ts > 0:
        cutoff_7d_ts = ts - 7 * 86400
        start_idx = bisect_left(push_timeline_ts, cutoff_7d_ts)
        end_idx = bisect_left(push_timeline_ts, ts)
        recent_ors = [push_timeline[i][1] for i in range(start_idx, min(end_idx, len(push_timeline)))]
        if len(recent_ors) > 2:
            or_mean = sum(recent_ors) / len(recent_ors)
            or_volatility_7d = math.sqrt(sum((o - or_mean) ** 2 for o in recent_ors) / len(recent_ors))
    feat["or_volatility_7d"] = or_volatility_7d

    # ── Interaction-Features (~5) ────────────────────────────────────────
    feat["eilmeldung_x_primetime"] = feat["is_eilmeldung"] * feat["is_prime_time"]
    feat["eilmeldung_x_hour"] = feat["is_eilmeldung"] * feat["hour"]
    feat["weekend_x_hour"] = feat["is_weekend"] * feat["hour"]
    feat["breaking_x_primetime"] = feat["is_breaking_style"] * feat["is_prime_time"]

    # ── Reichweite-Interactions ──
    feat["recipients_x_eilmeldung"] = feat["log_recipients"] * feat["is_eilmeldung"]
    feat["recipients_x_primetime"] = feat["log_recipients"] * feat["is_prime_time"]
    feat["n_apps_x_hour"] = feat["n_apps"] * feat["hour"]

    # Sättigungs-Interaktionen
    feat["rolling_3h_x_saturation"] = feat["rolling_or_3h"] * feat["saturation_score"]
    feat["push_rate_x_hour"] = feat["push_rate_3h"] * feat["hour"]
    # Zeitliche Trend-Differenzen (Momentum-Signale)
    feat["rolling_1h_vs_24h"] = feat["rolling_or_1h"] - feat["rolling_or_24h"]
    feat["rolling_3h_vs_24h"] = feat["rolling_or_3h"] - feat["rolling_or_24h"]
    # Month + Season (jahreszeit-abhängige Lesegewohnheiten)
    feat["month"] = float(month)
    feat["month_sin"] = math.sin(2 * math.pi * month / 12)
    feat["month_cos"] = math.cos(2 * math.pi * month / 12)
    feat["is_summer"] = 1.0 if month in (6, 7, 8) else 0.0
    feat["is_winter"] = 1.0 if month in (12, 1, 2) else 0.0

    # ── Kicker/Headline Split-Features ──
    kicker_text = push.get("kicker", "") or ""
    headline_text = push.get("headline", "") or ""
    feat["has_kicker"] = 1.0 if kicker_text.strip() else 0.0
    feat["kicker_len"] = float(len(kicker_text))
    feat["headline_len"] = float(len(headline_text))
    feat["kicker_headline_ratio"] = len(kicker_text) / max(len(headline_text), 1)

    # ── Keyword→OR Features (Historische Wort-Performance) ──────────────
    word_or = history_stats.get("word_or", {})
    bigram_or = history_stats.get("bigram_or", {})
    title_lower = title.lower()
    title_words = [w for w in title_lower.split() if len(w) >= 3 and w not in _TOPIC_STOPS]
    # Wort-Level: avg OR aller Wörter im Titel die wir kennen
    known_word_ors = []
    for w in title_words:
        w_stats = word_or.get(w)
        if w_stats and w_stats["n"] >= 5:
            known_word_ors.append(w_stats["avg"])
    feat["keyword_avg_or"] = sum(known_word_ors) / len(known_word_ors) if known_word_ors else global_avg
    feat["keyword_max_or"] = max(known_word_ors) if known_word_ors else global_avg
    feat["keyword_min_or"] = min(known_word_ors) if known_word_ors else global_avg
    feat["keyword_spread"] = feat["keyword_max_or"] - feat["keyword_min_or"]
    feat["n_known_keywords"] = float(len(known_word_ors))
    feat["keyword_coverage"] = len(known_word_ors) / max(len(title_words), 1)
    # Keyword Quantile: robustere Statistik
    if known_word_ors:
        sorted_kw = sorted(known_word_ors)
        feat["keyword_median_or"] = sorted_kw[len(sorted_kw) // 2]
        feat["keyword_q25_or"] = sorted_kw[len(sorted_kw) // 4] if len(sorted_kw) >= 4 else sorted_kw[0]
        feat["keyword_q75_or"] = sorted_kw[3 * len(sorted_kw) // 4] if len(sorted_kw) >= 4 else sorted_kw[-1]
    else:
        feat["keyword_median_or"] = global_avg
        feat["keyword_q25_or"] = global_avg
        feat["keyword_q75_or"] = global_avg
    # Keyword-Std (Spread der historischen ORs für die Wörter im Titel)
    known_word_stds = []
    for w in title_words:
        w_stats = word_or.get(w)
        if w_stats and w_stats["n"] >= 5:
            known_word_stds.append(w_stats.get("std", 0.0))
    feat["keyword_avg_std"] = sum(known_word_stds) / len(known_word_stds) if known_word_stds else 0.0
    # Bigram-Level
    known_bigram_ors = []
    if len(title_words) >= 2:
        for i in range(len(title_words) - 1):
            bg = f"{title_words[i]}_{title_words[i+1]}"
            bg_stats = bigram_or.get(bg)
            if bg_stats and bg_stats["n"] >= 5:
                known_bigram_ors.append(bg_stats["avg"])
    feat["bigram_avg_or"] = sum(known_bigram_ors) / len(known_bigram_ors) if known_bigram_ors else global_avg
    feat["bigram_max_or"] = max(known_bigram_ors) if known_bigram_ors else global_avg
    feat["n_known_bigrams"] = float(len(known_bigram_ors))
    # Keyword vs Category-Avg: Wie gut sind die Wörter relativ zur Kategorie?
    cat_avg = history_stats.get("cat_stats", {}).get(cat_lower, {}).get("avg_all", global_avg)
    feat["keyword_vs_cat"] = feat["keyword_avg_or"] - cat_avg
    # Heuristic Magnitude/Urgency (LLM-Ersatz)
    _urgency_words = {"eilmeldung", "breaking", "alarm", "warnung", "sofort", "jetzt",
                      "gerade", "aktuell", "liveticker", "notfall", "evakuierung"}
    _magnitude_words = {"krieg", "tod", "tote", "anschlag", "terror", "erdbeben", "tsunami",
                        "explosion", "absturz", "mord", "kanzler", "präsident", "papst",
                        "historisch", "erstmals", "rekord", "weltmeister", "olympia"}
    _click_words = {"geheimnis", "enthüllt", "wahrheit", "unfassbar", "schock", "skandal",
                    "so", "das", "diese", "jetzt", "mega", "hammer", "krass", "irre",
                    "unglaublich", "wahnsinn", "exklusiv"}
    title_word_set = set(title_words)
    feat["heur_urgency"] = float(len(title_word_set & _urgency_words))
    feat["heur_magnitude"] = float(len(title_word_set & _magnitude_words))
    feat["heur_clickbait"] = float(len(title_word_set & _click_words))

    # ── SHAP-guided Top-Feature Interactions ──
    # keyword_avg_or × zeitliche Features (Top-2 SHAP)
    feat["keyword_x_hour_avg"] = feat["keyword_avg_or"] * feat["weekday_hour_avg_or"] / max(global_avg, 1.0)
    feat["keyword_x_cat_hour"] = feat["keyword_avg_or"] * feat["cat_hour_avg_or"] / max(global_avg, 1.0)
    feat["keyword_x_saturation"] = feat["keyword_avg_or"] * feat["saturation_score"]
    feat["keyword_x_volatility"] = feat["keyword_avg_or"] * feat["or_volatility_7d"]
    # keyword_avg_or Abweichung von zeitlichem Durchschnitt
    feat["keyword_vs_hour"] = feat["keyword_avg_or"] - feat["weekday_hour_avg_or"]
    feat["keyword_vs_rolling24h"] = feat["keyword_avg_or"] - feat["rolling_or_24h"]

    # ── Sentence Embedding Similarity Features (Phase F) ──
    # _compute_embedding_features: 500 Cosine-Calls pro Push → NUR in Inference
    # Im Training (8000 Pushes) wären das 4M Cosine-Calls → skip, PCA ersetzt Signal
    _is_inference = state is not None  # state wird nur in Inference übergeben
    if _is_inference and _embedding_model is not None:
        try:
            emb_feats = _compute_embedding_features(title, history_stats)
            feat.update(emb_feats)
        except Exception:
            feat["emb_max_sim"] = 0.0
            feat["emb_avg_sim_top10"] = 0.0
            feat["emb_n_similar_50"] = 0.0
            feat["emb_similar_avg_or"] = 0.0
    else:
        feat["emb_max_sim"] = 0.0
        feat["emb_avg_sim_top10"] = 0.0
        feat["emb_n_similar_50"] = 0.0
        feat["emb_similar_avg_or"] = 0.0

    research_mods = state.get("research_modifiers", {}) if state else {}
    feat["heur_research_factor"] = float(research_mods.get("combined", 1.0)) if research_mods else 1.0
    feat["heur_phd_combined"] = 1.0

    # ── Granulare Wetter-Features (6) ──
    weather_data = ctx.get("weather", {}) if ctx.get("last_fetch", 0) > 0 else {}
    feat["weather_temp_c"] = float(weather_data.get("temp_c", 15)) / 40.0  # normalisiert auf ~0-1
    feat["weather_humidity"] = float(weather_data.get("humidity", 50)) / 100.0
    feat["weather_precip_mm"] = min(1.0, float(weather_data.get("precip_mm", 0)) / 10.0)
    feat["weather_wind_kmph"] = min(1.0, float(weather_data.get("wind_kmph", 10)) / 80.0)
    feat["weather_cloud_cover"] = float(weather_data.get("cloud_cover", 50)) / 100.0
    feat["weather_uv_index"] = min(1.0, float(weather_data.get("uv_index", 3)) / 11.0)

    # ── Titel-Embedding einmalig cachen (für PCA, Trends, Konkurrenz) ──
    # 1 Call pro Push (gecacht), nur in CV-Folds (fast_mode) übersprungen
    _title_emb = None
    if _embedding_model is not None and title:
        try:
            _title_emb = _get_embedding(title)  # Memory-cached, O(1) bei Wiederholung
        except Exception:
            pass

    # ── Google-Trends Embedding-Similarity (5) ──
    trends_list = ctx.get("trends", []) if ctx else []
    feat["trends_max_sim"] = 0.0
    feat["trends_avg_sim_top3"] = 0.0
    feat["trends_n_matching"] = 0.0
    feat["trends_is_trending"] = 0.0
    feat["trends_score_weighted"] = 0.0
    if trends_list and _title_emb is not None:
        try:
            trend_sims = []
            for trend_topic in trends_list[:20]:
                if not trend_topic or not isinstance(trend_topic, str):
                    continue
                trend_emb = _get_embedding(trend_topic)
                if trend_emb is not None:
                    sim = _cosine_similarity(_title_emb, trend_emb)
                    trend_sims.append(sim)
            if trend_sims:
                trend_sims.sort(reverse=True)
                feat["trends_max_sim"] = trend_sims[0]
                feat["trends_avg_sim_top3"] = sum(trend_sims[:3]) / min(3, len(trend_sims))
                feat["trends_n_matching"] = float(sum(1 for s in trend_sims if s > 0.4))
                feat["trends_is_trending"] = 1.0 if trend_sims[0] > 0.6 else 0.0
                feat["trends_score_weighted"] = sum(s for s in trend_sims if s > 0.3)
        except Exception:
            pass

    # ── Konkurrenz-Features (6) — nur Jaccard, kein Embedding pro Headline ──
    comp_cache = state.get("_competitor_cache", {}) if state else {}
    feat["comp_n_covering"] = 0.0
    feat["comp_max_sim"] = 0.0
    feat["comp_is_exclusive"] = 1.0
    feat["comp_lead_hours"] = 0.0
    feat["comp_german_coverage"] = 0.0
    feat["comp_saturation"] = 0.0
    if comp_cache and title:
        try:
            push_words_comp = set(re.findall(r'[a-zäöüß]{4,}', title.lower()))
            _stop_comp = {"der", "die", "das", "und", "von", "für", "mit", "auf", "den", "ist",
                          "ein", "eine", "sich", "auch", "noch", "nur", "jetzt", "alle", "neue",
                          "wird", "wurde", "nach", "über", "dass", "oder", "aber", "wenn", "weil"}
            push_words_comp -= _stop_comp
            total_sources = 0
            covering_sources = 0
            max_jacc = 0.0
            _german_sources = {"spiegel", "focus", "welt", "faz", "stern", "zeit", "tagesschau",
                               "ntv", "rtl", "bild", "sueddeutsche", "tagesspiegel", "morgenpost"}
            german_covering = 0
            for src, items in comp_cache.items():
                if not isinstance(items, list):
                    continue
                total_sources += 1
                is_german = any(g in src.lower() for g in _german_sources)
                src_covers = False
                for it in items[:10]:  # Max 10 statt 15 pro Quelle
                    comp_title = (it.get("title", "") if isinstance(it, dict) else str(it)).lower()
                    if not comp_title:
                        continue
                    # Nur Jaccard-Similarity (O(1) statt _get_embedding pro Headline)
                    comp_words = set(re.findall(r'[a-zäöüß]{4,}', comp_title)) - _stop_comp
                    if comp_words and push_words_comp:
                        jacc = len(push_words_comp & comp_words) / len(push_words_comp | comp_words)
                        if jacc > max_jacc:
                            max_jacc = jacc
                        if jacc > 0.2:
                            src_covers = True
                            break  # Eine Headline reicht pro Quelle
                if src_covers:
                    covering_sources += 1
                    if is_german:
                        german_covering += 1

            feat["comp_n_covering"] = float(covering_sources)
            feat["comp_max_sim"] = max_jacc
            feat["comp_is_exclusive"] = 1.0 if covering_sources == 0 else 0.0
            feat["comp_german_coverage"] = float(german_covering)
            feat["comp_saturation"] = covering_sources / max(1, total_sources)
        except Exception:
            pass

    # ── Embedding-PCA Features (25) ──
    for i in range(25):
        feat[f"emb_pca_{i}"] = 0.0
    if _embedding_pca is not None and _title_emb is not None and np is not None:
        try:
            emb_arr = np.array(_title_emb).reshape(1, -1)
            if _embedding_pca_mean is not None:
                emb_arr = emb_arr - _embedding_pca_mean
            pca_components = _embedding_pca.transform(emb_arr)[0]
            for i in range(min(25, len(pca_components))):
                feat[f"emb_pca_{i}"] = float(pca_components[i])
        except Exception:
            pass

    # ── LLM Magnitude Features (7) ──
    llm_data = push.get("_llm_scores", {})
    _has_llm = float(llm_data.get("magnitude", 0.0)) > 0
    feat["llm_has_score"] = 1.0 if _has_llm else 0.0
    feat["llm_magnitude"] = float(llm_data.get("magnitude", 0.0))
    feat["llm_clickability"] = float(llm_data.get("clickability", 0.0))
    feat["llm_relevanz"] = float(llm_data.get("relevanz", 0.0))
    feat["llm_dringlichkeit"] = float(llm_data.get("dringlichkeit", 0.0))
    feat["llm_emotionalitaet"] = float(llm_data.get("emotionalitaet", 0.0))
    # Composite: gewichtete Kombination
    feat["llm_composite"] = (
        feat["llm_magnitude"] * 0.35 +
        feat["llm_clickability"] * 0.25 +
        feat["llm_relevanz"] * 0.15 +
        feat["llm_dringlichkeit"] * 0.15 +
        feat["llm_emotionalitaet"] * 0.10
    )
    # Keyword-Heuristic als Fallback wenn kein LLM-Score (alle 5 Dimensionen)
    # Fix 2026-03-17: Dimensionen entkoppelt — nicht mehr alle linear von _heur_mag abgeleitet
    if not _has_llm and title:
        _heur_mag = _keyword_magnitude_heuristic(title, cat_lower, push.get("is_eilmeldung", 0))
        feat["llm_magnitude"] = _heur_mag
        feat["llm_clickability"] = max(3.0, min(8.0, feat.get("keyword_avg_or", 5.0)))
        # Relevanz: Eigenständig nach Kategorie-Breite, nur schwach an Magnitude gekoppelt
        _rel_base = {"politik": 7.0, "news": 6.5, "unterhaltung": 5.5, "sport": 4.5,
                     "geld": 5.0, "regional": 4.0, "auto": 4.0, "digital": 4.5,
                     "lifestyle": 3.5, "ratgeber": 3.5, "reise": 3.5}.get(cat_lower, 5.0)
        feat["llm_relevanz"] = min(10.0, _rel_base + min(2.5, (_heur_mag - 5.0) * 0.3))
        # Dringlichkeit: Eilmeldung=hoch, sonst nach Magnitude-Stufe (nicht linear)
        _urg_base = 3.0
        if push.get("is_eilmeldung", 0):
            _urg_base = 9.0
        elif _heur_mag >= 8:
            _urg_base = 6.0
        elif _heur_mag >= 6:
            _urg_base = 4.5
        feat["llm_dringlichkeit"] = min(10.0, _urg_base)
        # Emotionalitaet: Emotion-Words basiert
        _emo_score = 4.0
        _tl = title.lower()
        _tw = set(re.findall(r'[a-zäöüß]{3,}', _tl))
        for _ec, _ew in _GBRT_EMOTION_WORDS.items():
            if _tw & _ew:
                _emo_score = 7.0 if _ec in ("angst", "katastrophe", "sensation", "empoerung") else 5.5
                break
        feat["llm_emotionalitaet"] = _emo_score
        feat["llm_composite"] = (
            feat["llm_magnitude"] * 0.35 + feat["llm_clickability"] * 0.25 +
            feat["llm_relevanz"] * 0.15 + feat["llm_dringlichkeit"] * 0.15 +
            feat["llm_emotionalitaet"] * 0.10
        )

    # ── Interaction-Features für Direct Modeling ──
    ga = feat.get("global_avg_or", global_avg)
    feat["rolling_vs_cat_avg"] = feat.get("rolling_or_3h", ga) - feat.get("cat_avg_or_30d", ga)
    feat["saturation_x_cat"] = feat.get("saturation_score", 0) * feat.get("cat_avg_or_30d", ga)
    feat["keyword_vs_rolling"] = feat.get("keyword_avg_or", ga) - feat.get("rolling_or_3h", ga)
    # cat_hour_confidence: wie viele Samples stützen die Cat×Hour-Baseline?
    _ch_key = f"{cat_lower}_{push.get('hour', 12)}"
    _ch_n = ch_stats.get("n", 0) if ch_stats else 0
    feat["cat_hour_confidence"] = _ch_n / (_ch_n + 20.0)

    return feat


# Public aliases (tagesplan/builder.py importiert ohne Unterstrich)
gbrt_extract_features = _gbrt_extract_features
