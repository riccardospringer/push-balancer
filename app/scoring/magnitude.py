"""app/scoring/magnitude.py — Keyword-basierte Nachrichten-Magnitude-Heuristik.

Extrahiert aus push-balancer-server.py: _keyword_magnitude_heuristic()
"""
import re

# ── Emotion-Words (identisch mit push-balancer-server.py) ─────────────────
_GBRT_EMOTION_WORDS: dict = {
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

# ── BILD-Kernthemen Topic-Cluster (identisch mit push-balancer-server.py) ──
_GBRT_TOPIC_CLUSTERS: dict = {
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


def keyword_magnitude_heuristic(title: str, cat_lower: str, is_eilmeldung: int = 0) -> float:
    """Keyword-basierte Nachrichten-Magnitude 1-10 als LLM-Fallback.

    Extrahiert aus push-balancer-server.py: _keyword_magnitude_heuristic()

    Fixes (2026-03-17): Diminishing Returns bei Multi-Keyword-Hits,
    Emotion-Word/Magnitude Double-Counting eliminiert.

    Returns:
        float: Magnitude-Score 1.0–10.0
    """
    title_lower = title.lower()
    score = 3.0  # Basis-Score

    # Eilmeldung/Breaking
    if is_eilmeldung or "eilmeldung" in title_lower or "breaking" in title_lower:
        score += 4.0

    # Terror/Krieg/Katastrophe → hohe Magnitude
    _high_mag = {"terror", "anschlag", "krieg", "explosion", "tsunami", "erdbeben",
                 "tote", "opfer", "massaker", "attentat", "geisel", "amok"}
    _med_high = {"warnung", "alarm", "gefahr", "notfall", "evakuierung", "absturz",
                 "brand", "feuer", "mord", "erstmals", "historisch", "rekord"}
    _med = {"kanzler", "praesident", "papst", "trump", "putin", "skandal",
            "verhaftet", "festnahme", "verurteil", "rücktritt", "wahl"}
    _low = {"lifestyle", "rezept", "trend", "mode", "beauty", "fitness",
            "garten", "reise", "urlaub", "quiz", "rätsel", "horoskop"}

    words = set(re.findall(r'[a-zäöüß]{3,}', title_lower))
    # Diminishing Returns: Anzahl Hits zählt, nicht nur Existenz
    _mag_hits = words & _high_mag
    _mag_used: set = set()
    if _mag_hits:
        n = len(_mag_hits)
        score += 4.0 + (0.8 if n >= 2 else 0)
        _mag_used = _mag_hits
    elif words & _med_high:
        n = len(words & _med_high)
        score += 2.5 + (0.5 if n >= 2 else 0)
        _mag_used = words & _med_high
    elif words & _med:
        n = len(words & _med)
        score += 1.5 + (0.3 if n >= 2 else 0)
        _mag_used = words & _med

    if words & _low:
        score -= 1.5

    # Emotion-Words: NUR zählen wenn sie NICHT schon als Magnitude-Keyword gezählt wurden
    for emo_cat, emo_words in _GBRT_EMOTION_WORDS.items():
        new_emo = (words & emo_words) - _mag_used
        if new_emo:
            if emo_cat in ("angst", "katastrophe", "bedrohung"):
                score += 1.0
            elif emo_cat in ("sensation", "empoerung"):
                score += 0.7
            break

    # Topic-Cluster Bonus
    for cluster, cluster_words in _GBRT_TOPIC_CLUSTERS.items():
        if words & cluster_words:
            if cluster in ("crime", "wetter_extrem"):
                score += 0.5
            break

    # Kategorie-Adjustierung
    if cat_lower == "unterhaltung":
        score -= 1.0
    elif cat_lower in ("politik", "news"):
        score += 0.5
    elif cat_lower == "sport":
        _sport_high = {"gestorben", "ist tot", "tödlich", "herzstillstand",
                       "spielabbruch", "abgesagt", "in lebensgefahr"}
        _sport_med = {"verletzt", "verletzung", "ausfall", "entlassen", "feuert", "rauswurf",
                      "rücktritt", "suspendiert", "dopingsperre", "sperre"}
        _sport_low_boost = {"transfer", "wechsel", "abgang", "verpflichtet", "unterschreibt",
                            "verlängert", "aufstellung", "nominiert", "kader"}
        _sport_malus = {"überblick", "alle tore", "alle ergebnisse",
                        "tabelle", "spieltagsrückblick"}
        title_words = title_lower
        if any(w in title_words for w in _sport_high):
            score += 3.0
        elif any(w in title_words for w in _sport_med):
            score += 2.0
        elif any(w in title_words for w in _sport_low_boost):
            score += 0.8
        if any(w in title_words for w in _sport_malus):
            score -= 1.5

    return max(1.0, min(10.0, score))
