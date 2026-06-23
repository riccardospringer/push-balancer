"""Editorial push scoring for BILD candidate ranking.

The scorer uses local aggregate history only. It does not call external
services and does not expose historical example titles in its explanations.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
from collections import Counter
from typing import Any


_STOP_WORDS = {
    "aber", "alle", "auch", "auf", "aus", "bei", "das", "dass", "dem", "den",
    "der", "des", "die", "dies", "diese", "dieser", "doch", "ein", "eine",
    "einem", "einen", "einer", "fuer", "für", "hat", "haben", "hier", "ist",
    "jetzt", "kann", "mit", "nach", "nicht", "noch", "oder", "sich", "sie",
    "sind", "ueber", "über", "und", "von", "vor", "war", "was", "weil",
    "wenn", "wie", "wird", "wurde", "zum", "zur",
}

_TOPIC_CLUSTERS: dict[str, set[str]] = {
    "politik": {
        "bundestag", "kanzler", "merz", "scholz", "afd", "cdu", "spd",
        "regierung", "minister", "wahl", "gesetz", "koalition", "ukraine",
        "putin", "trump", "israel", "krieg",
    },
    "sport": {
        "bayern", "bvb", "dortmund", "bundesliga", "champions", "transfer",
        "trainer", "vertrag", "verletzung", "ausfall", "dfb", "wm", "em",
        "finale", "sieg", "niederlage",
    },
    "verbraucher": {
        "preise", "steuer", "rente", "miete", "geld", "kosten", "sparen",
        "krankenkasse", "versicherung", "strom", "gas", "heizung", "rueckruf",
        "rückruf", "warnung", "verbraucher", "auto", "blitzer",
    },
    "crime": {
        "mord", "messer", "polizei", "festnahme", "täter", "taeter",
        "razzia", "ueberfall", "überfall", "anschlag", "terror", "leiche",
        "prozess", "gericht", "verhaftet",
    },
    "wetter": {
        "wetter", "hitze", "regen", "unwetter", "sturm", "gewitter",
        "schnee", "glätte", "glaette", "hochwasser", "warnung",
    },
    "unterhaltung": {
        "star", "stars", "promi", "bohlen", "helene", "gottschalk",
        "dschungel", "gntm", "trennung", "hochzeit", "liebe", "royal",
        "prinz", "koenig", "könig",
    },
    "wirtschaft": {
        "dax", "boerse", "börse", "firma", "konzern", "job", "jobs",
        "wirtschaft", "insolvenz", "aktie", "milliarden", "millionen",
    },
    "digital": {
        "ki", "apple", "google", "meta", "microsoft", "tesla", "whatsapp",
        "tiktok", "internet", "daten", "smartphone",
    },
}

_BREAKING_RE = re.compile(
    r"(?i)\b(eilmeldung|breaking|alarm|anschlag|terror|krieg|tot|tod|stirbt|"
    r"gestorben|explosion|brand|absturz|festnahme|exklusiv)\b|\+\+"
)
_UTILITY_RE = re.compile(
    r"(?i)\b(warnung|rueckruf|rückruf|preise|steuer|rente|geld|sparen|kosten|"
    r"miete|strom|gas|heizung|wetter|verkehr|stau|blitzer|krankenkasse|"
    r"verbraucher|das bedeutet|müssen sie|muessen sie)\b"
)
_CONFLICT_RE = re.compile(
    r"(?i)\b(streit|skandal|krise|droht|attacke|angriff|kritik|prozess|"
    r"razzia|betrug|vorwurf|wut|zoff|aus|entlassen|ruecktritt|rücktritt)\b"
)
_EMOTION_RE = re.compile(
    r"(?i)\b(drama|schock|horror|tragödie|tragoedie|angst|wahnsinn|"
    r"unfassbar|hammer|bitter|traenen|tränen)\b"
)
_RESULT_RE = re.compile(r"(?i)\b(ergebnis|gewonnen|sieg|niederlage|finale|urteil|entscheidung)\b")
_CLICKBAIT_RE = re.compile(
    r"(?i)\b(das glaubt keiner|was dann passiert|dieser trick|irre|krass|"
    r"unglaublich|unfassbar|netz rastet aus|so reagiert)\b"
)

# ── Reuters Digital News Report 2025 — "Walking the notification tightrope" ──
# Kernbefund: Ueber-Sensationalismus/Clickbait und nicht-essenzielle, "nach Klicks
# suchende" Stories sind die groessten Overload-/Abmelde-Treiber bei Push-Alerts.
# Im bisherigen Score ist Clickbait nur mit 0.04 gewichtet, BILD-Reiz dagegen mit
# 0.18 — die folgende Korrektur wertet Overload-Risiko direkt im Score ab.
REUTERS_OVERLOAD_ENABLED: bool = os.environ.get(
    "PUSH_BALANCER_REUTERS_OVERLOAD_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
_OVERLOAD_SENSATION_RE = re.compile(
    r"(?i)\b(schock|wahnsinn|irre|unfassbar|unglaublich|skandal|horror|drama|"
    r"hammer|mega|krass|sensation|brutal|schock-?\w*)\b"
)
_OVERLOAD_CURIOSITY_RE = re.compile(
    r"(?i)(das steckt dahinter|sie werden (es )?nicht glauben|das m[uü]ssen sie sehen|"
    r"\bkurios\b|\bskurril\b|\bverr[uü]ckt\b|darum geht es jetzt|das ist der grund|"
    r"das m[uü]ssen sie wissen)"
)


def _reuters_overload_adjustment(
    title: str,
    tone: str,
    is_eil: bool,
    risks: list[str],
) -> float:
    """Score-Abwertung fuer Overload-Treiber nach Reuters DNR 2025.

    Ueber-Sensationalismus/Clickbait und nicht-essenzielle Neugier-/Klick-Frames
    treiben Abmeldungen; harte Breaking-Lagen sind ausgenommen (Breaking-
    Priorisierung funktioniert laut Studie). Begrenzt auf max. -12 Punkte.
    """
    if not REUTERS_OVERLOAD_ENABLED or is_eil or tone == "breaking":
        return 0.0
    penalty = 0.0
    sensation = len(_OVERLOAD_SENSATION_RE.findall(title))
    if sensation:
        penalty -= min(8.0, 4.0 + 2.0 * (sensation - 1))
        risks.append("Overload-Risiko: ueber-sensationelle Zuspitzung (Reuters DNR 2025)")
    if title.count("!") >= 2:
        penalty -= 3.0
    if _OVERLOAD_CURIOSITY_RE.search(title):
        penalty -= 4.0
        risks.append("Overload-Risiko: nicht-essenzieller Neugier-/Klick-Frame (Reuters DNR 2025)")
    return max(-12.0, penalty)
_FRESH_DEVELOPMENT_RE = re.compile(
    r"(?i)\b(heute|aktuell|neu|erstmals|plötzlich|ploetzlich|wende|"
    r"entscheidung|beschlossen|beschließt|beschliesst|festnahme|festgenommen|"
    r"angeklagt|urteil|eskaliert|tritt zurück|tritt zurueck|rücktritt|"
    r"ruecktritt|einigt|einigung|greift an|attacke|startet|stoppt|warnt)\b"
)
_POLITICS_RE = re.compile(
    r"(?i)\b(g7|bundestag|bundesrat|regierung|kanzler|minister|merz|scholz|"
    r"trump|putin|ukraine|russland|israel|iran|hormus|nato|eu|wahl|"
    r"staatsbürgerschaft|staatsbuergerschaft|pass|sanktionen|gesetz|"
    r"partei|afd|cdu|spd|grüne|gruene|fdp|koalition)\b"
)
_POLITICS_STRONG_RE = re.compile(
    r"(?i)\b(trump|putin|ukraine|russland|iran|israel|krieg|krise|gipfel|g7|"
    r"entscheidung|wende|eskalation|angriff|attacke|sanktionen|rücktritt|"
    r"ruecktritt|beschluss|beschließt|beschliesst|alarm|droht|plötzlich|"
    r"ploetzlich)\b"
)
_POLITICS_ABSTRACT_RE = re.compile(
    r"(?i)\b(fordert|fordern|soll|sollen|könnte|koennte|will|wollen|debatte|"
    r"diskussion|plan|pläne|plaene|strategie|konzept|programm|papier|"
    r"setzt.*thema|wirtschaftliche wende|verschärfen druck|verschaerfen druck)\b"
)
_VIDEO_RE = re.compile(r"(?i)\b(video|clip|stream|live|liveticker|gucken|ansehen|aufnahme)\b")
_VIDEO_STRONG_RE = re.compile(
    r"(?i)\b(live|jetzt gucken|jetzt sehen|spektakulär|spektakulaer|moment|"
    r"szene|aufnahme|kamera|explosion|brand|unfall|tor|rekord|promi|sport)\b"
)
_VIDEO_WEAK_RE = re.compile(r"(?i)\b(video|clip)\b")
_VAGUE_RE = re.compile(
    r"(?i)\b(das steckt dahinter|darum|so geht es|was dahinter steckt|"
    r"rätsel|raetsel|mysteriös|mysterioes|unklar|diese sache|dieses detail)\b"
)
_GENERIC_CASE_RE = re.compile(
    r"(?i)\b(beliebiger fall|passiert immer wieder|kein neues verbrechen|"
    r"prozessabschluss|zu speziell|ohne foto)\b"
)
_EXCLUSIVE_RE = re.compile(r"(?i)\b(bild-exklusiv|exklusiv|nur bei bild)\b")
_BILD_TRIGGER_PATTERNS: dict[str, tuple[re.Pattern[str], int, str]] = {
    "hae_moment": (
        re.compile(r"(?i)\b(hä\?|hae\?|kurios|rätsel|raetsel|skurril|verrückt|verrueckt|warum|wie kann)\b|\?"),
        10,
        "BILD-Reiz: guter Hä?-Moment oder klare Neugier",
    ),
    "outrage": (
        re.compile(r"(?i)\b(empörung|empoerung|wut|aufreger|skandal|abzocke|tricksen|gebühren|gebuehren|unfair|dreist)\b"),
        11,
        "BILD-Reiz: Aufreger mit Empörungs-Potenzial",
    ),
    "danger": (
        re.compile(r"(?i)\b(messer|kita|explosion|brand|feuerwehr|unfall|gefahr|alarm|warnung|terror|angriff|attacke|brandbombe)\b"),
        12,
        "BILD-Reiz: Gefahr, Sicherheit oder akute Betroffenheit",
    ),
    "crime": (
        re.compile(r"(?i)\b(mord|totschlag|polizei|razzia|festnahme|gericht|prozess|verbrechen|leiche|täter|taeter|opfer|messer)\b"),
        10,
        "BILD-Reiz: Crime/Polizei spricht starkes Push-Interesse an",
    ),
    "consumer": (
        re.compile(r"(?i)\b(geld|kosten|preise|rente|miete|steuer|gebühren|gebuehren|abzocke|rückruf|rueckruf|kunden|shops|strom|krankenkasse|haustiere)\b"),
        10,
        "BILD-Reiz: Verbraucher- oder Geld-Nutzwert für viele",
    ),
    "prominence": (
        re.compile(r"(?i)\b(star|promi|tv|lanz|bohlen|helene|gottschalk|messi|klose|klopp|bayern|bvb|trump|putin)\b"),
        8,
        "BILD-Reiz: prominente Namen erhöhen den Sofort-Klick",
    ),
    "sport_emotion": (
        re.compile(r"(?i)\b(fußball|fussball|bayern|bvb|messi|klopp|wm|em|tor|rekord|trainer|wechsel|transfer|star|finale)\b"),
        8,
        "BILD-Reiz: Sportmoment mit breiter Fan-Zielgruppe",
    ),
    "family": (
        re.compile(r"(?i)\b(kind|kinder|junge|mädchen|maedchen|kita|familie|mutter|vater|baby|haustier|hund|katze)\b"),
        8,
        "BILD-Reiz: Kinder/Familie/Tiere erzeugen Nähe und Betroffenheit",
    ),
    "exclusive": (_EXCLUSIVE_RE, 9, "BILD-Reiz: Exklusivität rechtfertigt Push auch ohne Breaking"),
    "broad_audience": (
        re.compile(r"(?i)\b(millionen|alle|kunden|patienten|pendler|fahrer|mieter|eltern|rentner|deutschland)\b"),
        8,
        "BILD-Reiz: große Zielgruppe unmittelbar betroffen",
    ),
}

_FEEDBACK_RULES: list[tuple[re.Pattern[str], int, str, str]] = [
    (re.compile(r"(?i)\b(top|weckt neugier|triggert gefühl|triggert gefuehl|guter hä|guter hae|kurios|starke zeile)\b"), 10, "driver", "Redaktionsfeedback: starke Zeile, Neugier oder Gefühl bestätigt"),
    (re.compile(r"(?i)\b(hohe relevanz|aktuelle entwicklung|aktuelle lage|gute keywords|jetzt-anlass)\b"), 8, "driver", "Redaktionsfeedback: Aktualität und Relevanz bestätigt"),
    (re.compile(r"(?i)\b(große zielgruppe|grosse zielgruppe|empörend|empoerend|aufreger)\b"), 8, "driver", "Redaktionsfeedback: breite Betroffenheit und Aufreger-Potenzial"),
    (re.compile(r"(?i)\b(bild-exklusiv|exklusiv geht immer)\b"), 7, "driver", "Redaktionsfeedback: Exklusivität als Push-Reiz"),
    (re.compile(r"(?i)\b(video,? aber ok|video.*aktuell|klar verständlich|klar verstaendlich|live)\b"), 5, "driver", "Redaktionsfeedback: Video hat klaren aktuellen Anlass"),
    (re.compile(r"(?i)\b(zu komplex|unkonkret|verrätselt|verraetselt|keine dringlichkeit|nichts passiert)\b"), -18, "risk", "Redaktionsfeedback: zu komplex, unkonkret oder ohne Dringlichkeit"),
    (re.compile(r"(?i)\b(nicht die erstmeldung|ohne aktuelle entwicklung|erwartbar|nur ein politisches thema)\b"), -10, "risk", "Redaktionsfeedback: keine Erstmeldung oder kein neuer Dreh"),
    (re.compile(r"(?i)\b(artikel aus der nacht|vom vorabend|vom vortag)\b"), -12, "risk", "Redaktionsfeedback: Artikel ist zeitlich verbraucht"),
    (re.compile(r"(?i)\b(beliebiger fall|passiert immer wieder|kein neues verbrechen|prozessabschluss)\b"), -10, "risk", "Redaktionsfeedback: Fall wirkt generisch oder nicht neu genug"),
    (re.compile(r"(?i)\b(video)\b"), -3, "risk", "Redaktionsfeedback: Video braucht einen klaren Push-Anlass"),
]


def score_push_candidate(
    push: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
    predicted_or: float | None = None,
) -> dict[str, Any]:
    """Score a single push candidate on a 0-100 editorial priority scale."""
    history = history or []
    state = state or {}
    title = _title(push)
    cat = _cat(push)
    target_dt = _target_dt(push)
    target_ts = int(target_dt.timestamp())
    weekday = target_dt.weekday()
    hour = int(push.get("hour", target_dt.hour) or target_dt.hour)
    is_eil = bool(push.get("is_eilmeldung") or push.get("isEilmeldung"))
    tone = _tone(title, is_eil)
    topic = _topic(title, cat)
    features = _extract_push_features(push, title, cat, target_dt)
    valid_history = _valid_history(history, target_ts)
    global_avg = _global_avg(valid_history, state)

    drivers: list[str] = []
    risks: list[str] = []

    bild_fit = _score_bild_fit(title, cat, tone, is_eil, features, drivers, risks)
    hist_score, hist_info = _score_history(
        valid_history, cat, hour, weekday, tone, topic, global_avg, drivers, risks
    )
    mix_score, mix_info = _score_mix(
        valid_history, title, cat, tone, topic, target_ts, drivers, risks
    )
    freshness_score = _score_freshness(push, title, cat, tone, features, target_dt, drivers, risks)
    bild_reiz = _score_bild_reiz(title, cat, tone, topic, features, drivers, risks)
    headline_strength = _score_headline_strength(title, tone, features, drivers, risks)
    politics_context = _score_politics_context(title, cat, features, freshness_score, drivers, risks)
    video_fit = _score_video_fit(title, features, drivers, risks)
    feedback_score = _score_editorial_feedback(push, features, drivers, risks)
    opening_score = _score_opening_potential(
        push,
        title,
        cat,
        tone,
        topic,
        global_avg,
        predicted_or,
        hist_info,
        features,
        freshness_score,
        bild_reiz,
        drivers,
        risks,
    )
    risk_score = _score_risk(title, cat, tone, mix_info, features, freshness_score, drivers, risks)

    raw_score = (
        bild_fit * 0.14
        + hist_score * 0.12
        + mix_score * 0.13
        + opening_score * 0.16
        + freshness_score * 0.17
        + bild_reiz * 0.18
        + headline_strength * 0.06
        + risk_score * 0.04
    )
    raw_score += (feedback_score - 60.0) * 0.18

    # Keep truly urgent stories from being buried, but still let fatigue matter.
    if is_eil or (tone == "breaking" and opening_score >= 72):
        raw_score += 4.0
    if tone == "neutral" and opening_score < 55 and hist_score < 55:
        raw_score -= 4.0
    if features["is_politics"]:
        raw_score += (politics_context - 60.0) * 0.23
    if features["is_video"]:
        raw_score += (video_fit - 60.0) * 0.14
    if features["stale_politics_without_development"]:
        raw_score -= 7.0
    if features["strong_non_politics"]:
        raw_score += 3.0

    # Reuters DNR 2025: Overload-Treiber (Sensationalismus/Clickbait/Neugier-Frame)
    # direkt abwerten, nicht nur im 0.04-gewichteten Risiko-Term.
    overload_adjustment = _reuters_overload_adjustment(title, tone, is_eil, risks)
    raw_score += overload_adjustment

    score = round(_clip(raw_score, 0.0, 100.0), 1)
    priority = _priority(score)

    drivers = _prioritize_notes(_dedupe(drivers), kind="driver")
    risks = _prioritize_notes(_dedupe(risks), kind="risk")
    if not drivers:
        drivers.append("Solider Basiswert ohne auffälligen Einzel-Treiber")
    if not risks:
        risks.append("Keine starken Ermüdungs- oder Klickigkeits-Signale")

    recommendation = _recommend_text(title, cat, tone, is_eil, risks)

    return {
        "score": score,
        "scoreReason": _reason(score, drivers, risks),
        "performanceDrivers": drivers[:4],
        "risks": risks[:4],
        "recommendedText": recommendation,
        "mixPriority": priority,
        "scoreBreakdown": {
            "bildFit": round(bild_fit, 1),
            "historicalTiming": round(hist_score, 1),
            "mixBalance": round(mix_score, 1),
            "openingRatePotential": round(opening_score, 1),
            "riskAndFatigue": round(risk_score, 1),
            "freshness": round(freshness_score, 1),
            "bildReiz": round(bild_reiz, 1),
            "headlineStrength": round(headline_strength, 1),
            "politicsContext": round(politics_context, 1),
            "videoFit": round(video_fit, 1),
            "editorialFeedback": round(feedback_score, 1),
            "overloadAdjustment": round(overload_adjustment, 1),
        },
    }


def rebalance_push_mix(
    candidates: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    target_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Apply a second-pass diversity adjustment across a candidate list."""
    if not candidates:
        return candidates

    ranked = sorted(candidates, key=lambda item: float(item.get("score", 0) or 0), reverse=True)
    cat_counts: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    adjusted: list[dict[str, Any]] = []

    for item in ranked:
        title = _title(item)
        cat = _cat(item)
        tone = _tone(title, bool(item.get("is_eilmeldung") or item.get("isEilmeldung")))
        topic = _topic(title, cat)
        features = _extract_push_features(item, title, cat, _target_dt(item))

        penalty = 0.0
        bonus = 0.0
        mix_risks: list[str] = []
        mix_drivers: list[str] = []

        cat_limit = 2 if cat == "politik" and not features.get("strong_politics") else 3
        if cat_counts[cat] >= cat_limit:
            penalty += min(12.0, (cat_counts[cat] - cat_limit + 1) * 3.0)
            mix_risks.append(f"Mix-Dopplung: Ressort {cat} ist bereits stark vertreten")
        if topic_counts[topic] >= 2:
            penalty += min(12.0, topic_counts[topic] * 4.0)
            mix_risks.append(f"Mix-Dopplung: Thema {topic} wiederholt sich")
        if tone in {"breaking", "emotion", "conflict"} and tone_counts[tone] >= 2:
            penalty += min(8.0, tone_counts[tone] * 3.0)
            mix_risks.append("Mix-Dopplung: ähnliche emotionale Mechanik")

        if cat_counts[cat] == 0 and topic_counts[topic] == 0:
            bonus += 2.0
            mix_drivers.append("Bringt Vielfalt in den aktuellen Kandidaten-Mix")
        if cat != "politik" and features.get("trigger_strength", 0) >= 18 and float(item.get("score", 0) or 0) >= 58:
            bonus += 2.5
            mix_drivers.append("BILD-starker Nicht-Politik-Kandidat bekommt im Mix eine echte Chance")
        if cat == "politik" and features.get("stale_politics_without_development"):
            penalty += 5.0
            mix_risks.append("Politik-Dichte: alte Politik ohne Entwicklung wird im Mix zurückgenommen")

        if penalty or bonus:
            item = dict(item)
            new_score = round(_clip(float(item.get("score", 0) or 0) - penalty + bonus, 0, 100), 1)
            item["score"] = new_score
            item["mixPriority"] = _priority(new_score)
            breakdown = dict(item.get("scoreBreakdown") or {})
            if "mixBalance" in breakdown:
                breakdown["mixBalance"] = round(_clip(float(breakdown["mixBalance"]) - penalty + bonus, 0, 100), 1)
            item["scoreBreakdown"] = breakdown
            item["risks"] = _prioritize_notes(_dedupe([*(item.get("risks") or []), *mix_risks]), kind="risk")[:4]
            item["performanceDrivers"] = _prioritize_notes(
                _dedupe([*(item.get("performanceDrivers") or []), *mix_drivers]),
                kind="driver",
            )[:4]
            if mix_risks:
                item["scoreReason"] = f"{item.get('scoreReason', '')} Mix-Abzug: {mix_risks[0]}".strip()

        adjusted.append(item)
        cat_counts[cat] += 1
        tone_counts[tone] += 1
        topic_counts[topic] += 1

    adjusted = _rebalance_politics_top10(adjusted)
    return sorted(adjusted, key=lambda item: float(item.get("score", 0) or 0), reverse=True)


def _rebalance_politics_top10(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < 6:
        return candidates
    ranked = sorted(candidates, key=lambda item: float(item.get("score", 0) or 0), reverse=True)
    top = ranked[:10]
    politics_count = sum(1 for item in top if _cat(item) == "politik")
    if politics_count < 6:
        return ranked

    floor = float(top[-1].get("score", 0) or 0) if len(top) >= 10 else float(top[-1].get("score", 0) or 0)
    surplus = max(1, politics_count - 5)
    adjusted: list[dict[str, Any]] = []
    politics_seen = 0

    for item in ranked:
        title = _title(item)
        cat = _cat(item)
        features = _extract_push_features(item, title, cat, _target_dt(item))
        score = float(item.get("score", 0) or 0)
        delta = 0.0
        drivers: list[str] = []
        risks: list[str] = []

        if cat == "politik":
            politics_seen += 1
            if politics_seen > 5 and not features.get("strong_politics"):
                delta -= min(7.0, 3.0 + surplus * 1.5)
                risks.append("Top-10-Balance: Politik ist bereits sehr dominant")
            if features.get("stale_politics_without_development"):
                delta -= 3.0
                risks.append("Top-10-Balance: alte Politik ohne neuen Dreh verliert Vorrang")
        elif score >= floor - 8 and features.get("trigger_strength", 0) >= 14:
            delta += min(7.0, 3.0 + surplus * 1.5)
            drivers.append("Top-10-Balance: BILD-starke Nicht-Politik liegt qualitativ nah dran")

        if delta:
            item = _score_adjusted_item(item, delta, drivers, risks)
        adjusted.append(item)
    return adjusted


def _score_adjusted_item(
    item: dict[str, Any],
    delta: float,
    drivers: list[str],
    risks: list[str],
) -> dict[str, Any]:
    updated = dict(item)
    new_score = round(_clip(float(updated.get("score", 0) or 0) + delta, 0, 100), 1)
    updated["score"] = new_score
    updated["mixPriority"] = _priority(new_score)
    breakdown = dict(updated.get("scoreBreakdown") or {})
    if "mixBalance" in breakdown:
        breakdown["mixBalance"] = round(_clip(float(breakdown["mixBalance"]) + delta, 0, 100), 1)
    updated["scoreBreakdown"] = breakdown
    updated["performanceDrivers"] = _prioritize_notes(
        _dedupe([*(updated.get("performanceDrivers") or []), *drivers]),
        kind="driver",
    )[:4]
    updated["risks"] = _prioritize_notes(_dedupe([*(updated.get("risks") or []), *risks]), kind="risk")[:4]
    if drivers or risks:
        updated["scoreReason"] = _reason(new_score, updated["performanceDrivers"], updated["risks"])
    return updated


def _title(push: dict[str, Any]) -> str:
    return str(push.get("title") or push.get("headline") or "").strip()


def _cat(push: dict[str, Any]) -> str:
    raw = str(push.get("cat") or push.get("category") or "news").lower().strip()
    mapping = {"geld": "wirtschaft", "leben": "verbraucher", "ratgeber": "verbraucher", "panorama": "news"}
    return mapping.get(raw, raw or "news")


def _target_dt(push: dict[str, Any]) -> _dt.datetime:
    ts = push.get("ts_num") or push.get("timestamp")
    try:
        ts_float = float(ts)
    except (TypeError, ValueError):
        ts_float = 0.0
    if ts_float > 0:
        return _dt.datetime.fromtimestamp(ts_float)

    pub = push.get("pubDate") or push.get("publishedAt")
    if isinstance(pub, str) and pub:
        try:
            return _dt.datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        except ValueError:
            pass
    return _dt.datetime.now()


def _valid_history(history: list[dict[str, Any]], target_ts: int) -> list[dict[str, Any]]:
    valid = []
    for item in history:
        try:
            orv = float(item.get("or", item.get("openRate", 0)) or 0)
            ts = int(item.get("ts_num", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 0 < orv <= 100 and ts > 0 and (target_ts <= 0 or ts < target_ts):
            valid.append(item)
    return valid


def _global_avg(history: list[dict[str, Any]], state: dict[str, Any]) -> float:
    state_avg = state.get("global_avg")
    if isinstance(state_avg, (int, float)) and state_avg > 0:
        return float(state_avg)
    vals = []
    for item in history:
        try:
            orv = float(item.get("or", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 0 < orv <= 100:
            vals.append(orv)
    return sum(vals) / len(vals) if vals else 4.77


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zäöüßaeoeue]{4,}", text.lower())) - _STOP_WORDS


def _tone(title: str, is_eil: bool = False) -> str:
    if is_eil or _BREAKING_RE.search(title):
        return "breaking"
    if _UTILITY_RE.search(title):
        return "utility"
    if _CONFLICT_RE.search(title):
        return "conflict"
    if _EMOTION_RE.search(title):
        return "emotion"
    if "?" in title:
        return "curiosity"
    if _RESULT_RE.search(title):
        return "result"
    return "neutral"


def _topic(title: str, cat: str) -> str:
    lower = title.lower()
    best_topic = cat if cat in _TOPIC_CLUSTERS else "news"
    best_hits = 0
    for topic, words in _TOPIC_CLUSTERS.items():
        hits = sum(1 for word in words if word in lower)
        if hits > best_hits:
            best_topic = topic
            best_hits = hits
    return best_topic


def _extract_push_features(
    push: dict[str, Any],
    title: str,
    cat: str,
    target_dt: _dt.datetime,
) -> dict[str, Any]:
    lower = title.lower()
    is_video = bool(push.get("isVideo") or push.get("video") or _VIDEO_RE.search(title))
    freshness_hours = _freshness_hours(push, target_dt)
    pub_dt = _publication_dt(push)
    trigger_hits: dict[str, int] = {}
    trigger_strength = 0
    for name, (pattern, weight, _label) in _BILD_TRIGGER_PATTERNS.items():
        if pattern.search(title):
            trigger_hits[name] = weight
            trigger_strength += weight

    is_politics = cat == "politik" or bool(_POLITICS_RE.search(title))
    has_development = bool(_FRESH_DEVELOPMENT_RE.search(title))
    strong_politics = is_politics and bool(_POLITICS_STRONG_RE.search(title)) and has_development
    abstract_politics = is_politics and bool(_POLITICS_ABSTRACT_RE.search(title)) and not has_development
    stale = freshness_hours is not None and freshness_hours > 6
    overnight = False
    if pub_dt is not None:
        overnight = pub_dt.hour < 6 or (pub_dt.date() < target_dt.date())

    strong_non_politics = (
        not is_politics
        and (trigger_strength >= 18 or cat in {"sport", "unterhaltung", "verbraucher", "crime", "news"})
        and not stale
    )

    feedback_text = " ".join(_collect_feedback_texts(push)).lower()
    if "artikel aus der nacht" in feedback_text or "vom vorabend" in feedback_text or "vom vortag" in feedback_text:
        stale = True
        overnight = True
    if "aktuelle entwicklung" in feedback_text or "erstmeldung" in feedback_text or "aktuelle lage" in feedback_text:
        has_development = True

    return {
        "lower": lower,
        "is_video": is_video,
        "video_strong": is_video and bool(_VIDEO_STRONG_RE.search(title)),
        "video_weak": is_video and not bool(_VIDEO_STRONG_RE.search(title)) and bool(_VIDEO_WEAK_RE.search(title)),
        "freshness_hours": freshness_hours,
        "published_dt": pub_dt,
        "is_stale": stale,
        "is_overnight": overnight,
        "trigger_hits": trigger_hits,
        "trigger_strength": trigger_strength,
        "is_politics": is_politics,
        "has_development": has_development,
        "strong_politics": strong_politics,
        "abstract_politics": abstract_politics,
        "stale_politics_without_development": is_politics and stale and not has_development and not _EXCLUSIVE_RE.search(title),
        "strong_non_politics": strong_non_politics,
        "is_exclusive": bool(_EXCLUSIVE_RE.search(title)),
        "is_vague": bool(_VAGUE_RE.search(title)),
        "is_generic_case": bool(_GENERIC_CASE_RE.search(title) or "passiert immer wieder" in feedback_text),
        "feedback_text": feedback_text,
    }


def _score_bild_fit(
    title: str,
    cat: str,
    tone: str,
    is_eil: bool,
    features: dict[str, Any],
    drivers: list[str],
    risks: list[str],
) -> float:
    score = 56.0
    length = len(title)
    words = title.split()
    word_count = len(words)

    if 34 <= length <= 86 and 5 <= word_count <= 12:
        score += 11
        drivers.append("BILD-Fit: kurz, konkret und schnell erfassbar")
    elif length > 105:
        score -= 10
        risks.append("Titel ist für Push-Verhältnisse zu lang")
    elif length < 18:
        score -= 7
        risks.append("Titel wirkt zu knapp und braucht mehr Kontext")

    if re.search(r"\d", title):
        score += 5
        drivers.append("Konkrete Zahl erhöht Orientierung und Klick-Anlass")

    if re.search(r"^[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß\s-]{2,24}[:|]", title) or ":" in title:
        score += 4

    if re.search(r"(?i)^(DAS|SO|HIER|JETZT|DIESE[RS]?)\b", title):
        score += 5
        drivers.append("BILD-Mechanik: direkter Einstieg mit klarem Fokus")

    if tone in {"breaking", "utility", "conflict"}:
        score += {"breaking": 12, "utility": 10, "conflict": 7}[tone]
        drivers.append(f"Starker Nachrichten- oder Nutzwert-Treiber: {tone}")
    elif tone == "emotion":
        score += 4

    if features.get("trigger_strength", 0) >= 18:
        score += 6
        drivers.append("BILD-Fit: Thema hat mehrere typische Push-Reize")
    if features.get("is_vague"):
        score -= 9
        risks.append("Headline wirkt unkonkret oder verrätselt")

    if is_eil and not re.search(r"(?i)eilmeldung|breaking|\+\+", title):
        score -= 4
        risks.append("Eilmeldungs-Flag ist nicht sauber im Text erkennbar")

    if cat in {"news", "verbraucher", "wirtschaft", "crime"}:
        score += 4
    elif cat == "politik" and not features.get("strong_politics"):
        score -= 4
        risks.append("Politik braucht für Push eine klare neue Entwicklung")
    elif cat == "sport" and features.get("trigger_hits", {}).get("sport_emotion"):
        score += 4
    elif cat == "unterhaltung" and tone == "neutral" and not features.get("trigger_hits", {}).get("prominence"):
        score -= 5
        risks.append("Unterhaltung ohne Prominenz, Konflikt oder Überraschung")

    if _CLICKBAIT_RE.search(title):
        score -= 12
        risks.append("Formulierung wirkt klickig statt redaktionell zwingend")

    if title.count("!") > 1 or "??" in title:
        score -= 8
        risks.append("Überzeichnung durch Satzzeichen kann Vertrauen kosten")

    return _clip(score, 0, 100)


def _score_history(
    history: list[dict[str, Any]],
    cat: str,
    hour: int,
    weekday: int,
    tone: str,
    topic: str,
    global_avg: float,
    drivers: list[str],
    risks: list[str],
) -> tuple[float, dict[str, Any]]:
    if not history:
        return 56.0, {"globalAvg": global_avg, "catHourAvg": global_avg, "catHourN": 0}

    cat_vals: list[float] = []
    hour_vals: list[float] = []
    weekday_hour_vals: list[float] = []
    cat_hour_vals: list[float] = []
    tone_hour_vals: list[float] = []
    topic_vals: list[float] = []

    for item in history:
        try:
            orv = float(item.get("or", 0) or 0)
            ts = int(item.get("ts_num", 0) or 0)
        except (TypeError, ValueError):
            continue
        dt = _dt.datetime.fromtimestamp(ts)
        item_cat = _cat(item)
        item_title = _title(item)
        item_hour = int(item.get("hour", dt.hour) or dt.hour)
        item_tone = _tone(item_title, bool(item.get("is_eilmeldung")))
        item_topic = _topic(item_title, item_cat)

        if item_cat == cat:
            cat_vals.append(orv)
        if item_hour == hour:
            hour_vals.append(orv)
        if dt.weekday() == weekday and item_hour == hour:
            weekday_hour_vals.append(orv)
        if item_cat == cat and item_hour == hour:
            cat_hour_vals.append(orv)
        if item_tone == tone and item_hour == hour:
            tone_hour_vals.append(orv)
        if item_topic == topic:
            topic_vals.append(orv)

    cat_avg = _bayes(cat_vals, global_avg, 10)
    hour_avg = _bayes(hour_vals, global_avg, 10)
    weekday_hour_avg = _bayes(weekday_hour_vals, global_avg, 8)
    cat_hour_avg = _bayes(cat_hour_vals, global_avg, 8)
    tone_hour_avg = _bayes(tone_hour_vals, global_avg, 8)
    topic_avg = _bayes(topic_vals, global_avg, 10)

    score = (
        _or_to_score(cat_hour_avg, global_avg) * 0.34
        + _or_to_score(hour_avg, global_avg) * 0.19
        + _or_to_score(cat_avg, global_avg) * 0.19
        + _or_to_score(weekday_hour_avg, global_avg) * 0.13
        + _or_to_score(tone_hour_avg, global_avg) * 0.08
        + _or_to_score(topic_avg, global_avg) * 0.07
    )

    if 6 <= hour <= 9 and (tone in {"breaking", "utility"} or cat in {"news", "verbraucher", "wirtschaft"}):
        score += 4
        drivers.append("Zeitfenster: Morgen funktioniert für frische News und Nutzwert")
    elif 10 <= hour <= 14 and (cat in {"verbraucher", "crime", "news"} or tone in {"utility", "conflict"}):
        score += 4
        drivers.append("Zeitfenster: Mittag begünstigt Verbraucher, Crime und Aufreger")
    elif 15 <= hour <= 18 and (cat in {"sport", "unterhaltung", "news", "crime"} or tone in {"curiosity", "emotion"}):
        score += 3
        drivers.append("Zeitfenster: Nachmittag öffnet Raum für Sport, Unterhaltung und kuriose News")
    elif 18 <= hour <= 22 and (cat in {"sport", "unterhaltung", "news"} or tone in {"breaking", "emotion"}):
        score += 4
        drivers.append("Zeitfenster: Abend begünstigt Sport, Unterhaltung und emotionale News")
    elif hour >= 23 or hour < 6:
        if tone != "breaking":
            score -= 8
            risks.append("Nachtzeit ohne Breaking-Druck senkt Push-Relevanz")
    if cat == "politik" and tone == "neutral":
        score -= 3
        risks.append("Zeitfenster: neutrale Politik braucht stärkeren aktuellen Anlass")

    if cat_hour_avg > global_avg + 0.6 and len(cat_hour_vals) >= 3:
        drivers.append(
            f"Historisches Muster: {cat} um {hour} Uhr liegt über Durchschnitt"
        )
    elif cat_hour_avg < global_avg - 0.6 and len(cat_hour_vals) >= 3:
        risks.append(f"Historisches Muster: {cat} um {hour} Uhr performt unter Durchschnitt")

    return _clip(score, 0, 100), {
        "globalAvg": round(global_avg, 3),
        "catHourAvg": round(cat_hour_avg, 3),
        "catHourN": len(cat_hour_vals),
        "hourAvg": round(hour_avg, 3),
        "catAvg": round(cat_avg, 3),
        "topicAvg": round(topic_avg, 3),
    }


def _score_mix(
    history: list[dict[str, Any]],
    title: str,
    cat: str,
    tone: str,
    topic: str,
    target_ts: int,
    drivers: list[str],
    risks: list[str],
) -> tuple[float, dict[str, Any]]:
    if not history or target_ts <= 0:
        return 68.0, {"sameCat6h": 0, "sameTone6h": 0, "similar6h": 0}

    words = _tokens(title)
    same_cat_6h = 0
    same_tone_6h = 0
    same_topic_6h = 0
    similar_6h = 0
    recent_total_6h = 0
    last_similar_hours = 999.0

    for item in history:
        try:
            ts = int(item.get("ts_num", 0) or 0)
        except (TypeError, ValueError):
            continue
        delta = target_ts - ts
        if delta <= 0 or delta > 6 * 3600:
            continue
        recent_total_6h += 1
        item_cat = _cat(item)
        item_title = _title(item)
        item_tone = _tone(item_title, bool(item.get("is_eilmeldung")))
        item_topic = _topic(item_title, item_cat)
        if item_cat == cat:
            same_cat_6h += 1
        if item_tone == tone:
            same_tone_6h += 1
        if item_topic == topic:
            same_topic_6h += 1
        item_words = _tokens(item_title)
        if words and item_words:
            jaccard = len(words & item_words) / max(1, len(words | item_words))
            if jaccard >= 0.24:
                similar_6h += 1
                last_similar_hours = min(last_similar_hours, delta / 3600)

    score = 76.0
    if same_cat_6h == 0:
        score += 7
        drivers.append("Mix: Ressort bringt frische Farbe in die letzten Stunden")
    elif same_cat_6h >= 3:
        score -= 13
        risks.append(f"Mix-Ermüdung: {same_cat_6h} Pushs aus ähnlichem Ressort in 6 Stunden")
    elif same_cat_6h >= 2:
        score -= 7

    if same_tone_6h >= 3 and tone in {"breaking", "emotion", "conflict"}:
        score -= 9
        risks.append("Ton-Ermüdung: zu viele ähnliche emotionale Trigger")

    if same_topic_6h >= 2 or similar_6h >= 1:
        score -= 10 + min(8, similar_6h * 3)
        risks.append("Themen-Sättigung: ähnliches Thema wurde kürzlich bereits gepusht")

    if recent_total_6h >= 8:
        score -= 5
        risks.append("Hohe Push-Dichte in den letzten Stunden")

    return _clip(score, 0, 100), {
        "sameCat6h": same_cat_6h,
        "sameTone6h": same_tone_6h,
        "sameTopic6h": same_topic_6h,
        "similar6h": similar_6h,
        "lastSimilarHours": round(last_similar_hours, 2),
    }


def _score_freshness(
    push: dict[str, Any],
    title: str,
    cat: str,
    tone: str,
    features: dict[str, Any],
    target_dt: _dt.datetime,
    drivers: list[str],
    risks: list[str],
) -> float:
    age = features.get("freshness_hours")
    if age is None:
        risks.append("Aktualität: keine belastbare Erstpublikation gefunden")
        return 60.0

    score = 58.0
    if age <= 0.5:
        score = 96.0
        drivers.append("Aktualität: sehr frisch veröffentlicht")
    elif age <= 1.5:
        score = 88.0
        drivers.append("Aktualität: frisch genug für einen Push")
    elif age <= 3:
        score = 74.0
        drivers.append("Aktualität: noch im Push-Fenster")
    elif age <= 6:
        score = 58.0
        risks.append("Aktualität: schon mehrere Stunden alt")
    elif age <= 18:
        score = 38.0
        risks.append("Aktualität: älter als 6 Stunden, braucht starken neuen Dreh")
    else:
        score = 26.0
        risks.append("Aktualität: Vortag/alt, nur mit Exklusivität oder starkem Evergreen-Reiz")

    if features.get("is_overnight"):
        score -= 10
        risks.append("Aktualität: Artikel aus Nacht/Vorabend wirkt verbraucht")
    if features.get("has_development"):
        score += 10
        drivers.append("Aktualität: erkennbare neue Entwicklung oder Erstmeldung")
    if features.get("is_exclusive"):
        score += 8
        drivers.append("Aktualität: Exklusivität kann Alter teilweise auffangen")
    if cat == "politik" and age > 3 and not features.get("has_development"):
        score -= 12
        risks.append("Politik: ohne neue Entwicklung verliert der Artikel schnell Push-Wert")
    if tone == "utility" and age <= 18:
        score += 4
    return _clip(score, 0, 100)


def _score_bild_reiz(
    title: str,
    cat: str,
    tone: str,
    topic: str,
    features: dict[str, Any],
    drivers: list[str],
    risks: list[str],
) -> float:
    score = 42.0
    hits: dict[str, int] = features.get("trigger_hits") or {}
    for name, weight in hits.items():
        label = _BILD_TRIGGER_PATTERNS[name][2]
        score += min(14, weight)
        drivers.append(label)

    if tone in {"breaking", "conflict", "utility", "emotion", "curiosity"}:
        score += {"breaking": 13, "conflict": 8, "utility": 8, "emotion": 7, "curiosity": 6}[tone]
    if topic in {"crime", "verbraucher", "wetter", "unterhaltung", "sport"} and hits:
        score += 5
    if cat == "politik" and not (features.get("strong_politics") or hits):
        score -= 10
        risks.append("BILD-Reiz: abstrakte Politik ohne klaren Sofort-Klick")
    if features.get("is_generic_case"):
        score -= 14
        risks.append("BILD-Reiz: Fall wirkt generisch oder schon bekannt")
    if features.get("is_vague"):
        score -= 9
    if not hits and tone == "neutral":
        score -= 4
        risks.append("BILD-Reiz: noch kein klarer Hä?-, Aufreger- oder Nutzwert-Moment")
    return _clip(score, 0, 100)


def _score_headline_strength(
    title: str,
    tone: str,
    features: dict[str, Any],
    drivers: list[str],
    risks: list[str],
) -> float:
    score = 58.0
    length = len(title)
    words = title.split()
    if 35 <= length <= 86 and 5 <= len(words) <= 12:
        score += 14
    elif length > 105:
        score -= 14
    elif length < 18:
        score -= 8

    if ":" in title:
        score += 5
    if "?" in title:
        score += 5
    if re.search(r"\d", title):
        score += 5
    if tone in {"breaking", "utility", "conflict", "curiosity"}:
        score += 6
    if features.get("is_vague"):
        score -= 15
        risks.append("Headline-Stärke: verrätselt statt konkret")
    if features.get("abstract_politics"):
        score -= 8
        risks.append("Headline-Stärke: politische Debatte schwer in einem Push-Satz")
    if score >= 75:
        drivers.append("Headline-Stärke: schnell verständlich und zuspitzbar")
    return _clip(score, 0, 100)


def _score_politics_context(
    title: str,
    cat: str,
    features: dict[str, Any],
    freshness_score: float,
    drivers: list[str],
    risks: list[str],
) -> float:
    if not features.get("is_politics"):
        return 66.0

    score = 54.0
    if features.get("strong_politics"):
        score += 26
        drivers.append("Politik: aktuelle Lage, prominente Akteure oder klare Wendung")
    elif _POLITICS_STRONG_RE.search(title):
        score += 10

    if features.get("has_development"):
        score += 10
    if freshness_score >= 80:
        score += 6
    if features.get("abstract_politics"):
        score -= 20
        risks.append("Politik: abstrakte Debatte ohne Ereignis oder Eskalation")
    if features.get("stale_politics_without_development"):
        score -= 24
        risks.append("Politik: alt/aus der Nacht und keine neue Entwicklung erkennbar")
    if not features.get("has_development") and not features.get("trigger_hits"):
        score -= 10
        risks.append("Politik: Nachrichtenwert noch zu gesetzt, nicht passiert")
    return _clip(score, 0, 100)


def _score_video_fit(
    title: str,
    features: dict[str, Any],
    drivers: list[str],
    risks: list[str],
) -> float:
    if not features.get("is_video"):
        return 68.0

    score = 56.0
    if features.get("video_strong"):
        score += 24
        drivers.append("Video: klarer Jetzt-Anlass oder starker Schauwert")
    if features.get("has_development"):
        score += 8
    if features.get("trigger_strength", 0) >= 16:
        score += 7
    if features.get("video_weak") and not features.get("video_strong"):
        score -= 16
        risks.append("Video: Bewegtbild allein liefert noch keinen Push-Anlass")
    if features.get("is_vague"):
        score -= 8
        risks.append("Video: Kontext muss ohne Rätsel sofort verständlich sein")
    return _clip(score, 0, 100)


def _score_editorial_feedback(
    push: dict[str, Any],
    features: dict[str, Any],
    drivers: list[str],
    risks: list[str],
) -> float:
    texts = _collect_feedback_texts(push)
    if not texts:
        return 60.0
    score = 60.0
    joined = " ".join(texts)
    for pattern, delta, kind, label in _FEEDBACK_RULES:
        if pattern.search(joined):
            score += delta
            if kind == "driver":
                drivers.append(label)
            else:
                risks.append(label)
    if features.get("is_video") and "video" in joined.lower() and score >= 60:
        drivers.append("Feedback-Logik: Video wird nicht pauschal abgewertet, sondern nach Anlass bewertet")
    return _clip(score, 0, 100)


def _score_opening_potential(
    push: dict[str, Any],
    title: str,
    cat: str,
    tone: str,
    topic: str,
    global_avg: float,
    predicted_or: float | None,
    hist_info: dict[str, Any],
    features: dict[str, Any],
    freshness_score: float,
    bild_reiz: float,
    drivers: list[str],
    risks: list[str],
) -> float:
    predicted = _predicted_percent(predicted_or)
    if predicted is None:
        predicted = float(hist_info.get("catHourAvg") or hist_info.get("catAvg") or global_avg)

    or_score = _or_to_score(predicted, global_avg)
    content = 48.0
    lower = title.lower()

    if tone == "breaking":
        content += 20
        drivers.append("Opening-Potenzial: hoher Aktualitäts- und Unterbrechungswert")
    elif tone == "utility":
        content += 17
        drivers.append("Opening-Potenzial: unmittelbarer Nutzwert")
    elif tone == "conflict":
        content += 12
    elif tone == "emotion":
        content += 8
    elif tone == "curiosity":
        content += 5

    if re.search(r"\d", title):
        content += 6
    if re.search(r"[A-ZÄÖÜ][a-zäöüß]{2,}", title):
        content += 5
    if cat == "regional" or any(word in lower for word in ("berlin", "hamburg", "muenchen", "münchen", "nrw")):
        content += 4
    if topic in {"crime", "verbraucher", "wetter"}:
        content += 5

    if freshness_score >= 85:
        content += 7
    elif freshness_score < 45 and tone != "utility":
        content -= 10

    if bild_reiz >= 78:
        content += 10
        drivers.append("Opening-Potenzial: BILD-Reiz stark genug für Sofort-Klick")
    elif bild_reiz < 45:
        content -= 8

    if features.get("is_politics") and not features.get("has_development"):
        content -= 9
    if features.get("abstract_politics"):
        content -= 8
    if features.get("strong_non_politics"):
        content += 6
    if features.get("is_video"):
        if features.get("video_strong"):
            content += 5
        else:
            content -= 6

    if tone == "neutral" and not re.search(r"\d|:|\?|!", title):
        content -= 7
        risks.append("Öffnungsanreiz ist noch zu allgemein")

    return _clip(or_score * 0.56 + content * 0.44, 0, 100)


def _score_risk(
    title: str,
    cat: str,
    tone: str,
    mix_info: dict[str, Any],
    features: dict[str, Any],
    freshness_score: float,
    drivers: list[str],
    risks: list[str],
) -> float:
    score = 82.0
    length = len(title)

    if _CLICKBAIT_RE.search(title):
        score -= 18
    if title.count("!") > 1:
        score -= 10
    if tone == "emotion" and _EMOTION_RE.findall(title) and len(_EMOTION_RE.findall(title)) >= 2:
        score -= 8
        risks.append("Emotion ist hoch, muss sauber belegbar bleiben")
    if length > 100:
        score -= 9
    if mix_info.get("similar6h", 0) >= 1:
        score -= 12
    if mix_info.get("sameCat6h", 0) >= 4:
        score -= 8
    if "?" in title and tone == "curiosity" and not re.search(r"\d|[A-ZÄÖÜ][a-zäöüß]{2,}", title):
        score -= 8
        risks.append("Frage erzeugt Neugier, aber noch zu wenig Substanz")
    if freshness_score < 45:
        score -= 12
    if cat == "politik" and features.get("abstract_politics"):
        score -= 14
    if features.get("stale_politics_without_development"):
        score -= 16
    if features.get("is_generic_case"):
        score -= 12
    if features.get("is_vague"):
        score -= 12
    if features.get("is_video") and not features.get("video_strong"):
        score -= 10
    if score >= 76:
        drivers.append("Risiko niedrig: kein klares Klickigkeits- oder Fatigue-Signal")
    return _clip(score, 0, 100)


def _recommend_text(title: str, cat: str, tone: str, is_eil: bool, risks: list[str]) -> str:
    text = re.sub(r"\s+", " ", title).strip(" -")
    text = re.sub(r"(?i)^(irre|krass|unfassbar|unglaublich|schock):\s*", "", text).strip()

    if len(text) > 92:
        text = _shorten(text, 88)

    if is_eil and not re.search(r"(?i)^eilmeldung|\+\+", text):
        return f"Eilmeldung: {text}"
    # Bewusst KEINE generischen Floskeln anhaengen ("Was jetzt wichtig ist",
    # "Was jetzt passiert", "Das bedeutet das fuer Sie"): sie verwaessern den
    # Titel und bringen keinen Mehrwert. Die konkrete Schlagzeile ist staerker.
    if any("zu lang" in risk for risk in risks):
        return _shorten(text, 82)
    return text


def _reason(score: float, drivers: list[str], risks: list[str]) -> str:
    if score >= 90:
        level = "außergewöhnlich stark"
    elif score >= 75:
        level = "stark"
    elif score >= 60:
        level = "solide"
    elif score >= 40:
        level = "eher schwach"
    else:
        level = "nicht pushwürdig"
    pro = _compact_reason_items(drivers, 2)
    contra = _compact_reason_items(risks, 1)
    if score >= 60:
        reason = f"{level}: hoch wegen {pro}" if pro else f"{level}: mehrere solide Push-Signale"
        if contra:
            reason += f". Risiko: {contra}"
        return reason + "."
    reason = f"{level}: nicht hochgerankt wegen {contra}" if contra else f"{level}: zu wenig klarer Push-Anlass"
    if pro:
        reason += f". Pluspunkt: {pro}"
    return reason + "."


def _compact_reason_items(items: list[str], max_items: int) -> str:
    cleaned = []
    for item in items:
        short = item
        for prefix in (
            "BILD-Reiz: ",
            "Aktualität: ",
            "Opening-Potenzial: ",
            "Redaktionsfeedback: ",
            "Headline-Stärke: ",
            "Politik: ",
            "Video: ",
        ):
            short = short.replace(prefix, "")
        short = short.strip(". ")
        if short and short not in cleaned:
            cleaned.append(short)
    return ", ".join(cleaned[:max_items])


def _priority(score: float) -> str:
    if score >= 75:
        return "hoch"
    if score >= 60:
        return "mittel"
    return "niedrig"


def _predicted_percent(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        predicted = float(value)
    except (TypeError, ValueError):
        return None
    if predicted <= 0:
        return None
    if predicted <= 1.0:
        predicted *= 100.0
    return predicted


def _publication_dt(push: dict[str, Any]) -> _dt.datetime | None:
    pub = push.get("pubDate") or push.get("publishedAt") or push.get("pub_date")
    if not isinstance(pub, str) or not pub:
        return None
    try:
        return _dt.datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    except ValueError:
        return None


def _freshness_hours(push: dict[str, Any], reference_dt: _dt.datetime | None = None) -> float | None:
    published = _publication_dt(push)
    if published is None:
        return None
    reference = reference_dt or _dt.datetime.now()
    return max(0.0, (reference - published).total_seconds() / 3600)


def _collect_feedback_texts(push: dict[str, Any]) -> list[str]:
    keys = (
        "feedback",
        "editorialFeedback",
        "manualFeedback",
        "comment",
        "comments",
        "scoreComment",
        "decisionComment",
        "approvalReason",
        "rejectReason",
        "rejectionReason",
        "holdReason",
        "freshnessNote",
        "headlineNote",
        "contentNote",
        "videoNote",
        "sectionBalanceNote",
        "feedbackTypes",
    )
    texts: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                texts.append(value.strip())
        elif isinstance(value, dict):
            for inner in value.values():
                add(inner)
        elif isinstance(value, (list, tuple, set)):
            for inner in value:
                add(inner)

    for key in keys:
        add(push.get(key))
    return texts


def _bayes(values: list[float], prior: float, prior_n: int) -> float:
    if not values:
        return prior
    return (sum(values) + prior * prior_n) / (len(values) + prior_n)


def _or_to_score(avg: float, global_avg: float) -> float:
    if global_avg <= 0:
        global_avg = 4.77
    diff = avg - global_avg
    ratio = avg / max(global_avg, 0.01)
    score = 55.0 + diff * 8.0 + math.log(max(0.2, ratio), 1.25) * 3.0
    return _clip(score, 20, 96)


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].strip(" ,;:-")
    return cut or text[:max_len].strip()


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _prioritize_notes(items: list[str], kind: str) -> list[str]:
    if kind == "risk":
        priority = (
            "Redaktionsfeedback",
            "Politik",
            "Aktualität",
            "Mix",
            "Top-10-Balance",
            "Video",
            "Headline",
            "BILD-Reiz",
            "Zeitfenster",
        )
    else:
        priority = (
            "Redaktionsfeedback",
            "Aktualität",
            "Politik",
            "Video",
            "Top-10-Balance",
            "BILD-Reiz",
            "Opening",
            "Headline",
            "Zeitfenster",
            "BILD-Fit",
            "Mix",
        )

    def key(item: str) -> tuple[int, str]:
        for idx, marker in enumerate(priority):
            if marker in item:
                return idx, item
        return len(priority), item

    return sorted(items, key=key)
