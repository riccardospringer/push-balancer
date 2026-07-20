"""Offline editorial stress test built from synthetic reader situations.

The panel represents combinations of editorial interests, attention states, and
usage motives. It does not represent observed people, estimate opening rate, read
production data, or participate in the Teams production decision path.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import product
from typing import Any

from app.scoring.editorial import (
    assess_germany_relevance,
    is_german_public_figure_parenthood_story,
)


PANEL_VERSION = "synthetic-reader-modes-v2"

FOCUS_MODES: dict[str, frozenset[str]] = {
    "national": frozenset({"national", "politics", "consumer", "safety", "weather", "mobility"}),
    "consumer": frozenset({"consumer", "utility", "national", "digital"}),
    "safety": frozenset({"safety", "weather", "crime", "mobility"}),
    "politics": frozenset({"politics", "national", "world"}),
    "regional": frozenset({"regional", "national", "crime", "weather"}),
    "sport_live": frozenset({"sport_live", "sport", "breaking"}),
    "sport_transfer": frozenset({"sport_transfer", "sport"}),
    "entertainment": frozenset({"entertainment", "emotion"}),
    "crime": frozenset({"crime", "safety", "regional"}),
    "digital": frozenset({"digital", "utility", "consumer"}),
    "weather_mobility": frozenset({"weather", "mobility", "safety", "national"}),
    "world": frozenset({"world", "breaking", "politics"}),
}

ATTENTION_MODES = (
    "essential_only",
    "quick_scan",
    "curiosity_open",
    "context_seeking",
)

MOTIVATION_MODES = (
    "direct_consequence",
    "urgency",
    "emotion",
)

PANEL_LIMITATIONS = (
    "Synthetische Situationszellen, keine beobachteten oder rekrutierten Nutzer.",
    "Der simulierte Interessenindex ist keine Opening Rate und keine Klickprognose.",
    "Keine individuelle Personalisierung, kein Nutzerprofil und keine Produktionsfreigabe.",
    "Lehren sind qualitative Shadow-QA-Hinweise und brauchen redaktionelle sowie Datenschutz-Freigabe.",
)

_CONSUMER_RE = re.compile(
    r"(?i)\b(rente|steuer|miete|preise|kosten|geld|krankenkasse|strom|gas|"
    r"verbraucher|kunden|schufa|filialen|insolvenz|rueckruf|rückruf)\b"
)
_SAFETY_RE = re.compile(
    r"(?i)\b(warnung|gefahr|evakuierung|keime|trinkwasser|brand|explosion|"
    r"anschlag|terror|gift|vermisst|notruf|alarm)\b"
)
_POLITICS_RE = re.compile(
    r"(?i)\b(bundestag|bundesrat|regierung|minister|kanzler|gesetz|wahl|"
    r"koalition|eu|nato|sanktionen)\b"
)
_SPORT_RE = re.compile(
    r"(?i)\b(bundesliga|dfb|nationalmannschaft|finale|trainer|spieler|verein|"
    r"elfmeter|tor|sieg|niederlage|transfer|wechsel|ausfall)\b"
)
_SPORT_LIVE_RE = re.compile(
    r"(?i)\b(finale|elfmeter|tor|sieg|niederlage|abpfiff|halbzeit|ergebnis|"
    r"live|gewinnt|verliert|erreicht)\b"
)
_SPORT_TRANSFER_RE = re.compile(r"(?i)\b(transfer|wechsel|wechselt|vertrag|verpflichtet|leihe)\b")
_ENTERTAINMENT_RE = re.compile(
    r"(?i)\b(tv-star|schauspieler|saenger|sänger|promi|trennung|hochzeit|"
    r"scheidung|show|moderator|royal)\b"
)
_CRIME_RE = re.compile(
    r"(?i)\b(polizei|gericht|prozess|festnahme|razzia|mord|messer|angriff|"
    r"taeter|täter|betrug|staatsanwalt)\b"
)
_DIGITAL_RE = re.compile(
    r"(?i)\b(ki|app|smartphone|handy|windows|software|daten|internet|"
    r"cyber|whatsapp|google|apple)\b"
)
_WEATHER_RE = re.compile(
    r"(?i)\b(wetter|hitze|unwetter|hochwasser|sturm|gewitter|schnee|" r"regen|trinkwasser)\b"
)
_MOBILITY_RE = re.compile(
    r"(?i)\b(bahn|zug|verkehr|flughafen|streik|stau|pendler|autobahn|" r"ausfall|bahnhof)\b"
)
_UTILITY_RE = re.compile(
    r"(?i)\b(was sich aendert|was sich ändert|das bedeutet|so viel|"
    r"frist|regel|warnung|preise|kosten|rente|steuer|kunden|verbraucher)\b"
)
_EMOTION_RE = re.compile(
    r"(?i)\b(baby|kind|kleinkind|mutter|vater|mama|mamas|papa|papas|eltern|"
    r"familie|stirbt|tot|trennung|"
    r"hochzeit|traenen|tränen|drama|streit)\b"
)
_TRAGEDY_RE = re.compile(
    r"(?i)\b(stirbt|gestorben|tot|todesfall|toedlich|tödlich|baby|kleinkind)\b"
)
_BROAD_SCOPE_RE = re.compile(
    r"(?i)\b(deutschland|bundesweit|millionen|viele|mehrere|landesweit|"
    r"pendler|verbraucher|kunden|patienten)\b"
)
_VAGUE_RE = re.compile(
    r"(?i)\b(dieses detail|das steckt dahinter|darum geht es|muessen sie kennen|"
    r"müssen sie kennen|so reagiert|was dann passiert|dieser trick)\b"
)
_CLICKBAIT_RE = re.compile(
    r"(?i)\b(irre|krass|unglaublich|unfassbar|schock|wahnsinn|netz rastet aus)\b"
)
_SPECULATIVE_RE = re.compile(
    r"(?i)\b(prueft|prüft|moeglich|möglich|koennte|könnte|soll angeblich|"
    r"denkt ueber|denkt über|plant wohl|transfer-pruefung|transfer-prüfung)\b"
)
_BREAKING_MARKER_RE = re.compile(r"(?i)\b(eilmeldung|breaking)\b")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _category(candidate: dict[str, Any]) -> str:
    raw = str(candidate.get("category") or candidate.get("cat") or "news").casefold()
    return {
        "geld": "wirtschaft",
        "inland": "news",
        "ausland": "news",
        "fussball": "sport",
        "fußball": "sport",
    }.get(raw, raw)


def _strict_breaking(candidate: dict[str, Any], title: str) -> bool:
    structured = bool(
        candidate.get("isBreaking")
        or candidate.get("isEilmeldung")
        or candidate.get("is_eilmeldung")
    )
    return bool(structured and _BREAKING_MARKER_RE.search(title))


def _candidate_signals(candidate: dict[str, Any]) -> dict[str, Any]:
    title = str(candidate.get("title") or candidate.get("headline") or "").strip()
    url = str(candidate.get("url") or candidate.get("link") or "").strip()
    category = _category(candidate)
    breaking = _strict_breaking(candidate, title)
    relevance_candidate = dict(candidate)
    for flag in ("isBreaking", "isEilmeldung", "is_eilmeldung"):
        relevance_candidate[flag] = breaking
    relevance = assess_germany_relevance(relevance_candidate)
    people_parenthood = is_german_public_figure_parenthood_story(relevance_candidate)
    tags: set[str] = set()

    relevance_level = str(relevance.get("level") or "neutral")
    if relevance_level in {"germany_broad", "germany_domestic", "germany_people"}:
        tags.add("national")
    if relevance_level.startswith("international") or relevance_level == "usa_domestic":
        tags.add("world")
    if breaking:
        tags.add("breaking")
    if _CONSUMER_RE.search(title) or category in {"verbraucher", "wirtschaft"}:
        tags.add("consumer")
    if _SAFETY_RE.search(title):
        tags.add("safety")
    if _POLITICS_RE.search(title) or category == "politik":
        tags.add("politics")
    if category == "regional" or "/regional/" in url.casefold():
        tags.add("regional")
    if _SPORT_RE.search(title) or category == "sport":
        tags.add("sport")
    if "sport" in tags and _SPORT_LIVE_RE.search(title):
        tags.add("sport_live")
    if "sport" in tags and _SPORT_TRANSFER_RE.search(title):
        tags.add("sport_transfer")
    if _ENTERTAINMENT_RE.search(title) or category == "unterhaltung":
        tags.add("entertainment")
    if people_parenthood:
        tags.update({"people_milestone", "entertainment", "emotion"})
    if _CRIME_RE.search(title) or category == "crime":
        tags.add("crime")
    if _DIGITAL_RE.search(title) or category == "digital":
        tags.add("digital")
    if _WEATHER_RE.search(title) or category == "wetter":
        tags.add("weather")
    if _MOBILITY_RE.search(title):
        tags.add("mobility")
    if _UTILITY_RE.search(title):
        tags.add("utility")
    if _EMOTION_RE.search(title):
        tags.add("emotion")

    broad_scope = bool(_BROAD_SCOPE_RE.search(title))
    isolated_tragedy = bool(
        _TRAGEDY_RE.search(title) and not breaking and not broad_scope and "safety" not in tags
    )
    vague = bool(_VAGUE_RE.search(title))
    clickbait = bool(_CLICKBAIT_RE.search(title))
    speculative = bool(_SPECULATIVE_RE.search(title))
    clear_title = bool(18 <= len(title) <= 125 and not vague and not clickbait)
    raw_score = _clamp(float(candidate.get("score") or 0.0), 0.0, 100.0)
    return {
        "title": title,
        "category": category,
        "score": raw_score,
        "breaking": breaking,
        "relevance": relevance,
        "relevanceLevel": relevance_level,
        "tags": tags,
        "broadScope": broad_scope,
        "isolatedTragedy": isolated_tragedy,
        "vague": vague,
        "clickbait": clickbait,
        "speculative": speculative,
        "clearTitle": clear_title,
        "peopleMilestone": people_parenthood,
        "directImpact": bool(
            tags & {"consumer", "utility", "safety", "weather", "mobility"}
            or (
                relevance_level == "germany_broad"
                and bool(tags & {"politics", "regional"})
                and "sport" not in tags
            )
        ),
    }


def _focus_adjustment(focus: str, tags: set[str]) -> tuple[float, str]:
    if focus in tags:
        return 2.5, f"direkter Fokus-Treffer {focus}"
    if tags & FOCUS_MODES[focus]:
        return 0.9, f"angrenzender Fokus-Treffer {focus}"
    return -0.8, f"kein Fokus-Treffer {focus}"


def _attention_adjustment(attention: str, signals: dict[str, Any]) -> tuple[float, str]:
    tags = signals["tags"]
    if attention == "essential_only":
        if signals["breaking"] or signals["directImpact"] or "safety" in tags:
            return 1.8, "Pflichtrelevanz fuer selektive Aufmerksamkeit"
        return -2.4, "kein Pflichtsignal fuer selektive Aufmerksamkeit"
    if attention == "quick_scan":
        if signals["clearTitle"]:
            return 1.0, "schnell erfassbare konkrete Zeile"
        return -1.8, "Zeile im schnellen Scan zu vage"
    if attention == "curiosity_open":
        if tags & {"emotion", "crime", "entertainment", "sport"}:
            return 1.1, "starker situativer Neugieranker"
        return -0.3, "wenig situativer Neugieranker"
    if tags & {"politics", "world", "consumer", "digital"}:
        return 1.1, "Thema traegt eine Kontextvertiefung"
    return -0.4, "geringer Zusatzwert fuer Kontextsuche"


def _motivation_adjustment(motivation: str, signals: dict[str, Any]) -> tuple[float, str]:
    tags = signals["tags"]
    if motivation == "direct_consequence":
        if signals["directImpact"]:
            return 1.6, "direkte persoenliche Konsequenz"
        return -1.0, "keine direkte Konsequenz erkennbar"
    if motivation == "urgency":
        if signals["breaking"] or tags & {"safety", "weather", "sport_live"}:
            return 1.9, "akute oder zeitkritische Lage"
        return -0.7, "geringe zeitliche Dringlichkeit"
    if tags & {"emotion", "crime", "entertainment", "sport"}:
        return 1.2, "emotionaler Zugang vorhanden"
    return -0.5, "geringer emotionaler Zugang"


def _time_adjustment(signals: dict[str, Any], hour: int, weekday: int) -> tuple[float, str]:
    tags = signals["tags"]
    if 5 <= hour <= 9:
        if signals["isolatedTragedy"]:
            return -4.5, "isolierte Tragoedie passt nicht in den Morgen"
        if signals["directImpact"] or tags & {"weather", "mobility", "safety"}:
            return 1.2, "Morgenfit durch Nutzwert oder direkte Lage"
        if tags & {"entertainment", "world"} and not signals["breaking"]:
            return -1.4, "schwacher Morgenfit ohne direkte Konsequenz"
    elif 17 <= hour <= 21 and tags & {"sport", "entertainment", "crime"}:
        return 1.1, "passender Abendfit fuer emotionale oder Live-Themen"
    elif hour >= 22:
        if signals["breaking"] or "sport_live" in tags:
            return 1.0, "spaeter Zeitpunkt durch Zeitkritik gerechtfertigt"
        return -1.0, "Routine-Thema ist fuer spaet zu schwach"
    elif 10 <= hour <= 15 and tags & {"consumer", "national", "digital"}:
        return 0.5, "solider Tagesfit fuer Nutzwert oder Inland"

    if weekday >= 5 and tags & {"sport", "entertainment"}:
        return 0.6, "Wochenendfit fuer Sport oder Unterhaltung"
    if weekday >= 5 and "politics" in tags and not signals["breaking"]:
        return -0.4, "Routine-Politik am Wochenende braucht mehr Druck"
    return 0.0, "neutraler Zeitfit"


def _scenario_verdict(
    signals: dict[str, Any],
    *,
    focus: str,
    attention: str,
    motivation: str,
    hour: int,
    weekday: int,
) -> tuple[str, float, list[str]]:
    relevance = signals["relevance"]
    if relevance.get("hardBlock"):
        return "skip", 0.0, [str(relevance.get("reason") or "harte Relevanzsperre")]

    # Keep the Push Score influential without letting a high score make every
    # unrelated situation look interested. At 75 the synthetic baseline is 2/10.
    value = _clamp((signals["score"] - 65.0) / 5.0, 0.0, 10.0)
    reasons = [f"Push-Score-Basis {signals['score']:.1f}"]
    relevance_adjustment = {
        "germany_broad": 1.8,
        "germany_domestic": 0.8,
        "germany_people": 1.2,
        "neutral": 0.0,
        "international_breaking": -0.2,
        "international": -2.2,
    }.get(signals["relevanceLevel"], 0.0)
    value += relevance_adjustment
    if relevance_adjustment > 0:
        reasons.append("direkte Deutschland-Relevanz")
    elif relevance_adjustment < 0:
        reasons.append("Auslandsdistanz ohne direkten Deutschland-Bezug")

    for adjustment, reason in (
        _focus_adjustment(focus, signals["tags"]),
        _attention_adjustment(attention, signals),
        _motivation_adjustment(motivation, signals),
        _time_adjustment(signals, hour, weekday),
    ):
        value += adjustment
        reasons.append(reason)

    if signals["score"] < 75.0:
        value -= 2.5
        reasons.append("Push-Score unter harter Teams-Schwelle")
    if signals["relevanceLevel"] == "international" and not signals["breaking"]:
        value -= 1.0
        reasons.append("nicht zeitkritisches Auslandsthema")
    if signals["breaking"]:
        value += 1.2
        reasons.append("verifiziertes Breaking-Signal")
    if signals["peopleMilestone"]:
        value += 0.8
        reasons.append("bestaetigte positive People-Ueberraschung")
    if signals["speculative"] and not signals["breaking"]:
        value -= 3.5
        reasons.append("spekulative statt vollzogene Lage")
    if signals["vague"]:
        value -= 1.5
        reasons.append("vage Informationszusage")
    if signals["clickbait"]:
        value -= 2.0
        reasons.append("Clickbait-Risiko")
    if signals["clearTitle"]:
        value += 0.5
    value = _clamp(value, 0.0, 14.0)
    if value >= 8.0:
        return "would_open", value, reasons
    if value >= 5.25:
        return "would_consider", value, reasons
    return "skip", value, reasons


def _candidate_drivers(signals: dict[str, Any]) -> list[str]:
    tags = signals["tags"]
    drivers: list[str] = []
    if signals["relevanceLevel"] == "germany_broad":
        drivers.append("breite direkte Deutschland-Relevanz")
    elif signals["relevanceLevel"] == "germany_domestic":
        drivers.append("klarer Inlandskontext")
    elif signals["relevanceLevel"] == "germany_people":
        drivers.append("bestaetigtes Lebensereignis einer benannten deutschen oeffentlichen Person")
    if signals["score"] >= 85.0:
        drivers.append("sehr hoher Push-Score")
    if signals["directImpact"]:
        drivers.append("erkennbare direkte Konsequenz oder Nutzwert")
    if tags & {"safety", "weather", "mobility"}:
        drivers.append("akute Warn-, Sicherheits- oder Mobilitaetslage")
    if signals["breaking"]:
        drivers.append("verifiziertes Breaking-Signal")
    if signals["peopleMilestone"]:
        drivers.append("positive People-News mit konkretem Ueberraschungsmoment")
    if "sport_live" in tags or ("sport_transfer" in tags and not signals["speculative"]):
        drivers.append("materielles Sportereignis")
    if signals["clearTitle"]:
        drivers.append("konkrete, schnell erfassbare Zeile")
    return drivers[:5]


def _candidate_barriers(signals: dict[str, Any], hour: int) -> list[str]:
    barriers: list[str] = []
    if signals["relevance"].get("hardBlock"):
        barriers.append(str(signals["relevance"].get("reason") or "harte Relevanzsperre"))
    elif signals["relevanceLevel"] == "international" and not signals["breaking"]:
        barriers.append("Auslandsthema ohne unmittelbaren Deutschland-Bezug")
    if signals["score"] < 75.0:
        barriers.append("Push-Score unter 75")
    if signals["isolatedTragedy"] and 5 <= hour <= 9:
        barriers.append("isolierte Tragoedie ohne Morgen-Nutzwert")
    if signals["vague"]:
        barriers.append("vage Informationszusage")
    if signals["clickbait"]:
        barriers.append("Clickbait-Risiko")
    if signals["speculative"] and not signals["breaking"]:
        barriers.append("spekulative statt vollzogene Lage")
    if not signals["directImpact"] and not signals["breaking"] and not signals["peopleMilestone"]:
        barriers.append("keine breite direkte Konsequenz")
    return barriers[:5]


def evaluate_synthetic_reader_modes(
    candidate: dict[str, Any],
    *,
    hour: int | None = None,
    weekday: int | None = None,
) -> dict[str, Any]:
    """Evaluate one article against 144 non-personal synthetic situations."""
    signals = _candidate_signals(candidate)
    evaluation_hour = int(
        hour if hour is not None else candidate.get("recommendedHour", candidate.get("hour", 12))
    )
    evaluation_hour = int(_clamp(evaluation_hour, 0, 23))
    evaluation_weekday = int(weekday if weekday is not None else candidate.get("weekday", 2))
    evaluation_weekday = int(_clamp(evaluation_weekday, 0, 6))

    verdicts: Counter[str] = Counter()
    focus_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    score_total = 0.0
    for focus, attention, motivation in product(
        FOCUS_MODES,
        ATTENTION_MODES,
        MOTIVATION_MODES,
    ):
        verdict, scenario_score, _reasons = _scenario_verdict(
            signals,
            focus=focus,
            attention=attention,
            motivation=motivation,
            hour=evaluation_hour,
            weekday=evaluation_weekday,
        )
        verdicts[verdict] += 1
        focus_verdicts[focus][verdict] += 1
        score_total += scenario_score

    scenario_count = len(FOCUS_MODES) * len(ATTENTION_MODES) * len(MOTIVATION_MODES)
    open_cells = int(verdicts["would_open"])
    consider_cells = int(verdicts["would_consider"])
    skip_cells = int(verdicts["skip"])
    interest_index = round(
        100.0 * (open_cells + 0.45 * consider_cells) / max(1, scenario_count),
        1,
    )
    if open_cells / scenario_count >= 0.40 and skip_cells / scenario_count <= 0.35:
        editorial_band = "broad_support"
    elif interest_index >= 42.0:
        editorial_band = "selective_support"
    else:
        editorial_band = "weak_support"

    focus_summary = []
    for focus, counts in focus_verdicts.items():
        total = sum(counts.values())
        focus_summary.append(
            {
                "focus": focus,
                "wouldOpenCells": int(counts["would_open"]),
                "wouldConsiderCells": int(counts["would_consider"]),
                "wouldSkipCells": int(counts["skip"]),
                "syntheticInterestIndex": round(
                    100.0
                    * (counts["would_open"] + 0.45 * counts["would_consider"])
                    / max(1, total),
                    1,
                ),
            }
        )
    focus_summary.sort(
        key=lambda item: (
            float(item["syntheticInterestIndex"]),
            int(item["wouldOpenCells"]),
            str(item["focus"]),
        ),
        reverse=True,
    )

    baseline_recommended = candidate.get("baselineRecommended")
    if baseline_recommended is True and editorial_band == "weak_support":
        baseline_comparison = "challenge_legacy_recommendation"
    elif baseline_recommended is False and editorial_band == "broad_support":
        baseline_comparison = "synthetic_missed_opportunity"
    elif baseline_recommended in {True, False}:
        baseline_comparison = "broadly_aligned_or_review"
    else:
        baseline_comparison = "no_baseline"

    return {
        "panelVersion": PANEL_VERSION,
        "studyId": str(candidate.get("studyId") or candidate.get("id") or "synthetic-case"),
        "title": signals["title"],
        "category": signals["category"],
        "pushScore": round(signals["score"], 1),
        "recommendedHour": evaluation_hour,
        "weekday": evaluation_weekday,
        "scenarioCount": scenario_count,
        "wouldOpenCells": open_cells,
        "wouldConsiderCells": consider_cells,
        "wouldSkipCells": skip_cells,
        "syntheticInterestIndex": interest_index,
        "averageScenarioScore": round(score_total / max(1, scenario_count), 2),
        "editorialBand": editorial_band,
        "baselineRecommended": baseline_recommended,
        "baselineComparison": baseline_comparison,
        "germanyRelevance": signals["relevanceLevel"],
        "drivers": _candidate_drivers(signals),
        "barriers": _candidate_barriers(signals, evaluation_hour),
        "focusBreakdown": focus_summary,
        "strongestFocusModes": focus_summary[:3],
        "weakestFocusModes": list(reversed(focus_summary[-3:])),
        "signalTags": sorted(signals["tags"]),
        "shadowOnly": True,
        "productionUseAllowed": False,
        "representsObservedUsers": False,
        "canEstimateOpeningRate": False,
        "notOpeningRate": True,
        "limitations": list(PANEL_LIMITATIONS),
    }


def _derive_study_lessons(results: list[dict[str, Any]]) -> list[str]:
    lessons: list[str] = []
    broad_germany = [item for item in results if item["germanyRelevance"] == "germany_broad"]
    international = [item for item in results if item["germanyRelevance"] == "international"]
    if broad_germany and international:
        german_average = sum(item["syntheticInterestIndex"] for item in broad_germany) / len(
            broad_germany
        )
        international_average = sum(item["syntheticInterestIndex"] for item in international) / len(
            international
        )
        if german_average > international_average:
            lessons.append(
                "Direkte Deutschland-Relevanz traegt synthetisch breiter als nicht zeitkritisches Ausland."
            )
    if any(item["germanyRelevance"] == "usa_domestic" for item in results):
        lessons.append(
            "Ein hoher Push-Score darf reine US-Inlands-People-/Crime-Stoffe nicht retten."
        )
    if any(item["germanyRelevance"] == "germany_people" for item in results):
        lessons.append(
            "Bestaetigte positive People-Ereignisse benannter deutscher oeffentlicher Personen koennen "
            "ohne Alarmismus breite Neugier tragen."
        )
    if any("isolierte Tragoedie ohne Morgen-Nutzwert" in item["barriers"] for item in results):
        lessons.append(
            "Isolierte Tragoedien am Morgen brauchen einen akuten Warn- oder Handlungsnutzen."
        )
    if any(
        "sport_live" in item["signalTags"] and item["recommendedHour"] >= 17 for item in results
    ):
        lessons.append(
            "Materielle Sportereignisse gewinnen vor allem im zeitnahen Abend-/Live-Kontext."
        )
    if any("vage Informationszusage" in item["barriers"] for item in results):
        lessons.append("Ein hoher Score kompensiert keine vage Informationszusage im Push-Titel.")
    if any(item["baselineComparison"] == "synthetic_missed_opportunity" for item in results):
        lessons.append(
            "Breite nationale Warn- und Nutzwertlagen sollten im Kandidatenfeld frueher sichtbar werden."
        )
    lessons.append(
        "Die Simulation liefert keine reale OR; jede Produktionsaenderung braucht echte aggregierte Evidenz."
    )
    return lessons


def run_synthetic_reader_panel_study(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a deterministic shadow study without reading or persisting any data."""
    results = [evaluate_synthetic_reader_modes(candidate) for candidate in candidates]
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["syntheticInterestIndex"]),
            float(item["pushScore"]),
            str(item["studyId"]),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["syntheticRank"] = rank
    return {
        "panelVersion": PANEL_VERSION,
        "studyType": "synthetic_editorial_shadow_qa",
        "candidateCount": len(results),
        "scenarioCellsPerCandidate": (
            len(FOCUS_MODES) * len(ATTENTION_MODES) * len(MOTIVATION_MODES)
        ),
        "totalScenarioDecisions": sum(item["scenarioCount"] for item in results),
        "results": ranked,
        "lessons": _derive_study_lessons(ranked),
        "shadowOnly": True,
        "productionUseAllowed": False,
        "representsObservedUsers": False,
        "canEstimateOpeningRate": False,
        "limitations": list(PANEL_LIMITATIONS),
    }


def render_synthetic_reader_panel_markdown(study: dict[str, Any]) -> str:
    """Render a compact, explicitly synthetic study report."""
    lines = [
        "# Synthetisches Reader-Mode-Panel",
        "",
        "> Shadow-QA mit synthetischen Situationszellen. Keine echten BILD-Nutzer,",
        "> keine Opening Rate, keine Produktionsfreigabe.",
        "",
        f"- Panel-Version: `{study.get('panelVersion')}`",
        f"- Synthetische Artikelfaelle: {int(study.get('candidateCount') or 0)}",
        f"- Zellen je Fall: {int(study.get('scenarioCellsPerCandidate') or 0)}",
        f"- Ausgewertete Szenarioentscheidungen: {int(study.get('totalScenarioDecisions') or 0)}",
        "",
        "| Rang | Synthetischer Fall | Zeit | Push-Score | Oeffnen | Pruefen | Ablehnen | Index* | Urteil | Legacy-Vergleich |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in study.get("results") or []:
        title = str(item.get("title") or "").replace("|", "/")
        lines.append(
            "| {rank} | {title} | {hour:02d}:00 | {score:.1f} | {open_cells} | "
            "{consider_cells} | {skip_cells} | {index:.1f} | {band} | {comparison} |".format(
                rank=int(item.get("syntheticRank") or 0),
                title=title,
                hour=int(item.get("recommendedHour") or 0),
                score=float(item.get("pushScore") or 0.0),
                open_cells=int(item.get("wouldOpenCells") or 0),
                consider_cells=int(item.get("wouldConsiderCells") or 0),
                skip_cells=int(item.get("wouldSkipCells") or 0),
                index=float(item.get("syntheticInterestIndex") or 0.0),
                band=str(item.get("editorialBand") or ""),
                comparison=str(item.get("baselineComparison") or ""),
            )
        )
    lines.extend(
        [
            "",
            "Hinweis: Der synthetische Interessenindex ist nur eine interne Szenarioabdeckung, keine OR.",
            "",
            "## Qualitative Lehren",
            "",
            *[f"- {lesson}" for lesson in study.get("lessons") or []],
            "",
            "## Grenzen",
            "",
            *[f"- {item}" for item in study.get("limitations") or []],
            "",
        ]
    )
    return "\n".join(lines)
