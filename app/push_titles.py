"""Mehrstufige lokale Push-Titel-Logik fuer Render ohne externe Abhaengigkeiten."""

from __future__ import annotations

from dataclasses import dataclass
import re

MAX_TITLE_LENGTH = 78
IDEAL_MIN_LENGTH = 28
IDEAL_MAX_LENGTH = 58
HARD_MIN_LENGTH = 18

_VIDEO_MARKERS = ("/video/", "/videos/", "-video-", " im video", "video:", "clip", "aufnahmen")
_BREAKING_MARKERS = ("eil", "breaking", "live", "exklusiv", "warnung", "enthüllt", "enthuellt")
_WEAK_PHRASES = (
    "das ist jetzt wichtig",
    "das musst du jetzt wissen",
    "das musst du wissen",
    "hier alle infos",
    "so reagiert das netz",
)
_HYPE_WORDS = ("mega", "hammer", "krass", "irre", "sensation", "wahnsinn")
_GENERIC_PRONOUNS = (" sie ", " er ", " ihr ", " ihm ", " ihnen ", " ihn ")
_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "fuer", "für",
    "mit", "ohne", "auf", "im", "in", "am", "an", "zu", "zum", "zur", "bei", "nach", "vor",
    "gilt", "jetzt", "enthüllt", "enthuellt", "exklusiv", "live",
}

_CATEGORY_PREFIXES: dict[str, str] = {
    "sport": "Sport",
    "politik": "Politik",
    "wirtschaft": "Wirtschaft",
    "unterhaltung": "Unterhaltung",
    "regional": "Regional",
    "digital": "Digital",
    "news": "News",
}


@dataclass
class TitleBrief:
    original_title: str
    cleaned_title: str
    category: str
    content_type: str
    is_breaking: bool
    subject: str
    detail: str
    focus_term: str
    hook: str
    audience_value: str


def _clean(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    compact = compact.replace(" ,", ",").replace(" .", ".")
    return compact[:180]


def infer_content_type(url: str = "", title: str = "") -> str:
    haystack = f"{url} {title}".lower()
    return "video" if any(marker in haystack for marker in _VIDEO_MARKERS) else "editorial"


def _trim_push_title(text: str, max_len: int = MAX_TITLE_LENGTH) -> str:
    compact = _clean(text).strip(" -")
    if len(compact) <= max_len:
        return compact
    shortened = compact[: max_len - 1].rsplit(" ", 1)[0].strip()
    return (shortened or compact[: max_len - 1]).rstrip(":,;.-") + "..."


def _category_prefix(category: str) -> str:
    key = (category or "news").strip().lower()
    return _CATEGORY_PREFIXES.get(key, "News")


def _strip_noise_prefixes(detail: str) -> str:
    cleaned = _clean(detail)
    cleaned = re.sub(r"^(enthüllt|enthuellt|exklusiv|live|neu|jetzt)\W+\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(MEGA|mega|Hammer|hammer|irre|krass)\W+", "", cleaned)
    cleaned = cleaned.strip(" -:!?,")
    return _clean(cleaned)


def _split_subject_detail(title: str) -> tuple[str, str]:
    compact = _clean(title)
    if ":" in compact:
        subject, detail = compact.split(":", 1)
        if len(subject.split()) <= 5 and detail.strip():
            return subject.strip(), detail.strip()
    if " - " in compact:
        subject, detail = compact.split(" - ", 1)
        if len(subject.split()) <= 5 and detail.strip():
            return subject.strip(), detail.strip()
    return "", compact


def _is_weak_subject(subject: str) -> bool:
    lowered = f" {subject.lower()} "
    return (
        not subject
        or subject.startswith("„")
        or subject.startswith('"')
        or any(pronoun in lowered for pronoun in _GENERIC_PRONOUNS)
        or len(re.findall(r"[A-Za-zÄÖÜäöüß0-9-]+", subject)) > 6
    )


def _compact_fact(detail: str) -> str:
    compact = _clean(detail)
    compact = re.sub(r"^(jetzt|nun|plötzlich|ploetzlich)\s+", "", compact, flags=re.I)
    compact = re.sub(r"^„[^“]+“:\s*", "", compact)
    compact = re.sub(r'^"[^"]+":\s*', "", compact)
    compact = re.sub(r"^(so|darum|deshalb)\s+", "", compact, flags=re.I)
    compact = re.sub(
        r"^(rechnet|plant|warnt|greift|schliesst|schließt|fordert|kritisiert|attackiert)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+)(.*)$",
        r"\2 \1\3",
        compact,
        flags=re.I,
    )
    return _clean(compact)


def _extract_focus_term(detail: str, subject: str) -> str:
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9-]+", f"{subject} {detail}")
    candidates = [token for token in tokens if token.lower() not in _STOPWORDS and len(token) > 3]
    if not candidates:
        return _clean(detail)
    return candidates[-1]


def _derive_audience_value(detail: str, focus_term: str) -> str:
    lowered = detail.lower()
    transforms = [
        (r"für diese (.+) gilt (.+)", r"Diese \1 betrifft \2"),
        (r"fuer diese (.+) gilt (.+)", r"Diese \1 betrifft \2"),
        (r"so (.+) (geht|funktioniert|läuft|laeuft) (.+)", r"So \1 \3"),
        (r"was (.+) jetzt bedeutet", r"Was \1 jetzt bedeutet"),
        (r"warum (.+)", r"Warum \1"),
    ]
    for pattern, replacement in transforms:
        if re.search(pattern, lowered):
            rewritten = re.sub(pattern, replacement, detail, flags=re.I)
            return _clean(rewritten)
    if focus_term and focus_term.lower() not in lowered:
        return _clean(f"{focus_term}: Was jetzt wichtig ist")
    return _clean(detail)


def _build_brief(title: str, category: str = "news", url: str = "") -> TitleBrief:
    original = _clean(title)
    if not original:
        raise ValueError("title is required")

    subject, detail = _split_subject_detail(original)
    detail = _strip_noise_prefixes(detail or original)
    if not detail:
        detail = original
    if _is_weak_subject(subject):
        subject = ""
    detail = _compact_fact(detail)

    content_type = infer_content_type(url, original)
    is_breaking = any(marker in original.lower() for marker in _BREAKING_MARKERS)
    focus_term = _extract_focus_term(detail, subject)
    audience_value = _derive_audience_value(detail, focus_term)
    hook = focus_term or subject or detail

    return TitleBrief(
        original_title=original,
        cleaned_title=original,
        category=(category or "news").lower(),
        content_type=content_type,
        is_breaking=is_breaking,
        subject=_clean(subject),
        detail=detail,
        focus_term=focus_term,
        hook=hook,
        audience_value=audience_value,
    )


def _dedupe_keep_order(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for text, angle in items:
        cleaned = _trim_push_title(text)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append((cleaned, angle))
    return ordered


def _generate_candidates(brief: TitleBrief) -> list[dict]:
    prefix = _category_prefix(brief.category)
    subject = brief.subject
    detail = brief.detail
    focus = brief.focus_term
    audience_value = brief.audience_value

    candidates: list[tuple[str, str]] = []
    compact_fact = _compact_fact(detail)

    if brief.content_type == "video":
        candidates.extend(
            [
                (f"Im Video: {detail}", "video"),
                (f"Video: {detail}", "video"),
            ]
        )

    if subject:
        candidates.extend(
            [
                (f"{subject}: {detail}", "direkt"),
                (f"{subject}: {audience_value}", "konsequenz"),
            ]
        )
    else:
        candidates.extend(
            [
                (compact_fact, "direkt"),
                (audience_value, "konsequenz"),
            ]
        )

    if compact_fact and compact_fact != detail:
        candidates.append((compact_fact, "fakt"))

    if focus:
        candidates.extend(
            [
                (f"{focus}: Was jetzt wichtig ist", "nutzwert"),
                (f"{focus}: Darum geht es jetzt", "erklaerer"),
            ]
        )

    if brief.is_breaking:
        candidates.extend(
            [
                (f"EIL: {detail}", "breaking"),
                (f"{subject + ': ' if subject else ''}{detail}", "breaking"),
            ]
        )

    candidates.extend(
        [
            (f"{prefix}: {detail}", "kontext"),
            (brief.original_title, "original"),
        ]
    )

    return [
        {"titel": title, "ansatz": angle, "laenge": len(title)}
        for title, angle in _dedupe_keep_order(candidates)
    ]


def _score_candidate(candidate: str, brief: TitleBrief) -> tuple[float, list[str], list[str]]:
    score = 5.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    signal_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9-]+", brief.original_title)
        if len(token) > 3 and token.lower() not in _STOPWORDS and (token[:1].isupper() or "-" in token)
    ]

    length = len(candidate)
    if IDEAL_MIN_LENGTH <= length <= IDEAL_MAX_LENGTH:
        score += 1.6
        strengths.append("liegt im kompakten Push-Laengenfenster")
    elif HARD_MIN_LENGTH <= length <= MAX_TITLE_LENGTH:
        score += 0.6
    else:
        score -= 1.0
        weaknesses.append("ist zu lang oder zu knapp fuer einen Lock-Screen-Titel")

    lowered = candidate.lower()
    if brief.subject and brief.subject.lower() in lowered:
        score += 1.0
        strengths.append("haelt das zentrale Subjekt sichtbar")
    if brief.focus_term and brief.focus_term.lower() in lowered:
        score += 1.0
        strengths.append("enthaelt den wichtigsten inhaltlichen Hook")

    signal_matches = sum(1 for token in signal_tokens if token in lowered)
    if signal_matches >= 2:
        score += min(1.5, 0.5 * signal_matches)
        strengths.append("haelt mehrere konkrete Signalwoerter der Geschichte")

    if any(marker in lowered for marker in ("betrifft", "gilt", "droht", "fehlt", "kommt", "sagt", "plant", "warnt", "enthüllt", "enthuellt")):
        score += 0.8
        strengths.append("arbeitet mit einer konkreten Bewegung statt nur mit Buzzwords")

    if lowered.endswith("?"):
        score -= 0.5
        weaknesses.append("wirkt als Frage schwächer und weniger konkret")

    if candidate.startswith("„") or candidate.startswith('"'):
        score -= 1.8
        weaknesses.append("beginnt mit einem anonymen Zitat statt mit der eigentlichen Nachricht")

    if any(pronoun in f" {lowered} " for pronoun in _GENERIC_PRONOUNS):
        score -= 1.2
        weaknesses.append("enthaelt ein Pronomen ohne klaren Bezug")

    if any(phrase in lowered for phrase in _WEAK_PHRASES):
        score -= 1.1
        weaknesses.append("enthaelt zu viel generischen Nutzwert-Text")
        if signal_matches < 2:
            score -= 0.8
            weaknesses.append("verliert gegenueber dem Original zu viel konkrete Information")

    if sum(word in lowered for word in _HYPE_WORDS) >= 1:
        score -= 0.7
        weaknesses.append("ist sprachlich zu aufgeheizt")

    if candidate.count("!") > 1:
        score -= 0.6
        weaknesses.append("setzt zu stark auf Ausrufezeichen")

    if brief.content_type == "video" and "video" in lowered:
        score += 0.8
        strengths.append("kennzeichnet den Video-Charakter sichtbar")

    if brief.category == "sport" and any(token in lowered for token in ("bvb", "fc", "tor", "trainer", "klubs", "klausel", "transfer")):
        score += 0.7
        strengths.append("transportiert einen sporttypischen Trigger")

    if re.match(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+ .*?(rechnet|plant|schliesst|schließt|greift|betrifft|gilt|warnt|droht|kommt|fordert)\b", candidate):
        score += 1.1
        strengths.append("startet mit Akteur und klarer Aktion")

    category_prefix = f"{_category_prefix(brief.category).lower()}:"
    if lowered.startswith(category_prefix):
        score -= 0.4
        weaknesses.append("nutzt nur einen Ressort-Prefix statt direkt mit der Nachricht zu starten")

    score += max(-0.6, 0.6 - abs(((IDEAL_MIN_LENGTH + IDEAL_MAX_LENGTH) / 2) - length) / 20)
    return round(max(0.0, min(score, 10.0)), 1), strengths, weaknesses


def _select_candidates(brief: TitleBrief, candidates: list[dict]) -> tuple[list[dict], dict, dict]:
    rated: list[dict] = []
    for candidate in candidates:
        score, strengths, weaknesses = _score_candidate(candidate["titel"], brief)
        rated.append(
            {
                "titel": candidate["titel"],
                "ansatz": candidate["ansatz"],
                "laenge": len(candidate["titel"]),
                "gesamt": score,
                "staerken": strengths,
                "schwaeche": weaknesses[0] if weaknesses else "",
            }
        )

    rated.sort(key=lambda item: (-item["gesamt"], item["laenge"]))
    winner = rated[0] if rated else {
        "titel": brief.original_title,
        "laenge": len(brief.original_title),
        "gesamt": 5.0,
        "staerken": [],
        "schwaeche": "",
        "ansatz": "fallback",
    }
    alternative = rated[1] if len(rated) > 1 else winner

    winner_reason = winner["staerken"][0] if winner.get("staerken") else "liefert die klarste, kompakteste Version des Themas"
    alt_reason = alternative["schwaeche"] or "setzt einen anderen Schwerpunkt"

    gewinner = {
        "titel": winner["titel"],
        "laenge": winner["laenge"],
        "gesamt_score": winner["gesamt"],
        "warum_dieser": winner_reason,
    }
    alternative_payload = {
        "titel": alternative["titel"],
        "laenge": alternative["laenge"],
        "warum": alt_reason,
    }
    return rated[:5], gewinner, alternative_payload


def build_push_title_suggestions(title: str, category: str = "news", url: str = "") -> dict:
    brief = _build_brief(title, category, url)
    candidates = _generate_candidates(brief)
    rated, winner, alternative = _select_candidates(brief, candidates)
    alternative_titles = [
        candidate["titel"]
        for candidate in rated
        if candidate["titel"] != winner["titel"]
    ][:4]

    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["ansatz"], []).append(
            {"titel": candidate["titel"], "laenge": candidate["laenge"]}
        )

    reasoning = winner["warum_dieser"]
    return {
        "title": winner["titel"],
        "alternativeTitles": alternative_titles,
        "reasoning": reasoning,
        "advisoryOnly": True,
        "contentType": brief.content_type,
        "gewinner": winner,
        "alternative": alternative,
        "alle_kandidaten": grouped,
        "bewertungen": rated,
        "meta": {
            "content_type": brief.content_type,
            "analyse": {
                "kern": brief.detail,
                "hook": brief.hook,
                "emotion": "dringlich" if brief.is_breaking else "konkret-direkt",
                "leserwert": brief.audience_value,
            },
            "anzahl_kandidaten": len(candidates),
            "dauer_gesamt_s": 0.0,
            "dauer_call1_s": 0.0,
            "dauer_call2_s": 0.0,
            "modell": "local-editorial-chain",
            "modus": "local-editorial-chain",
        },
    }
