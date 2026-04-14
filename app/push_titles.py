"""Deterministische Push-Titel-Vorschlaege ohne externe Abhaengigkeiten."""

from __future__ import annotations

import re

MAX_TITLE_LENGTH = 78

_VIDEO_MARKERS = ("/video/", "/videos/", "-video-", " im video", "video:", "clip", "aufnahmen")

_CATEGORY_PREFIXES: dict[str, str] = {
    "sport": "Sport",
    "politik": "Politik",
    "wirtschaft": "Wirtschaft",
    "unterhaltung": "Unterhaltung",
    "regional": "Regional",
    "digital": "Digital",
    "news": "News",
}


def _clean(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    compact = compact.replace(" ,", ",").replace(" .", ".")
    return compact[:140]


def infer_content_type(url: str = "", title: str = "") -> str:
    haystack = f"{url} {title}".lower()
    return "video" if any(marker in haystack for marker in _VIDEO_MARKERS) else "editorial"


def _trim_push_title(text: str, max_len: int = MAX_TITLE_LENGTH) -> str:
    compact = _clean(text)
    if len(compact) <= max_len:
        return compact
    shortened = compact[: max_len - 1].rsplit(" ", 1)[0].strip()
    return (shortened or compact[: max_len - 1]).rstrip(":,;.-") + "..."


def _short_core(title: str) -> str:
    words = _clean(title).split()
    if len(words) <= 8:
        return " ".join(words)
    return " ".join(words[:8])


def _is_breaking(title: str) -> bool:
    upper = title.upper()
    return any(marker in upper for marker in ("EIL", "BREAKING", "LIVE", "EXKLUSIV", "WARNUNG"))


def _category_prefix(category: str) -> str:
    key = (category or "news").strip().lower()
    return _CATEGORY_PREFIXES.get(key, "News")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = _trim_push_title(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def build_push_title_suggestions(title: str, category: str = "news", url: str = "") -> dict:
    source_title = _clean(title)
    if not source_title:
        raise ValueError("title is required")

    content_type = infer_content_type(url, source_title)
    core = _short_core(source_title)
    prefix = _category_prefix(category)
    is_breaking = _is_breaking(source_title)

    candidates = [
        source_title,
        f"{core}: Das ist jetzt wichtig",
        f"{core}: Das musst du jetzt wissen",
        f"{prefix}: {core}",
    ]

    if is_breaking:
        candidates.insert(0, f"EIL: {core}")

    if content_type == "video":
        candidates = [
            f"Im Video: {core}",
            f"{core}: Die Szenen im Video",
            f"Video: {core}",
            *candidates,
        ]

    alternatives = _dedupe_keep_order(candidates)
    winner = alternatives[0]
    rest = alternatives[1:5]

    if content_type == "video":
        reasoning = "Lokaler Vorschlag mit sichtbarem Video-Kontext und kurzer, pushbarer Formulierung."
    elif is_breaking:
        reasoning = "Lokaler Vorschlag mit Breaking-Signal, kurzer Struktur und klarer Dringlichkeit."
    else:
        reasoning = "Lokaler Vorschlag mit kurzer, direkter Formulierung ohne externe KI-Abhaengigkeiten."

    all_candidates = {
        "fallback": [{"titel": candidate} for candidate in alternatives],
    }

    return {
        "title": winner,
        "alternativeTitles": rest,
        "reasoning": reasoning,
        "advisoryOnly": True,
        "contentType": content_type,
        "gewinner": {
            "titel": winner,
            "laenge": len(winner),
            "gesamt_score": 7.2 if content_type == "video" else 7.0,
            "warum_dieser": reasoning,
        },
        "alternative": {
            "titel": rest[0] if rest else winner,
            "laenge": len(rest[0]) if rest else len(winner),
            "warum": "Alternative Formulierung mit aehnlichem Fokus.",
        },
        "alle_kandidaten": all_candidates,
        "meta": {
            "content_type": content_type,
            "analyse": {
                "kern": core,
                "hook": prefix,
                "emotion": "bildstark-direkt" if content_type == "video" else "sachlich-direkt",
            },
            "anzahl_kandidaten": len(alternatives),
            "dauer_gesamt_s": 0.0,
            "dauer_call1_s": 0.0,
            "dauer_call2_s": 0.0,
            "modell": "local-fallback",
            "modus": "local-fallback",
        },
    }
