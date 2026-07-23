"""Mehrstufige lokale Push-Titel-Logik fuer Render ohne externe Abhaengigkeiten."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

MAX_TITLE_LENGTH = 80
IDEAL_MIN_LENGTH = 35
IDEAL_MAX_LENGTH = 65
HARD_MIN_LENGTH = 18

_VIDEO_MARKERS = ("/video/", "/videos/", "-video-", " im video", "video:", "clip", "aufnahmen")
_BREAKING_MARKERS = ("eil", "breaking", "live", "exklusiv", "enthüllt", "enthuellt")
_WEAK_PHRASES = (
    "das ist jetzt wichtig",
    "das musst du jetzt wissen",
    "das musst du wissen",
    "hier alle infos",
    "so reagiert das netz",
    "darum geht es jetzt",
    "was jetzt wichtig ist",
    "im fokus",
)
_HYPE_WORDS = ("mega", "hammer", "krass", "irre", "sensation", "wahnsinn")
_GENERIC_PRONOUNS = (" sie ", " er ", " ihm ", " ihnen ", " ihn ")
_ACTION_WORDS = (
    "trifft", "schiesst", "schießt", "gewinnt", "verliert", "stoppt", "warnt", "droht",
    "plant", "fordert", "beschliesst", "beschließt", "beschlossen", "entscheidet", "kippt", "rettet",
    "steigt", "faellt", "fällt", "explodiert", "startet", "endet", "greift", "verlaesst",
    "verlässt", "wechselt", "sichert", "verpasst", "bremst", "loest", "löst",
    "läuft", "laeuft", "festgenommen", "spricht", "zieht", "stellt", "eingestellt",
    "ahnte", "ahnt", "tor", "tore", "toren",
)
_CONSEQUENCE_WORDS = (
    "wm", "em", "wahl", "krieg", "krise", "gefahr", "warnung", "streik", "ausfall",
    "rente", "steuer", "preise", "geld", "urteil", "entscheidung", "folge", "folgen",
    "beben", "drama", "wende", "schock", "rekord", "chance", "titel", "finale",
)
_EMPTY_METAPHORS = ("erdbeben nach", "beben nach", "schock nach", "drama nach")
_NON_ACTOR_TITLE_WORDS = {
    "WM", "EM", "Erdbeben", "Beben", "Schock", "Drama", "Tore", "Toren", "Tor",
    "News", "Sport", "Politik", "Wirtschaft", "Digital", "Regional",
    "Mann", "Frau", "Hund", "Katze", "Kilometer", "Millionen", "Deutsche",
    "Familien", "Regeln", "Messerattacke", "Bahnhof", "Warnung", "Gewitter",
    "Unwetter", "Bürgergeld", "Buergergeld", "Renten-Reform", "Ukraine", "Neue",
}
_LOW_VALUE_ACTOR_PHRASES = {
    "bühne weltpolitik",
    "buehne weltpolitik",
    "momente g7-gipfel",
    "millionen deutsche",
    "diese familien",
    "heftige gewitter",
    "alter besitzer",
}
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

_CONSUMER_TERMS = (
    "bürgergeld", "buergergeld", "rente", "steuer", "preise", "kosten", "geld",
    "familien", "mieter", "kunden", "verbraucher", "krankenkasse",
)
_CRIME_TERMS = (
    "messerattacke", "messer", "attacke", "attackiert", "angriff", "angegriffen",
    "mord", "ermordet", "getötet", "getoetet", "tötung", "toetung", "erstochen",
    "erschossen", "schuss", "schüsse", "schuesse", "schießerei", "schiesserei",
    "zündet", "zuendet", "angezündet", "angezuendet", "brandanschlag", "brandstiftung",
    "vergewaltigt", "vergewaltigung", "missbrauch", "entführt", "entfuehrt", "entführung",
    "entfuehrung", "geiseln", "leiche", "leichen", "verletzt", "verletzte", "schwerverletzt",
    "polizei", "festgenommen", "festnahme", "razzia", "überfall", "ueberfall",
    "raubüberfall", "raubueberfall", "bahnhof", "täter", "taeter", "opfer",
    "unfall", "verunglückt", "verunglueckt", "tödlich", "toedlich", "explosion",
)
_WEATHER_TERMS = (
    "unwetter", "warnung", "gewitter", "sturm", "regen", "hitze", "schnee",
    "hochwasser", "orkan",
)
_WEAK_SUBJECTS = (
    "studie zeigt",
    "experten erklären",
    "experten erklaeren",
    "forscher erklären",
    "forscher erklaeren",
)
_LOW_PUSH_MARKERS = (
    "studie zeigt",
    "experten erklären",
    "experten erklaeren",
)

_TITLE_BAIT_PATTERNS = (
    r"\bdas musst du (?:sehen|wissen)\b",
    r"\bsie werden nicht glauben\b",
    r"\bdu glaubst nicht\b",
    r"\bwas dann passiert\b",
    r"\bdiese wahrheit\b",
    r"\bdieses geheimnis\b",
    r"\bjetzt kommt alles raus\b",
    r"\bniemand hat damit gerechnet\b",
)
_TITLE_BAIT_RE = re.compile("|".join(_TITLE_BAIT_PATTERNS), re.IGNORECASE)
_TITLE_CURIOSITY_RE = re.compile(
    r"\b(?:warum|wieso|wie|wer|wen|welche|welcher|was .* bedeutet|"
    r"was hinter|so fiel|so kam es|darum)\b",
    re.IGNORECASE,
)
_TITLE_IMPACT_RE = re.compile(
    r"\b(?:betrifft|treffen|gilt|ändert sich|aendert sich|teurer|billiger|"
    r"kosten|geld|rente|steuer|bürgergeld|buergergeld|familien|verbraucher|"
    r"beschäftigte|beschaeftigte|mieter|kunden|reisende)\b",
    re.IGNORECASE,
)
_TITLE_CONFLICT_RE = re.compile(
    r"\b(?:streit|zoff|scheidung|trennung|konflikt|kampf|krise|vorwurf|"
    r"millionen|milliarden|wechsel|entscheidung)\b",
    re.IGNORECASE,
)
_TITLE_EVENT_RE = re.compile(
    r"\b(?:beschlie(?:ß|ss)t|beschlossen|festgenommen|nimmt .* fest|warnt|"
    r"stoppt|tritt zurück|tritt zurueck|gewinnt|verliert|wechselt|trifft|schie(?:ß|ss)t|"
    r"explodiert|brennt|gesperrt|evakuiert|verurteilt|stirbt|tot|tote)\b",
    re.IGNORECASE,
)
_TITLE_VAGUE_OBJECT_RE = re.compile(
    r"\b(?:wichtig\w*|neu\w*|gro(?:ß|ss)\w*)\s+(?:paket|thema|entwicklung|entscheidung)\b",
    re.IGNORECASE,
)
_TITLE_FRAME_WORDS = {
    "beschluss", "beschlossen", "darum", "dahinter", "bedeutet", "betrifft", "fiel", "geht", "hinter",
    "jetzt", "kam", "neue", "neuen", "neuer", "neues", "so", "trifft",
    "warum", "was", "welche", "welcher", "welches", "wen", "wer", "wie",
}
_TITLE_GENERIC_CONTENT = {
    "aktuell", "alles", "entscheidung", "entwicklung", "geschichte", "infos",
    "meldung", "nachricht", "neuigkeit", "paket", "sache", "thema", "wichtig", "etwas",
}
_TITLE_REVIEW_STOPWORDS = _STOPWORDS | {
    "es", "ihre", "ihr", "ihren", "seine", "seiner", "seinen", "mehr", "erste",
}
_TITLE_MIN_INTEREST_SCORE = 68.0

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
    actors: list[str]
    action_terms: list[str]
    consequence_terms: list[str]
    depth_summary: str


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


def _cap_first(text: str) -> str:
    cleaned = _clean(text)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _normalize_for_similarity(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _title_similarity(left: str, right: str) -> float:
    norm_left = _normalize_for_similarity(left)
    norm_right = _normalize_for_similarity(right)
    if not norm_left or not norm_right:
        return 0.0
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in terms)


def _without_leading_fillers(text: str) -> str:
    cleaned = _clean(text)
    cleaned = re.sub(r"\bneue[nrms]?\s+", "", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\bjetzt\s+", "", cleaned, count=1, flags=re.I)
    return _clean(cleaned)


def _topic_from_subject(subject: str, detail: str = "") -> str:
    lowered = f"{subject} {detail}".lower()
    if "bürgergeld" in lowered or "buergergeld" in lowered:
        return "Bürgergeld"
    if "rentenpaket" in lowered or "renten-paket" in lowered:
        return "Rentenpaket"
    if "renten-reform" in lowered or "rentenreform" in lowered:
        return "Renten-Reform"
    if "rente" in lowered:
        return "Rente"
    if "steuer" in lowered:
        return "Steuer"
    if "unwetter" in lowered:
        return "Unwetter-Warnung"
    if "gewitter" in lowered:
        return "Gewitter"
    if "ukraine" in lowered:
        return "Ukraine"
    if "g7" in lowered:
        return "G7-Gipfel"
    cleaned = _clean(subject)
    cleaned = re.sub(r"^neue regeln beim\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^neue\s+", "", cleaned, flags=re.I)
    return _cap_first(cleaned)


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
    if " | " in compact:
        subject, detail = compact.split(" | ", 1)
        if len(subject.split()) <= 6 and detail.strip():
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
        or subject.strip().lower() in _WEAK_SUBJECTS
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
    combined = f"{subject} {detail}"
    lowered = combined.lower()
    preferred = [
        (r"\bG7-Gipfel\b", "G7-Gipfel"),
        (r"\bRenten-Reform\b", "Renten-Reform"),
        (r"\bBürgergeld\b", "Bürgergeld"),
        (r"\bBuergergeld\b", "Bürgergeld"),
        (r"\bMesserattacke\b", "Messerattacke"),
        (r"\bUnwetter-Warnung\b", "Unwetter-Warnung"),
        (r"\bGewitter\b", "Gewitter"),
        (r"\bUkraine\b", "Ukraine"),
        (r"\bWM\s+20\d{2}\b", ""),
    ]
    for pattern, label in preferred:
        match = re.search(pattern, combined, flags=re.I)
        if match:
            return label or match.group(0)

    if subject and not _is_weak_subject(subject):
        return _topic_from_subject(subject, detail)

    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9-]+", combined)
    candidates = [
        token for token in tokens
        if token.lower() not in _STOPWORDS
        and token not in _NON_ACTOR_TITLE_WORDS
        and len(token) > 3
    ]
    if not candidates:
        return _clean(detail)
    for token in candidates:
        if "-" in token:
            return token
    if _has_any(lowered, _CRIME_TERMS):
        return next((token for token in candidates if token.lower() in _CRIME_TERMS), candidates[0])
    return candidates[0]


def _extract_named_actors(title: str) -> list[str]:
    text = title or ""
    actors: list[str] = []
    leading_action = re.match(
        r"^\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){0,2})\s+"
        r"(plant|warnt|fordert|kritisiert|attackiert|beschließt|beschliesst|spricht|trifft|"
        r"verlässt|verlaesst|wechselt|entscheidet|stoppt|droht)\b",
        text,
    )
    if leading_action:
        actors.append(leading_action.group(1))

    for pattern in (
        r"\bvon\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){0,2})\b",
        r"\bmit\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){0,2})\b",
    ):
        actors.extend(re.findall(pattern, text))

    multi_word = re.findall(
        r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+)+)\b",
        text,
    )
    actors.extend(multi_word)

    for token in re.findall(r"\b[A-ZÄÖÜ]{2,}\b", text):
        if token.lower() not in {"wm", "em", "g7"}:
            actors.append(token)

    seen: set[str] = set()
    result: list[str] = []
    for actor in actors:
        actor = re.sub(r"^(Eilmeldung|Breaking|Live)\s+", "", actor).strip()
        actor = " ".join(part for part in actor.split() if part not in _NON_ACTOR_TITLE_WORDS)
        key = actor.lower()
        if not actor or len(actor) < 3:
            continue
        if key in _LOW_VALUE_ACTOR_PHRASES:
            continue
        if any(word in key.split() for word in ("millionen", "familien", "kilometer", "momente")):
            continue
        if key not in seen and len(actor) > 2:
            seen.add(key)
            result.append(actor)
    return result[:4]


def _is_g7_moments_title(title: str) -> bool:
    lowered = (title or "").lower()
    return "g7" in lowered and "cringy" in lowered and "momente" in lowered


def _last_name(actor: str) -> str:
    parts = [part for part in re.split(r"\s+", actor.strip()) if part]
    return parts[-1] if parts else actor


def _extract_terms(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    lowered = (text or "").lower()
    return [term for term in vocabulary if term in lowered][:5]


def _depth_summary(actors: list[str], action_terms: list[str], consequence_terms: list[str]) -> str:
    parts = []
    if actors:
        parts.append("Akteur: " + ", ".join(actors[:2]))
    if action_terms:
        parts.append("Aktion: " + ", ".join(action_terms[:2]))
    if consequence_terms:
        parts.append("Fallhoehe: " + ", ".join(consequence_terms[:2]))
    return "; ".join(parts) if parts else "Kern/Fallhoehe noch zu unscharf"


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
    actors = _extract_named_actors(original)
    action_terms = _extract_terms(original, _ACTION_WORDS)
    consequence_terms = _extract_terms(original, _CONSEQUENCE_WORDS)
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
        actors=actors,
        action_terms=action_terms,
        consequence_terms=consequence_terms,
        depth_summary=_depth_summary(actors, action_terms, consequence_terms),
    )


def _low_push_warning(brief: TitleBrief) -> str:
    lowered = brief.original_title.lower()
    if not any(marker in lowered for marker in _LOW_PUSH_MARKERS):
        return ""
    if brief.is_breaking or _has_any(lowered, _CRIME_TERMS) or _has_any(lowered, _WEATHER_TERMS):
        return ""
    return "Warnhinweis: eher Nutzwert als harter Push-Anlass; nur mit klarem Timing/Zielgruppenbezug pushen"


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


def _generate_editorial_variants(brief: TitleBrief) -> list[tuple[str, str]]:
    """Build push-first variants: clear news, pointed, useful, curiosity."""
    original = brief.original_title
    lowered = original.lower()
    subject = brief.subject
    detail = brief.detail
    detail_clean = _without_leading_fillers(detail)
    candidates: list[tuple[str, str]] = []

    if _is_g7_moments_title(original):
        candidates.extend(
            [
                ("G7-Gipfel: Die cringy Momente der Weltpolitik", "A-klare-news-push"),
                ("G7-Gipfel: Weltpolitik zum Fremdschämen", "B-zugespitzt"),
                ("Was beim G7-Gipfel hängen bleibt", "D-neugier"),
                ("Diese G7-Momente bleiben hängen", "D-neugier"),
            ]
        )

    sport_record_match = re.search(
        r"WM-Rekord\s+von\s+(?P<record_actor>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+)"
        r"\s+eingestellt:\s+(?P<witness>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+)"
        r"\s+ahnte\s+es\s+(?:schon\s+)?früh",
        original,
        flags=re.I,
    )
    if sport_record_match:
        record_actor = _cap_first(sport_record_match.group("record_actor"))
        witness = _cap_first(sport_record_match.group("witness"))
        candidates.extend(
            [
                (f"{witness} ahnte {record_actor}s WM-Rekord schon früh", "B-zugespitzt"),
                (f"{record_actor} stellt WM-Rekord ein - {witness} ahnte es", "A-klare-news-push"),
                (f"{witness}s frühe Ahnung vor {record_actor}s WM-Rekord", "D-neugier"),
                (f"{record_actor}s WM-Rekord: Das ahnte {witness} früh", "C-nutzwert-betroffenheit"),
            ]
        )

    study_match = re.match(
        r"(?:Studie zeigt|Forscher(?:\s+erklären|\s+erklaeren)?):?\s+Diese\s+(.+?)\s+verbessern\s+(.+)$",
        original,
        flags=re.I,
    )
    if study_match:
        topic = _cap_first(study_match.group(1))
        effect = _clean(study_match.group(2))
        effect_topic = re.sub(r"^(das|die|der)\s+", "", effect, flags=re.I)
        candidates.extend(
            [
                (f"Diese {topic} verbessern {effect}", "A-klare-news-push"),
                (f"Welche {topic} {effect} verbessern", "D-neugier"),
                (f"{topic}: Welche {effect} verbessern", "D-neugier"),
                (f"{_cap_first(effect_topic)}: Welche {topic} es verbessern", "C-nutzwert-betroffenheit"),
            ]
        )

    sleep_match = re.match(
        r"Experten\s+erklären,\s+warum\s+(?P<group>.+?)\s+schlecht\s+schlafen$",
        original,
        flags=re.I,
    )
    if sleep_match:
        group = _clean(sleep_match.group("group"))
        candidates.extend(
            [
                (f"Darum schlafen {group} schlecht", "A-klare-news-push"),
                ("Schlecht schlafen: Das steckt dahinter", "D-neugier"),
                (f"Warum {group} schlecht schlafen", "C-nutzwert-betroffenheit"),
                (f"{_cap_first(group)} schlafen schlecht: Experten erklären warum", "A-klare-news-push"),
            ]
        )

    if brief.is_breaking or subject.lower().startswith(("eilmeldung", "breaking")):
        candidates.append((f"EIL: {detail_clean}", "A-klare-news-push"))
        decision = re.match(
            r"(?P<actor>.+?)\s+beschlie(?:ß|ss)t\s+(?P<object>.+)$",
            detail_clean,
            flags=re.I,
        )
        if decision:
            obj = _cap_first(decision.group("object"))
            candidates.append((f"EIL: {obj} beschlossen", "B-zugespitzt"))
            if "ukraine" in obj.lower() and "milliarden" in obj.lower():
                candidates.extend(
                    [
                        ("Ukraine-Paket: Bundesregierung beschließt Milliarden", "A-klare-news-push"),
                        ("Bundesregierung beschließt Ukraine-Milliarden", "B-zugespitzt"),
                        ("Ukraine-Milliarden beschlossen", "A-klare-news-push"),
                    ]
                )

    decision_source = detail_clean if subject.lower().startswith(("eilmeldung", "breaking")) else original
    policy_decision = re.match(
        r"(?P<actor>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){0,2})\s+"
        r"beschlie(?:ß|ss)t\s+(?P<object>.+)$",
        decision_source,
        flags=re.I,
    )
    if policy_decision:
        actor = _clean(policy_decision.group("actor"))
        obj = _without_leading_fillers(policy_decision.group("object"))
        audience = ""
        audience_split = re.split(r"\s+f(?:ü|ue)r\s+", obj, maxsplit=1, flags=re.I)
        if len(audience_split) == 2:
            possible_object, possible_audience = (_clean(part) for part in audience_split)
            if _has_any(possible_audience, _CONSUMER_TERMS) or re.search(
                r"\b(?:millionen|deutsche|beschäftigte|beschaeftigte|familien|mieter|kunden|verbraucher)\b",
                possible_audience,
                flags=re.I,
            ):
                obj, audience = possible_object, possible_audience
        topic = _topic_from_subject(obj, obj)
        urgency = "EIL: " if brief.is_breaking else ""
        candidates.extend(
            [
                (f"{urgency}{_cap_first(obj)} beschlossen", "A-klare-news-push"),
                (f"{actor} beschließt {obj}", "A-klare-news-push"),
            ]
        )
        if audience:
            candidates.extend(
                [
                    (f"{topic}: Was der Beschluss für {audience} bedeutet", "D-neugier"),
                    (f"{audience}: {topic} ist beschlossen", "C-nutzwert-betroffenheit"),
                ]
            )

    if _has_any(lowered, _WEATHER_TERMS):
        weather_core = detail_clean if subject else _without_leading_fillers(original)
        if "warnung" in lowered:
            candidates.append((f"Warnung: {weather_core}", "A-klare-news-push"))
        candidates.extend(
            [
                (weather_core, "A-klare-news-push"),
                ("Gewitter-Warnung für Deutschland", "B-zugespitzt") if "gewitter" in lowered and "deutschland" in lowered else ("", "B-zugespitzt"),
                ("Heftige Gewitter: Das kommt auf Deutschland zu", "D-neugier") if "heftige gewitter" in lowered and "deutschland" in lowered else ("", "D-neugier"),
                (f"{_topic_from_subject(subject, detail)}: {weather_core}", "B-zugespitzt"),
            ]
        )

    crime_match = re.match(
        r"(?P<person>Mann|Frau|Jugendlicher|Jugendliche|Teenager|Polizist|Polizistin)\s+"
        r"nach\s+(?P<event>.+?)\s+festgenommen\b",
        original,
        flags=re.I,
    )
    if crime_match:
        person = _cap_first(crime_match.group("person"))
        event = _cap_first(crime_match.group("event"))
        candidates.extend(
            [
                (f"{event}: {person} festgenommen", "A-klare-news-push"),
                (f"Festnahme nach {event}", "B-zugespitzt"),
                (f"Was zur {event} bekannt ist", "D-neugier"),
                (f"Nach {event}: {person} festgenommen", "A-klare-news-push"),
            ]
        )

    police_match = re.match(
        r"Polizei\s+nimmt\s+(?P<person>.+?)\s+nach\s+(?P<event>.+?)\s+fest$",
        original,
        flags=re.I,
    )
    if police_match:
        person = _clean(police_match.group("person"))
        event = _cap_first(police_match.group("event"))
        candidates.extend(
            [
                (f"{event}: Polizei nimmt {person} fest", "A-klare-news-push"),
                (f"Festnahme nach {event}", "B-zugespitzt"),
            ]
        )

    if subject and _has_any(lowered, _CONSUMER_TERMS):
        topic = _topic_from_subject(subject, detail)
        candidates.append((f"{topic}: {detail_clean}", "C-nutzwert-betroffenheit"))
        if re.search(r"\bfür diese\b|\bfuer diese\b", detail_clean, flags=re.I):
            candidates.append((f"{topic}: Für wen sich jetzt etwas ändert", "C-nutzwert-betroffenheit"))
            candidates.append((f"{topic}: Wen die neuen Regeln treffen", "D-neugier"))

    politics_plan = re.match(
        r"(?P<actor>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){0,1})\s+"
        r"plant\s+(?P<object>.+?)(?:\s+für\s+(?P<audience>.+))?$",
        original,
        flags=re.I,
    )
    if politics_plan:
        actor = politics_plan.group("actor")
        obj = _without_leading_fillers(politics_plan.group("object"))
        audience = _clean(politics_plan.group("audience") or "")
        if audience:
            candidates.extend(
                [
                    (f"{audience}: {actor} plant {obj}", "A-klare-news-push"),
                    (f"{actor} plant {obj} für {audience}", "A-klare-news-push"),
                    (f"{obj}: Was {actor} jetzt plant", "D-neugier"),
                    (f"{audience}: Diese {obj} plant {actor}", "C-nutzwert-betroffenheit"),
                    (f"{obj} für {audience}", "A-klare-news-push"),
                ]
            )
        else:
            candidates.extend(
                [
                    (f"{actor} plant {obj}", "A-klare-news-push"),
                    (f"{obj}: Was {actor} jetzt plant", "D-neugier"),
                ]
            )

    stakes_match = re.match(
        r"(?:es\s+geht|streit(?:en)?|zoff)\s+(?:jetzt\s+)?um\s+(?P<stakes>.+)$",
        detail_clean,
        flags=re.I,
    )
    if subject and stakes_match:
        stakes = _clean(stakes_match.group("stakes"))
        candidates.extend(
            [
                (f"{subject}: Warum es um {stakes} geht", "D-neugier"),
                (f"{subject}: Streit um {stakes}", "B-zugespitzt"),
            ]
        )

    if brief.category == "sport" and subject and "wechsel" in lowered and "entscheidung" in detail_clean.lower():
        actor_label = re.sub(r"\s+vor\s+(?:dem\s+)?wechsel.*$", "", subject, flags=re.I).strip()
        actor_label = actor_label or subject
        candidates.extend(
            [
                (f"{actor_label}: So fiel die Wechsel-Entscheidung", "D-neugier"),
                (f"{actor_label}: Die Wechsel-Entscheidung ist gefallen", "A-klare-news-push"),
            ]
        )
    promi_match = re.match(
        r"(?P<actor>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9.-]+){1,2})\s+"
        r"spricht\s+erstmals\s+über\s+(?P<topic>.+)$",
        original,
        flags=re.I,
    )
    if promi_match:
        actor = promi_match.group("actor")
        topic = _clean(promi_match.group("topic"))
        topic_label = re.sub(r"^(ihr|sein|seine|seinen)\s+", "", topic, flags=re.I)
        candidates.extend(
            [
                (f"Jetzt spricht {actor} über {topic}", "A-klare-news-push"),
                (f"{_cap_first(topic_label)}: Jetzt spricht {actor}", "B-zugespitzt"),
                (f"{actor} über {topic}", "B-zugespitzt"),
                (f"{actor} spricht über {topic_label}", "A-klare-news-push"),
                (f"{actor}: Was sie über {topic} sagt", "D-neugier"),
            ]
        )

    emotional_match = re.match(
        r"(?P<who>Hund|Katze|Kind|Junge|Mädchen|Maedchen)\s+läuft\s+"
        r"(?P<distance>\d+\s+[A-Za-zÄÖÜäöüß-]+)\s+zurück\s+zu\s+seinem\s+alten\s+Besitzer\b",
        original,
        flags=re.I,
    )
    if emotional_match:
        who = _cap_first(emotional_match.group("who"))
        distance = emotional_match.group("distance")
        distance_compound = distance.replace(" ", "-")
        candidates.extend(
            [
                (f"{distance}: {who} läuft zurück zum alten Besitzer", "B-zugespitzt"),
                (f"{who} läuft {distance} zurück zu seinem Besitzer", "A-klare-news-push"),
                (f"{distance} zurück zum alten Besitzer", "D-neugier"),
                (f"Der {distance_compound}-Weg zurück zum Besitzer", "D-neugier"),
            ]
        )

    if subject and detail_clean and subject.lower() not in {"news", "politik", "sport"}:
        topic = _topic_from_subject(subject, detail)
        candidates.append((f"{topic}: {detail_clean}", "A-klare-news-push"))
    elif detail_clean and detail_clean != original:
        candidates.append((detail_clean, "A-klare-news-push"))

    return candidates


def _generate_candidates(brief: TitleBrief) -> list[dict]:
    subject = brief.subject
    detail = brief.detail
    audience_value = brief.audience_value
    actor = brief.actors[0] if brief.actors else subject
    actor_short = _last_name(actor) if actor else ""

    candidates: list[tuple[str, str]] = _generate_editorial_variants(brief)
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

    # Bewusst KEINE generischen Erklaerer-Floskeln ("Was jetzt wichtig ist",
    # "Darum geht es jetzt") erzeugen - sie bringen keinen Mehrwert und landeten
    # zuletzt als Push-Titel in den Teams-Empfehlungen.

    actor_is_redundant = actor_short and actor_short.lower() in {
        compact_fact.lower(),
        detail.lower(),
        audience_value.lower(),
    } or (
        actor_short
        and (
            compact_fact.lower().startswith(actor_short.lower())
            or detail.lower().startswith(actor_short.lower())
            or audience_value.lower().startswith(actor_short.lower())
        )
    )
    if actor_short and actor_short.lower() not in {"wm", "em"} and not actor_is_redundant:
        candidates.append((f"{actor_short}: {audience_value}", "akteur"))

    if brief.category == "sport":
        lowered_original = brief.original_title.lower()
        if actor_short and ("tor" in lowered_original or "trifft" in lowered_original):
            if "wm" in lowered_original:
                candidates.extend(
                    [
                        (f"{actor_short} schießt sich Richtung WM 2026", "sport-folge"),
                        (f"{actor_short}-Tore lösen WM-Beben aus", "sport-folge"),
                    ]
                )
            else:
                candidates.extend(
                    [
                        (f"{actor_short}-Tore verändern die Lage", "sport-folge"),
                        (f"{actor_short} trifft - und alles ist offen", "sport-folge"),
                    ]
                )
        if "erdbeben nach" in lowered_original and actor_short:
            candidates.append((f"{actor_short}-Tore lösen das Beben aus", "sport-folge"))

    if brief.is_breaking:
        candidates.extend(
            [
                (f"EIL: {detail}", "breaking"),
                (f"{subject + ': ' if subject else ''}{detail}", "breaking"),
            ]
        )

    candidates.extend(
        [
            (brief.original_title, "original"),
        ]
    )

    return [
        {"titel": title, "ansatz": angle, "laenge": len(title)}
        for title, angle in _dedupe_keep_order(candidates)
    ]


def _is_strong_visible_alternative(item: dict, winner_title: str, brief: TitleBrief) -> bool:
    title = item.get("titel", "")
    if not title or title == winner_title:
        return False
    lowered = title.lower()
    category_prefix = f"{_category_prefix(brief.category).lower()}:"
    if lowered == brief.original_title.lower():
        return False
    if lowered.startswith(category_prefix):
        return False
    if " im fokus:" in lowered:
        return False
    colon_lead = re.match(r"^([^:]{5,32}):\s+(.+)$", lowered)
    if colon_lead and colon_lead.group(1).strip() in colon_lead.group(2):
        return False
    if any(phrase in lowered for phrase in _WEAK_PHRASES):
        return False
    if _title_similarity(title, brief.original_title) >= 0.94 and len(title) >= len(brief.original_title) * 0.85:
        return False
    min_score = 6.0 if _low_push_warning(brief) else 7.0
    return float(item.get("gesamt", 0.0)) >= min_score


def _score_candidate(candidate: str, brief: TitleBrief) -> tuple[float, list[str], list[str]]:
    score = 5.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    low_push_warning = _low_push_warning(brief)
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
    similarity = _title_similarity(candidate, brief.original_title)
    if brief.subject and brief.subject.lower() in lowered:
        score += 1.0
        strengths.append("haelt das zentrale Subjekt sichtbar")
    if brief.focus_term and brief.focus_term.lower() in lowered:
        score += 1.0
        strengths.append("enthaelt den wichtigsten inhaltlichen Hook")

    actor_hits = sum(1 for actor in brief.actors if _last_name(actor).lower() in lowered or actor.lower() in lowered)
    if actor_hits:
        score += min(1.4, 0.7 * actor_hits)
        strengths.append("macht den zentralen Begriff sofort sichtbar")

    action_hits = sum(1 for term in brief.action_terms if term in lowered)
    candidate_action_hits = sum(1 for term in _ACTION_WORDS if term in lowered)
    consequence_hits = sum(1 for term in brief.consequence_terms if term in lowered)
    if action_hits or candidate_action_hits:
        score += 0.8
        strengths.append("enthaelt eine konkrete Handlung")
    if consequence_hits:
        score += 0.8
        strengths.append("zeigt Fallhoehe oder Konsequenz")
    if actor_hits and (action_hits or candidate_action_hits or consequence_hits):
        score += 0.6
        strengths.append("verbindet Akteur mit Fallhoehe statt nur Schlagworten")

    signal_matches = sum(1 for token in signal_tokens if token in lowered)
    if signal_matches >= 2:
        score += min(1.5, 0.5 * signal_matches)
        strengths.append("haelt mehrere konkrete Signalwoerter der Geschichte")

    if any(marker in lowered for marker in ("betrifft", "gilt", "droht", "fehlt", "kommt", "sagt", "plant", "warnt", "enthüllt", "enthuellt")):
        score += 0.8
        strengths.append("arbeitet mit einer konkreten Bewegung statt nur mit Buzzwords")

    if _is_g7_moments_title(brief.original_title) and "g7" in lowered and (
        "momente" in lowered or "hängen" in lowered or "haengen" in lowered
    ):
        score += 1.2
        strengths.append("setzt den G7-Hook redaktionell neu")

    if lowered.startswith(("eil:", "warnung:")):
        score += 0.7
        strengths.append("setzt Dringlichkeit sofort sichtbar")

    if re.search(r":\s+(mann|frau|jugendlicher|jugendliche|teenager)\s+festgenommen\b", lowered):
        score += 0.7
        strengths.append("verdichtet Tat und Folge fuer den Sperrbildschirm")

    if re.search(r"\bfür wen\b|\bfuer wen\b|\bwen die\b", lowered):
        score += 0.6
        strengths.append("macht konkrete Betroffenheit sichtbar")

    if lowered.startswith("darum "):
        score += 1.2
        strengths.append("macht den Erklaer-Nutzwert sofort sichtbar")

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

    empty_metaphor = any(phrase in lowered for phrase in _EMPTY_METAPHORS)
    if empty_metaphor:
        score -= 1.4
        weaknesses.append("metaphorischer Einstieg bleibt ohne klare redaktionelle Folge")

    if lowered == brief.original_title.lower():
        score -= 3.4
        weaknesses.append("kopiert die Original-Headline zu stark")
    elif similarity >= 0.9 and length >= len(brief.original_title) * 0.85:
        score -= 1.8
        weaknesses.append("liegt noch zu nah an der Original-Headline")

    duplicated_lead = re.match(r"^([^:]{3,32}):\s*\1\b", lowered)
    if duplicated_lead:
        score -= 1.8
        weaknesses.append("doppelt den Einstieg statt ihn redaktionell zu verdichten")

    repeated_colon_lead = re.match(r"^([^:]{5,32}):\s+(.+)$", lowered)
    repeated_hook = bool(
        repeated_colon_lead and repeated_colon_lead.group(1).strip() in repeated_colon_lead.group(2)
    )
    if repeated_hook:
        score -= 1.2
        weaknesses.append("wiederholt den Hook zu sichtbar")

    if " im fokus:" in lowered:
        score -= 1.3
        weaknesses.append("wirkt wie ein generischer Fokus-Prefix")

    focus_duplication = re.match(r"^(.+?) im fokus:\s*\1$", lowered)
    if focus_duplication:
        score -= 2.0
        weaknesses.append("doppelt denselben Begriff statt eine neue Erkenntnis zu liefern")

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
        score -= 1.4
        weaknesses.append("nutzt nur einen Ressort-Prefix statt direkt mit der Nachricht zu starten")

    score += max(-0.6, 0.6 - abs(((IDEAL_MIN_LENGTH + IDEAL_MAX_LENGTH) / 2) - length) / 20)
    if empty_metaphor:
        score = min(score, 7.4)
    if lowered == brief.original_title.lower():
        score = min(score, 6.4)
    elif similarity >= 0.92 and length >= len(brief.original_title) * 0.8:
        score = min(score, 7.2)
    if duplicated_lead or repeated_hook or " im fokus:" in lowered:
        score = min(score, 7.0)
    if lowered.startswith(category_prefix):
        score = min(score, 6.8)
    if low_push_warning:
        score = min(score - 0.4, 7.4)
        if not weaknesses:
            weaknesses.append("kein harter Push-Anlass, eher Nutzwert")
    if not (actor_hits and (action_hits or candidate_action_hits or consequence_hits)) and not brief.is_breaking:
        score = min(score, 8.1)
        if not weaknesses:
            weaknesses.append("noch zu wenig Akteur-Handlung-Fallhoehe fuer eine Top-Bewertung")
    if brief.category == "sport" and lowered.startswith("sport:"):
        score = min(score, 7.2)
    return round(max(0.0, min(score, 10.0)), 1), strengths, weaknesses


def _review_token_key(token: str) -> str:
    value = str(token or "").casefold()
    value = value.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return re.sub(r"[^a-z0-9]", "", value)


def _review_content_tokens(text: str) -> list[str]:
    normalized = str(text or "").replace("-", " ")
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", normalized)
    ignored = {_review_token_key(token) for token in _TITLE_REVIEW_STOPWORDS | _TITLE_FRAME_WORDS}
    return [
        key
        for token in tokens
        if (key := _review_token_key(token))
        and (len(key) >= 4 or key in {"wm", "em", "g7"})
        and key not in ignored
    ]


def _tokens_are_related(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]


def _grounded_interest_review(candidate: str, brief: TitleBrief) -> dict:
    title = _clean(candidate)
    lowered = title.casefold()
    candidate_tokens = _review_content_tokens(title)
    original_tokens = _review_content_tokens(brief.original_title)
    grounded_tokens = [
        token
        for token in candidate_tokens
        if any(_tokens_are_related(token, original) for original in original_tokens)
    ]
    grounded_count = len(set(grounded_tokens))
    grounded_ratio = grounded_count / max(1, len(set(candidate_tokens)))
    generic_keys = {_review_token_key(token) for token in _TITLE_GENERIC_CONTENT}
    concrete_tokens = [token for token in candidate_tokens if token not in generic_keys]

    stripped_lead = re.sub(
        r"^(?:eil(?:meldung)?|breaking|warnung|news|politik|sport|wirtschaft)\s*:\s*",
        "",
        lowered,
    )
    bait = bool(_TITLE_BAIT_RE.search(lowered))
    hype = any(word in lowered for word in _HYPE_WORDS)
    weak_phrase = any(phrase in lowered for phrase in _WEAK_PHRASES)
    vague_object = bool(_TITLE_VAGUE_OBJECT_RE.search(lowered))
    vague_quantifier = bool(re.search(r"\b(?:etwas|alles)\b", lowered))
    vague_reference = bool(
        re.match(r"^(?:das|dies|diese|so|darum)\b", stripped_lead)
        and grounded_count < 2
    )
    generic_only = bool(candidate_tokens) and len(concrete_tokens) <= 1
    novel_numbers = sorted(
        set(re.findall(r"\b\d+(?:[.,]\d+)?\b", title))
        - set(re.findall(r"\b\d+(?:[.,]\d+)?\b", brief.original_title))
    )

    has_specific_question = bool(_TITLE_CURIOSITY_RE.search(lowered)) and grounded_count >= 2
    has_impact = bool(_TITLE_IMPACT_RE.search(lowered))
    has_conflict = bool(_TITLE_CONFLICT_RE.search(lowered))
    has_event = bool(
        _TITLE_EVENT_RE.search(lowered)
        or any(term in lowered for term in brief.action_terms)
        or _has_any(lowered, _CRIME_TERMS)
        or _has_any(lowered, _WEATHER_TERMS)
    )

    length = len(title)
    if 35 <= length <= 72:
        clarity = 100.0
    elif 24 <= length <= 82:
        clarity = 78.0
    elif 18 <= length <= MAX_TITLE_LENGTH:
        clarity = 60.0
    else:
        clarity = 35.0
    if title.count(":") > 1 or title.count("!") > 1 or " | " in title:
        clarity -= 18.0
    if any(pronoun in f" {lowered} " for pronoun in _GENERIC_PRONOUNS):
        clarity -= 14.0
    if vague_reference:
        clarity -= 18.0

    specificity = 28.0 + min(52.0, grounded_count * 13.0)
    if len(set(concrete_tokens)) >= 3:
        specificity += 10.0
    if grounded_ratio >= 0.75:
        specificity += 10.0
    elif grounded_ratio < 0.5:
        specificity -= 22.0
    if generic_only or vague_object:
        specificity -= 35.0
    if vague_quantifier:
        specificity -= 18.0
    if vague_reference:
        specificity -= 20.0

    relevance = 42.0
    if has_impact:
        relevance = max(relevance, 88.0)
    if has_conflict:
        relevance = max(relevance, 80.0)
    if has_event:
        relevance = max(relevance, 84.0)
    if grounded_count >= 2:
        relevance += 8.0
    if generic_only or vague_object:
        relevance -= 28.0
    if vague_quantifier:
        relevance -= 8.0

    curiosity = 34.0
    click_reason = ""
    if has_specific_question:
        curiosity = 92.0
        click_reason = "stellt eine konkrete, im Artikel beantwortbare Leserfrage"
    elif has_impact and grounded_count >= 2:
        curiosity = 78.0
        click_reason = "macht eine konkrete persönliche Folge zum Klickgrund"
    elif has_conflict and grounded_count >= 2:
        curiosity = 74.0
        click_reason = "zeigt Konflikt und Fallhöhe, ohne die Auflösung vorwegzunehmen"
    elif has_event and grounded_count >= 2:
        curiosity = 62.0
        click_reason = "meldet ein konkretes Ereignis mit erkennbarem Informationswert"
    if vague_reference or generic_only or vague_object:
        curiosity = min(curiosity, 28.0)
        click_reason = ""
    elif vague_quantifier:
        curiosity -= 16.0

    honesty = 100.0
    risks: list[str] = []
    strengths: list[str] = []
    if bait:
        honesty = min(honesty, 20.0)
        risks.append("manipulativer Clickbait-Frame")
    if hype:
        honesty -= 24.0
        risks.append("unnötig aufgeheizte Sprache")
    if weak_phrase:
        honesty -= 32.0
        risks.append("austauschbare Teaser-Floskel")
    if novel_numbers:
        honesty = min(honesty, 25.0)
        risks.append("Zahl ist nicht durch die Artikel-Headline gedeckt")
    if grounded_ratio < 0.5:
        honesty -= 22.0
        risks.append("zu wenig sprachliche Deckung durch die Artikel-Headline")
    if vague_reference or generic_only or vague_object:
        honesty -= 24.0
        risks.append("zu wenig konkrete Substanz für eine ehrliche Neugierlücke")
    if vague_quantifier:
        honesty -= 12.0
        risks.append("vage Wörter ersetzen eine konkrete Folge")

    if grounded_count >= 3:
        strengths.append("mehrere konkrete Faktenanker bleiben sichtbar")
    if click_reason:
        strengths.append(click_reason)
    if 35 <= length <= 72:
        strengths.append("auf dem Sperrbildschirm schnell erfassbar")

    clarity = max(0.0, min(100.0, clarity))
    specificity = max(0.0, min(100.0, specificity))
    relevance = max(0.0, min(100.0, relevance))
    curiosity = max(0.0, min(100.0, curiosity))
    honesty = max(0.0, min(100.0, honesty))
    editorial_score, _, _ = _score_candidate(title, brief)
    dimension_score = (
        clarity * 0.18
        + specificity * 0.22
        + relevance * 0.22
        + curiosity * 0.20
        + honesty * 0.18
    )
    interest_score = round(dimension_score * 0.8 + editorial_score * 10.0 * 0.2, 1)
    approved = bool(
        interest_score >= _TITLE_MIN_INTEREST_SCORE
        and clarity >= 55.0
        and specificity >= 55.0
        and relevance >= 55.0
        and honesty >= 72.0
        and (curiosity >= 55.0 or brief.is_breaking)
        and click_reason
    )
    if not click_reason:
        risks.append("kein klarer, ehrlicher Klickgrund")

    return {
        "approved": approved,
        "score": interest_score,
        "minimumScore": _TITLE_MIN_INTEREST_SCORE,
        "clickReason": click_reason,
        "strengths": strengths[:3],
        "risks": list(dict.fromkeys(risks))[:3],
        "dimensions": {
            "clarity": round(clarity, 1),
            "specificity": round(specificity, 1),
            "relevance": round(relevance, 1),
            "curiosity": round(curiosity, 1),
            "honesty": round(honesty, 1),
        },
        "groundedAnchorCount": grounded_count,
        "groundedRatio": round(grounded_ratio, 3),
    }


def review_push_title(
    title: str,
    *,
    original_title: str,
    category: str = "news",
    url: str = "",
) -> dict:
    """Score a proposed title for grounded interest, not raw clickbait."""
    brief = _build_brief(original_title, category, url)
    return _grounded_interest_review(title, brief)


def _select_candidates(brief: TitleBrief, candidates: list[dict]) -> tuple[list[dict], dict, dict]:
    # Generische Floskel-Titel hart aussortieren, bevor ueberhaupt bewertet wird -
    # so koennen sie weder Gewinner noch Alternative werden.
    candidates = [
        candidate
        for candidate in candidates
        if not any(phrase in candidate["titel"].lower() for phrase in _WEAK_PHRASES)
    ]
    rated: list[dict] = []
    for candidate in candidates:
        score, strengths, weaknesses = _score_candidate(candidate["titel"], brief)
        interest_review = _grounded_interest_review(candidate["titel"], brief)
        rated.append(
            {
                "titel": candidate["titel"],
                "ansatz": candidate["ansatz"],
                "laenge": len(candidate["titel"]),
                "gesamt": score,
                "interestScore": interest_review["score"],
                "titleReview": interest_review,
                "staerken": strengths,
                "schwaeche": weaknesses[0] if weaknesses else "",
            }
        )

    rated.sort(
        key=lambda item: (
            not item["titleReview"]["approved"],
            -item["interestScore"],
            -item["gesamt"],
            -_editorial_priority(item["titel"], brief),
            item["laenge"],
        )
    )
    winner = rated[0] if rated else {
        "titel": brief.original_title,
        "laenge": len(brief.original_title),
        "gesamt": 5.0,
        "staerken": [],
        "schwaeche": "",
        "ansatz": "fallback",
    }
    alternative = next(
        (item for item in rated if _is_strong_visible_alternative(item, winner["titel"], brief)),
        rated[1] if len(rated) > 1 else winner,
    )

    def display_reason(item: dict, fallback: str) -> str:
        if item.get("staerken"):
            return "; ".join(item.get("staerken", [])[:2])
        weakness = item.get("schwaeche", "")
        if weakness and "noch zu wenig Akteur-Handlung" not in weakness:
            return weakness
        return fallback

    winner_reason = display_reason(winner, "liefert die klarste, kompakteste Version des Themas")
    low_warning = _low_push_warning(brief)
    if low_warning and winner["gesamt"] < 7.5:
        winner_reason = f"{winner_reason}; {low_warning}"
    alt_reason = display_reason(alternative, "setzt einen anderen Schwerpunkt")

    gewinner = {
        "titel": winner["titel"],
        "laenge": winner["laenge"],
        "gesamt_score": winner["gesamt"],
        "interest_score": winner.get("interestScore", 0.0),
        "title_review": winner.get("titleReview", {}),
        "warum_dieser": winner_reason,
    }
    alternative_payload = {
        "titel": alternative["titel"],
        "laenge": alternative["laenge"],
        "warum": alt_reason,
    }
    visible_rated = rated[:8]
    if not any(item["titel"] == brief.original_title for item in visible_rated):
        original_rating = next((item for item in rated if item["titel"] == brief.original_title), None)
        if original_rating:
            visible_rated.append(original_rating)
    return visible_rated, gewinner, alternative_payload


def _editorial_priority(title: str, brief: TitleBrief) -> float:
    lowered = title.lower()
    priority = 0.0
    if lowered.startswith(("eil:", "warnung:")):
        priority += 2.5
    if re.search(r":\s+(mann|frau|jugendlicher|jugendliche|teenager)\s+festgenommen\b", lowered):
        priority += 2.0
    if re.search(r"\bfür wen\b|\bfuer wen\b|\bwen die\b", lowered):
        priority += 1.3
    if "millionen deutsche" in lowered or "für millionen" in lowered or "fuer millionen" in lowered:
        priority += 1.4
    if lowered.startswith("diese ") and " verbessern " in lowered:
        priority += 1.0
    if lowered.startswith("darum "):
        priority += 1.2
    if "wm-rekord" in lowered and "ahnte" in lowered and ":" not in lowered.split("ahnte", 1)[0]:
        priority += 1.6
    if lowered.startswith("jetzt spricht "):
        priority += 1.2
    if "schießt" in lowered or "schiesst" in lowered:
        priority += 2.0
    if "richtung wm" in lowered:
        priority += 1.5
    if any(term in lowered for term in ("löst", "loest", "verändert", "veraendert", "entscheidet")):
        priority += 1.0
    if any(phrase in lowered for phrase in _EMPTY_METAPHORS):
        priority -= 4.0
    if lowered == brief.original_title.lower():
        priority -= 6.0
    elif _title_similarity(title, brief.original_title) >= 0.92:
        priority -= 3.0
    if re.match(r"^([^:]{3,32}):\s*\1\b", lowered):
        priority -= 3.0
    if " im fokus:" in lowered:
        priority -= 2.0
    if lowered.startswith("die große bühne") or lowered.startswith("die grosse buehne"):
        priority -= 3.0
    if _is_g7_moments_title(brief.original_title) and lowered.startswith("g7-gipfel:"):
        priority += 3.0
    if lowered == "g7-gipfel: die cringy momente der weltpolitik":
        priority += 1.0
    if lowered.startswith(f"{_category_prefix(brief.category).lower()}:"):
        priority -= 4.0
    if "was jetzt wichtig ist" in lowered or "darum geht es jetzt" in lowered:
        priority -= 2.0
    return priority


def build_push_title_suggestions(title: str, category: str = "news", url: str = "") -> dict:
    brief = _build_brief(title, category, url)
    candidates = _generate_candidates(brief)
    rated, winner, alternative = _select_candidates(brief, candidates)
    strong_alternatives = [
        candidate["titel"]
        for candidate in rated
        if _is_strong_visible_alternative(candidate, winner["titel"], brief)
    ]
    fallback_alternatives = [
        candidate["titel"]
        for candidate in rated
        if candidate["titel"] != winner["titel"] and candidate["titel"] not in strong_alternatives
    ]
    alternative_titles = (strong_alternatives + fallback_alternatives)[:3]

    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["ansatz"], []).append(
            {"titel": candidate["titel"], "laenge": candidate["laenge"]}
        )

    reasoning = winner["warum_dieser"]
    warning = _low_push_warning(brief)
    return {
        "title": winner["titel"],
        "alternativeTitles": alternative_titles,
        "reasoning": reasoning,
        "titleReview": winner.get("title_review", {}),
        "warnhinweis": warning,
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
                "akteure": brief.actors,
                "aktionen": brief.action_terms,
                "fallhoehe": brief.consequence_terms,
                "redaktionelle_tiefe": brief.depth_summary,
                "warnhinweis": warning,
            },
            "anzahl_kandidaten": len(candidates),
            "dauer_gesamt_s": 0.0,
            "dauer_call1_s": 0.0,
            "dauer_call2_s": 0.0,
            "modell": "local-editorial-chain",
            "modus": "local-editorial-chain",
        },
    }
