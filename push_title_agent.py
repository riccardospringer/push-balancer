#!/usr/bin/env python3
"""Push-Zeilen Generator v5 — Editorial-One-Brain (Single Call).

Ein einzelner LLM-Call erstellt Analyse, Kandidaten und Gewinnerauswahl.
"""

import os
import json
import logging
import time
import re

from app.cost_controls import allow_calls

log = logging.getLogger("push-title-agent")

DEFAULT_TITLE_MODEL = "gpt-5.6-luna"


def _resolve_title_model(configured_model: str) -> str:
    candidate = (configured_model or "").strip()
    if not candidate or candidate.lower() == "gpt-4o-mini":
        return DEFAULT_TITLE_MODEL
    return candidate


MODEL = _resolve_title_model(os.environ.get("OPENAI_TITLE_GENERATION_MODEL", ""))
MAX_PUSH_LENGTH = 100
# Interaktiver Button-Pfad: ein einzelner kurzer GPT-5.6-Call statt langer
# Reasoning-/Retry-Ketten. Hoehere alte Render-Werte werden bewusst gedeckelt.
AGENT_TIMEOUT = min(
    float(os.environ.get("OPENAI_TITLE_GENERATION_TIMEOUT_S", "8.0")),
    8.0,
)
DEFAULT_MAX_TOKENS = min(
    int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_TOKENS", "600")),
    600,
)
REASONING_EFFORT = "none"
_OPENAI_CLIENT = None
_OPENAI_CLIENT_KEY = ""


def _openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AI_API_KEY", "")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _llm_unavailable_reason() -> str:
    if not _openai_api_key():
        return "OPENAI_API_KEY/AI_API_KEY fehlt"
    if not _env_enabled("PAID_EXTERNAL_APIS_ENABLED"):
        return "PAID_EXTERNAL_APIS_ENABLED ist deaktiviert"
    if not _env_enabled("OPENAI_TITLE_GENERATION_ENABLED"):
        return "OPENAI_TITLE_GENERATION_ENABLED ist deaktiviert"
    hourly = int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR", "0") or "0")
    daily = int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY", "0") or "0")
    if hourly <= 0 or daily <= 0:
        return "OPENAI_TITLE_GENERATION Call-Budget ist 0"
    return ""


def _clean_title(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" | ", ": ")
    text = text.replace("  ", " ")
    return text[:MAX_PUSH_LENGTH].strip(" ,-")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clean_title(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _score_local_candidate(title: str) -> float:
    title = _clean_title(title)
    if not title:
        return 0.0
    length = len(title)
    score = 7.0 - min(3.0, abs(length - 72) / 12)
    if "|" in title:
        score -= 1.5
    if "," in title:
        score -= 0.3
    if "!" in title:
        score += 0.2
    if ":" in title:
        score += 0.3
    if any(word in title.lower() for word in ("live", "eilmeldung", "breaking", "warnung")):
        score += 0.4
    return round(max(3.5, min(9.2, score)), 1)


def _is_video_context(article_type: str, title: str, text: str = "") -> bool:
    haystack = f"{article_type} {title} {text}".lower()
    return any(marker in haystack for marker in ("video", "/video/", "im video", "aufnahmen", "clip"))


def _local_editorial_one_brain(
    title,
    text,
    category,
    kicker="",
    headline="",
    article_type="editorial",
):
    base_title = _clean_title(title)
    base_headline = _clean_title(headline) if headline else ""
    base_kicker = _clean_title(kicker) if kicker else ""
    base_category = (category or "news").replace("(video)", "").strip() or "news"
    try:
        from app.push_titles import build_push_title_suggestions

        local = build_push_title_suggestions(
            title=base_title,
            category=base_category,
            url="/video/" if _is_video_context(article_type, title, text) else "",
        )
        grouped = local.get("alle_kandidaten", {}) if isinstance(local, dict) else {}
        kandidaten = []
        for ansatz, items in grouped.items():
            for item in items:
                titel = item.get("titel")
                if titel:
                    kandidaten.append(
                        {
                            "titel": titel,
                            "laenge": len(titel),
                            "ansatz": ansatz,
                        }
                    )
        return {
            "analyse": (local.get("meta") or {}).get("analyse", {}),
            "kandidaten": kandidaten[:8],
            "bewertungen": local.get("bewertungen", [])[:5],
            "gewinner": _with_video_reason(local.get("gewinner", {}), article_type, title, text),
            "alternative": local.get("alternative", {}),
            "alternativeTitles": local.get("alternativeTitles", []),
            "warnhinweis": local.get("warnhinweis", ""),
        }
    except Exception as exc:
        log.warning("[PushTitle] Deep local fallback failed, using simple fallback: %s", exc)

    words = base_title.split()
    short_core = _clean_title(" ".join(words[: min(len(words), 8)])) or base_title
    is_video = _is_video_context(article_type, title, text)

    raw_candidates = [
        base_title,
        f"{base_kicker}: {base_headline or base_title}" if base_kicker else "",
        f"{base_headline}: {short_core}" if base_headline and base_headline != base_title else "",
        short_core,
        f"{short_core}: Das ist jetzt wichtig" if len(short_core) < 70 else short_core,
        f"{base_title}: Die wichtigsten Fakten" if len(base_title) < 72 else base_title,
        f"{short_core}: Das müssen Leser jetzt wissen" if len(short_core) < 62 else "",
        f"{category.title()}: {short_core}" if category else "",
    ]
    if is_video:
        raw_candidates.extend(
            [
                f"Im Video: {short_core}" if len(short_core) < 70 else "",
                f"{short_core}: Die Szenen im Video" if len(short_core) < 60 else "",
            ]
        )
    candidate_titles = _dedupe_keep_order(raw_candidates)[:6]
    if not candidate_titles:
        candidate_titles = [base_title]

    candidate_payload = []
    labels = ["sprachlich", "sprachlich", "psychologisch", "psychologisch", "datenbasiert", "datenbasiert"]
    for idx, candidate in enumerate(candidate_titles):
        candidate_payload.append(
            {
                "titel": candidate,
                "laenge": len(candidate),
                "ansatz": labels[idx] if idx < len(labels) else "fallback",
            }
        )

    scored = sorted(
        (
            {
                "titel": candidate["titel"],
                "gesamt": _score_local_candidate(candidate["titel"]),
                "schwaeche": (
                    "Lokaler Fallback ohne LLM-Feinschliff"
                    if not is_video
                    else "Lokaler Fallback fuer Video-Kontext ohne LLM-Feinschliff"
                ),
            }
            for candidate in candidate_payload
        ),
        key=lambda item: item["gesamt"],
        reverse=True,
    )
    winner = scored[0]
    alternative = scored[1] if len(scored) > 1 else scored[0]

    return {
        "analyse": {
            "kern": short_core or base_title,
            "hook": base_kicker or category,
            "emotion": "bildstark-direkt" if is_video else "sachlich-direkt",
        },
        "kandidaten": candidate_payload,
        "bewertungen": scored[:3],
        "gewinner": {
            "titel": winner["titel"],
            "laenge": len(winner["titel"]),
            "gesamt_score": winner["gesamt"],
            "warum_dieser": (
                "Lokaler Fallback: kurze, klare und sofort verwendbare Push-Zeile."
                if not is_video
                else "Lokaler Fallback: klare Video-Zeile, die das Format sichtbar macht."
            ),
        },
        "alternative": {
            "titel": alternative["titel"],
            "laenge": len(alternative["titel"]),
            "warum": "Lokaler Fallback mit ähnlicher journalistischer Lesbarkeit.",
        },
    }


def _llm_call(
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY/AI_API_KEY nicht gesetzt")
    global _OPENAI_CLIENT, _OPENAI_CLIENT_KEY
    if _OPENAI_CLIENT is None or _OPENAI_CLIENT_KEY != api_key:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        _OPENAI_CLIENT_KEY = api_key
    client = _OPENAI_CLIENT
    token_limit = min(max_tokens, DEFAULT_MAX_TOKENS)
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "timeout": AGENT_TIMEOUT,
        "store": False,
    }
    if MODEL.lower().startswith("gpt-5"):
        request.update(
            {
                "max_completion_tokens": token_limit,
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "reasoning_effort": REASONING_EFFORT,
                    "verbosity": "low",
                },
            }
        )
    else:
        request.update(
            {
                "max_tokens": token_limit,
                "temperature": temperature,
            }
        )

    try:
        resp = client.chat.completions.create(**request)
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI lieferte keinen Push-Titel")
        return content.strip()
    except Exception as exc:
        log.warning("[PushTitle] Schneller LLM-Call fehlgeschlagen: %s", exc)
        raise


EDITORIAL_ONE_BRAIN_SYS = """Du bist erfahrener BILD-Push-Redakteur. Arbeite nach dem
Push-Headline-Prompt v1.4. Erstelle aus dem belegten Artikelinhalt genau drei
unterschiedliche Push-Varianten mit jeweils Headline und Zeile 2.

FELDLOGIK:
- Headline 25-45 Zeichen; die ersten 25 Zeichen tragen Akteur und Kernbegriff.
- Zeile 2: 20-35 Zeichen, elliptisch und hart. Sie darf die Headline nicht doppeln.
- Die Headline muss allein funktionieren. Information, Modalität oder Zuschreibung darf
  nie ausschließlich in Zeile 2 stehen.
- iOS liest Zeile 2 vor der Headline; auch in dieser Reihenfolge muss es funktionieren.
- Kein Marken-, Ressort- oder Eilmeldungs-Prefix, keine Dachzeile.

HANDWERK UND FAKTENTREUE:
- Finde den Core Claim: Akteur, Handlung in Artikelmodalität und Konsequenz.
- Nichts erfinden, dramatisieren oder als dringlicher darstellen. Keine Fragen,
  Ausrufezeichen, Wortspiele, vagen Pronomen, Clickbait-Formeln, Bewertungsadjektive,
  unbelegten Superlative oder zurückgehaltenen Kerninformationen.
- Nutze Ellipsen, starke Verben, harte Substantive und kurze Wörter.
- Enthält der Artikel eine belegte zentrale Zahl, muss genau eine zentrale Zahl im Push stehen.
- Bei Bericht, Schätzung, Prognose oder eigener Rechnung steht die kurze Zuschreibung am
  Anfang der Headline (z. B. "Berichte:", "Analysten:", "Studie:").
- Ein Eigenname gehört nur bei bundesweitem Signal in die Headline, sonst Kategorie vorn.
- Bei Tod, Gewalt, Terror, Trauer oder Katastrophen nüchtern und faktenorientiert schreiben.

STUFE:
- 1: sofort wissen; ungeklickt vollständig, nüchtern, keine offene Implikation.
- 2: sollte man wissen; Kern vollständig, genau ein belegter Aspekt darf offen bleiben.
- 3: Recherche/Analyse/People; konkrete belegte Tatsache in der Headline, keine Frage.

VARIANTEN: Nutze drei verschiedene Typen aus FAKT, BETROFFENHEIT, FOLGE und
OFFENE IMPLIKATION (letztere nie bei Stufe 1). Jede Variante muss denselben Core Claim tragen.

Finde INDIVIDUELL den staerksten gedeckten Hook: Neuigkeit, Betroffenheit, Konflikt, Ueberraschung,
Nutzen oder Fallhoehe. Nutze konkrete Namen, Orte, Zahlen, Folgen oder Ereignisse aus dem Input.
Jeder Titel muss vollstaendig durch Kicker, Headline oder Text gedeckt sein: nichts erfinden,
nicht dramatisieren, keine falsche Dringlichkeit und keine Clickbait-Luege.

Antworte NUR als kompaktes JSON:
{{
  "stufe":1,
  "stufe_begruendung":"max 8 Woerter",
  "kandidaten":[
    {{"ansatz":"FAKT","titel":"25-45 Zeichen","zeile2":"20-35 Zeichen","gesamt":0.0}},
    {{"ansatz":"BETROFFENHEIT","titel":"25-45 Zeichen","zeile2":"20-35 Zeichen","gesamt":0.0}},
    {{"ansatz":"FOLGE","titel":"25-45 Zeichen","zeile2":"20-35 Zeichen","gesamt":0.0}}
  ],
  "gewinner_index":0,
  "warum_dieser":"ein bis zwei kurze Saetze",
  "warnhinweis":""
}}"""


def _bounded_score(value) -> float:
    try:
        return round(max(0.0, min(10.0, float(value))), 1)
    except (TypeError, ValueError):
        return 0.0


def _headline_level(value) -> int:
    try:
        return max(1, min(3, int(value)))
    except (TypeError, ValueError):
        return 2


def _parse_compact_editorial_response(data: dict) -> dict | None:
    raw_candidates = data.get("kandidaten", [])
    if not isinstance(raw_candidates, list):
        return None

    parsed = []
    for item in raw_candidates[:4]:
        if not isinstance(item, dict):
            continue
        titel = _clean_title(str(item.get("titel", "")))[:80].strip(" ,-")
        ansatz = str(item.get("ansatz", "")).strip()
        zeile2 = _clean_title(str(item.get("zeile2", "")))[:35].strip(" ,-")
        if not titel or not ansatz:
            continue
        parsed.append(
            {
                "kandidat": {
                    "titel": titel,
                    "zeile2": zeile2,
                    "laenge": len(titel),
                    "ansatz": ansatz,
                },
                "gesamt": _bounded_score(item.get("gesamt", 0)),
                "schwaeche": str(item.get("schwaeche", "")).strip(),
            }
        )
    if not parsed:
        return None

    winner_ansatz = str(data.get("gewinner_ansatz", "")).strip()
    try:
        winner_index = int(data.get("gewinner_index", -1))
    except (TypeError, ValueError):
        winner_index = -1
    if winner_index < 0 and winner_ansatz:
        winner_index = next(
            (
                index
                for index, item in enumerate(parsed)
                if item["kandidat"]["ansatz"] == winner_ansatz
            ),
            -1,
        )
    if winner_index < 0 or winner_index >= len(parsed):
        winner_index = max(range(len(parsed)), key=lambda idx: parsed[idx]["gesamt"])
    winner_item = parsed[winner_index]
    alternatives = sorted(
        (item for idx, item in enumerate(parsed) if idx != winner_index),
        key=lambda item: item["gesamt"],
        reverse=True,
    )
    alternative_item = alternatives[0] if alternatives else winner_item

    return {
        "analyse": data.get("analyse", {}),
        "kandidaten": [item["kandidat"] for item in parsed],
        "bewertungen": [
            {
                "titel": item["kandidat"]["titel"],
                "gesamt": item["gesamt"],
                "schwaeche": item["schwaeche"],
            }
            for item in parsed
        ],
        "gewinner": {
            "titel": winner_item["kandidat"]["titel"],
            "zeile2": winner_item["kandidat"]["zeile2"],
            "laenge": winner_item["kandidat"]["laenge"],
            "gesamt_score": winner_item["gesamt"],
            "warum_dieser": str(data.get("warum_dieser", "")).strip(),
        },
        "alternative": {
            "titel": alternative_item["kandidat"]["titel"],
            "zeile2": alternative_item["kandidat"]["zeile2"],
            "laenge": alternative_item["kandidat"]["laenge"],
            "warum": str(data.get("warum_alternative", "")).strip(),
        },
        "warnhinweis": str(data.get("warnhinweis", "")).strip(),
        "stufe": _headline_level(data.get("stufe", 2)),
        "stufe_begruendung": str(data.get("stufe_begruendung", "")).strip(),
    }


def _editorial_one_brain(title, text, category, kicker="", headline=""):
    parts = []
    if kicker:
        parts.append(f"Kicker: {kicker}")
    parts.append(f"Titel: {title}")
    if headline and headline != title:
        parts.append(f"Headline: {headline}")
    parts.append(f"Kategorie: {category}")
    if text:
        parts.append(f"\nText:\n{text[:1500]}")

    raw = _llm_call(
        EDITORIAL_ONE_BRAIN_SYS,
        "\n".join(parts),
        temperature=0.4,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    try:
        if "{" in raw:
            data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            if data.get("gewinner_index") is not None or data.get("gewinner_ansatz"):
                compact_result = _parse_compact_editorial_response(data)
                if compact_result:
                    return compact_result

            analyse = data.get("analyse", {})
            kandidaten = data.get("kandidaten", [])
            for k in kandidaten:
                if k.get("titel"):
                    k["laenge"] = len(k["titel"])

            for b in data.get("bewertungen", []):
                try:
                    b["gesamt"] = round(float(b.get("gesamt", 0)), 1)
                except (TypeError, ValueError):
                    b["gesamt"] = 0.0

            for key in ("gewinner", "alternative"):
                entry = data.get(key, {})
                if entry.get("titel"):
                    entry["laenge"] = len(entry["titel"])

            winner = data.get("gewinner", {})
            try:
                winner["gesamt_score"] = round(float(winner.get("gesamt_score", 0)), 1)
            except (TypeError, ValueError):
                winner["gesamt_score"] = 0.0

            return {
                "analyse": analyse,
                "kandidaten": kandidaten,
                "bewertungen": data.get("bewertungen", []),
                "gewinner": data.get("gewinner", {}),
                "alternative": data.get("alternative", {}),
                "warnhinweis": data.get("warnhinweis", ""),
            }
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[EditorialOneBrain] JSON-Parse: {e}")

    titles = re.findall(r'"titel"\s*:\s*"([^"]+)"', raw)
    kandidaten = [{"titel": t, "laenge": len(t), "ansatz": "fallback"} for t in titles[:6]]
    winner_titel = kandidaten[0]["titel"] if kandidaten else title
    return {
        "analyse": {"kern": title},
        "kandidaten": kandidaten,
        "bewertungen": [],
        "gewinner": {
            "titel": winner_titel,
            "laenge": len(winner_titel),
            "gesamt_score": 5.0,
            "warum_dieser": "Fallback wegen nicht parsbarer LLM-Antwort",
        },
        "alternative": {
            "titel": title,
            "laenge": len(title),
            "warum": "Fallback",
        },
    }


def _with_video_reason(entry: dict, article_type: str, title: str, text: str = "") -> dict:
    item = dict(entry or {})
    if _is_video_context(article_type, title, text):
        reason = str(item.get("warum_dieser") or item.get("warum") or "")
        if "video" not in reason.lower():
            if "warum_dieser" in item:
                item["warum_dieser"] = (reason + "; Video-Kontext klar erkannt").strip("; ")
            elif "warum" in item:
                item["warum"] = (reason + "; Video-Kontext klar erkannt").strip("; ")
    return item


# ═══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_push_title(article_title, article_text="", category="news",
                        kicker="", headline="", model=None, article_type="editorial",
                        force_llm=False):
    """Editorial-One-Brain Pipeline (Single Call)."""
    t0 = time.monotonic()
    log.info(f"[PushTitle] Start: '{article_title[:60]}' ({category})")

    llm_unavailable_reason = _llm_unavailable_reason()
    use_llm = (
        not llm_unavailable_reason
        and allow_calls(
            [
                (
                    "openai_title_generation_hour",
                    int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR", "0") or "0"),
                    3600,
                ),
                (
                    "openai_title_generation_day",
                    int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY", "0") or "0"),
                    86400,
                ),
            ]
        )
    )
    category_label = f"{category} (video)" if _is_video_context(article_type, article_title, article_text) else category
    if use_llm:
        log.info("[PushTitle] LLM-Call gestartet: individuelle Headline-Generierung")
        one_brain = _editorial_one_brain(article_title, article_text, category_label, kicker, headline)
    else:
        if force_llm:
            log.warning("[PushTitle] LLM angefordert, aber nicht verfuegbar: %s", llm_unavailable_reason or "Budget erschoepft")
        one_brain = _local_editorial_one_brain(
            article_title,
            article_text,
            category_label,
            kicker,
            headline,
            article_type=article_type,
        )
    analyse = one_brain.get("analyse", {})
    kandidaten = one_brain.get("kandidaten", [])
    t1 = time.monotonic()
    log.info(f"[PushTitle] One-Brain: {t1-t0:.1f}s — {len(kandidaten)} Kandidaten")

    grouped = {"sprachlich": [], "psychologisch": [], "datenbasiert": []}
    for k in kandidaten:
        a = k.get("ansatz", "sprachlich")
        grouped.setdefault(a, []).append(k)

    result = {
        "bewertungen": one_brain.get("bewertungen", []),
        "gewinner": one_brain.get("gewinner", {}),
        "alternative": one_brain.get("alternative", {}),
        "warnhinweis": one_brain.get("warnhinweis", ""),
        "stufe": _headline_level(one_brain.get("stufe", 2)),
        "stufe_begruendung": one_brain.get("stufe_begruendung", ""),
    }

    if not result["gewinner"].get("titel"):
        result["gewinner"] = {
            "titel": article_title,
            "laenge": len(article_title),
            "gesamt_score": 5.0,
            "warum_dieser": "Fallback ohne Gewinner",
        }
    if not result["alternative"].get("titel"):
        result["alternative"] = {
            "titel": article_title,
            "laenge": len(article_title),
            "warum": "Fallback",
        }
    result["alternative"] = _with_video_reason(result["alternative"], article_type, article_title, article_text)

    content_type = "video" if _is_video_context(article_type, article_title, article_text) else "editorial"
    result["meta"] = {
        "original_titel": article_title,
        "kategorie": category_label,
        "content_type": content_type,
        "dauer_gesamt_s": round(t1 - t0, 1),
        "dauer_call1_s": round(t1 - t0, 1),
        "dauer_call2_s": 0.0,
        "anzahl_kandidaten": len(kandidaten),
        "modell": MODEL if use_llm else "local-fallback",
        "analyse": analyse,
        "modus": "llm-individual-headline" if use_llm else "local-fallback",
        "llm_requested": bool(force_llm),
        "llm_call_started": bool(use_llm),
        "llm_unavailable_reason": "" if use_llm else (llm_unavailable_reason or "Budget erschoepft"),
    }
    result["alle_kandidaten"] = grouped
    result["title"] = result["gewinner"].get("titel", article_title)
    preferred_alternatives = [
        title for title in one_brain.get("alternativeTitles", [])
        if title and title != result["title"]
    ]
    fallback_alternatives = [
        item.get("titel", "")
        for items in grouped.values()
        for item in items
        if item.get("titel") and item.get("titel") != result["title"]
    ]
    result["alternativeTitles"] = (preferred_alternatives + [
        title for title in fallback_alternatives if title not in preferred_alternatives
    ])[:3]
    result["reasoning"] = result["gewinner"].get("warum_dieser", "")
    result["advisoryOnly"] = True
    result["contentType"] = content_type

    w = result.get("gewinner", {})
    log.info(f"[PushTitle] ERGEBNIS: '{w.get('titel', '?')[:80]}' "
             f"(Score: {w.get('gesamt_score', '?')}/10, {t1-t0:.1f}s)")
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        print("Usage: python3 push_title_agent.py 'Titel' ['Text'] ['Kategorie']")
        sys.exit(1)
    title = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    cat = sys.argv[3] if len(sys.argv) > 3 else "news"
    result = generate_push_title(title, text, cat)
    w = result.get("gewinner", {})
    print(f"\n{'='*70}")
    print(f"ORIGINAL:    {title}")
    print(f"GEWINNER:    {w.get('titel', '?')}")
    print(f"Score:       {w.get('gesamt_score', '?')}/10")
    print(f"Dauer:       {result['meta']['dauer_gesamt_s']}s")
    alt = result.get("alternative", {})
    if alt.get("titel"):
        print(f"ALT:         {alt['titel']}")
    for grp, vs in result.get("alle_kandidaten", {}).items():
        if vs:
            print(f"\n  [{grp.upper()}]")
            for v in vs:
                print(f"    {v.get('titel', '?')} ({v.get('laenge', '?')}Z)")
