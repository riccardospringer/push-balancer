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

MODEL = os.environ.get("OPENAI_TITLE_GENERATION_MODEL", "gpt-4o-mini")
MAX_PUSH_LENGTH = 100
AGENT_TIMEOUT = float(os.environ.get("OPENAI_TITLE_GENERATION_TIMEOUT_S", "8.0"))
DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_TITLE_GENERATION_MAX_TOKENS", "320"))
_OPENAI_CLIENT = None
_OPENAI_CLIENT_KEY = ""


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


def _llm_call(system: str, user: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt")
    global _OPENAI_CLIENT, _OPENAI_CLIENT_KEY
    if _OPENAI_CLIENT is None or _OPENAI_CLIENT_KEY != api_key:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        _OPENAI_CLIENT_KEY = api_key
    client = _OPENAI_CLIENT
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=min(max_tokens, DEFAULT_MAX_TOKENS),
        temperature=temperature,
        timeout=AGENT_TIMEOUT,
    )
    return resp.choices[0].message.content.strip()


EDITORIAL_ONE_BRAIN_SYS = f"""Du bist die zentrale Editorial-One-Brain Instanz fuer BILD Push.
Du arbeitest in EINEM Durchlauf: analysieren, Varianten bauen, bewerten, Gewinner waehlen.

REGELN:
- 60-80 Zeichen ideal, max {MAX_PUSH_LENGTH}
- Praesens + aktive Verben
- KEIN Pipe-Format ("|"), keine Emojis, kein Passiv, max 1 Komma
- Titel muss journalistisch sauber und faktentreu sein

LIEFERE:
- analyse: kern/hook/emotion
- kandidaten: exakt 6 Titel (2x sprachlich, 2x psychologisch, 2x datenbasiert)
- bewertungen: bis zu 3 bewertete Top-Titel
- gewinner: bester Titel inkl. Begruendung und Score 0-10
- alternative: zweitbester Titel

Antworte NUR als JSON:
{{
  "analyse": {{"kern":"...","hook":"...","emotion":"..."}},
  "kandidaten":[{{"titel":"...","ansatz":"sprachlich|psychologisch|datenbasiert"}}],
  "bewertungen":[{{"titel":"...","gesamt":0.0,"schwaeche":"..."}}],
  "gewinner":{{"titel":"...","laenge":0,"gesamt_score":0.0,"warum_dieser":"..."}},
  "alternative":{{"titel":"...","laenge":0,"warum":"..."}}
}}"""


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

    raw = _llm_call(EDITORIAL_ONE_BRAIN_SYS, "\n".join(parts), temperature=0.4, max_tokens=320)

    try:
        if "{" in raw:
            data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
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


# ═══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_push_title(article_title, article_text="", category="news",
                        kicker="", headline="", model=None, article_type="editorial"):
    """Editorial-One-Brain Pipeline (Single Call)."""
    t0 = time.monotonic()
    log.info(f"[PushTitle] Start: '{article_title[:60]}' ({category})")

    use_llm = (
        os.environ.get("PAID_EXTERNAL_APIS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
        and os.environ.get("OPENAI_TITLE_GENERATION_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
        and bool(os.environ.get("OPENAI_API_KEY", ""))
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
        one_brain = _editorial_one_brain(article_title, article_text, category_label, kicker, headline)
    else:
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

    result["meta"] = {
        "original_titel": article_title,
        "kategorie": category_label,
        "content_type": article_type,
        "dauer_gesamt_s": round(t1 - t0, 1),
        "dauer_call1_s": round(t1 - t0, 1),
        "dauer_call2_s": 0.0,
        "anzahl_kandidaten": len(kandidaten),
        "modell": MODEL if use_llm else "local-fallback",
        "analyse": analyse,
        "modus": "editorial-one-brain" if use_llm else "local-fallback",
    }
    result["alle_kandidaten"] = grouped

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
