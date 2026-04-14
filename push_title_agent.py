#!/usr/bin/env python3
"""Mehrstufige Push-Titel-Chain fuer hochwertige Vorschlaege.

Wenn OPENAI_API_KEY gesetzt ist, laeuft eine 3-stufige Editorial-Chain:
1. Briefing / Angle Extraction
2. Candidate Generation
3. Critic + Selection

Ohne API-Key faellt der Aufrufer kontrolliert auf die lokale Chain in app.push_titles zurueck.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from app.push_titles import build_push_title_suggestions

log = logging.getLogger("push-title-agent")

MODEL = os.environ.get("PUSH_TITLE_MODEL", "gpt-4.1")
TEMPERATURE = float(os.environ.get("PUSH_TITLE_TEMPERATURE", "0.4"))
MAX_PUSH_LENGTH = 78
AGENT_TIMEOUT = 20


def _extract_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found")
    return json.loads(raw[start : end + 1])


def _llm_call(system: str, user: str, temperature: float = TEMPERATURE, max_tokens: int = 900) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=AGENT_TIMEOUT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return _extract_json(content)


BRIEF_SYSTEM = f"""Du bist Push-Copy-Chef fuer BILD.
Analysiere eine Geschichte fuer Push-Nachrichten auf Lock-Screen-Niveau.

ZIELE:
- maximale Klarheit
- konkrete Leser-Relevanz
- aktive, faktennahe Sprache
- keine billige Clickbait-Mechanik

REGELN:
- Fokus auf Neues, Konsequenz, Betroffenheit oder Konflikt
- nur reale Informationen aus Titel/Text verwenden
- kennzeichne Video sichtbar als Video
- halte die spaetere Push-Zeile idealerweise zwischen {28} und {58} Zeichen, hart max {MAX_PUSH_LENGTH}

Antworte nur als JSON:
{{
  "subject":"...",
  "core_fact":"...",
  "reader_value":"...",
  "best_angle":"neues|konsequenz|konflikt|erklaerer|video",
  "tone":"direkt|dringlich|sachlich",
  "content_type":"editorial|video",
  "must_keep":["...","..."],
  "avoid":["...","..."]
}}"""


CANDIDATE_SYSTEM = f"""Du schreibst starke BILD-Push-Zeilen.
Erzeuge auf Basis eines Briefings exakt 8 Kandidaten.

REGELN:
- journalistisch sauber, konkret und klickstark
- keine Emojis, kein Pipe-Zeichen, kein leerer Alarmismus
- kein Kandidat ueber {MAX_PUSH_LENGTH} Zeichen
- variiere die Strategien: direkt, konsequenz, konflikt, nutzwert, breaking/video falls passend

Antworte nur als JSON:
{{
  "kandidaten":[
    {{"titel":"...","ansatz":"direkt"}},
    {{"titel":"...","ansatz":"konsequenz"}}
  ]
}}"""


CRITIC_SYSTEM = """Du bist der letzte Push-Critic fuer BILD.
Bewerte Kandidaten strikt nach:
1. Klarheit auf dem Lock Screen
2. Konkretheit / faktische Substanz
3. Leser-Relevanz
4. Sprachliche Spannung ohne billiges Clickbait
5. Laenge / mobile Tauglichkeit

Waehle einen Gewinner und eine starke Alternative.

Antworte nur als JSON:
{
  "bewertungen":[
    {"titel":"...","gesamt":0.0,"staerke":"...","schwaeche":"..."}
  ],
  "gewinner":{"titel":"...","gesamt_score":0.0,"warum_dieser":"..."},
  "alternative":{"titel":"...","warum":"..."}
}"""


def _brief_payload(title: str, text: str, category: str, kicker: str = "", headline: str = "") -> str:
    parts = []
    if kicker:
        parts.append(f"Kicker: {kicker}")
    parts.append(f"Titel: {title}")
    if headline and headline != title:
        parts.append(f"Headline: {headline}")
    parts.append(f"Kategorie: {category}")
    if text:
        parts.append(f"Textauszug: {text[:1600]}")
    return "\n".join(parts)


def _generate_with_llm_chain(article_title: str, article_text: str, category: str, kicker: str = "", headline: str = "") -> dict:
    t0 = time.monotonic()
    brief = _llm_call(BRIEF_SYSTEM, _brief_payload(article_title, article_text, category, kicker, headline), temperature=0.2, max_tokens=500)
    t1 = time.monotonic()

    candidate_payload = json.dumps(
        {
            "briefing": brief,
            "original_title": article_title,
            "category": category,
        },
        ensure_ascii=False,
    )
    candidate_response = _llm_call(CANDIDATE_SYSTEM, candidate_payload, temperature=0.7, max_tokens=700)
    t2 = time.monotonic()

    candidates = candidate_response.get("kandidaten", [])[:8]
    if not candidates:
        raise RuntimeError("LLM title chain returned no candidates")

    critic_payload = json.dumps(
        {
            "briefing": brief,
            "original_title": article_title,
            "category": category,
            "kandidaten": candidates,
        },
        ensure_ascii=False,
    )
    critic_response = _llm_call(CRITIC_SYSTEM, critic_payload, temperature=0.1, max_tokens=900)
    t3 = time.monotonic()

    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        title = candidate.get("titel", "").strip()
        if not title:
            continue
        angle = candidate.get("ansatz", "llm")
        grouped.setdefault(angle, []).append({"titel": title, "laenge": len(title)})

    winner = critic_response.get("gewinner", {}) or {}
    alternative = critic_response.get("alternative", {}) or {}
    ratings = critic_response.get("bewertungen", []) or []

    if not winner.get("titel"):
        winner = {"titel": candidates[0]["titel"], "gesamt_score": 7.0, "warum_dieser": "Bester Kandidat nach LLM-Critic."}
    if not alternative.get("titel"):
        fallback_alt = candidates[1]["titel"] if len(candidates) > 1 else candidates[0]["titel"]
        alternative = {"titel": fallback_alt, "warum": "Zweitstarker Kandidat der LLM-Chain."}

    winner["laenge"] = len(winner["titel"])
    alternative["laenge"] = len(alternative["titel"])

    return {
        "title": winner["titel"],
        "alternativeTitles": [
            entry.get("titel")
            for entry in ratings
            if entry.get("titel") and entry.get("titel") != winner["titel"]
        ][:4],
        "reasoning": winner.get("warum_dieser", ""),
        "advisoryOnly": True,
        "contentType": brief.get("content_type", "editorial"),
        "bewertungen": ratings,
        "gewinner": winner,
        "alternative": alternative,
        "alle_kandidaten": grouped,
        "meta": {
            "original_titel": article_title,
            "kategorie": category,
            "dauer_gesamt_s": round(t3 - t0, 1),
            "dauer_call1_s": round(t1 - t0, 1),
            "dauer_call2_s": round(t3 - t1, 1),
            "anzahl_kandidaten": sum(len(items) for items in grouped.values()),
            "modell": MODEL,
            "analyse": {
                "kern": brief.get("core_fact", article_title),
                "hook": brief.get("reader_value", ""),
                "emotion": brief.get("tone", "direkt"),
                "winkel": brief.get("best_angle", ""),
            },
            "modus": "editorial-prompt-chain",
        },
    }


def generate_push_title(article_title, article_text="", category="news", kicker="", headline="", model=None):
    """LLM-Chain mit robustem lokalem Fallback."""
    if not os.environ.get("OPENAI_API_KEY", ""):
        return build_push_title_suggestions(article_title, category=category, url="")

    try:
        return _generate_with_llm_chain(article_title, article_text, category, kicker=kicker, headline=headline)
    except Exception:
        log.exception("[PushTitle] LLM-Chain fehlgeschlagen, falle lokal zurueck")
        return build_push_title_suggestions(article_title, category=category, url="")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        print("Usage: python3 push_title_agent.py 'Titel' ['Text'] ['Kategorie']")
        raise SystemExit(1)

    title = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""
    category = sys.argv[3] if len(sys.argv) > 3 else "news"
    print(json.dumps(generate_push_title(title, text, category), ensure_ascii=False, indent=2))
