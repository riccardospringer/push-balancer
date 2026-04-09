#!/usr/bin/env python3
"""Push-Zeilen Generator v5 — Editorial-One-Brain (Single Call).

Ein einzelner LLM-Call erstellt Analyse, Kandidaten und Gewinnerauswahl.
"""

import os
import json
import logging
import time
import re

log = logging.getLogger("push-title-agent")

MODEL = "gpt-4o"
MAX_PUSH_LENGTH = 100
AGENT_TIMEOUT = 20


def _llm_call(system: str, user: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
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

    raw = _llm_call(EDITORIAL_ONE_BRAIN_SYS, "\n".join(parts), temperature=0.7, max_tokens=1000)

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
                        kicker="", headline="", model=None):
    """Editorial-One-Brain Pipeline (Single Call)."""
    t0 = time.monotonic()
    log.info(f"[PushTitle] Start: '{article_title[:60]}' ({category})")

    one_brain = _editorial_one_brain(article_title, article_text, category, kicker, headline)
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
        "kategorie": category,
        "dauer_gesamt_s": round(t1 - t0, 1),
        "dauer_call1_s": round(t1 - t0, 1),
        "dauer_call2_s": 0.0,
        "anzahl_kandidaten": len(kandidaten),
        "modell": MODEL,
        "analyse": analyse,
        "modus": "editorial-one-brain",
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
