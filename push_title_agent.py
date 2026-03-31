#!/usr/bin/env python3
"""Push-Zeilen Generator v4 — 2 schnelle GPT-4o Calls (~6s).

Call 1: Kreativteam generiert 6 Titel-Kandidaten mit Analyse
Call 2: Chefredakteur waehlt den besten aus
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


# ═══════════════════════════════════════════════════════════════════════════════
#  CALL 1: KREATIVTEAM
# ═══════════════════════════════════════════════════════════════════════════════

KREATIV_SYS = f"""Du bist das BILD Push-Kreativteam. Analysiere den Artikel und generiere 6 Push-Titel.

REGELN:
- 60-80 Zeichen ideal, max {MAX_PUSH_LENGTH}
- JEDER Titel braucht einen offenen Loop (Frage die nur durch Tippen aufgeloest wird)
- Praesens + aktive Verben ("greift an" statt "hat angegriffen")
- KEIN Pipe-Format ("|"), keine Emojis, kein Passiv, max 1 Komma

TECHNIKEN (aus Top-Pushes mit 20% OR):
- Gedankenstrich-Cliffhanger: "Fakt — Schock-Wende" (staerkste Technik)
- Doppelpunkt-Kicker: "Kontext: Die Nachricht"
- Name+Alter: "Leon (6)" macht Opfer zu Menschen
- Zitat-Einstieg: Woertliche Rede = Leser IST in der Szene
- Informationsluecke: GENUG verraten, EINE Sache offenlassen

6 Titel: 2x sprachlich, 2x psychologisch (Curiosity Gap/Verlust-Aversion), 2x datenbasiert.

Antworte NUR als JSON:
{{"analyse":{{"kern":"...","hook":"...","emotion":"..."}},"kandidaten":[{{"titel":"...","ansatz":"sprachlich|psychologisch|datenbasiert"}}]}}"""


def _kreativteam(title, text, category, kicker="", headline=""):
    parts = []
    if kicker:
        parts.append(f"Kicker: {kicker}")
    parts.append(f"Titel: {title}")
    if headline and headline != title:
        parts.append(f"Headline: {headline}")
    parts.append(f"Kategorie: {category}")
    if text:
        parts.append(f"\nText:\n{text[:1500]}")

    raw = _llm_call(KREATIV_SYS, "\n".join(parts), temperature=0.8, max_tokens=900)

    try:
        if "{{" not in raw and "{" in raw:
            data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            analyse = data.get("analyse", {})
            kandidaten = data.get("kandidaten", [])
            for k in kandidaten:
                if "titel" in k:
                    k["laenge"] = len(k["titel"])
            return analyse, kandidaten
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[Kreativteam] JSON-Parse: {e}")

    titles = re.findall(r'"titel"\s*:\s*"([^"]+)"', raw)
    return {"kern": title}, [{"titel": t, "laenge": len(t), "ansatz": "fallback"} for t in titles[:6]]


# ═══════════════════════════════════════════════════════════════════════════════
#  CALL 2: CHEFREDAKTEUR
# ═══════════════════════════════════════════════════════════════════════════════

CHEF_SYS = f"""Du bist der BILD Push-Chefredakteur. Waehle den Titel mit der hoechsten Opening Rate.

Bewertung: Klick-Impuls (40%), Emotion (25%), Klarheit (20%), BILD-DNA (10%), Faktentreue (5%).
Max {MAX_PUSH_LENGTH} Zeichen. MUSS offenen Loop haben. KEIN Pipe "|".

Antworte NUR als JSON:
{{"bewertungen":[{{"titel":"...","gesamt":0.0,"schwaeche":"..."}}],"gewinner":{{"titel":"...","laenge":0,"gesamt_score":0.0,"warum_dieser":"..."}},"alternative":{{"titel":"...","laenge":0,"warum":"..."}}}}"""


def _chefredakteur(analyse, kandidaten, original_title):
    if not kandidaten:
        return {
            "gewinner": {"titel": original_title, "laenge": len(original_title),
                         "gesamt_score": 5.0, "warum_dieser": "Keine Kandidaten"},
            "alternative": {"titel": original_title, "laenge": len(original_title), "warum": "Fallback"},
            "bewertungen": [],
        }

    cand_text = "\n".join(
        f"[{c.get('ansatz','?')}] \"{c['titel']}\" ({len(c.get('titel',''))}Z)"
        for c in kandidaten if c.get("titel")
    )

    user = f"""ORIGINAL: "{original_title}"
KERN: {analyse.get('kern', '')}
HOOK: {analyse.get('hook', '')}
EMOTION: {analyse.get('emotion', '')}

KANDIDATEN:
{cand_text}

Bewerte die besten 3. Waehle oder synthetisiere den optimalen Titel."""

    raw = _llm_call(CHEF_SYS, user, temperature=0.2, max_tokens=800)

    try:
        if "{" in raw:
            result = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            for key in ("gewinner", "alternative"):
                entry = result.get(key, {})
                if entry.get("titel"):
                    entry["laenge"] = len(entry["titel"])
            for b in result.get("bewertungen", []):
                try:
                    b["gesamt"] = round(float(b.get("gesamt", 0)), 1)
                except (TypeError, ValueError):
                    b["gesamt"] = 0.0
            w = result.get("gewinner", {})
            try:
                w["gesamt_score"] = round(float(w.get("gesamt_score", 0)), 1)
            except (TypeError, ValueError):
                w["gesamt_score"] = 0.0
            return result
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"[Chef] JSON-Parse: {e}")

    return {
        "gewinner": {"titel": original_title, "laenge": len(original_title),
                     "gesamt_score": 5.0, "warum_dieser": "Nicht parsbar"},
        "alternative": {"titel": original_title, "laenge": len(original_title), "warum": "Fallback"},
        "bewertungen": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_push_title(article_title, article_text="", category="news",
                        kicker="", headline="", model=None):
    """2-Call Pipeline: Kreativteam → Chefredakteur."""
    t0 = time.monotonic()
    log.info(f"[PushTitle] Start: '{article_title[:60]}' ({category})")

    analyse, kandidaten = _kreativteam(article_title, article_text, category, kicker, headline)
    t1 = time.monotonic()
    log.info(f"[PushTitle] Call 1: {t1-t0:.1f}s — {len(kandidaten)} Kandidaten")

    result = _chefredakteur(analyse, kandidaten, article_title)
    t2 = time.monotonic()

    grouped = {"sprachlich": [], "psychologisch": [], "datenbasiert": []}
    for k in kandidaten:
        a = k.get("ansatz", "sprachlich")
        grouped.setdefault(a, []).append(k)

    result["meta"] = {
        "original_titel": article_title,
        "kategorie": category,
        "dauer_gesamt_s": round(t2 - t0, 1),
        "dauer_call1_s": round(t1 - t0, 1),
        "dauer_call2_s": round(t2 - t1, 1),
        "anzahl_kandidaten": len(kandidaten),
        "modell": MODEL,
        "analyse": analyse,
    }
    result["alle_kandidaten"] = grouped

    w = result.get("gewinner", {})
    log.info(f"[PushTitle] ERGEBNIS: '{w.get('titel', '?')[:80]}' "
             f"(Score: {w.get('gesamt_score', '?')}/10, {t2-t0:.1f}s)")
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
