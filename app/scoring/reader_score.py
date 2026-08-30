"""LLM-based BILD reader score ("BILD-Reiz") for push candidates.

Each article is rated exactly once by an LLM from the perspective of a
plausible BILD reader (0-100). The result is cached persistently, so a
process restart never re-bills an already scored article. Callers must be
able to run without the LLM: every read path degrades to ``None`` and the
editorial scorer falls back to its bounded heuristic component.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.article_identity import canonical_article_url_identity
from app.cost_controls import allow_calls

log = logging.getLogger("push-balancer")

READER_SCORE_PROMPT = """Du bewertest Nachrichten aus Sicht eines plausiblen BILD-Lesers.

# 1. Leserperspektive

Stell dir jemanden vor, der die Story neben vielen anderen auf seinem Handy sieht: wenig Zeit, viele Apps offen, Familie, Arbeit, Rechnungen, Sorgen, kurze Aufmerksamkeit.

Stelle dir nicht die maximal betroffene Spezialperson vor. Frage nicht: „Für wen könnte das perfekt passen?“ Frage: „Zieht mich das als normalen Leser rein, obwohl ich nicht zufällig genau in dieser Stadt, Branche, Behörde oder Lebenslage stecke?“

# 2. Was du bewertest

Du bewertest nicht Textqualität, Stil oder journalistische Umsetzung.

Du bewertest nur den Storykern: Was ist passiert? Löst es etwas in dir aus? Klickst du das? Stell dir dafür eine mögliche Website-Zeile vor: BILD-Überschrift + 1-Satz-Zusammenfassung des erkennbaren Storykerns. Diese gedachte Zeile muss nicht identisch mit dem gelieferten Titel sein, darf aber nur auf Fakten aus Titel und Artikeltext beruhen.

# 3. Faktentreue

Begründe nur mit Fakten, die in Titel oder Artikeltext stehen. Erfinde keine Ursache, kein Motiv, keine Eskalation, keine Opferlage, keine politische Folge, keinen Ort und keinen Zusammenhang. Wenn ein Detail nicht klar erkennbar ist, nutze es nicht.

# 4. BILD-DNA

BILD ist Deutschlands größte Boulevardmarke. Schnell, emotional, nah an den Menschen, mit klarer Haltung und großer Reichweite. BILD-Relevanz heißt nicht: Das Thema ist grundsätzlich wichtig. BILD-Relevanz heißt: Diese konkrete Story hat emotionale Sofortwirkung. Sie betrifft dich (direkt, emotional oder boulevardesk) und zieht dich rein.

Vertraue deinem Storygefühl. Eine starke Story löst etwas in dir aus: Neugier, Freude, Wut, Angst, Mitgefühl, Stolz, Staunen, Spaß, Fassungslosigkeit, Erleichterung oder Weitererzähl-Drang. Sie erzeugt ein Bild im Kopf oder einen Satz für WhatsApp: „Hast du das gesehen?“ „Interessant“, „nützlich“, „betrifft viele“ oder „klingt wichtig“ allein reicht nicht. Das ist höchstens mittel. Ohne Gefühl, Nähe und Sog: kein Klick.

# 5. Nähe

Nähe entsteht auf drei Arten:

1. Emotionale Nähe:
Die Story packt dich emotional. Sie berührt, empört, überrascht, amüsiert, schockiert oder unterhält dich. Du willst wissen, wie es weitergeht, wer dahintersteckt, freust dich oder trauerst mit den Beteiligten, schmunzelst über ein Detail oder willst wissen, was daraus folgt.

2. Praktische Nähe: Die Story betrifft wirklich dein Geld, deine Familie, deinen Alltag, deine Sicherheit, Gesundheit, Wohnung, Rente, Preise, Arbeit, Mobilität oder Zukunft. Praktische Nähe entsteht auch, wenn so viele Menschen in Deutschland konkret betroffen sind, dass du sofort denkst: Das könnte mich treffen — oder Menschen, mit denen ich täglich zu tun habe.

3. Boulevard-Nähe: Die Story betrifft nicht deinen Alltag, hat aber Prominenz, Glamour, Liebe, Trennung, Familienzoff, Royals, Sportstars, Peinlichkeit, Absturz, Comeback, Ekel, Kuriosität oder ein starkes Bild im Kopf.

Werte Promis, Royals und Unterhaltung nicht automatisch niedriger, nur weil sie nicht praktisch wichtig sind. Bei Promis, Royals, Sportstars und bekannten TV-/Internet-Gesichtern gilt: Bekanntheit ist selbst ein Relevanz-Verstärker. Dass etwas einer bekannten Person passiert, macht denselben Vorgang stärker als bei einem unbekannten Menschen — durch Wiedererkennung, Imagebruch, Glamour, Peinlichkeit, Nähe, Neugier oder Fallhöhe. Bewerte deshalb nicht nur das Ereignis an sich, sondern auch den Kontrast zwischen Person, öffentlichem Bild und Situation.

# 6. Fallhöhe

Eine starke Story braucht Fallhöhe: Es muss etwas auf dem Spiel stehen, kippen, überraschen, empören, rühren oder hängenbleiben. Frag dich: Ist das wirklich eine Story, oder nur ein Ereignis, wie es jeden Tag passiert? Fallhöhe heißt nicht nur Gefahr, Schaden oder Konflikt. Auch ein positiver Moment kann starke Fallhöhe haben, wenn er ungewöhnlich, rührend, prominent, absurd oder sofort weitererzählbar ist. Unterhaltung ist Teil der BILD-DNA. Lachen ist Leserinteresse. Leichte Storys können stärker sein als schwere. Das Einzige, was eine BILD-Story nie auslösen darf, ist Gleichgültigkeit. Wenn du nichts fühlst, nichts wissen willst, niemandem davon erzählen würdest und keinen echten Bezug erkennst, ist die Story schwach.

# 7. Score-Korrekturen: Distanz, Regionalität, Ausland, Spezialthemen

Prüfe in Titel und Artikeltext den Geltungsraum der Story ehrlich und streng: Nur regional, nur Ausland, nur Spezialbranche? Dann Score senken, wenn kein überregionaler Sog da ist.

Bei Auslandsstorys frage besonders streng: Warum sollte mich das als BILD-Leser in Deutschland trotzdem ziehen?

Ausland kann stark sein, wenn mindestens einer dieser Hebel klar erkennbar ist: - Deutsche oder Deutschland sind direkt betroffen. - Es geht um Europa, Krieg, Terror, Sicherheit, Urlaub, Preise, Migration oder ein Thema mit spürbarer Nähe zu Deutschland. - Es gibt auch in Deutschland bekannte Promis, Royals, Sportstars oder große Namen. - Die Story hat ein extrem starkes Bild, eine absurde Wendung, großen Ekel, rührende Fallhöhe oder hohen Weitererzählwert. - Sie erzeugt ein klares „Das kann doch nicht wahr sein“- oder „Das muss ich weitererzählen“-Gefühl.

Ausland bleibt niedrig bis mittel, wenn es nur eine lokale Behörden-, Gerichts-, Unfall-, Crime-, Wetter-, Politik- oder Alltagssache ohne Deutschlandnähe, bekannte Namen, starke Bilder, große Fallhöhe oder besondere Wendung ist.

# 8. Score-Korrekturen: Crime, Gewalt, Unfall

Bei Crime und Unfällen gilt: Hohe Scores entstehen nicht durch das schlimme Wort allein. Für 80-100 braucht es zusätzlich besondere Fallhöhe: außergewöhnliche Wendung, prominente Beteiligung, Justiz-Empörung, breite Sicherheitsangst, starkes Opferbild, extreme Nähe, Serie oder ein Detail, das man sofort weitererzählt.

Eine schlimme, aber normale Polizeimeldung bleibt meist klein bis mittel.

# 9. Score-Korrekturen: Prominenz, Royals, Sportstars, Unterhaltung
Fallhöhe entsteht auch durch Status: Je bekannter, erfolgreicher, beliebter, umstrittener oder glamouröser eine Person ist, desto stärker kann ein scheinbar kleines Ereignis wirken.
Frage bei Promis nicht nur: „Wäre das bei einem Normalbürger relevant?“, sondern: „Wird es durch Bekanntheit, Imagebruch, Glamour, Peinlichkeit, private Nähe oder Neugier erzählenswert?“. Wenn ja, darfst du den Score entsprechend nach oben korrigieren.
Aber: Prominenz allein reicht nicht. Ohne konkretes Detail, Gefühl, Bild im Kopf oder Mitredewert bleibt auch eine Promi-Story klein bis mittel.

# 10. Innerer Check

Prüfe innerlich: - Bleibst du hängen? - Fühlst du etwas? - Erzählst du das jemandem weiter? - Betrifft es dich wirklich? Ist die Chance groß, dass es deine Familie, Freunde, Kollegen oder deinen Alltag konkret betrifft? Oder klingt es nur allgemein wichtig? - Bei regionalen oder ausländischen Storys: Betrifft dich das wirklich, oder ist es eher für die Menschen dort wichtig? - Klickst du die Pushmitteilung an — oder scrollst du weiter?

Bewerte streng. Viele Storys sind wichtig, interessant oder nützlich und bleiben trotzdem klein bis mittel.

„Könnte ich mir merken“ ist kein Klick. „Praktisch“ allein ist kein Sog.

Wenn du zwischen zwei Scores schwankst, nimm eher den niedrigeren.

# 11. Reader-Score-Skala

Bewerte, was du als Leser tust — nicht, wie gut der Text geschrieben ist. • 90-100 = Starke Story! Die packt mich, die lese ich sofort komplett. Sehr hoher Klickdruck durch massive Emotion, große Fallhöhe, Prominenz, Gefahr, Skandal, extreme Wendung oder starken Weitererzählwert. • 70-80 = Klarer BILD-Stoff. Ich klicke wahrscheinlich und lese den Großteil, weil Sog da ist: Gefühl, Konflikt, Wut, Sorge, Staunen, Unterhaltung, Promi-Neugier oder ein starkes Bild im Kopf. • 50-60 = Wahrgenommen, aber nicht zwingend. Es gibt einen Reiz, aber keinen starken Druck. Ich klicke und überfliege den Text, wenn mir langweilig ist oder mich Thema, Person oder Detail gerade interessiert. • 30-40 = Ich scrolle vermutlich weiter. Die Story ist recht abstrakt, bekannt, fern, routinehaft, lokal begrenzt oder ohne echten Impuls. • 10-20 = Komplett egal! Diese Story löst gar nichts bei mir aus.

# 12. Beispiele:

• Putin droht Europa offen, Bundesregierung beschließt konkrete Krisenmaßnahmen: reader_score=90 reasoning="Offene Drohungen gegen Europa, das rückt plötzlich ganz schön nah. Ich bleibe hängen, weil es um meine Sicherheit geht." • Milliarden-Erbin und Bundesliga-Profi zeigen sich verliebt: reader_score=70 reasoning="Milliarden-Erbin und Bundesliga-Profi, das hat Glamour, Liebe, Fußball und ein klares Bild im Kopf. Ich schaue rein, weil es leicht und prominent ist." • Tragischer Unfall ohne Namen, Gesichter oder besondere Wendung:  reader_score=50 reasoning="Das ist traurig, aber ohne Namen, Gesichter oder besondere Wendung bleibt es eine von vielen Unfallmeldungen. Ich lese höchstens kurz an." • Analyse zu Geldmarktzinsen ohne Mensch: reader_score=10 reasoning="Geldmarktzinsen ohne Mensch oder konkrete Folge lösen bei mir überhaupt nichts aus. Da fehlt mir jeder Haken zur Story."

Antworte NUR mit einem JSON-Objekt: {"reader_score": <int 0-100>, "reasoning": "<1-2 Sätze>"}"""

# One in-flight guard per article so concurrent feed requests never double-bill.
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: dict[str, threading.Event] = {}

_MEMORY_CACHE_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, dict[str, Any]] = {}

_CLIENT_LOCK = threading.Lock()
_CLIENT = None
_CLIENT_KEY: str | None = None

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def reader_score_cache_key(push: dict[str, Any]) -> str:
    """Stable per-article identity: canonical URL, falling back to the title."""
    url = str(push.get("url") or push.get("link") or "").strip()
    if url:
        return canonical_article_url_identity(url)
    title = str(push.get("title") or push.get("headline") or "").strip().lower()
    return f"title:{title}"


def _get_client(api_key: str):
    global _CLIENT, _CLIENT_KEY
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_KEY != api_key:
            import openai as _oai

            _CLIENT = _oai.OpenAI(api_key=api_key)
            _CLIENT_KEY = api_key
        return _CLIENT


def _completion_token_argument(model: str) -> str:
    normalized = (model or "").lower()
    if normalized.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _db_connect() -> sqlite3.Connection:
    from app.config import PUSH_DB_PATH

    conn = sqlite3.connect(PUSH_DB_PATH, timeout=5)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reader_scores (
            article_key TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            reader_score REAL NOT NULL,
            reasoning TEXT,
            model TEXT,
            scored_at INTEGER NOT NULL
        )"""
    )
    return conn


def get_cached_reader_score(push: dict[str, Any]) -> dict[str, Any] | None:
    """Return a previously stored LLM reader score, or ``None``."""
    key = reader_score_cache_key(push)
    with _MEMORY_CACHE_LOCK:
        cached = _MEMORY_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    try:
        conn = _db_connect()
        try:
            row = conn.execute(
                "SELECT reader_score, reasoning, model, scored_at FROM reader_scores"
                " WHERE article_key = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log.debug("[reader-score] cache read failed: %s", exc)
        return None
    if row is None:
        return None
    entry = {
        "readerScore": float(row[0]),
        "readerScoreReasoning": str(row[1] or ""),
        "readerScoreModel": str(row[2] or ""),
        "readerScoreScoredAt": int(row[3] or 0),
    }
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE[key] = dict(entry)
    return entry


def _store_reader_score(
    key: str, push: dict[str, Any], score: float, reasoning: str, model: str
) -> dict[str, Any]:
    entry = {
        "readerScore": round(float(score), 1),
        "readerScoreReasoning": reasoning,
        "readerScoreModel": model,
        "readerScoreScoredAt": int(time.time()),
    }
    with _MEMORY_CACHE_LOCK:
        _MEMORY_CACHE[key] = dict(entry)
    try:
        conn = _db_connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO reader_scores"
                " (article_key, url, title, reader_score, reasoning, model, scored_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    str(push.get("url") or push.get("link") or ""),
                    str(push.get("title") or push.get("headline") or ""),
                    entry["readerScore"],
                    reasoning,
                    model,
                    entry["readerScoreScoredAt"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("[reader-score] cache write failed: %s", exc)
    return entry


def _reader_score_enabled() -> bool:
    # Wie die Titel-Generierung: eigener Schalter + Key, unabhaengig vom
    # PAID_EXTERNAL_APIS_ENABLED-Flag (das gilt fuer Adobe/Backfill).
    from app.config import OPENAI_API_KEY, OPENAI_READER_SCORE_ENABLED

    return bool(OPENAI_READER_SCORE_ENABLED and OPENAI_API_KEY)


def _call_llm(push: dict[str, Any]) -> dict[str, Any] | None:
    from app import config

    title = str(push.get("title") or push.get("headline") or "").strip()
    if not title:
        return None
    if not allow_calls(
        [
            (
                "openai_reader_score_hour",
                config.OPENAI_READER_SCORE_MAX_CALLS_PER_HOUR,
                3600,
            ),
            (
                "openai_reader_score_day",
                config.OPENAI_READER_SCORE_MAX_CALLS_PER_DAY,
                86400,
            ),
        ]
    ):
        log.warning("[reader-score] call budget exhausted; falling back to heuristic")
        return None

    body_text = str(
        push.get("text") or push.get("articleText") or push.get("description") or ""
    ).strip()
    user_content = f"Titel: {title}"
    if body_text:
        user_content += f"\n\nArtikeltext: {body_text[:4000]}"

    model = config.OPENAI_READER_SCORE_MODEL
    token_argument = _completion_token_argument(model)
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": READER_SCORE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "timeout": config.OPENAI_READER_SCORE_TIMEOUT_S,
        "store": False,
        token_argument: config.OPENAI_READER_SCORE_MAX_TOKENS,
    }
    if token_argument == "max_tokens":
        request["temperature"] = 0.1
    else:
        request["extra_body"] = {
            "reasoning_effort": config.OPENAI_READER_SCORE_REASONING_EFFORT,
            "verbosity": "low",
        }

    client = _get_client(config.OPENAI_API_KEY)
    response = client.chat.completions.create(**request)
    text = (response.choices[0].message.content or "").strip()
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"reader score response is not JSON: {text[:120]!r}")
    data = json.loads(match.group())
    score = float(data.get("reader_score"))
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"reader score out of range: {score}")
    return {
        "score": score,
        "reasoning": str(data.get("reasoning") or "").strip(),
        "model": model,
    }


def get_or_create_reader_score(push: dict[str, Any]) -> dict[str, Any] | None:
    """Return the article's LLM reader score, calling the LLM at most once.

    Returns ``None`` when the LLM is disabled, over budget, or failing; the
    editorial scorer then uses its heuristic fallback for this article.
    """
    cached = get_cached_reader_score(push)
    if cached is not None:
        return cached
    if not _reader_score_enabled():
        return None

    key = reader_score_cache_key(push)
    with _INFLIGHT_LOCK:
        event = _INFLIGHT.get(key)
        if event is None:
            _INFLIGHT[key] = event = threading.Event()
            owner = True
        else:
            owner = False

    if not owner:
        from app import config

        event.wait(timeout=config.OPENAI_READER_SCORE_TIMEOUT_S + 2.0)
        return get_cached_reader_score(push)

    try:
        result = _call_llm(push)
        if result is None:
            return None
        return _store_reader_score(
            key, push, result["score"], result["reasoning"], result["model"]
        )
    except Exception as exc:
        log.warning("[reader-score] LLM scoring failed for %s: %s", key, exc)
        return None
    finally:
        event.set()
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(key, None)


def enrich_articles_with_reader_scores(
    articles: list[dict[str, Any]],
    *,
    max_new_calls: int | None = None,
    max_workers: int = 4,
) -> None:
    """Attach ``readerScore*`` fields to each article in place.

    Cached scores are attached for free; at most ``max_new_calls`` uncached
    articles trigger an LLM call (concurrently), so one slow feed request can
    never stack unbounded latency. Uncached leftovers simply stay heuristic
    until the next poll.
    """
    from app import config

    if max_new_calls is None:
        max_new_calls = config.OPENAI_READER_SCORE_MAX_CALLS_PER_REQUEST

    pending: list[dict[str, Any]] = []
    for article in articles:
        cached = get_cached_reader_score(article)
        if cached is not None:
            article.update(cached)
        else:
            pending.append(article)

    if not pending or max_new_calls <= 0 or not _reader_score_enabled():
        return

    batch = pending[:max_new_calls]
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(batch)))) as pool:
        futures = {pool.submit(get_or_create_reader_score, article): article for article in batch}
        for future in as_completed(futures):
            article = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                log.warning("[reader-score] enrichment worker failed: %s", exc)
                continue
            if entry is not None:
                article.update(entry)
