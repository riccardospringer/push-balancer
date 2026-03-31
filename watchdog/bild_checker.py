#!/usr/bin/env python3
"""
BILD-Overlap-Checker: Prüft ob ein Event bereits von BILD berichtet wurde.
Fetcht BILD News-Sitemap und cached Titel + Keyword-Sets im Memory.
"""

import asyncio
import logging
import re
import time
from typing import Optional
from xml.etree import ElementTree

import aiohttp
import certifi
import ssl

log = logging.getLogger("watchdog.bild_checker")

# ---------------------------------------------------------------------------
# Deutsche Stoppwörter (für Keyword-Overlap)
# ---------------------------------------------------------------------------

STOP_WORDS = frozenset([
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "und", "oder", "aber", "doch", "auch", "nur", "noch", "schon",
    "sehr", "mehr", "nach", "vor", "bei", "mit", "von", "aus", "auf", "für",
    "über", "unter", "zwischen", "durch", "gegen", "ohne", "bis", "seit",
    "wird", "wurde", "werden", "worden", "ist", "sind", "war", "waren",
    "hat", "haben", "hatte", "hatten", "kann", "können", "soll", "sollen",
    "muss", "müssen", "will", "wollen", "darf", "dürfen",
    "sich", "nicht", "kein", "keine", "keinen", "keinem",
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "man", "wie", "was", "wer", "wen", "wem", "wo", "wann", "warum",
    "hier", "dort", "jetzt", "dann", "so", "als", "wenn", "dass",
    "im", "am", "zum", "zur", "ins", "ans", "vom", "beim",
    "um", "zu", "in", "an",
])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_titles: list[str] = []
_keywords: list[set[str]] = []
_last_refresh: float = 0.0
_refresh_lock = asyncio.Lock()

SITEMAP_URL = "https://www.bild.de/sitemap-news.xml"
REFRESH_INTERVAL = 600  # 10 Minuten


def _tokenize(text: str) -> set[str]:
    """Text in Keyword-Set umwandeln (lowercase, ohne Stoppwörter, min 3 Zeichen)."""
    words = re.findall(r"[a-zäöüß]{3,}", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard-Similarity zwischen zwei Sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def needs_refresh() -> bool:
    """True wenn letzter Refresh > 10 Min her."""
    return (time.time() - _last_refresh) > REFRESH_INTERVAL


async def refresh(session: Optional[aiohttp.ClientSession] = None) -> int:
    """BILD News-Sitemap laden und Titel cachen. Gibt Anzahl Titel zurück."""
    global _titles, _keywords, _last_refresh

    async with _refresh_lock:
        # Double-check nach Lock
        if not needs_refresh():
            return len(_titles)

        own_session = False
        if session is None:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            session = aiohttp.ClientSession(connector=connector)
            own_session = True

        try:
            async with session.get(
                SITEMAP_URL,
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "NewsWatchdog/1.0"},
            ) as resp:
                if resp.status != 200:
                    log.warning("BILD Sitemap HTTP %d", resp.status)
                    return len(_titles)
                xml_text = await resp.text()
        except Exception as e:
            log.warning("BILD Sitemap Fehler: %s", e)
            return len(_titles)
        finally:
            if own_session:
                await session.close()

        # XML parsen — News-Sitemap Namespace
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as e:
            log.warning("BILD Sitemap XML-Fehler: %s", e)
            return len(_titles)

        ns = {
            "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "news": "http://www.google.com/schemas/sitemap-news/0.9",
        }

        new_titles = []
        new_keywords = []

        for url_elem in root.findall(".//sm:url", ns):
            news_elem = url_elem.find("news:news", ns)
            if news_elem is None:
                continue
            title_elem = news_elem.find("news:title", ns)
            if title_elem is None or not title_elem.text:
                continue
            title = title_elem.text.strip()
            new_titles.append(title)
            new_keywords.append(_tokenize(title))

        if new_titles:
            _titles = new_titles
            _keywords = new_keywords
            _last_refresh = time.time()
            log.info("BILD-Checker: %d Titel geladen", len(_titles))
        else:
            log.warning("BILD-Checker: Keine Titel in Sitemap gefunden")

        return len(_titles)


def check_overlap(title: str) -> dict:
    """
    Prüft ob ein Titel mit BILD-Berichterstattung überlappt.

    Returns:
        {
            "has_overlap": bool,
            "score": float (0.0-1.0),
            "type": "exact" | "topic" | "none",
            "bild_title": str | None
        }
    """
    if not _titles:
        return {"has_overlap": False, "score": 0.0, "type": "none", "bild_title": None}

    title_kw = _tokenize(title)
    if not title_kw:
        return {"has_overlap": False, "score": 0.0, "type": "none", "bild_title": None}

    best_score = 0.0
    best_title = None

    for i, bild_kw in enumerate(_keywords):
        score = _jaccard(title_kw, bild_kw)
        if score > best_score:
            best_score = score
            best_title = _titles[i]

    if best_score >= 0.85:
        return {"has_overlap": True, "score": best_score, "type": "exact", "bild_title": best_title}
    elif best_score >= 0.35:
        return {"has_overlap": True, "score": best_score, "type": "topic", "bild_title": best_title}
    else:
        return {"has_overlap": False, "score": best_score, "type": "none", "bild_title": best_title}
