"""Tests fuer Re-Publish-Schutz, Video-Null und Event-Modus (Feedback 30.08.2026)."""

from __future__ import annotations

import datetime as dt
import importlib
import time

import pytest

from app.scoring import first_seen as fs
from app.scoring.editorial import (
    is_video_article,
    rebalance_push_mix,
    score_push_candidate,
)
from app.scoring.freshness import publication_age_hours

NOW_TS = 1_800_000_000


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.PUSH_DB_PATH", str(tmp_path / "first_seen.db"))
    fs._SCHEMA_READY = False
    with fs._MEMORY_LOCK:
        fs._MEMORY_FIRST_SEEN.clear()
    yield
    with fs._MEMORY_LOCK:
        fs._MEMORY_FIRST_SEEN.clear()


# ── Re-Publish-Schutz ───────────────────────────────────────────────────────


def test_republished_article_keeps_its_original_age():
    """Ein 20h alter Artikel darf durch Neu-Publikation nicht frisch werden."""
    article = {
        "url": "https://www.bild.de/regional/zwei-kleinkinder-am-bahnhof",
        "title": "Zwei Kleinkinder (2 und 3) in Windeln alleine am Bahnhof",
        "pubDate": _iso(NOW_TS - 20 * 3600),
    }

    # Erstsichtung gestern Nachmittag.
    fs.apply_first_seen_publication_floor([article], now_ts=NOW_TS - 20 * 3600)
    assert article["pubDate"] == _iso(NOW_TS - 20 * 3600)

    # Heute erneut publiziert: die Sitemap meldet "vor 10 Minuten".
    republished = {
        "url": article["url"],
        "title": article["title"],
        "pubDate": _iso(NOW_TS - 600),
    }
    fs.apply_first_seen_publication_floor([republished], now_ts=NOW_TS)

    assert republished["republishDetected"] is True
    assert republished["sitemapPubDate"] == _iso(NOW_TS - 600)
    age = publication_age_hours(republished["pubDate"], now_ts=NOW_TS)
    assert age is not None and age >= 19.5


def test_genuinely_new_article_keeps_its_publication_time():
    article = {
        "url": "https://www.bild.de/news/echte-neuigkeit",
        "title": "Echte Neuigkeit",
        "pubDate": _iso(NOW_TS - 300),
    }
    fs.apply_first_seen_publication_floor([article], now_ts=NOW_TS)

    assert "republishDetected" not in article
    assert article["pubDate"] == _iso(NOW_TS - 300)


def test_first_sighting_without_valid_date_anchors_the_article_now():
    """Ohne brauchbares Datum zaehlt die Sichtung — spaeteres Re-Publish faellt auf."""
    article = {"url": "https://www.bild.de/news/ohne-datum", "title": "Ohne Datum"}
    fs.apply_first_seen_publication_floor([article], now_ts=NOW_TS - 8 * 3600)

    later = {
        "url": article["url"],
        "title": article["title"],
        "pubDate": _iso(NOW_TS - 60),
    }
    fs.apply_first_seen_publication_floor([later], now_ts=NOW_TS)

    age = publication_age_hours(later["pubDate"], now_ts=NOW_TS)
    assert age is not None and age >= 7.5


def test_republished_article_is_dropped_by_the_twelve_hour_gate():
    from app.routers.feed import _fresh_article_candidates

    article = {
        "id": "kleinkinder",
        "url": "https://www.bild.de/regional/kleinkinder-bahnhof-gate",
        "title": "Zwei Kleinkinder alleine am Bahnhof",
        "pubDate": _iso(NOW_TS - 20 * 3600),
    }
    fs.apply_first_seen_publication_floor([article], now_ts=NOW_TS - 20 * 3600)

    republished = {
        "id": "kleinkinder",
        "url": article["url"],
        "title": article["title"],
        "pubDate": _iso(NOW_TS - 600),
    }
    fs.apply_first_seen_publication_floor([republished], now_ts=NOW_TS)

    assert _fresh_article_candidates([republished], now_ts=NOW_TS) == []


# ── Video-Null ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "article",
    [
        {"title": "Tor des Monats", "isVideo": True},
        {"title": "Tor des Monats", "type": "video"},
        {"title": "Normale Zeile", "url": "https://www.bild.de/sport/video/abc"},
        {"title": "Das Tor im Video"},
        {"title": "Hier sehen Sie den Moment"},
        {"title": "Der Clip zeigt die Szene"},
    ],
)
def test_video_articles_always_score_zero(article):
    now = int(time.time())
    assert is_video_article(article) is True
    scored = score_push_candidate(
        {**article, "cat": "sport", "hour": 12, "ts_num": now}, reader_score=99.0
    )
    assert scored["score"] == 0.0
    assert scored["isVideo"] is True
    assert scored["mixPriority"] == "niedrig"
    assert any("Video" in risk for risk in scored["risks"])


def test_text_article_is_not_treated_as_video():
    now = int(time.time())
    article = {
        "title": "Bund beschliesst Strombonus fuer Millionen Haushalte",
        "url": "https://www.bild.de/politik/strombonus",
        "cat": "politik",
        "hour": 12,
        "ts_num": now,
    }
    assert is_video_article(article) is False
    assert score_push_candidate(article, reader_score=80.0)["score"] > 0


def test_video_gets_zero_in_the_sitemap_base_score():
    from app.routers.feed import _build_article_score

    score, reason = _build_article_score("sport", "Das Tor im Video", _iso(NOW_TS), "video")
    assert score == 0.0
    assert "Video" in reason


def test_reader_score_enrichment_skips_videos(monkeypatch):
    from app.scoring import reader_score as rs

    monkeypatch.setattr("app.config.OPENAI_READER_SCORE_ENABLED", True)
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "sk-test")
    calls: list[str] = []

    def fake_call(push):
        calls.append(push["title"])
        return {"score": 70.0, "reasoning": "", "model": "test"}

    monkeypatch.setattr(rs, "_call_llm", fake_call)
    articles = [
        {"url": "https://www.bild.de/sport/video/tor", "title": "Tor im Video"},
        {"url": "https://www.bild.de/politik/rente", "title": "Rentenpaket beschlossen"},
    ]
    rs.enrich_articles_with_reader_scores(articles, max_new_calls=5)

    assert calls == ["Rentenpaket beschlossen"]


# ── Event-Modus ─────────────────────────────────────────────────────────────


def _reload_scoring(monkeypatch, enabled: bool):
    monkeypatch.setenv("PUSH_BALANCER_EVENT_MODE_ENABLED", "1" if enabled else "0")
    from app import config

    importlib.reload(config)
    from app.scoring import editorial

    importlib.reload(editorial)
    return editorial


def test_event_mode_lifts_election_articles_and_is_off_by_default(monkeypatch):
    from app import config

    assert config.PUSH_BALANCER_EVENT_MODE_ENABLED is False

    now = int(time.time())
    candidate = {
        "title": "Wahl-Hochrechnung: Regierung vor dem Aus",
        "cat": "politik",
        "hour": 20,
        "ts_num": now,
    }
    try:
        editorial = _reload_scoring(monkeypatch, True)
        active = editorial.score_push_candidate(dict(candidate), reader_score=85.0)
        sport = editorial.score_push_candidate(
            {"title": "Bayern gewinnt souveraen", "cat": "sport", "hour": 20, "ts_num": now},
            reader_score=85.0,
        )
        assert active["scoreBreakdown"]["eventModeAdjustment"] > 0
        assert sport["scoreBreakdown"]["eventModeAdjustment"] == 0.0
        assert active["score"] > sport["score"]
        assert any("Event-Modus" in d for d in active["performanceDrivers"])
    finally:
        _reload_scoring(monkeypatch, False)


def test_event_mode_suspends_the_politics_mix_cap(monkeypatch):
    now = int(time.time())
    politics = [
        {
            "title": f"Wahl-Analyse Nummer {index}",
            "cat": "politik",
            "url": f"https://www.bild.de/politik/wahl-{index}",
            "ts_num": now,
            "score": 90.0 - index,
            "scoreBreakdown": {"mixBalance": 70.0},
            "performanceDrivers": [],
            "risks": [],
        }
        for index in range(6)
    ]

    normal = rebalance_push_mix([dict(item) for item in politics], target_ts=now)
    normal_top = [item for item in normal[:6] if item["cat"] == "politik"]
    assert any(
        "Ressort politik" in " ".join(item.get("risks") or []) for item in normal_top
    )

    try:
        editorial = _reload_scoring(monkeypatch, True)
        event = editorial.rebalance_push_mix([dict(item) for item in politics], target_ts=now)
        joined_risks = [" ".join(item.get("risks") or []) for item in event]
        assert not any("Ressort politik" in risks for risks in joined_risks)
        assert not any("Thema politik" in risks for risks in joined_risks)
        # Reihenfolge bleibt die des rohen Push Scores.
        assert [item["title"] for item in event] == [item["title"] for item in politics]
    finally:
        _reload_scoring(monkeypatch, False)
