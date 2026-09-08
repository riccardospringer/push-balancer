"""Tests fuer Re-Publish-Schutz, Video-Null und Event-Modus (Feedback 30.08.2026)."""

from __future__ import annotations

import datetime as dt
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


# ── Automatische Grosslagen-Erkennung ───────────────────────────────────────


def _event_article(title: str, *, age_h: float = 0.5, ticker: bool = False, breaking: bool = False):
    slug = "liveticker/lage" if ticker else "artikel"
    return {
        "title": title,
        "url": f"https://www.bild.de/politik/{slug}",
        "pubDate": _iso(int(NOW_TS - age_h * 3600)),
        "isBreaking": breaking,
    }


def _election_night_field() -> list[dict]:
    return [
        _event_article("Wahlabend: Erste Hochrechnung sieht die CDU vorn", age_h=0.3, ticker=True),
        _event_article("Hochrechnung 20:15 Uhr: Koalitionsverhandlung wird schwierig", age_h=0.6),
        _event_article("Wahllokale geschlossen: Die Auszaehlung laeuft", age_h=1.0),
        _event_article("Wahlergebnis in Sachsen-Anhalt: AfD deutlich vorn", age_h=1.5),
        _event_article("Wahlsieg gefeiert: Jubel in der Parteizentrale", age_h=2.5),
    ]


def test_election_night_is_detected_automatically():
    from app.scoring.events import detect_active_event_groups

    assert detect_active_event_groups(_election_night_field(), now_ts=NOW_TS) == {"wahl"}


def test_routine_coverage_does_not_trigger_event_mode():
    """Dauerthemen duerfen die Mix-Deckel nicht permanent aushebeln."""
    from app.scoring.events import detect_active_event_groups

    routine = [
        _event_article("Luftangriff auf Kiew gemeldet", age_h=5.0),
        _event_article("Waffenruhe bleibt weiter unklar", age_h=7.0),
        _event_article("Invasion dauert an", age_h=9.0),
        _event_article("Raketenangriff auf Odessa", age_h=10.0),
        _event_article("Grossangriff abgewehrt", age_h=11.0),
    ]
    assert detect_active_event_groups(routine, now_ts=NOW_TS) == set()


def test_event_needs_enough_articles_and_a_running_signal():
    from app.scoring.events import detect_active_event_groups

    # Zu wenige Artikel.
    assert detect_active_event_groups(_election_night_field()[:3], now_ts=NOW_TS) == set()

    # Genug Artikel und frisch, aber kein Ticker/keine Eilmeldung.
    without_signal = [
        {**article, "url": "https://www.bild.de/politik/artikel", "isBreaking": False}
        for article in _election_night_field()
    ]
    assert detect_active_event_groups(without_signal, now_ts=NOW_TS) == set()

    # Genug Artikel und Signal, aber nichts davon frisch.
    stale = [
        {**article, "pubDate": _iso(NOW_TS - 8 * 3600)} for article in _election_night_field()
    ]
    assert detect_active_event_groups(stale, now_ts=NOW_TS) == set()


def test_detected_event_lifts_its_articles_but_not_unrelated_ones():
    from app.scoring.events import detect_active_event_groups

    now = int(time.time())
    active = detect_active_event_groups(_election_night_field(), now_ts=NOW_TS)

    election = score_push_candidate(
        {
            "title": "Wahlergebnis in Sachsen-Anhalt: AfD deutlich vorn",
            "cat": "politik",
            "hour": 20,
            "ts_num": now,
        },
        reader_score=80.0,
        active_events=active,
    )
    sport = score_push_candidate(
        {"title": "Bayern gewinnt souveraen", "cat": "sport", "hour": 20, "ts_num": now},
        reader_score=80.0,
        active_events=active,
    )

    assert election["scoreBreakdown"]["eventModeAdjustment"] > 0
    assert sport["scoreBreakdown"]["eventModeAdjustment"] == 0.0
    assert election["score"] > sport["score"]
    assert any("Event-Modus" in driver for driver in election["performanceDrivers"])


def test_detected_event_suspends_the_politics_mix_caps():
    now = int(time.time())
    field = _election_night_field()
    candidates = [
        {
            **article,
            "cat": "politik",
            "ts_num": now,
            "score": 90.0 - index,
            "scoreBreakdown": {"mixBalance": 70.0},
            "performanceDrivers": [],
            "risks": [],
        }
        for index, article in enumerate(field)
    ]

    balanced = rebalance_push_mix([dict(item) for item in candidates], target_ts=now)
    joined = [" ".join(item.get("risks") or []) for item in balanced]

    assert not any("Ressort politik" in risks for risks in joined)
    assert not any("Thema politik" in risks for risks in joined)
    assert [item["title"] for item in balanced] == [item["title"] for item in candidates]


def test_ordinary_politics_field_keeps_its_mix_caps():
    """Ohne Grosslage bleiben die Deckel unveraendert wirksam."""
    now = int(time.time())
    candidates = [
        {
            "title": f"Debatte im Bundestag Teil {index}",
            "url": f"https://www.bild.de/politik/debatte-{index}",
            "cat": "politik",
            "pubDate": _iso(NOW_TS - 3 * 3600),
            "ts_num": now,
            "score": 90.0 - index,
            "scoreBreakdown": {"mixBalance": 70.0},
            "performanceDrivers": [],
            "risks": [],
        }
        for index in range(6)
    ]

    balanced = rebalance_push_mix(candidates, target_ts=now)
    joined = " ".join(" ".join(item.get("risks") or []) for item in balanced)
    assert "Ressort politik" in joined


def test_event_mode_can_be_forced_off(monkeypatch):
    from app.scoring.events import detect_active_event_groups

    monkeypatch.setattr("app.config.PUSH_BALANCER_EVENT_MODE", "off")
    assert detect_active_event_groups(_election_night_field(), now_ts=NOW_TS) == set()


# ── Speicher-Deckel ─────────────────────────────────────────────────────────


def test_first_seen_memory_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(fs, "_MEMORY_MAX_ENTRIES", 10)
    for index in range(40):
        fs._remember(f"key-{index}", NOW_TS + index)

    with fs._MEMORY_LOCK:
        assert len(fs._MEMORY_FIRST_SEEN) == 10
        # Die juengsten Eintraege bleiben erhalten.
        assert "key-39" in fs._MEMORY_FIRST_SEEN
        assert "key-0" not in fs._MEMORY_FIRST_SEEN


def test_reader_score_memory_cache_is_bounded(monkeypatch):
    from app.scoring import reader_score as rs

    monkeypatch.setattr(rs, "_MEMORY_CACHE_MAX_ENTRIES", 10)
    with rs._MEMORY_CACHE_LOCK:
        rs._MEMORY_CACHE.clear()
    for index in range(40):
        rs._remember_reader_score(f"key-{index}", {"readerScore": float(index)})

    with rs._MEMORY_CACHE_LOCK:
        assert len(rs._MEMORY_CACHE) == 10
        assert "key-39" in rs._MEMORY_CACHE
        assert "key-0" not in rs._MEMORY_CACHE
    with rs._MEMORY_CACHE_LOCK:
        rs._MEMORY_CACHE.clear()
