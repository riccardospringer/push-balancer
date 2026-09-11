"""Tests for the LLM-based BILD reader score (one call per article)."""

from __future__ import annotations

import time

import pytest

from app.scoring import reader_score as rs
from app.scoring.editorial import score_push_candidate


@pytest.fixture(autouse=True)
def _isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.PUSH_DB_PATH", str(tmp_path / "reader.db"))
    with rs._MEMORY_CACHE_LOCK:
        rs._MEMORY_CACHE.clear()
    yield
    with rs._MEMORY_CACHE_LOCK:
        rs._MEMORY_CACHE.clear()


def _enable_llm(monkeypatch):
    monkeypatch.setattr("app.config.PAID_EXTERNAL_APIS_ENABLED", True)
    monkeypatch.setattr("app.config.OPENAI_READER_SCORE_ENABLED", True)
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "sk-test")


def test_reader_score_is_called_exactly_once_per_article(monkeypatch):
    _enable_llm(monkeypatch)
    calls: list[str] = []

    def fake_call(push):
        calls.append(push["title"])
        return {"score": 77.0, "reasoning": "Testbegruendung", "model": "test-model"}

    monkeypatch.setattr(rs, "_call_llm", fake_call)

    push = {"url": "https://www.bild.de/news/test-artikel", "title": "Test-Artikel"}
    first = rs.get_or_create_reader_score(push)
    second = rs.get_or_create_reader_score(push)

    assert first is not None and first["readerScore"] == 77.0
    assert second is not None and second["readerScore"] == 77.0
    assert len(calls) == 1

    # Persistenz: auch nach Leeren des Memory-Caches keine zweite Abrechnung.
    with rs._MEMORY_CACHE_LOCK:
        rs._MEMORY_CACHE.clear()
    third = rs.get_or_create_reader_score(push)
    assert third is not None and third["readerScore"] == 77.0
    assert len(calls) == 1


def test_reader_score_disabled_returns_none(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_READER_SCORE_ENABLED", False)
    assert rs.get_or_create_reader_score({"url": "https://x.invalid/a", "title": "A"}) is None


def test_reader_score_failure_falls_back_to_none(monkeypatch):
    _enable_llm(monkeypatch)

    def broken_call(push):
        raise RuntimeError("api down")

    monkeypatch.setattr(rs, "_call_llm", broken_call)
    assert rs.get_or_create_reader_score({"url": "https://x.invalid/b", "title": "B"}) is None


def test_enrichment_is_bounded_per_request(monkeypatch):
    _enable_llm(monkeypatch)
    calls: list[str] = []

    def fake_call(push):
        calls.append(push["title"])
        return {"score": 50.0, "reasoning": "", "model": "test-model"}

    monkeypatch.setattr(rs, "_call_llm", fake_call)

    articles = [
        {"url": f"https://www.bild.de/news/artikel-{index}", "title": f"Artikel {index}"}
        for index in range(6)
    ]
    rs.enrich_articles_with_reader_scores(articles, max_new_calls=3)

    assert len(calls) == 3
    assert sum(1 for article in articles if article.get("readerScore") is not None) == 3


def test_llm_reader_score_drives_forty_percent_of_the_push_score():
    now = int(time.time())
    base = {"title": "Neutraler Artikel ohne Signalwoerter", "cat": "news", "hour": 10, "ts_num": now}
    low = score_push_candidate(dict(base), reader_score=0.0)
    high = score_push_candidate(dict(base), reader_score=100.0)

    assert high["scoreBreakdown"]["bildReiz"] == 100.0
    assert low["scoreBreakdown"]["bildReiz"] == 0.0
    assert high["scoreBreakdown"]["bildReizSource"] == "llm_reader_score"
    # 100 Punkte Unterschied im Reader-Score = 40 Punkte im Gesamtscore
    # (BILD-Reiz-Gewicht 40 %), plus dem indirekten Anteil im
    # Öffnungs-Potenzial. Der direkte Anteil ist die Untergrenze.
    assert high["score"] - low["score"] >= 40.0
    assert high["readerScore"] == 100.0


def test_missing_reader_score_uses_heuristic_fallback():
    now = int(time.time())
    scored = score_push_candidate(
        {"title": "Messer-Angriff: Polizei fasst Täter", "cat": "news", "hour": 10, "ts_num": now}
    )
    assert scored["scoreBreakdown"]["bildReizSource"] == "heuristik_fallback"
    assert scored["readerScore"] is None


def test_invalid_reader_score_is_rejected():
    now = int(time.time())
    scored = score_push_candidate(
        {"title": "Irgendein Artikel", "cat": "news", "hour": 10, "ts_num": now},
        reader_score=140.0,
    )
    assert scored["scoreBreakdown"]["bildReizSource"] == "heuristik_fallback"


def test_breaking_news_taxonomy_marks_article_as_eilmeldung():
    from app.scoring.editorial import is_breaking_news_taxonomy

    assert is_breaking_news_taxonomy({"taxonomy": ["Politik", "Breaking News"]})
    assert is_breaking_news_taxonomy({"taxonomyNodes": [{"name": "Eilmeldung"}]})
    assert not is_breaking_news_taxonomy({"taxonomy": ["Fussball", "Spielbericht"]})


def test_reader_scores_endpoint_returns_cached_scores(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    _enable_llm(monkeypatch)

    def fake_call(push):
        return {"score": 81.0, "reasoning": "Starke Story", "model": "test-model"}

    monkeypatch.setattr(rs, "_call_llm", fake_call)

    client = TestClient(app)
    response = client.post(
        "/api/reader-scores",
        json={
            "articles": [
                {"url": "https://www.bild.de/news/endpoint-artikel", "title": "Endpoint-Artikel"}
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["weight"] == 0.4
    entry = payload["scores"]["https://www.bild.de/news/endpoint-artikel"]
    assert entry["readerScore"] == 81.0
    assert entry["readerScoreReasoning"] == "Starke Story"


def test_cache_generation_separates_prompt_and_model_changes(monkeypatch):
    """A prompt or model swap must not serve judgements from the old setup.

    The cache has no expiry, so a stale generation would silently keep scoring
    articles with the retired prompt/model for as long as they stay in the feed.
    """
    article = {"url": "https://www.bild.de/politik/beispiel-123.bild.html"}

    monkeypatch.setattr("app.config.OPENAI_READER_SCORE_MODEL", "gpt-5.6-luna")
    rs._reader_score_generation.cache_clear()
    luna_key = rs.reader_score_cache_key(article)

    monkeypatch.setattr("app.config.OPENAI_READER_SCORE_MODEL", "gpt-5.6-terra")
    rs._reader_score_generation.cache_clear()
    terra_key = rs.reader_score_cache_key(article)

    assert luna_key != terra_key
    # Same article identity, different generation prefix.
    assert luna_key.split(":", 1)[1] == terra_key.split(":", 1)[1]

    changed_prompt = rs._reader_score_generation("gpt-5.6-terra")
    monkeypatch.setattr(rs, "READER_SCORE_PROMPT", rs.READER_SCORE_PROMPT + " Zusatzregel.")
    rs._reader_score_generation.cache_clear()
    assert rs._reader_score_generation("gpt-5.6-terra") != changed_prompt


def test_reader_score_model_uses_gpt5_parameter_family():
    """Terra must take the gpt-5 argument shape, like the model it replaces."""
    from app import config

    assert config.OPENAI_READER_SCORE_MODEL == "gpt-5.6-terra"
    assert rs._completion_token_argument(config.OPENAI_READER_SCORE_MODEL) == (
        "max_completion_tokens"
    )
