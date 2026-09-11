"""Tests fuer die CMS-Dokumenttyp-Erkennung (Artikel vs. Video)."""

from __future__ import annotations

import io
import time

import pytest

from app.scoring import document_type as dt
from app.scoring.editorial import score_push_candidate

VIDEO_URL = "https://www.bild.de/politik/ausland/reporter-berichten-ich-dachte-ich-ersticke-6a9a84d2"
ARTICLE_URL = "https://www.bild.de/sport/fussball/hertha-legende-adelt-thorsteinsson-6aa3be1a"

VIDEO_HEAD = b'<html><head><meta property="og:type" content="video"><title>x</title>'
ARTICLE_HEAD = b'<html><head><meta property="og:type" content="article"><title>x</title>'
SCHEMA_ONLY_HEAD = b'<html><head><script type="application/ld+json">{"@type":"VideoObject"}</script>'


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.PUSH_DB_PATH", str(tmp_path / "doctype.db"))
    dt._SCHEMA_READY = False
    with dt._MEMORY_LOCK:
        dt._MEMORY_CACHE.clear()
    yield
    with dt._MEMORY_LOCK:
        dt._MEMORY_CACHE.clear()


def _fake_urlopen(bodies: dict[str, bytes], calls: list[str] | None = None):
    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def _open(request, timeout=None, context=None):
        url = request.full_url
        if calls is not None:
            calls.append(url)
        assert request.headers.get("Range", "").startswith("bytes=0-")
        if url not in bodies:
            raise OSError("not found")
        return _Response(bodies[url])

    return _open


def test_video_document_is_detected_from_the_public_page(monkeypatch):
    monkeypatch.setattr(
        dt.urllib.request,
        "urlopen",
        _fake_urlopen({VIDEO_URL: VIDEO_HEAD, ARTICLE_URL: ARTICLE_HEAD}),
    )
    articles = [
        {"url": VIDEO_URL, "title": "Reporter berichten"},
        {"url": ARTICLE_URL, "title": "Hertha-Legende adelt Thorsteinsson"},
    ]
    dt.annotate_document_types(articles, max_new_probes=10, wait_budget_s=10)

    assert articles[0]["documentType"] == "video"
    assert articles[0]["isVideo"] is True
    assert articles[0]["type"] == "video"
    assert articles[1]["documentType"] == "article"
    assert not articles[1].get("isVideo")


def test_schema_video_object_counts_as_video(monkeypatch):
    monkeypatch.setattr(
        dt.urllib.request, "urlopen", _fake_urlopen({VIDEO_URL: SCHEMA_ONLY_HEAD})
    )
    article = {"url": VIDEO_URL, "title": "Ohne og:type"}
    dt.annotate_document_types([article], max_new_probes=5, wait_budget_s=10)

    assert article["isVideo"] is True


def test_each_article_is_probed_only_once(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        dt.urllib.request, "urlopen", _fake_urlopen({VIDEO_URL: VIDEO_HEAD}, calls)
    )
    for _ in range(3):
        article = {"url": VIDEO_URL, "title": "Reporter berichten"}
        dt.annotate_document_types([article], max_new_probes=5, wait_budget_s=10)
        assert article["isVideo"] is True

    assert len(calls) == 1

    # Auch nach Verlust des Speicher-Caches bleibt es bei einem Abruf.
    with dt._MEMORY_LOCK:
        dt._MEMORY_CACHE.clear()
    again = {"url": VIDEO_URL, "title": "Reporter berichten"}
    dt.annotate_document_types([again], max_new_probes=5, wait_budget_s=10)
    assert again["isVideo"] is True
    assert len(calls) == 1


def test_probe_failure_leaves_the_article_untouched(monkeypatch):
    def _boom(request, timeout=None, context=None):
        raise OSError("network down")

    monkeypatch.setattr(dt.urllib.request, "urlopen", _boom)
    article = {"url": ARTICLE_URL, "title": "Hertha-Legende"}
    dt.annotate_document_types([article], max_new_probes=5, wait_budget_s=5)

    assert "documentType" not in article
    assert not article.get("isVideo")


def test_probe_budget_limits_new_lookups(monkeypatch):
    calls: list[str] = []
    bodies = {f"https://www.bild.de/news/artikel-{i}": ARTICLE_HEAD for i in range(10)}
    monkeypatch.setattr(dt.urllib.request, "urlopen", _fake_urlopen(bodies, calls))
    articles = [{"url": url, "title": "x"} for url in bodies]

    dt.annotate_document_types(articles, max_new_probes=3, wait_budget_s=10)

    assert len(calls) == 3


def test_detected_video_scores_zero_end_to_end(monkeypatch):
    monkeypatch.setattr(dt.urllib.request, "urlopen", _fake_urlopen({VIDEO_URL: VIDEO_HEAD}))
    article = {"url": VIDEO_URL, "title": "Reporter berichten über ihre Erfahrungen"}
    dt.annotate_document_types([article], max_new_probes=5, wait_budget_s=10)

    scored = score_push_candidate(
        {
            "title": article["title"],
            "url": article["url"],
            "isVideo": article.get("isVideo"),
            "type": article.get("type"),
            "cat": "politik",
            "hour": 14,
            "ts_num": int(time.time()),
        },
        reader_score=90.0,
    )

    assert scored["score"] == 0.0
    assert scored["isVideo"] is True


def test_feed_payload_zeroes_detected_videos_even_with_a_captured_score(monkeypatch):
    """Ein altes Browser-Capture darf ein erkanntes Video nicht wiederbeleben."""
    import datetime as _dt

    from app.routers import feed

    published = (
        _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=30)
    ).isoformat()
    sitemap = f"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
        xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
  <url>
    <loc>{VIDEO_URL}</loc>
    <news:news>
      <news:title>Reporter berichten über ihre Erfahrungen</news:title>
      <news:publication_date>{published}</news:publication_date>
    </news:news>
  </url>
</urlset>""".encode()

    monkeypatch.setattr(feed, "_fetch_url", lambda _url: sitemap)
    monkeypatch.setattr(feed, "ARTICLE_PREDICTION_ENRICHMENT_ENABLED", False)
    monkeypatch.setattr(dt.urllib.request, "urlopen", _fake_urlopen({VIDEO_URL: VIDEO_HEAD}))
    monkeypatch.setattr(
        "app.routers.score_capture.get_score_snapshot_for_url",
        lambda *_a, **_k: {
            "score": 88.0,
            "capturedAt": int(time.time()),
            "ageSeconds": 10,
            "source": "memory",
        },
    )

    payload = feed.build_articles_payload(include_teams_decisions=False)
    article = payload["articles"][0]

    assert article["isVideo"] is True
    assert article["score"] == 0.0
    assert article["scoreSource"] == "server_editorial_video"
