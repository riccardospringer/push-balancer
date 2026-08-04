"""Wirkungsmessung: Annahmequote und Opening-Rate-Vergleich."""

import sqlite3
import time

import app.database as _db
from app.database import push_db_upsert
from app.notifications.teams_effectiveness import build_effectiveness_report


def _push(message_id: str, ts: int, *, or_val: float, title: str, link: str, cat="news"):
    return {
        "message_id": message_id,
        "ts_num": ts,
        "or": or_val,
        "title": title,
        "headline": title,
        "kicker": "",
        "cat": cat,
        "link": link,
        "type": "editorial",
        "hour": 12,
        "title_len": len(title),
        "opened": 100,
        "received": 1000,
        "channel": "",
        "channels": [],
        "is_eilmeldung": False,
        "target_stats": {},
        "app_list": [],
        "n_apps": 0,
        "total_recipients": 1000,
    }


def _record_recommendation(url: str, title: str, sent_at: int, score: float = 88.0):
    conn = sqlite3.connect(_db.PUSH_DB_PATH)
    conn.execute(
        """INSERT INTO teams_recommendations (
               id, article_key, article_url, article_title, section,
               recommendation_type, status, should_notify, score, predicted_or,
               decided_at_ts, sent_at_ts, send_status, created_ts, updated_ts
           ) VALUES (?, ?, ?, ?, 'news', 'teams_alert', 'sent', 1, ?, 6.0,
                     ?, ?, 'sent', ?, ?)""",
        (f"rec-{sent_at}-{url[-12:]}", url, url, title, score, sent_at, sent_at,
         sent_at, sent_at),
    )
    conn.commit()
    conn.close()


def test_report_is_empty_but_valid_without_data(tmp_db):
    report = build_effectiveness_report(7)

    assert report["ok"] is True
    assert report["recommendationsSent"] == 0
    assert "Noch keine gesendeten Empfehlungen" in report["summary"]


def test_followed_recommendation_is_detected_by_url(tmp_db):
    now = int(time.time())
    url = "https://www.bild.de/politik/rentenpaket-beschlossen"
    _record_recommendation(url, "Rentenpaket beschlossen", now - 3600)
    push_db_upsert([
        _push("p1", now - 3000, or_val=7.4, title="Rentenpaket beschlossen", link=url),
        _push("p2", now - 2000, or_val=4.1, title="Ganz andere Meldung heute",
              link="https://www.bild.de/news/andere-meldung"),
    ])

    report = build_effectiveness_report(7)

    assert report["recommendationsSent"] == 1
    assert report["recommendationsFollowed"] == 1
    assert report["acceptanceRatePercent"] == 100.0
    assert report["openingRate"]["followedAvg"] == 7.4
    assert report["openingRate"]["otherAvg"] == 4.1
    assert report["openingRate"]["uplift"] > 0
    assert "ueber den uebrigen Pushes" in report["summary"]


def test_ignored_recommendation_lowers_the_acceptance_rate(tmp_db):
    now = int(time.time())
    followed_url = "https://www.bild.de/politik/umgesetzt"
    ignored_url = "https://www.bild.de/politik/ignoriert"
    _record_recommendation(followed_url, "Kanzler kuendigt Reform an", now - 7200)
    _record_recommendation(ignored_url, "Voellig anderes Thema ohne Umsetzung", now - 5400)
    push_db_upsert([
        _push("p1", now - 7000, or_val=6.5, title="Kanzler kuendigt Reform an",
              link=followed_url),
    ])

    report = build_effectiveness_report(7)

    assert report["recommendationsSent"] == 2
    assert report["recommendationsFollowed"] == 1
    assert report["acceptanceRatePercent"] == 50.0


def test_push_outside_the_follow_window_does_not_count(tmp_db):
    now = int(time.time())
    url = "https://www.bild.de/politik/spaeter-push"
    _record_recommendation(url, "Sehr wichtige Meldung heute", now - 20 * 3600)
    push_db_upsert([
        _push("p1", now - 15 * 3600, or_val=6.0, title="Sehr wichtige Meldung heute",
              link=url),
    ])

    report = build_effectiveness_report(7)

    assert report["recommendationsFollowed"] == 0
    assert report["acceptanceRatePercent"] == 0.0


def test_similar_title_counts_even_with_a_different_url(tmp_db):
    """Die Redaktion pusht oft eine andere URL zur selben Story."""
    now = int(time.time())
    _record_recommendation(
        "https://www.bild.de/politik/bahnstreik-artikel-a",
        "Bahnstreik legt Fernverkehr bundesweit lahm",
        now - 3600,
    )
    push_db_upsert([
        _push("p1", now - 3000, or_val=7.0,
              title="Bahnstreik legt Fernverkehr bundesweit lahm",
              link="https://www.bild.de/news/bahnstreik-artikel-b"),
    ])

    report = build_effectiveness_report(7)

    assert report["recommendationsFollowed"] == 1


def test_negative_uplift_is_reported_honestly(tmp_db):
    """Wenn befolgte Pushes schlechter laufen, muss das im Klartext stehen."""
    now = int(time.time())
    url = "https://www.bild.de/politik/schwache-empfehlung"
    _record_recommendation(url, "Empfohlene aber schwache Meldung", now - 3600)
    push_db_upsert([
        _push("p1", now - 3000, or_val=3.0, title="Empfohlene aber schwache Meldung",
              link=url),
        _push("p2", now - 2000, or_val=8.0, title="Starke redaktionelle Eigenwahl",
              link="https://www.bild.de/news/eigenwahl"),
    ])

    report = build_effectiveness_report(7)

    assert report["openingRate"]["uplift"] < 0
    assert "unter den uebrigen Pushes" in report["summary"]
