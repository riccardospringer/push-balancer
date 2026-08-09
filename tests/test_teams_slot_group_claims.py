"""Synthetic tests for durable five-article Teams slot claims."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from app import database
from app.teams_slot_claims import (
    teams_recommendation_slot_fail_if_owned,
    teams_recommendation_slot_get,
    teams_recommendation_slot_group_get,
    teams_recommendation_slot_group_owned,
    teams_recommendation_slot_record,
    teams_recommendation_slot_record_group_receipt,
    teams_recommendation_slot_record_receipt,
    teams_recommendation_slot_cleanup,
    teams_recommendation_slot_try_claim,
    teams_recommendation_slot_try_claim_group,
)


def _articles() -> list[dict]:
    return [
        {
            "article_key": f"https://www.bild.de/news/synthetic-group-{position}",
            "article_id": f"synthetic-group-{position}",
            "article_url": f"https://www.bild.de/news/synthetic-group-{position}",
            "article_title": f"Synthetische Gruppenmeldung {position}",
            "title_hash": hashlib.sha256(
                f"synthetische gruppenmeldung {position}".encode()
            ).hexdigest(),
            "score": 95.0 - position,
            "predicted_or": 0.05,
            "candidate_updated_at": 1_700_000_000 + position,
            "is_breaking": position == 1,
            "reason": f"Synthetischer Gruppenrang {position}",
        }
        for position in range(1, 6)
    ]


def _payload(slot_ts: int) -> dict:
    return {
        "ready": "yes",
        "contractVersion": 2,
        "slotId": f"teams-recommendation-{slot_ts}",
        "recommendationCount": 5,
        "messageHtml": "".join(
            f"<p><strong>Top {position}:</strong> Synthetisch</p>" for position in range(1, 6)
        ),
    }


def _claim_group(slot_ts: int, *, request_id: str = "synthetic-group-run") -> dict:
    return teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=_articles(),
        request_id=request_id,
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )


def test_group_claim_binds_exactly_five_pseudonymous_articles(tmp_db):
    slot_ts = int(time.time())

    claimed = _claim_group(slot_ts)
    replayed = _claim_group(slot_ts)
    items = teams_recommendation_slot_group_get(slot_ts)
    owned = teams_recommendation_slot_group_owned(
        slot_ts,
        request_id="synthetic-group-run",
        now_ts=slot_ts + 1,
    )

    assert claimed == {
        "claimed": True,
        "reason": "claimed",
        "bindingSlotTs": slot_ts,
        "itemCount": 5,
    }
    assert replayed["claimed"] is True
    assert replayed["reason"] == "replayed"
    assert replayed["replayPayload"] == _payload(slot_ts)
    assert [item["position"] for item in items] == [1, 2, 3, 4, 5]
    assert {item["status"] for item in items} == {"sending"}
    assert all(len(item["article_ref"]) == 64 for item in items)
    assert all("bild.de" not in item["article_ref"] for item in items)
    assert owned == {"owned": True, "reason": "owned", "itemCount": 5}

    alerts = [database.teams_alert_get(item["article_key"]) for item in _articles()]
    assert all(alert is not None for alert in alerts)
    assert {alert["status"] for alert in alerts if alert} == {"sending"}
    assert {alert["last_decision_ts"] for alert in alerts if alert} == {slot_ts}


def test_group_claim_rejects_wrong_count_and_duplicate_without_writes(tmp_db):
    slot_ts = int(time.time())
    four = _articles()[:4]
    duplicate = _articles()
    duplicate[-1] = dict(duplicate[0])

    wrong_count = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=four,
        request_id="synthetic-four-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )
    duplicate_result = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=duplicate,
        request_id="synthetic-duplicate-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )

    assert wrong_count == {"claimed": False, "reason": "invalid_slot_group_claim"}
    assert duplicate_result == {
        "claimed": False,
        "reason": "invalid_slot_group_claim",
    }
    assert teams_recommendation_slot_get(slot_ts) is None
    assert teams_recommendation_slot_group_get(slot_ts) == []


def test_group_claim_rejects_cms_and_canonical_url_aliases(tmp_db):
    slot_ts = int(time.time())
    same_cms = _articles()
    same_cms[-1] = {
        **same_cms[-1],
        "article_key": "https://www.bild.de/news/synthetic-cms-alias",
        "article_id": same_cms[0]["article_id"],
        "article_url": "https://www.bild.de/news/synthetic-cms-alias",
    }
    same_url = _articles()
    same_url[-1] = {
        **same_url[-1],
        "article_key": "https://bild.de/news/synthetic-group-1/amp?output=1",
        "article_id": "",
        "article_url": "https://bild.de/news/synthetic-group-1/amp?output=1",
    }
    same_url[0] = {**same_url[0], "article_id": ""}

    cms_result = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=same_cms,
        request_id="synthetic-cms-alias-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )
    url_result = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=same_url,
        request_id="synthetic-url-alias-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )

    assert cms_result == {"claimed": False, "reason": "invalid_slot_group_claim"}
    assert url_result == {"claimed": False, "reason": "invalid_slot_group_claim"}
    assert teams_recommendation_slot_get(slot_ts) is None
    assert teams_recommendation_slot_group_get(slot_ts) == []


def test_group_claim_is_all_or_nothing_when_one_article_is_terminal(tmp_db):
    slot_ts = int(time.time())
    articles = _articles()
    terminal = articles[-1]
    database.teams_alert_record(
        article_key=terminal["article_key"],
        article_id=terminal["article_id"],
        article_url=terminal["article_url"],
        article_title=terminal["article_title"],
        title_hash=terminal["title_hash"],
        score=terminal["score"],
        predicted_or=terminal["predicted_or"],
        candidate_updated_at=terminal["candidate_updated_at"],
        is_breaking=False,
        reason="synthetic terminal state",
        status="sent",
        decision_ts=slot_ts - 1,
    )

    result = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=articles,
        request_id="synthetic-blocked-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )

    assert result == {"claimed": False, "reason": "article_already_sent"}
    assert teams_recommendation_slot_get(slot_ts) is None
    assert teams_recommendation_slot_group_get(slot_ts) == []
    assert all(
        database.teams_alert_get(article["article_key"]) is None for article in articles[:-1]
    )


def test_parallel_group_claims_have_one_winner_and_one_complete_group(tmp_db):
    slot_ts = int(time.time())

    def claim(request_id: str) -> dict:
        return teams_recommendation_slot_try_claim_group(
            slot_ts,
            articles=_articles(),
            request_id=request_id,
            claim_payload=_payload(slot_ts),
            now_ts=slot_ts,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, ("synthetic-parallel-a", "synthetic-parallel-b")))

    assert sum(result.get("claimed") is True for result in results) == 1
    assert {result["reason"] for result in results} == {
        "claimed",
        "slot_send_in_progress",
    }
    assert len(teams_recommendation_slot_group_get(slot_ts)) == 5
    assert {
        database.teams_alert_get(article["article_key"])["status"] for article in _articles()
    } == {"sending"}


def test_group_sent_receipt_finalizes_all_five_once_via_legacy_wrapper(tmp_db):
    slot_ts = int(time.time())
    articles = _articles()
    assert _claim_group(slot_ts)["claimed"] is True

    first = teams_recommendation_slot_record_receipt(
        slot_ts,
        status="sent",
        request_id="synthetic-group-run",
        article_key=articles[0]["article_key"],
        now_ts=slot_ts + 2,
    )
    repeated = teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="sent",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 3,
    )

    assert first["recorded"] is True and first["itemCount"] == 5
    assert repeated == {
        "recorded": True,
        "reason": "already_recorded",
        "itemCount": 5,
    }
    assert teams_recommendation_slot_get(slot_ts)["status"] == "sent"
    assert {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)} == {"sent"}
    alerts = [database.teams_alert_get(article["article_key"]) for article in articles]
    assert {alert["status"] for alert in alerts if alert} == {"sent"}
    assert {alert["alert_count"] for alert in alerts if alert} == {1}


def test_parallel_group_receipts_are_idempotent(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True

    def receipt() -> dict:
        return teams_recommendation_slot_record_group_receipt(
            slot_ts,
            status="sent",
            request_id="synthetic-group-run",
            now_ts=slot_ts + 2,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: receipt(), range(2)))

    assert all(result["recorded"] is True for result in results)
    assert {result["reason"] for result in results} == {
        "recorded",
        "already_recorded",
    }
    assert {
        database.teams_alert_get(article["article_key"])["alert_count"] for article in _articles()
    } == {1}


def test_failed_group_receipt_releases_all_five_for_retry(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True

    receipt = teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="failed",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 1,
    )
    failed_article_statuses = {
        database.teams_alert_get(article["article_key"])["status"] for article in _articles()
    }
    failed_item_statuses = {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)}
    retried = teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=_articles(),
        request_id="synthetic-retry-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts + 2,
    )

    assert receipt["recorded"] is True
    assert failed_article_statuses == {"transport_failed"}
    assert failed_item_statuses == {"failed"}
    assert {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)} == {"sending"}
    assert {
        database.teams_alert_get(article["article_key"])["status"] for article in _articles()
    } == {"sending"}
    assert retried["claimed"] is True


def test_delivery_uncertain_is_terminal_for_every_group_article(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True
    receipt = teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="delivery_uncertain",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 1,
    )
    later_slot = slot_ts + 600
    duplicate = teams_recommendation_slot_try_claim_group(
        later_slot,
        articles=_articles(),
        request_id="synthetic-later-run",
        claim_payload=_payload(later_slot),
        now_ts=later_slot,
    )

    assert receipt["recorded"] is True
    assert duplicate == {"claimed": False, "reason": "article_already_sent"}
    assert {
        database.teams_alert_get(article["article_key"])["status"] for article in _articles()
    } == {"delivery_uncertain"}


def test_delivery_uncertain_blocks_same_cms_under_a_changed_url(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True
    assert teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="delivery_uncertain",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 1,
    )["recorded"] is True
    aliased = _articles()
    aliased[0] = {
        **aliased[0],
        "article_key": "https://m.bild.de/news/synthetic-group-1-renamed",
        "article_url": "https://m.bild.de/news/synthetic-group-1-renamed",
    }
    later_slot = slot_ts + 600

    duplicate = teams_recommendation_slot_try_claim_group(
        later_slot,
        articles=aliased,
        request_id="synthetic-uncertain-cms-alias-run",
        claim_payload=_payload(later_slot),
        now_ts=later_slot,
    )

    assert duplicate == {"claimed": False, "reason": "article_already_sent"}
    assert teams_recommendation_slot_get(later_slot) is None


def test_missing_group_receipt_never_recycles_articles_after_lease(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True

    later_slot = slot_ts + 600
    duplicate = teams_recommendation_slot_try_claim_group(
        later_slot,
        articles=_articles(),
        request_id="synthetic-missing-receipt-run",
        claim_payload=_payload(later_slot),
        now_ts=later_slot,
    )

    assert duplicate == {
        "claimed": False,
        "reason": "article_delivery_unresolved",
    }
    assert teams_recommendation_slot_get(slot_ts)["status"] == "sending"
    assert {
        database.teams_alert_get(article["article_key"])["status"]
        for article in _articles()
    } == {"sending"}


def test_missing_receipt_blocks_same_article_under_a_url_alias(tmp_db):
    slot_ts = int(time.time())
    original = _articles()
    original[0] = {**original[0], "article_id": ""}
    assert teams_recommendation_slot_try_claim_group(
        slot_ts,
        articles=original,
        request_id="synthetic-url-original-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )["claimed"] is True
    aliased = _articles()
    aliased[0] = {
        **aliased[0],
        "article_key": "https://bild.de/news/synthetic-group-1/amp?output=1",
        "article_id": "",
        "article_url": "https://bild.de/news/synthetic-group-1/amp?output=1",
    }
    later_slot = slot_ts + 600

    duplicate = teams_recommendation_slot_try_claim_group(
        later_slot,
        articles=aliased,
        request_id="synthetic-missing-url-alias-run",
        claim_payload=_payload(later_slot),
        now_ts=later_slot,
    )

    assert duplicate == {
        "claimed": False,
        "reason": "article_delivery_unresolved",
    }
    assert teams_recommendation_slot_get(later_slot) is None


def test_failed_group_allows_same_cms_under_a_changed_url(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True
    assert teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="failed",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 1,
    )["recorded"] is True
    aliased = _articles()
    aliased[0] = {
        **aliased[0],
        "article_key": "https://m.bild.de/news/synthetic-group-1-retry",
        "article_url": "https://m.bild.de/news/synthetic-group-1-retry",
    }
    later_slot = slot_ts + 600

    retried = teams_recommendation_slot_try_claim_group(
        later_slot,
        articles=aliased,
        request_id="synthetic-failed-cms-alias-run",
        claim_payload=_payload(later_slot),
        now_ts=later_slot,
    )

    assert retried["claimed"] is True
    assert retried["itemCount"] == 5


def test_legacy_record_cannot_mutate_an_exact_five_group(tmp_db):
    slot_ts = int(time.time())
    articles = _articles()
    assert _claim_group(slot_ts)["claimed"] is True

    for legacy_status in ("sent", "failed", "delivery_uncertain"):
        teams_recommendation_slot_record(
            slot_ts,
            article_key=articles[0]["article_key"],
            status=legacy_status,
            error="synthetic legacy writer",
            now_ts=slot_ts + 1,
        )

    assert teams_recommendation_slot_get(slot_ts)["status"] == "sending"
    assert {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)} == {
        "sending"
    }
    assert {
        database.teams_alert_get(article["article_key"])["status"] for article in articles
    } == {"sending"}


def test_group_receipt_rolls_back_if_one_article_is_no_longer_owned(tmp_db):
    slot_ts = int(time.time())
    articles = _articles()
    assert _claim_group(slot_ts)["claimed"] is True
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "UPDATE teams_alerts SET last_decision_ts = ? WHERE article_key = ?",
            (slot_ts + 20, articles[-1]["article_key"]),
        )

    receipt = teams_recommendation_slot_record_group_receipt(
        slot_ts,
        status="sent",
        request_id="synthetic-group-run",
        now_ts=slot_ts + 2,
    )

    assert receipt == {
        "recorded": False,
        "reason": "article_claim_not_owned_by_slot",
    }
    assert teams_recommendation_slot_get(slot_ts)["status"] == "sending"
    assert {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)} == {"sending"}
    assert all(
        database.teams_alert_get(article["article_key"])["alert_count"] == 0 for article in articles
    )


def test_owned_stale_group_release_clears_all_sending_article_claims(tmp_db):
    slot_ts = int(time.time())
    assert _claim_group(slot_ts)["claimed"] is True

    wrong_owner = teams_recommendation_slot_fail_if_owned(
        slot_ts,
        request_id="synthetic-wrong-owner",
        error="synthetic stale claim",
        now_ts=slot_ts + 1,
    )
    released = teams_recommendation_slot_fail_if_owned(
        slot_ts,
        request_id="synthetic-group-run",
        error="synthetic stale claim",
        now_ts=slot_ts + 2,
    )

    assert wrong_owner == {
        "released": False,
        "reason": "claim_not_owned_or_final",
    }
    assert released == {"released": True, "reason": "released"}
    assert teams_recommendation_slot_get(slot_ts)["status"] == "failed"
    assert {item["status"] for item in teams_recommendation_slot_group_get(slot_ts)} == {"failed"}
    assert {
        database.teams_alert_get(article["article_key"])["status"] for article in _articles()
    } == {"claim_released"}


def test_owned_stale_legacy_top1_release_clears_orphan(tmp_db):
    slot_ts = int(time.time())
    article = _articles()[0]
    slot = teams_recommendation_slot_try_claim(
        slot_ts,
        article_key=article["article_key"],
        request_id="synthetic-legacy-run",
        claim_payload=_payload(slot_ts),
        now_ts=slot_ts,
    )
    article_claim = database.teams_alert_try_claim_send(
        article_key=article["article_key"],
        article_id=article["article_id"],
        article_url=article["article_url"],
        article_title=article["article_title"],
        title_hash=article["title_hash"],
        score=article["score"],
        predicted_or=article["predicted_or"],
        candidate_updated_at=article["candidate_updated_at"],
        is_breaking=False,
        reason="synthetic legacy claim",
        decision_ts=slot_ts,
        alert_cooldown_minutes=0,
        global_cooldown_minutes=0,
        in_progress_cooldown_minutes=5,
        failed_cooldown_minutes=0,
        transport_failure_cooldown_minutes=0,
    )
    released = teams_recommendation_slot_fail_if_owned(
        slot_ts,
        request_id="synthetic-legacy-run",
        error="stale legacy contract",
        now_ts=slot_ts + 1,
    )

    assert slot["claimed"] is True and article_claim["claimed"] is True
    assert released == {"released": True, "reason": "released"}
    assert database.teams_alert_get(article["article_key"])["status"] == "claim_released"
    assert teams_recommendation_slot_group_get(slot_ts) == []


def test_group_retention_removes_children_with_expired_parent(tmp_db):
    old_slot_ts = int(time.time()) - 46 * 86400
    assert _claim_group(old_slot_ts)["claimed"] is True
    with sqlite3.connect(tmp_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM teams_recommendation_slot_articles").fetchone()[0]
            == 5
        )

    assert teams_recommendation_slot_group_get(old_slot_ts) == []
    assert teams_recommendation_slot_get(old_slot_ts) is None
    assert all(
        database.teams_alert_get(article["article_key"]) is None
        for article in _articles()
    )
    with sqlite3.connect(tmp_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM teams_recommendation_slot_articles").fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM teams_alerts").fetchone()[0] == 0


def test_explicit_maintenance_cleans_expired_payload_without_a_claim_read(tmp_db):
    old_slot_ts = int(time.time()) - 46 * 86400
    assert _claim_group(old_slot_ts)["claimed"] is True

    teams_recommendation_slot_cleanup(now_ts=int(time.time()))

    with sqlite3.connect(tmp_db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM teams_recommendation_slot_claims"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM teams_recommendation_slot_articles"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM teams_alerts").fetchone()[0] == 0
