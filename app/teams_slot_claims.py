"""Durable exactly-one claim for each calculated Teams recommendation slot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from app import database

_RETENTION_SECONDS = 45 * 86400
_CLAIM_REPLAY_SECONDS = 5 * 60


def _ensure_schema(conn: sqlite3.Connection, *, now_ts: int) -> None:
    """Create and age out the tiny slot ledger without changing score storage."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS teams_recommendation_slot_claims (
            binding_slot_ts INTEGER PRIMARY KEY,
            article_ref TEXT NOT NULL DEFAULT '',
            request_ref TEXT NOT NULL DEFAULT '',
            claim_payload_json TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            claimed_at INTEGER DEFAULT 0,
            sent_at INTEGER DEFAULT 0,
            last_error TEXT DEFAULT ''
        )"""
    )
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(teams_recommendation_slot_claims)")
    }
    if "request_ref" not in columns:
        conn.execute(
            "ALTER TABLE teams_recommendation_slot_claims "
            "ADD COLUMN request_ref TEXT NOT NULL DEFAULT ''"
        )
    if "claim_payload_json" not in columns:
        conn.execute(
            "ALTER TABLE teams_recommendation_slot_claims "
            "ADD COLUMN claim_payload_json TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_teams_recommendation_slot_status "
        "ON teams_recommendation_slot_claims(status, claimed_at)"
    )
    conn.execute(
        """UPDATE teams_recommendation_slot_claims
           SET claim_payload_json = ''
           WHERE binding_slot_ts + ? <= ?
             AND claim_payload_json != ''""",
        (_CLAIM_REPLAY_SECONDS, int(now_ts)),
    )
    conn.execute(
        """DELETE FROM teams_recommendation_slot_claims
           WHERE MAX(claimed_at, sent_at) < ?""",
        (int(now_ts) - _RETENTION_SECONDS,),
    )


def _article_ref(article_key: str) -> str:
    raw = str(article_key or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""


def _request_ref(request_id: str) -> str:
    raw = str(request_id or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""


def _claim_payload_json(claim_payload: dict | None) -> str:
    if not isinstance(claim_payload, dict):
        return ""
    return json.dumps(claim_payload, ensure_ascii=False, separators=(",", ":"))


def _decode_claim_payload(raw_payload: str) -> dict | None:
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def teams_recommendation_slot_replay(
    binding_slot_ts: int,
    *,
    request_id: str,
    now_ts: int | None = None,
) -> dict | None:
    """Return a prior successful claim for the same run within the slot window."""
    slot_ts = int(binding_slot_ts or 0)
    request_ref = _request_ref(request_id)
    if slot_ts <= 0 or not request_ref:
        return None
    now = int(now_ts or time.time())
    if now >= slot_ts + _CLAIM_REPLAY_SECONDS:
        return None
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_schema(conn, now_ts=now)
            row = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims "
                "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            conn.commit()
        finally:
            conn.close()
    if row is None or str(row["request_ref"] or "") != request_ref:
        return None
    if str(row["status"] or "") != "sending":
        return None
    return _decode_claim_payload(str(row["claim_payload_json"] or ""))


def teams_recommendation_slot_try_claim(
    binding_slot_ts: int,
    *,
    article_key: str,
    request_id: str = "",
    claim_payload: dict | None = None,
    now_ts: int | None = None,
    lease_seconds: int = 300,
) -> dict:
    """Atomically reserve the only recommendation delivery for a fixed slot."""
    slot_ts = int(binding_slot_ts or 0)
    article_ref = _article_ref(article_key)
    request_ref = _request_ref(request_id)
    payload_json = _claim_payload_json(claim_payload)
    if slot_ts <= 0 or not article_ref or bool(request_ref) != bool(payload_json):
        return {"claimed": False, "reason": "invalid_slot_claim"}
    now = int(now_ts or time.time())
    lease = max(30, int(lease_seconds or 300))
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            existing = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims "
                "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            if existing:
                existing_status = str(existing["status"] or "")
                claimed_at = int(existing["claimed_at"] or 0)
                if (
                    request_ref
                    and str(existing["request_ref"] or "") == request_ref
                    and existing_status == "sending"
                    and now < slot_ts + _CLAIM_REPLAY_SECONDS
                ):
                    replay_payload = _decode_claim_payload(
                        str(existing["claim_payload_json"] or "")
                    )
                    if replay_payload is not None:
                        conn.execute("ROLLBACK")
                        return {
                            "claimed": True,
                            "reason": "replayed",
                            "bindingSlotTs": slot_ts,
                            "replayPayload": replay_payload,
                        }
                if existing_status in {"sent", "delivery_uncertain"}:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_already_sent"}
                if existing_status == "sending" and now - claimed_at < lease:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_send_in_progress"}

            conn.execute(
                """INSERT INTO teams_recommendation_slot_claims (
                       binding_slot_ts, article_ref, request_ref, claim_payload_json,
                       status, claimed_at, sent_at, last_error
                   ) VALUES (?, ?, ?, ?, 'sending', ?, 0, '')
                   ON CONFLICT(binding_slot_ts) DO UPDATE SET
                       article_ref = excluded.article_ref,
                       request_ref = excluded.request_ref,
                       claim_payload_json = excluded.claim_payload_json,
                       status = 'sending',
                       claimed_at = excluded.claimed_at,
                       sent_at = 0,
                       last_error = ''""",
                (slot_ts, article_ref, request_ref, payload_json, now),
            )
            conn.execute("COMMIT")
            return {"claimed": True, "reason": "claimed", "bindingSlotTs": slot_ts}
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()


def teams_recommendation_slot_record(
    binding_slot_ts: int,
    *,
    article_key: str,
    status: str,
    error: str = "",
    now_ts: int | None = None,
) -> None:
    """Record success or release a failed claim for another in-window attempt."""
    slot_ts = int(binding_slot_ts or 0)
    if slot_ts <= 0:
        return
    now = int(now_ts or time.time())
    normalized_status = (
        status if status in {"sent", "delivery_uncertain"} else "failed"
    )
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        _ensure_schema(conn, now_ts=now)
        conn.execute(
            """INSERT INTO teams_recommendation_slot_claims (
                   binding_slot_ts, article_ref, status, claimed_at, sent_at, last_error
               ) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(binding_slot_ts) DO UPDATE SET
                   article_ref = excluded.article_ref,
                   status = excluded.status,
                   sent_at = excluded.sent_at,
                   request_ref = CASE
                       WHEN excluded.status = 'failed' THEN ''
                       ELSE teams_recommendation_slot_claims.request_ref
                   END,
                   claim_payload_json = CASE
                       WHEN excluded.status = 'failed' THEN ''
                       ELSE teams_recommendation_slot_claims.claim_payload_json
                   END,
                   last_error = excluded.last_error""",
            (
                slot_ts,
                _article_ref(article_key),
                normalized_status,
                now,
                now if normalized_status in {"sent", "delivery_uncertain"} else 0,
                str(error or "")[:500],
            ),
        )
        conn.commit()
        conn.close()


def teams_recommendation_slot_fail_if_owned(
    binding_slot_ts: int,
    *,
    request_id: str,
    error: str = "",
    now_ts: int | None = None,
) -> dict:
    """Release only the caller's still-sending claim; never downgrade a final slot."""
    slot_ts = int(binding_slot_ts or 0)
    request_ref = _request_ref(request_id)
    if slot_ts <= 0 or not request_ref:
        return {"released": False, "reason": "invalid_slot_claim"}
    now = int(now_ts or time.time())
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            conn.execute(
                """UPDATE teams_recommendation_slot_claims
                   SET status = 'failed', sent_at = 0,
                       request_ref = '', claim_payload_json = '', last_error = ?
                   WHERE binding_slot_ts = ?
                     AND status = 'sending'
                     AND request_ref = ?""",
                (str(error or "")[:500], slot_ts, request_ref),
            )
            changed = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            conn.execute("COMMIT")
            return {
                "released": changed == 1,
                "reason": "released" if changed == 1 else "claim_not_owned_or_final",
            }
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()


def teams_recommendation_slot_get(binding_slot_ts: int) -> dict | None:
    """Load one slot claim for authenticated delivery-receipt handling."""
    slot_ts = int(binding_slot_ts or 0)
    if slot_ts <= 0:
        return None
    now = int(time.time())
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_schema(conn, now_ts=now)
            row = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims "
                "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            conn.commit()
        finally:
            conn.close()
    return dict(row) if row else None


def teams_alert_get_by_ref(article_ref: str) -> dict | None:
    """Resolve a retained article claim by its pseudonymous slot-ledger hash."""
    normalized_ref = str(article_ref or "").strip()
    if not normalized_ref:
        return None
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM teams_alerts").fetchall()
        finally:
            conn.close()
    for row in rows:
        item = dict(row)
        article_key = str(item.get("article_key") or "")
        if article_key and _article_ref(article_key) == normalized_ref:
            return item
    return None


def teams_recommendation_slot_record_receipt(
    binding_slot_ts: int,
    *,
    status: str,
    request_id: str,
    article_key: str,
    now_ts: int | None = None,
) -> dict:
    """Atomically finalize one slot and its exact article claim.

    The article identity stays pseudonymised in this ledger. Repeated receipts
    with the same final state are idempotent. The article claim timestamp must
    still belong to this slot, so a late receipt cannot finalize a later slot.
    """
    slot_ts = int(binding_slot_ts or 0)
    normalized_status = str(status or "").strip().lower()
    request_ref = _request_ref(request_id)
    expected_article_ref = _article_ref(article_key)
    if slot_ts <= 0 or normalized_status not in {
        "sent",
        "failed",
        "delivery_uncertain",
    } or not request_ref or not expected_article_ref:
        return {"recorded": False, "reason": "invalid_receipt"}
    now = int(now_ts or time.time())
    article_status = {
        "sent": "sent",
        "failed": "transport_failed",
        "delivery_uncertain": "delivery_uncertain",
    }[normalized_status]

    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            existing = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims "
                "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            if existing is None:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "unknown_slot"}

            existing_status = str(existing["status"] or "")
            article_ref = str(existing["article_ref"] or "")
            if str(existing["request_ref"] or "") != request_ref:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "receipt_owner_mismatch",
                    "articleRef": article_ref,
                }
            if article_ref != expected_article_ref:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "article_claim_mismatch",
                    "articleRef": article_ref,
                }

            article = conn.execute(
                "SELECT status, last_decision_ts FROM teams_alerts "
                "WHERE article_key = ?",
                (article_key,),
            ).fetchone()
            if article is None:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "article_claim_missing",
                    "articleRef": article_ref,
                }
            current_article_status = str(article["status"] or "")
            claimed_at = int(existing["claimed_at"] or 0)
            article_claimed_at = int(article["last_decision_ts"] or 0)

            if existing_status == normalized_status:
                conn.execute("ROLLBACK")
                if current_article_status == article_status:
                    return {
                        "recorded": True,
                        "reason": "already_recorded",
                        "articleRef": article_ref,
                    }
                return {
                    "recorded": False,
                    "reason": "article_claim_conflict",
                    "articleRef": article_ref,
                }
            if existing_status != "sending":
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "slot_state_conflict",
                    "articleRef": article_ref,
                }
            if current_article_status != "sending" or article_claimed_at != claimed_at:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "article_claim_not_owned_by_slot",
                    "articleRef": article_ref,
                }

            if article_status == "sent":
                conn.execute(
                    """UPDATE teams_alerts
                       SET first_alert_ts = CASE
                               WHEN first_alert_ts > 0 THEN first_alert_ts ELSE ?
                           END,
                           last_alert_ts = ?, last_decision_ts = ?,
                           status = 'sent', alert_count = alert_count + 1,
                           last_error = ''
                       WHERE article_key = ? AND status = 'sending'
                         AND last_decision_ts = ?""",
                    (now, now, now, article_key, claimed_at),
                )
            else:
                conn.execute(
                    """UPDATE teams_alerts
                       SET last_decision_ts = ?, status = ?, last_error = ''
                       WHERE article_key = ? AND status = 'sending'
                         AND last_decision_ts = ?""",
                    (now, article_status, article_key, claimed_at),
                )
            article_changed = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            if article_changed != 1:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "article_claim_race",
                    "articleRef": article_ref,
                }

            conn.execute(
                """UPDATE teams_recommendation_slot_claims
                   SET status = ?, sent_at = ?, last_error = ''
                   WHERE binding_slot_ts = ? AND status = 'sending'
                     AND request_ref = ?""",
                (
                    normalized_status,
                    now if normalized_status in {"sent", "delivery_uncertain"} else 0,
                    slot_ts,
                    request_ref,
                ),
            )
            slot_changed = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            if slot_changed != 1:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "slot_claim_race",
                    "articleRef": article_ref,
                }
            conn.execute("COMMIT")
            return {
                "recorded": True,
                "reason": "recorded",
                "articleRef": article_ref,
            }
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
