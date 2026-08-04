"""Durable exactly-one claim for each calculated Teams recommendation slot."""

from __future__ import annotations

import hashlib
import sqlite3
import time

from app import database

_RETENTION_SECONDS = 45 * 86400


def _ensure_schema(conn: sqlite3.Connection, *, now_ts: int) -> None:
    """Create and age out the tiny slot ledger without changing score storage."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS teams_recommendation_slot_claims (
            binding_slot_ts INTEGER PRIMARY KEY,
            article_ref TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            claimed_at INTEGER DEFAULT 0,
            sent_at INTEGER DEFAULT 0,
            last_error TEXT DEFAULT ''
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_teams_recommendation_slot_status "
        "ON teams_recommendation_slot_claims(status, claimed_at)"
    )
    conn.execute(
        """DELETE FROM teams_recommendation_slot_claims
           WHERE MAX(claimed_at, sent_at) < ?""",
        (int(now_ts) - _RETENTION_SECONDS,),
    )


def _article_ref(article_key: str) -> str:
    raw = str(article_key or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""


def teams_recommendation_slot_try_claim(
    binding_slot_ts: int,
    *,
    article_key: str,
    now_ts: int | None = None,
    lease_seconds: int = 300,
) -> dict:
    """Atomically reserve the only recommendation delivery for a fixed slot."""
    slot_ts = int(binding_slot_ts or 0)
    article_ref = _article_ref(article_key)
    if slot_ts <= 0 or not article_ref:
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
                status = str(existing["status"] or "")
                claimed_at = int(existing["claimed_at"] or 0)
                if status in {"sent", "delivery_uncertain"}:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_already_sent"}
                if status == "sending" and now - claimed_at < lease:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_send_in_progress"}

            conn.execute(
                """INSERT INTO teams_recommendation_slot_claims (
                       binding_slot_ts, article_ref, status, claimed_at, sent_at, last_error
                   ) VALUES (?, ?, 'sending', ?, 0, '')
                   ON CONFLICT(binding_slot_ts) DO UPDATE SET
                       article_ref = excluded.article_ref,
                       status = 'sending',
                       claimed_at = excluded.claimed_at,
                       sent_at = 0,
                       last_error = ''""",
                (slot_ts, article_ref, now),
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
