"""Durable exactly-one claim for each calculated Teams recommendation slot."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time

from app import database
from app.article_identity import canonical_article_id, canonical_article_url_identity

_RETENTION_SECONDS = 45 * 86400
_CLAIM_REPLAY_SECONDS = 5 * 60
_GROUP_ARTICLE_COUNT = 5
_CLAIM_RELEASED_STATUS = "claim_released"
_MAINTENANCE_INTERVAL_SECONDS = 3600
_maintenance_lock = threading.Lock()
_last_maintenance_ts = 0


def _ensure_schema(conn: sqlite3.Connection, *, now_ts: int) -> None:
    """Create and age out the tiny slot ledger without changing score storage."""
    conn.row_factory = sqlite3.Row
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
        str(row[1]) for row in conn.execute("PRAGMA table_info(teams_recommendation_slot_claims)")
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
    # One Teams post contains five recommendations. Keep their relationship to
    # the delivery slot without copying article URLs or titles into this
    # operational ledger. ``article_ref`` is the same SHA-256 reference used by
    # the parent table.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS teams_recommendation_slot_articles (
            binding_slot_ts INTEGER NOT NULL,
            position INTEGER NOT NULL,
            article_ref TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            claimed_at INTEGER DEFAULT 0,
            finalized_at INTEGER DEFAULT 0,
            last_error TEXT DEFAULT '',
            PRIMARY KEY (binding_slot_ts, position),
            UNIQUE (binding_slot_ts, article_ref)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_teams_recommendation_slot_articles_status "
        "ON teams_recommendation_slot_articles(status, claimed_at)"
    )
    conn.execute(
        """UPDATE teams_recommendation_slot_claims
           SET claim_payload_json = ''
           WHERE binding_slot_ts + ? <= ?
             AND claim_payload_json != ''""",
        (_CLAIM_REPLAY_SECONDS, int(now_ts)),
    )
    retention_cutoff = int(now_ts) - _RETENTION_SECONDS
    expired_slots = conn.execute(
        """SELECT * FROM teams_recommendation_slot_claims
           WHERE MAX(claimed_at, sent_at) < ?""",
        (retention_cutoff,),
    ).fetchall()
    for expired_slot in expired_slots:
        _release_owned_article_claims_conn(
            conn,
            expired_slot,
            error="expired_slot_retention_cleanup",
            # Preserve the old timestamp so the normal teams_alerts retention
            # cleanup can remove this non-delivery metadata as intended.
            now_ts=int(expired_slot["claimed_at"] or 0),
        )
    conn.execute(
        """DELETE FROM teams_recommendation_slot_articles
           WHERE binding_slot_ts IN (
               SELECT binding_slot_ts
               FROM teams_recommendation_slot_claims
               WHERE MAX(claimed_at, sent_at) < ?
           )""",
        (retention_cutoff,),
    )
    conn.execute(
        """DELETE FROM teams_recommendation_slot_claims
           WHERE MAX(claimed_at, sent_at) < ?""",
        (retention_cutoff,),
    )
    conn.execute(
        """DELETE FROM teams_recommendation_slot_articles
           WHERE NOT EXISTS (
               SELECT 1
               FROM teams_recommendation_slot_claims AS slot
               WHERE slot.binding_slot_ts =
                     teams_recommendation_slot_articles.binding_slot_ts
           )"""
    )
    # Slot-ledger access happens on every scheduled run, so enforce the same
    # 45-day limit for the raw title/URL rows even when the process never
    # restarts (the broader database cleanup otherwise runs only at startup).
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'teams_alerts'"
    ).fetchone():
        conn.execute(
            "DELETE FROM teams_alerts WHERE last_decision_ts < ?",
            (retention_cutoff,),
        )


def teams_recommendation_slot_cleanup(*, now_ts: int | None = None) -> None:
    """Enforce scheduled-ledger and raw-alert retention without requiring a claim."""
    now = int(now_ts or time.time())
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()


def teams_recommendation_slot_cleanup_if_due(
    *,
    now_ts: int | None = None,
) -> bool:
    """Run at most hourly; safe to call from the continuously polled health route."""
    global _last_maintenance_ts
    now = int(now_ts or time.time())
    with _maintenance_lock:
        if _last_maintenance_ts and now - _last_maintenance_ts < _MAINTENANCE_INTERVAL_SECONDS:
            return False
        teams_recommendation_slot_cleanup(now_ts=now)
        _last_maintenance_ts = now
        return True


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


def _finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return default


def _normalize_group_articles(articles: list[dict] | None) -> list[dict] | None:
    """Validate exactly five private article claims in their displayed order."""
    if not isinstance(articles, list) or len(articles) != _GROUP_ARTICLE_COUNT:
        return None
    normalized: list[dict] = []
    seen_refs: set[str] = set()
    seen_article_ids: set[str] = set()
    seen_url_identities: set[str] = set()
    for position, raw in enumerate(articles, start=1):
        if not isinstance(raw, dict):
            return None
        article_key = str(raw.get("article_key") or "").strip()
        article_url = str(raw.get("article_url") or "").strip()
        article_title = str(raw.get("article_title") or "").strip()
        article_id = str(raw.get("article_id") or article_key).strip()
        article_ref = _article_ref(article_key)
        canonical_id = canonical_article_id(article_id)
        url_identity = canonical_article_url_identity(article_url)
        if (
            not article_ref
            or not article_url
            or not article_title
            or not url_identity
            or article_ref in seen_refs
            or (canonical_id and canonical_id in seen_article_ids)
            or url_identity in seen_url_identities
        ):
            return None
        seen_refs.add(article_ref)
        if canonical_id:
            seen_article_ids.add(canonical_id)
        seen_url_identities.add(url_identity)
        normalized.append(
            {
                "position": position,
                "article_ref": article_ref,
                "article_key": article_key,
                "article_id": article_id,
                "article_url": article_url,
                "article_title": article_title[:500],
                "title_hash": str(raw.get("title_hash") or "").strip()
                or hashlib.sha256(article_title.casefold().encode("utf-8")).hexdigest(),
                "score": _finite_float(raw.get("score")),
                "predicted_or": _finite_float(raw.get("predicted_or")),
                "candidate_updated_at": _safe_int(raw.get("candidate_updated_at")),
                "is_breaking": bool(raw.get("is_breaking")),
                "reason": str(raw.get("reason") or "Push empfohlen")[:500],
            }
        )
    return normalized


def _matching_alert_rows_conn(
    conn: sqlite3.Connection,
    article: dict,
) -> list[sqlite3.Row]:
    """Match legacy and current claims by key, CMS ID, or canonical URL."""
    article_key = str(article.get("article_key") or "").strip()
    article_id = canonical_article_id(article.get("article_id"))
    url_identity = canonical_article_url_identity(article.get("article_url"))
    matches: list[sqlite3.Row] = []
    for row in conn.execute("SELECT * FROM teams_alerts").fetchall():
        row_key = str(row["article_key"] or "").strip()
        row_id = canonical_article_id(row["article_id"])
        row_url_identity = canonical_article_url_identity(
            row["article_url"] or row_key
        )
        if (
            (article_key and row_key == article_key)
            or (article_id and row_id and row_id == article_id)
            or (url_identity and row_url_identity and row_url_identity == url_identity)
        ):
            matches.append(row)
    return matches


def _table_exists_conn(conn: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (str(table_name),),
        ).fetchone()
    )


def _slot_article_rows(
    conn: sqlite3.Connection,
    binding_slot_ts: int,
) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM teams_recommendation_slot_articles
           WHERE binding_slot_ts = ?
           ORDER BY position""",
        (int(binding_slot_ts),),
    ).fetchall()


def _alert_rows_by_refs(
    conn: sqlite3.Connection,
    article_refs: set[str],
) -> dict[str, sqlite3.Row]:
    """Resolve pseudonymous refs once without persisting another raw identity."""
    if not article_refs:
        return {}
    matches: dict[str, sqlite3.Row] = {}
    for row in conn.execute("SELECT * FROM teams_alerts").fetchall():
        article_key = str(row["article_key"] or "")
        article_ref = _article_ref(article_key)
        if article_ref in article_refs:
            # A SHA-256 collision is not an identity we can safely finalize.
            if article_ref in matches:
                return {}
            matches[article_ref] = row
    return matches


def _group_owned_conn(
    conn: sqlite3.Connection,
    slot: sqlite3.Row,
    *,
    expected_count: int = _GROUP_ARTICLE_COUNT,
) -> tuple[bool, str, list[sqlite3.Row], dict[str, sqlite3.Row]]:
    items = _slot_article_rows(conn, int(slot["binding_slot_ts"] or 0))
    if len(items) != expected_count:
        return False, "slot_group_incomplete", items, {}
    positions = [int(item["position"] or 0) for item in items]
    refs = [str(item["article_ref"] or "") for item in items]
    claimed_at = int(slot["claimed_at"] or 0)
    if (
        positions != list(range(1, expected_count + 1))
        or len(set(refs)) != expected_count
        or refs[0] != str(slot["article_ref"] or "")
        or any(str(item["status"] or "") != "sending" for item in items)
        or any(int(item["claimed_at"] or 0) != claimed_at for item in items)
    ):
        return False, "slot_group_conflict", items, {}
    alerts = _alert_rows_by_refs(conn, set(refs))
    if len(alerts) != expected_count:
        return False, "article_claim_missing", items, alerts
    if any(
        str(alerts[article_ref]["status"] or "") != "sending"
        or int(alerts[article_ref]["last_decision_ts"] or 0) != claimed_at
        for article_ref in refs
    ):
        return False, "article_claim_not_owned_by_slot", items, alerts
    return True, "owned", items, alerts


def _release_owned_article_claims_conn(
    conn: sqlite3.Connection,
    slot: sqlite3.Row,
    *,
    error: str,
    now_ts: int,
) -> int:
    """Release only sending article rows still owned by this exact slot lease."""
    slot_ts = int(slot["binding_slot_ts"] or 0)
    claimed_at = int(slot["claimed_at"] or 0)
    items = _slot_article_rows(conn, slot_ts)
    refs = {str(item["article_ref"] or "") for item in items if str(item["article_ref"] or "")}
    # The parent reference is also the legacy Top-1 fallback and protects a
    # partially corrupted child group from stranding its primary claim.
    legacy_ref = str(slot["article_ref"] or "")
    if legacy_ref:
        refs.add(legacy_ref)
    alerts = _alert_rows_by_refs(conn, refs)
    released = 0
    for article_ref, alert in alerts.items():
        article_key = str(alert["article_key"] or "")
        conn.execute(
            """UPDATE teams_alerts
               SET status = ?, last_decision_ts = ?, last_error = ?
               WHERE article_key = ? AND status = 'sending'
                 AND last_decision_ts = ?""",
            (
                _CLAIM_RELEASED_STATUS,
                int(now_ts),
                str(error or "")[:500],
                article_key,
                claimed_at,
            ),
        )
        released += int(conn.execute("SELECT changes()").fetchone()[0] or 0)
    if items:
        conn.execute(
            """UPDATE teams_recommendation_slot_articles
               SET status = 'failed', finalized_at = ?, last_error = ?
               WHERE binding_slot_ts = ? AND status = 'sending'
                 AND claimed_at = ?""",
            (int(now_ts), str(error or "")[:500], slot_ts, claimed_at),
        )
    return released


def _group_article_block_reason_conn(
    conn: sqlite3.Connection,
    article: dict,
    *,
    now_ts: int,
    in_progress_seconds: int,
) -> str:
    article_key = str(article["article_key"])
    for existing in _matching_alert_rows_conn(conn, article):
        existing_status = str(existing["status"] or "")
        existing_decision_ts = int(existing["last_decision_ts"] or 0)
        retained_ts = max(
            existing_decision_ts,
            int(existing["last_alert_ts"] or 0),
            int(existing["first_alert_ts"] or 0),
        )
        if retained_ts and now_ts - retained_ts >= _RETENTION_SECONDS:
            continue
        if existing_status in {"sent", "delivery_uncertain"}:
            return "article_already_sent"
        if existing_status == "sending":
            existing_key = str(existing["article_key"] or article_key)
            article_ref = _article_ref(existing_key)
            unresolved_group = None
            if _table_exists_conn(
                conn, "teams_recommendation_slot_articles"
            ) and _table_exists_conn(conn, "teams_recommendation_slot_claims"):
                unresolved_group = conn.execute(
                    """SELECT 1
                       FROM teams_recommendation_slot_articles AS item
                       JOIN teams_recommendation_slot_claims AS slot
                         ON slot.binding_slot_ts = item.binding_slot_ts
                       WHERE item.article_ref = ?
                         AND item.status = 'sending'
                         AND slot.status = 'sending'
                         AND item.claimed_at = slot.claimed_at
                       LIMIT 1""",
                    (article_ref,),
                ).fetchone()
            if unresolved_group is not None:
                # Once the exact-five payload has left the backend, an absent
                # receipt is acknowledgement-ambiguous. Never recycle one of
                # those identities into a later slot merely because its lease
                # elapsed; an explicit receipt or retained-state cleanup must
                # resolve it first.
                return "article_delivery_unresolved"
        if (
            existing_status == "sending"
            and existing_decision_ts
            and now_ts - existing_decision_ts < in_progress_seconds
        ):
            return "article_send_in_progress"
    return ""


def teams_recommendation_article_identity_block_reasons(
    articles: list[dict],
    *,
    now_ts: int | None = None,
    in_progress_seconds: int = 300,
) -> dict[str, str]:
    """Read retained identity conflicts before ranking; the claim rechecks atomically."""
    if not isinstance(articles, list) or not articles:
        return {}
    now = int(now_ts or time.time())
    lease = max(30, int(in_progress_seconds or 300))
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            if not _table_exists_conn(conn, "teams_alerts"):
                return {}
            blockers: dict[str, str] = {}
            for article in articles:
                if not isinstance(article, dict):
                    continue
                lookup_key = str(article.get("lookup_key") or "").strip()
                if not lookup_key:
                    continue
                reason = _group_article_block_reason_conn(
                    conn,
                    article,
                    now_ts=now,
                    in_progress_seconds=lease,
                )
                if reason:
                    blockers[lookup_key] = reason
            return blockers
        finally:
            conn.close()


def _claim_group_article_conn(
    conn: sqlite3.Connection,
    article: dict,
    *,
    now_ts: int,
    in_progress_seconds: int,
) -> dict:
    """Claim one group member inside the caller's existing transaction."""
    blocked_reason = _group_article_block_reason_conn(
        conn,
        article,
        now_ts=now_ts,
        in_progress_seconds=in_progress_seconds,
    )
    if blocked_reason:
        return {"claimed": False, "reason": blocked_reason}
    article_key = str(article["article_key"])
    existing = conn.execute(
        "SELECT * FROM teams_alerts WHERE article_key = ?",
        (article_key,),
    ).fetchone()

    first_alert_ts = int(existing["first_alert_ts"] or 0) if existing else 0
    last_alert_ts = int(existing["last_alert_ts"] or 0) if existing else 0
    alert_count = int(existing["alert_count"] or 0) if existing else 0
    conn.execute(
        """INSERT INTO teams_alerts (
            article_key, article_id, article_url, article_title, title_hash,
            first_alert_ts, last_alert_ts, last_decision_ts, last_score,
            last_predicted_or, last_candidate_updated_at, last_is_breaking,
            last_reason, status, alert_count, last_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'sending', ?, '')
        ON CONFLICT(article_key) DO UPDATE SET
            article_id = excluded.article_id,
            article_url = excluded.article_url,
            article_title = excluded.article_title,
            title_hash = excluded.title_hash,
            last_decision_ts = excluded.last_decision_ts,
            last_score = excluded.last_score,
            last_predicted_or = excluded.last_predicted_or,
            last_candidate_updated_at = excluded.last_candidate_updated_at,
            last_is_breaking = excluded.last_is_breaking,
            last_reason = excluded.last_reason,
            status = excluded.status,
            last_error = ''""",
        (
            article_key,
            str(article["article_id"]),
            str(article["article_url"]),
            str(article["article_title"]),
            str(article["title_hash"]),
            first_alert_ts,
            last_alert_ts,
            int(now_ts),
            float(article["score"]),
            float(article["predicted_or"]),
            int(article["candidate_updated_at"]),
            1 if article["is_breaking"] else 0,
            str(article["reason"]),
            alert_count,
        ),
    )
    return {"claimed": True, "reason": "claimed"}


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
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
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


def teams_recommendation_slot_group_get(binding_slot_ts: int) -> list[dict]:
    """Return pseudonymous group rows for diagnostics and authenticated callers."""
    slot_ts = int(binding_slot_ts or 0)
    if slot_ts <= 0:
        return []
    now = int(time.time())
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_schema(conn, now_ts=now)
            rows = _slot_article_rows(conn, slot_ts)
            conn.commit()
        finally:
            conn.close()
    return [dict(row) for row in rows]


def teams_recommendation_slot_group_owned(
    binding_slot_ts: int,
    *,
    request_id: str,
    now_ts: int | None = None,
) -> dict:
    """Verify that the caller still owns all five durable article claims."""
    slot_ts = int(binding_slot_ts or 0)
    request_ref = _request_ref(request_id)
    if slot_ts <= 0 or not request_ref:
        return {"owned": False, "reason": "invalid_slot_claim", "itemCount": 0}
    now = int(now_ts or time.time())
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_schema(conn, now_ts=now)
            slot = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            if (
                slot is None
                or str(slot["status"] or "") != "sending"
                or str(slot["request_ref"] or "") != request_ref
            ):
                result = {
                    "owned": False,
                    "reason": "claim_not_owned_or_final",
                    "itemCount": 0,
                }
            else:
                owned, reason, items, _alerts = _group_owned_conn(conn, slot)
                result = {
                    "owned": owned,
                    "reason": reason,
                    "itemCount": len(items),
                }
            conn.commit()
            return result
        finally:
            conn.close()


def teams_recommendation_slot_try_claim_group(
    binding_slot_ts: int,
    *,
    articles: list[dict],
    request_id: str,
    claim_payload: dict,
    now_ts: int | None = None,
    lease_seconds: int = 300,
) -> dict:
    """Atomically reserve one slot and exactly five displayed recommendations."""
    slot_ts = int(binding_slot_ts or 0)
    request_ref = _request_ref(request_id)
    payload_json = _claim_payload_json(claim_payload)
    normalized_articles = _normalize_group_articles(articles)
    if (
        slot_ts <= 0
        or not request_ref
        or not payload_json
        or normalized_articles is None
        or _safe_int(claim_payload.get("recommendationCount")) != _GROUP_ARTICLE_COUNT
    ):
        return {"claimed": False, "reason": "invalid_slot_group_claim"}

    now = int(now_ts or time.time())
    lease = max(30, int(lease_seconds or 300))
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            existing = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            released_expired_slot = False
            if existing:
                existing_status = str(existing["status"] or "")
                claimed_at = int(existing["claimed_at"] or 0)
                if (
                    str(existing["request_ref"] or "") == request_ref
                    and existing_status == "sending"
                    and now < slot_ts + _CLAIM_REPLAY_SECONDS
                ):
                    replay_payload = _decode_claim_payload(
                        str(existing["claim_payload_json"] or "")
                    )
                    owned, reason, items, _alerts = _group_owned_conn(conn, existing)
                    if replay_payload is not None and owned:
                        conn.execute("ROLLBACK")
                        return {
                            "claimed": True,
                            "reason": "replayed",
                            "bindingSlotTs": slot_ts,
                            "replayPayload": replay_payload,
                            "itemCount": len(items),
                        }
                    conn.execute("ROLLBACK")
                    return {
                        "claimed": False,
                        "reason": reason if not owned else "slot_replay_unavailable",
                    }
                if existing_status in {"sent", "delivery_uncertain"}:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_already_sent"}
                if existing_status == "sending" and now - claimed_at < lease:
                    conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": "slot_send_in_progress"}
                if existing_status == "sending":
                    _release_owned_article_claims_conn(
                        conn,
                        existing,
                        error="expired_slot_group_claim",
                        now_ts=now,
                    )
                    released_expired_slot = True

            # Validate and write every article in the same SQLite transaction.
            # If any member is terminal or concurrently sending, the rollback
            # also removes all earlier members of this attempt.
            for article in normalized_articles:
                blocked_reason = _group_article_block_reason_conn(
                    conn,
                    article,
                    now_ts=now,
                    in_progress_seconds=lease,
                )
                if blocked_reason:
                    if released_expired_slot:
                        conn.execute(
                            """UPDATE teams_recommendation_slot_claims
                               SET status = 'failed', sent_at = 0,
                                   request_ref = '', claim_payload_json = '',
                                   last_error = ?
                               WHERE binding_slot_ts = ? AND status = 'sending'""",
                            (blocked_reason[:500], slot_ts),
                        )
                        conn.execute("COMMIT")
                    else:
                        conn.execute("ROLLBACK")
                    return {"claimed": False, "reason": blocked_reason}
            for article in normalized_articles:
                article_claim = _claim_group_article_conn(
                    conn,
                    article,
                    now_ts=now,
                    in_progress_seconds=lease,
                )
                if not article_claim.get("claimed"):
                    conn.execute("ROLLBACK")
                    return {
                        "claimed": False,
                        "reason": str(article_claim.get("reason") or "article_claim_blocked"),
                    }

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
                (
                    slot_ts,
                    str(normalized_articles[0]["article_ref"]),
                    request_ref,
                    payload_json,
                    now,
                ),
            )
            conn.execute(
                "DELETE FROM teams_recommendation_slot_articles " "WHERE binding_slot_ts = ?",
                (slot_ts,),
            )
            conn.executemany(
                """INSERT INTO teams_recommendation_slot_articles (
                       binding_slot_ts, position, article_ref, status,
                       claimed_at, finalized_at, last_error
                   ) VALUES (?, ?, ?, 'sending', ?, 0, '')""",
                [
                    (
                        slot_ts,
                        int(article["position"]),
                        str(article["article_ref"]),
                        now,
                    )
                    for article in normalized_articles
                ],
            )
            conn.execute("COMMIT")
            return {
                "claimed": True,
                "reason": "claimed",
                "bindingSlotTs": slot_ts,
                "itemCount": _GROUP_ARTICLE_COUNT,
            }
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()


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
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
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
                if existing_status == "sending":
                    _release_owned_article_claims_conn(
                        conn,
                        existing,
                        error="expired_slot_claim",
                        now_ts=now,
                    )

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
            # This is the backwards-compatible single-article API. Any child
            # rows left by an expired group no longer describe this claim.
            conn.execute(
                "DELETE FROM teams_recommendation_slot_articles " "WHERE binding_slot_ts = ?",
                (slot_ts,),
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
    normalized_status = status if status in {"sent", "delivery_uncertain"} else "failed"
    with database._push_db_lock:
        conn = sqlite3.connect(database.PUSH_DB_PATH, timeout=30, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            group_member = conn.execute(
                """SELECT 1 FROM teams_recommendation_slot_articles
                   WHERE binding_slot_ts = ? LIMIT 1""",
                (slot_ts,),
            ).fetchone()
            if group_member is not None:
                # The legacy writer cannot atomically finalize all five group
                # members. Ignore it instead of corrupting the exact-five
                # invariant; the authenticated receipt path owns this slot.
                conn.commit()
                return
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
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
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
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_schema(conn, now_ts=now)
            existing = conn.execute(
                """SELECT * FROM teams_recommendation_slot_claims
                   WHERE binding_slot_ts = ? AND status = 'sending'
                     AND request_ref = ?""",
                (slot_ts, request_ref),
            ).fetchone()
            if existing is None:
                conn.execute("ROLLBACK")
                return {"released": False, "reason": "claim_not_owned_or_final"}
            _release_owned_article_claims_conn(
                conn,
                existing,
                error=str(error or ""),
                now_ts=now,
            )
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
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
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


def teams_recommendation_slot_record_group_receipt(
    binding_slot_ts: int,
    *,
    status: str,
    request_id: str,
    now_ts: int | None = None,
) -> dict:
    """Atomically finalize a slot and all five article claims it delivered."""
    slot_ts = int(binding_slot_ts or 0)
    normalized_status = str(status or "").strip().lower()
    request_ref = _request_ref(request_id)
    if (
        slot_ts <= 0
        or normalized_status not in {"sent", "failed", "delivery_uncertain"}
        or not request_ref
    ):
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
            slot = conn.execute(
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
                (slot_ts,),
            ).fetchone()
            if slot is None:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "unknown_slot"}
            if str(slot["request_ref"] or "") != request_ref:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "receipt_owner_mismatch"}

            items = _slot_article_rows(conn, slot_ts)
            if len(items) != _GROUP_ARTICLE_COUNT:
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "slot_group_incomplete",
                    "itemCount": len(items),
                }
            refs = [str(item["article_ref"] or "") for item in items]
            positions = [int(item["position"] or 0) for item in items]
            if (
                positions != list(range(1, _GROUP_ARTICLE_COUNT + 1))
                or len(set(refs)) != _GROUP_ARTICLE_COUNT
                or refs[0] != str(slot["article_ref"] or "")
            ):
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "slot_group_conflict"}
            alerts = _alert_rows_by_refs(conn, set(refs))
            if len(alerts) != _GROUP_ARTICLE_COUNT:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "article_claim_missing"}

            current_slot_status = str(slot["status"] or "")
            if current_slot_status == normalized_status:
                group_matches = all(
                    str(item["status"] or "") == normalized_status
                    and str(alerts[article_ref]["status"] or "") == article_status
                    for item, article_ref in zip(items, refs)
                )
                conn.execute("ROLLBACK")
                return {
                    "recorded": group_matches,
                    "reason": ("already_recorded" if group_matches else "article_claim_conflict"),
                    "itemCount": len(items),
                }
            if current_slot_status != "sending":
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "slot_state_conflict"}

            claimed_at = int(slot["claimed_at"] or 0)
            if any(
                str(item["status"] or "") != "sending"
                or int(item["claimed_at"] or 0) != claimed_at
                or str(alerts[article_ref]["status"] or "") != "sending"
                or int(alerts[article_ref]["last_decision_ts"] or 0) != claimed_at
                for item, article_ref in zip(items, refs)
            ):
                conn.execute("ROLLBACK")
                return {
                    "recorded": False,
                    "reason": "article_claim_not_owned_by_slot",
                }

            for article_ref in refs:
                article_key = str(alerts[article_ref]["article_key"] or "")
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
                if int(conn.execute("SELECT changes()").fetchone()[0] or 0) != 1:
                    conn.execute("ROLLBACK")
                    return {"recorded": False, "reason": "article_claim_race"}

            conn.execute(
                """UPDATE teams_recommendation_slot_articles
                   SET status = ?, finalized_at = ?, last_error = ''
                   WHERE binding_slot_ts = ? AND status = 'sending'
                     AND claimed_at = ?""",
                (normalized_status, now, slot_ts, claimed_at),
            )
            if int(conn.execute("SELECT changes()").fetchone()[0] or 0) != _GROUP_ARTICLE_COUNT:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "slot_group_race"}

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
            if int(conn.execute("SELECT changes()").fetchone()[0] or 0) != 1:
                conn.execute("ROLLBACK")
                return {"recorded": False, "reason": "slot_claim_race"}
            conn.execute("COMMIT")
            return {
                "recorded": True,
                "reason": "recorded",
                "articleRef": refs[0],
                "itemCount": _GROUP_ARTICLE_COUNT,
            }
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()


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
    if (
        slot_ts <= 0
        or normalized_status
        not in {
            "sent",
            "failed",
            "delivery_uncertain",
        }
        or not request_ref
        or not expected_article_ref
    ):
        return {"recorded": False, "reason": "invalid_receipt"}
    now = int(now_ts or time.time())
    group_items = teams_recommendation_slot_group_get(slot_ts)
    if group_items:
        if str(group_items[0].get("article_ref") or "") != expected_article_ref:
            return {
                "recorded": False,
                "reason": "article_claim_mismatch",
                "articleRef": str(group_items[0].get("article_ref") or ""),
            }
        return teams_recommendation_slot_record_group_receipt(
            slot_ts,
            status=normalized_status,
            request_id=request_id,
            now_ts=now,
        )
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
                "SELECT * FROM teams_recommendation_slot_claims " "WHERE binding_slot_ts = ?",
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
                "SELECT status, last_decision_ts FROM teams_alerts " "WHERE article_key = ?",
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
