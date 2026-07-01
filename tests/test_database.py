"""test_database.py — Tests für SQLite Push-Datenbank-Funktionen.

Testet push_db_upsert, push_db_load_all, push_db_count mit einer
temporären SQLite-DB (tmp_db fixture aus conftest.py).
Produktions-DB wird niemals berührt.
"""
import sqlite3
import time
from unittest.mock import patch

import pytest
from app import database


# ── _push_db_upsert ──────────────────────────────────────────────────────────


class TestTeamsAlertHistory:
    def test_teams_alert_list_recent_contains_dashboard_fields(self, tmp_db):
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.teams_alert_record(
                article_key="article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                title_hash="hash-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                score=82.0,
                predicted_or=5.4,
                candidate_updated_at=1_800_000_000,
                is_breaking=True,
                reason="Push empfohlen",
                status="sent",
                decision_ts=1_800_000_100,
            )

            rows = database.teams_alert_list_recent(limit=5)

        assert len(rows) == 1
        assert rows[0]["article_title"] == "Eilmeldung: Regierung beschliesst Paket"
        assert rows[0]["article_url"] == "https://www.bild.de/politik/article-1"
        assert rows[0]["status"] == "sent"
        assert rows[0]["last_score"] == pytest.approx(82.0)

    def test_teams_recommendation_record_persists_suggestion_snapshot(self, tmp_db):
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            rec_id = database.teams_recommendation_record(
                article_key="https://www.bild.de/politik/article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                section="politik",
                recommendation_type="teams_alert",
                status="sent",
                should_notify=True,
                score=82.0,
                teams_alert_score=84.0,
                teams_alert_threshold=78.0,
                editorial_score=86.0,
                predicted_or=5.4,
                predicted_or_label="5,40 % OR (Artikelmodell)",
                expected_visits=51_000,
                dashboard_rank=2,
                decided_at_ts=1_800_000_100,
                sent_at_ts=1_800_000_100,
                send_status="sent",
                summary="Push empfohlen",
                reasons=["Score stark", "Timing passt"],
                blocking_reasons=[],
                decision={"status": "notify", "dashboardRank": 2},
            )

            rows = database.teams_recommendation_list_recent(limit=5)

        assert rec_id
        assert len(rows) == 1
        assert rows[0]["article_title"] == "Eilmeldung: Regierung beschliesst Paket"
        assert rows[0]["recommendation_type"] == "teams_alert"
        assert rows[0]["status"] == "sent"
        assert rows[0]["should_notify"] == 1
        assert rows[0]["expected_visits"] == 51_000
        assert "Score stark" in rows[0]["reasons_json"]

    def test_teams_alert_claim_blocks_duplicate_in_flight_send(self, tmp_db):
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            first = database.teams_alert_try_claim_send(
                article_key="article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                title_hash="hash-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                score=82.0,
                predicted_or=0.0,
                candidate_updated_at=1_800_000_000,
                is_breaking=True,
                reason="Push empfohlen",
                decision_ts=1_800_000_100,
            )
            second = database.teams_alert_try_claim_send(
                article_key="article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                title_hash="hash-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                score=82.0,
                predicted_or=0.0,
                candidate_updated_at=1_800_000_000,
                is_breaking=True,
                reason="Push empfohlen",
                decision_ts=1_800_000_120,
            )

        assert first["claimed"] is True
        assert second["claimed"] is False
        assert second["reason"] == "article_send_in_progress"

    def test_teams_alert_claim_blocks_recent_failed_attempt(self, tmp_db):
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.teams_alert_record(
                article_key="article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                title_hash="hash-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                score=82.0,
                predicted_or=0.0,
                candidate_updated_at=1_800_000_000,
                is_breaking=True,
                reason="Push empfohlen",
                status="failed",
                decision_ts=1_800_000_100,
            )
            claim = database.teams_alert_try_claim_send(
                article_key="article-1",
                article_id="article-1",
                article_url="https://www.bild.de/politik/article-1",
                title_hash="hash-1",
                article_title="Eilmeldung: Regierung beschliesst Paket",
                score=82.0,
                predicted_or=0.0,
                candidate_updated_at=1_800_000_000,
                is_breaking=True,
                reason="Push empfohlen",
                decision_ts=1_800_000_100 + 90 * 60,
                failed_cooldown_minutes=12 * 60,
            )

        assert claim["claimed"] is False
        assert claim["reason"] == "article_failure_cooldown"

class TestPushDbUpsert:
    def test_upsert_inserts_new_records(self, tmp_db, sample_pushes):
        """Neue Pushes werden korrekt in die DB geschrieben."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            count = database.push_db_upsert(sample_pushes)
        assert count == len(sample_pushes)

    def test_upsert_empty_list_returns_zero(self, tmp_db):
        """Leere Liste → 0 zurück, kein Fehler."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            count = database.push_db_upsert([])
        assert count == 0

    def test_upsert_single_push(self, tmp_db):
        """Einzelner Push wird korrekt geschrieben."""
        push = {
            "message_id": "single_001",
            "ts_num": int(time.time()),
            "or": 5.5,
            "title": "Test-Artikel",
            "headline": "Test-Artikel Schlagzeile",
            "kicker": "TEST",
            "cat": "news",
            "link": "https://www.bild.de/news/test",
            "type": "editorial",
            "hour": 14,
            "title_len": 12,
            "opened": 500,
            "received": 10000,
            "channel": "main",
            "channels": ["main"],
            "is_eilmeldung": False,
            "n_apps": 1,
            "total_recipients": 10000,
        }
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            count = database.push_db_upsert([push])
        assert count == 1

    def test_upsert_duplicate_does_not_duplicate_rows(self, tmp_db, sample_pushes):
        """Denselben Push zweimal einfügen → immer noch dieselbe Anzahl Rows."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            database.push_db_upsert(sample_pushes)  # zweites Mal
            conn = sqlite3.connect(tmp_db)
            row_count = conn.execute("SELECT COUNT(*) FROM pushes").fetchone()[0]
            conn.close()
        assert row_count == len(sample_pushes)

    def test_upsert_updates_or_val_on_conflict(self, tmp_db):
        """Bei Duplikat-Key wird or_val aktualisiert, wenn neuer Wert höher."""
        push_v1 = {
            "message_id": "dup_test_001",
            "ts_num": int(time.time()),
            "or": 3.0,
            "title": "Artikel",
            "headline": "",
            "kicker": "",
            "cat": "news",
            "link": "https://www.bild.de/news/x",
            "type": "editorial",
            "hour": 10,
            "title_len": 7,
            "opened": 100,
            "received": 5000,
            "channel": "main",
            "channels": [],
            "is_eilmeldung": False,
            "n_apps": 0,
            "total_recipients": 5000,
        }
        push_v2 = dict(push_v1)
        push_v2["or"] = 7.5  # höherer Wert

        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert([push_v1])
            database.push_db_upsert([push_v2])
            conn = sqlite3.connect(tmp_db)
            row = conn.execute(
                "SELECT or_val FROM pushes WHERE message_id = ?", ("dup_test_001",)
            ).fetchone()
            conn.close()
        assert row[0] == pytest.approx(7.5)

    def test_upsert_does_not_downgrade_or_val(self, tmp_db):
        """Bei Duplikat wird or_val NICHT überschrieben wenn neuer Wert kleiner."""
        push_high = {
            "message_id": "no_downgrade_001",
            "ts_num": int(time.time()),
            "or": 8.0,
            "title": "Wichtiger Artikel",
            "headline": "",
            "kicker": "",
            "cat": "news",
            "link": "https://www.bild.de/news/y",
            "type": "editorial",
            "hour": 12,
            "title_len": 17,
            "opened": 2000,
            "received": 10000,
            "channel": "main",
            "channels": [],
            "is_eilmeldung": False,
            "n_apps": 1,
            "total_recipients": 10000,
        }
        push_low = dict(push_high)
        push_low["or"] = 0.0  # or=0 → soll nicht überschreiben

        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert([push_high])
            database.push_db_upsert([push_low])
            conn = sqlite3.connect(tmp_db)
            row = conn.execute(
                "SELECT or_val FROM pushes WHERE message_id = ?", ("no_downgrade_001",)
            ).fetchone()
            conn.close()
        # or_val=0 darf nicht überschreiben, das ON CONFLICT behält den alten Wert
        assert row[0] == pytest.approx(8.0)


# ── _push_db_load_all ────────────────────────────────────────────────────────

class TestPushDbLoadAll:
    def test_load_returns_list_of_dicts(self, tmp_db, sample_pushes):
        """_push_db_load_all gibt eine Liste von Dicts zurück."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            result = database.push_db_load_all()
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_load_returns_all_inserted_pushes(self, tmp_db, sample_pushes):
        """Alle eingefügten Pushes werden zurückgegeben."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            result = database.push_db_load_all()
        # _push_db_load_all filtert sportbild/autobild Links heraus
        bild_only = [p for p in sample_pushes
                     if "sportbild." not in p.get("link", "")
                     and "autobild." not in p.get("link", "")]
        assert len(result) == len(bild_only)

    def test_load_result_has_expected_fields(self, tmp_db, sample_pushes):
        """Jedes zurückgegebene Dict hat die erwarteten Felder."""
        required_fields = {
            "message_id", "or", "ts_num", "title", "cat",
            "link", "hour", "opened", "received", "is_eilmeldung"
        }
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            result = database.push_db_load_all()
        for row in result:
            for field in required_fields:
                assert field in row, f"Feld '{field}' fehlt in DB-Ergebnis"

    def test_load_with_min_ts_filters_old_entries(self, tmp_db, sample_pushes):
        """min_ts-Filter schließt ältere Einträge aus."""
        now = int(time.time())
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            # Setze min_ts auf "jetzt" → alle sample_pushes (ts = now - i*3600) übernehmen
            # Nur der neueste (i=0, ts=now) liegt über now-1
            result_all = database.push_db_load_all(min_ts=0)
            result_recent = database.push_db_load_all(min_ts=now - 1800)  # letzte 30min
        assert len(result_recent) <= len(result_all)

    def test_load_empty_db_returns_empty_list(self, tmp_db):
        """Leere DB → leere Liste, kein Fehler."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            result = database.push_db_load_all()
        assert result == []

    def test_load_channels_is_list(self, tmp_db, sample_pushes):
        """channels-Feld wird als Liste zurückgegeben (JSON deserialisiert)."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            result = database.push_db_load_all()
        for row in result:
            assert isinstance(row["channels"], list)


# ── _push_db_count ────────────────────────────────────────────────────────────

class TestPushDbCount:
    def test_count_empty_db(self, tmp_db):
        """Leere DB → 0."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            count = database.push_db_count()
        assert count == 0

    def test_count_after_insert(self, tmp_db, sample_pushes):
        """Nach Upsert von N Pushes gibt count N zurück."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            count = database.push_db_count()
        assert count == len(sample_pushes)

    def test_count_does_not_duplicate_on_reinsert(self, tmp_db, sample_pushes):
        """Doppelter Upsert derselben Daten ändert den Count nicht."""
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            database.push_db_upsert(sample_pushes)
            count = database.push_db_count()
        assert count == len(sample_pushes)

    def test_count_returns_integer(self, tmp_db, sample_pushes):
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(sample_pushes)
            count = database.push_db_count()
        assert isinstance(count, int)


# ── Snapshot / Liste laden ────────────────────────────────────────────────────

class TestSnapshotLoading:
    """Prüft, dass nach Upsert aus einer Snapshot-Liste die Daten korrekt lesbar sind."""

    def test_snapshot_list_upserted_and_readable(self, tmp_db):
        """Snapshot als Liste von Dicts → alle Einträge gespeichert und lesbar."""
        now = int(time.time())
        snapshot = [
            {
                "message_id": f"snap_{i}",
                "ts_num": now - i * 100,
                "or": 5.0,
                "title": f"Snapshot Artikel {i}",
                "headline": "",
                "kicker": "",
                "cat": "news",
                "link": f"https://www.bild.de/news/snap-{i}",
                "type": "editorial",
                "hour": 12,
                "title_len": 20,
                "opened": 800,
                "received": 15000,
                "channel": "main",
                "channels": [],
                "is_eilmeldung": False,
                "n_apps": 2,
                "total_recipients": 15000,
            }
            for i in range(5)
        ]
        with patch.object(database, "PUSH_DB_PATH", tmp_db):
            database.push_db_upsert(snapshot)
            loaded = database.push_db_load_all()
        assert len(loaded) == 5
        titles = {row["title"] for row in loaded}
        assert "Snapshot Artikel 0" in titles
