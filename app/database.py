"""app/database.py — SQLite-Datenbankfunktionen für den Push Balancer.

Alle DB-Funktionen aus push-balancer-server.py extrahiert und importierbar gemacht.
Die Funktionen sind thread-safe via _push_db_lock (threading.Lock).
"""
import json
import logging
import sqlite3
import threading
import time

from app.config import PUSH_DB_PATH

log = logging.getLogger("push-balancer")

_push_db_lock = threading.Lock()


# ── Init ───────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Initialisiert die SQLite-Datenbank (alle Tabellen + Indizes, idempotent)."""
    conn = sqlite3.connect(PUSH_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS pushes (
        message_id TEXT PRIMARY KEY,
        ts_num INTEGER NOT NULL,
        or_val REAL DEFAULT 0,
        title TEXT,
        headline TEXT,
        kicker TEXT,
        cat TEXT,
        link TEXT,
        type TEXT DEFAULT 'editorial',
        hour INTEGER DEFAULT -1,
        title_len INTEGER DEFAULT 0,
        opened INTEGER DEFAULT 0,
        received INTEGER DEFAULT 0,
        channel TEXT DEFAULT '',
        channels TEXT DEFAULT '[]',
        is_eilmeldung INTEGER DEFAULT 0,
        updated_at INTEGER DEFAULT 0,
        target_stats TEXT DEFAULT '{}',
        app_list TEXT DEFAULT '[]',
        n_apps INTEGER DEFAULT 0,
        total_recipients INTEGER DEFAULT 0
    )""")
    # Migrate: add new columns if they don't exist
    for _col, _type, _default in [
        ("target_stats", "TEXT", "'{}'"), ("app_list", "TEXT", "'[]'"),
        ("n_apps", "INTEGER", "0"), ("total_recipients", "INTEGER", "0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE pushes ADD COLUMN {_col} {_type} DEFAULT {_default}")
        except Exception:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_ts ON pushes(ts_num)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_cat ON pushes(cat)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_or_ts ON pushes(or_val, ts_num)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pushes_hour_or ON pushes(hour, or_val)")

    # Prediction log
    conn.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        push_id TEXT NOT NULL,
        predicted_or REAL,
        actual_or REAL,
        basis_method TEXT DEFAULT '',
        methods_detail TEXT DEFAULT '{}',
        features TEXT DEFAULT '{}',
        model_version INTEGER DEFAULT 0,
        predicted_at INTEGER NOT NULL,
        actual_recorded_at INTEGER DEFAULT 0,
        title TEXT DEFAULT '',
        confidence REAL DEFAULT 0,
        q10 REAL DEFAULT 0,
        q90 REAL DEFAULT 0,
        UNIQUE(push_id)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predlog_ts ON prediction_log(predicted_at)")

    # Experiment Tracking
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        hyperparams TEXT DEFAULT '{}',
        metrics TEXT DEFAULT '{}',
        baselines TEXT DEFAULT '{}',
        cv_results TEXT DEFAULT '{}',
        n_features INTEGER DEFAULT 0,
        n_samples INTEGER DEFAULT 0,
        model_hash TEXT DEFAULT '',
        promoted INTEGER DEFAULT 0,
        training_duration_s REAL DEFAULT 0
    )""")

    # Promotion Gates Log
    conn.execute("""CREATE TABLE IF NOT EXISTS promotion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        passed INTEGER DEFAULT 0,
        gates TEXT DEFAULT '{}',
        champion_mae REAL DEFAULT 0,
        challenger_mae REAL DEFAULT 0,
        reason TEXT DEFAULT ''
    )""")

    # Embedding Cache
    conn.execute("""CREATE TABLE IF NOT EXISTS embedding_cache (
        title_hash TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        embedding TEXT DEFAULT '[]',
        created_at INTEGER NOT NULL
    )""")

    # Monitoring Events
    conn.execute("""CREATE TABLE IF NOT EXISTS monitoring_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT DEFAULT 'info',
        message TEXT DEFAULT '',
        metrics_json TEXT DEFAULT '{}'
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_ts ON monitoring_events(timestamp)")

    # Tagesplan Suggestion Snapshots
    conn.execute("""CREATE TABLE IF NOT EXISTS tagesplan_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_iso TEXT NOT NULL,
        slot_hour INTEGER NOT NULL,
        suggestion_num INTEGER NOT NULL,
        article_title TEXT,
        article_link TEXT,
        article_category TEXT,
        article_score REAL DEFAULT 0,
        expected_or REAL DEFAULT 0,
        best_cat TEXT DEFAULT '',
        captured_at INTEGER NOT NULL,
        UNIQUE(date_iso, slot_hour, suggestion_num)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tpsug_date ON tagesplan_suggestions(date_iso)")

    # ML v2: LLM-Score Spalten (idempotent via ALTER TABLE)
    _llm_columns = [
        ("llm_magnitude", "REAL DEFAULT 0"),
        ("llm_clickability", "REAL DEFAULT 0"),
        ("llm_relevanz", "REAL DEFAULT 0"),
        ("llm_dringlichkeit", "REAL DEFAULT 0"),
        ("llm_emotionalitaet", "REAL DEFAULT 0"),
        ("llm_scored_at", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_type in _llm_columns:
        try:
            conn.execute(f"ALTER TABLE pushes ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Spalte existiert bereits

    conn.commit()
    conn.close()
    log.info("[PushDB] Initialized at %s", PUSH_DB_PATH)


# ── Upsert / Laden ─────────────────────────────────────────────────────────

def push_db_upsert(parsed_pushes: list) -> int:
    """Insert or update parsed pushes into SQLite. Returns count of new/updated rows."""
    if not parsed_pushes:
        return 0
    now = int(time.time())
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        cur = conn.cursor()
        count = 0
        for p in parsed_pushes:
            mid = p.get("message_id") or f"{p['ts_num']}_{p.get('title', '')[:30]}"
            _ts_json = json.dumps(p.get("target_stats", {})) if isinstance(p.get("target_stats"), dict) else "{}"
            _apps_json = json.dumps(p.get("app_list", [])) if isinstance(p.get("app_list"), list) else "[]"
            cur.execute("""INSERT INTO pushes (message_id, ts_num, or_val, title, headline, kicker,
                cat, link, type, hour, title_len, opened, received, channel, channels, is_eilmeldung, updated_at,
                target_stats, app_list, n_apps, total_recipients)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    or_val = CASE WHEN excluded.or_val > 0 THEN excluded.or_val ELSE or_val END,
                    opened = CASE WHEN excluded.opened > opened THEN excluded.opened ELSE opened END,
                    received = CASE WHEN excluded.received > received THEN excluded.received ELSE received END,
                    target_stats = CASE WHEN length(excluded.target_stats) > 2 THEN excluded.target_stats ELSE target_stats END,
                    n_apps = CASE WHEN excluded.n_apps > 0 THEN excluded.n_apps ELSE n_apps END,
                    total_recipients = CASE WHEN excluded.total_recipients > 0 THEN excluded.total_recipients ELSE total_recipients END,
                    updated_at = ?
            """, (mid, p["ts_num"], p.get("or", p.get("or_val", 0)), p.get("title", ""), p.get("headline", ""),
                  p.get("kicker", ""), p.get("cat", ""), p.get("link", ""), p.get("type", "editorial"),
                  p.get("hour", -1), p.get("title_len", 0), p.get("opened", 0), p.get("received", 0),
                  p.get("channel", ""), json.dumps(p.get("channels", [])), 1 if p.get("is_eilmeldung") else 0,
                  now, _ts_json, _apps_json, p.get("n_apps", 0), p.get("total_recipients", 0), now))
            count += cur.rowcount
        conn.commit()
        conn.close()
    return count


def push_db_load_all(min_ts: int = 0) -> list:
    """Lädt alle Pushes aus SQLite, optional gefiltert nach min_ts (Unix-Timestamp).

    Nutzt eigene Connection mit WAL-Mode für nicht-blockierendes Lesen.
    Filtert SportBILD und AutoBILD Links heraus.
    """
    conn = sqlite3.connect(PUSH_DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM pushes WHERE ts_num > ? AND link NOT LIKE '%sportbild.%' AND link NOT LIKE '%autobild.%' ORDER BY ts_num DESC",
            (min_ts,),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for r in rows:
        keys = r.keys()
        result.append({
            "message_id": r["message_id"],
            "or": r["or_val"],
            "ts": str(r["ts_num"]),
            "ts_num": r["ts_num"],
            "title": r["title"],
            "headline": r["headline"],
            "kicker": r["kicker"],
            "cat": r["cat"],
            "link": r["link"],
            "type": r["type"],
            "hour": r["hour"],
            "title_len": r["title_len"],
            "opened": r["opened"],
            "received": r["received"],
            "channel": r["channel"],
            "channels": json.loads(r["channels"] or "[]"),
            "is_eilmeldung": bool(r["is_eilmeldung"]),
            "target_stats": json.loads(r["target_stats"] or "{}") if "target_stats" in keys else {},
            "app_list": json.loads(r["app_list"] or "[]") if "app_list" in keys else [],
            "n_apps": r["n_apps"] if "n_apps" in keys else 0,
            "total_recipients": r["total_recipients"] if "total_recipients" in keys else 0,
        })
    return result


def push_db_max_ts() -> int:
    """Gibt den höchsten ts_num-Wert in der Datenbank zurück."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        row = conn.execute("SELECT MAX(ts_num) FROM pushes").fetchone()
        conn.close()
    return row[0] or 0


def push_db_count() -> int:
    """Gibt die Gesamtanzahl der Pushes in der Datenbank zurück."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        row = conn.execute("SELECT COUNT(*) FROM pushes").fetchone()
        conn.close()
    return row[0] or 0


# ── Prediction Log ─────────────────────────────────────────────────────────

def push_db_log_prediction(
    push_id: str,
    predicted_or: float,
    actual_or: float,
    basis_method: str = "",
    methods_detail: dict | None = None,
    features: dict | None = None,
    model_version: int = 0,
    title: str = "",
    confidence: float = 0.0,
    q10: float = 0.0,
    q90: float = 0.0,
) -> None:
    """Speichert eine Prediction im prediction_log für ML-Training.

    Upsert: aktualisiert actual_or wenn der Eintrag bereits existiert.
    """
    now = int(time.time())
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.execute("""INSERT INTO prediction_log
            (push_id, predicted_or, actual_or, basis_method, methods_detail, features,
             model_version, predicted_at, actual_recorded_at, title, confidence, q10, q90)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(push_id) DO UPDATE SET
                actual_or = CASE WHEN excluded.actual_or > 0 THEN excluded.actual_or ELSE actual_or END,
                actual_recorded_at = CASE WHEN excluded.actual_or > 0 THEN ? ELSE actual_recorded_at END
        """, (push_id, predicted_or, actual_or, basis_method,
              json.dumps(methods_detail or {}), json.dumps(features or {}),
              model_version, now, now, title, confidence, q10, q90, now))
        conn.commit()
        conn.close()


def push_db_get_training_data(min_ts: int = 0, limit: int = 5000) -> list:
    """Lädt Prediction-Log-Einträge mit predicted + actual OR für ML-Training."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT * FROM prediction_log
            WHERE predicted_or > 0 AND actual_or > 0 AND predicted_at > ?
            ORDER BY predicted_at DESC LIMIT ?""", (min_ts, limit)).fetchall()
        conn.close()
    return [dict(r) for r in rows]


# ── Monitoring ─────────────────────────────────────────────────────────────

def log_monitoring_event(
    event_type: str,
    severity: str,
    message: str,
    metrics: dict | None = None,
) -> None:
    """Speichert ein Monitoring-Event in der DB.

    event_type: drift/calibration_shift/mae_spike/ab_result/online_pause
    severity: info/warning/critical
    """
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            conn.execute(
                "INSERT INTO monitoring_events (timestamp, event_type, severity, message, metrics_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(time.time()), event_type, severity, message,
                 json.dumps(metrics or {}, default=str)),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        log.warning("[Monitoring] Event-Log Fehler: %s", e)


def load_monitoring_events(limit: int = 100) -> list:
    """Lädt die letzten N Monitoring-Events aus der DB."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM monitoring_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def load_experiments(limit: int = 50) -> list:
    """Lädt die letzten N ML-Experimente."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        for json_col in ("hyperparams", "metrics", "baselines", "cv_results"):
            if d.get(json_col):
                try:
                    d[json_col] = json.loads(d[json_col])
                except Exception:
                    pass
        result.append(d)
    return result


def load_llm_scores_for_push(push_id: str) -> dict:
    """Lädt LLM-Scores aus DB für einen Push (für Feature-Extraktion im Training)."""
    if not push_id:
        return {}
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            row = conn.execute(
                """SELECT llm_magnitude, llm_clickability, llm_relevanz,
                llm_dringlichkeit, llm_emotionalitaet FROM pushes WHERE message_id=?""",
                (push_id,),
            ).fetchone()
            conn.close()
        if row and row[0] and row[0] > 0:
            return {
                "magnitude": float(row[0]),
                "clickability": float(row[1] or 0),
                "relevanz": float(row[2] or 0),
                "dringlichkeit": float(row[3] or 0),
                "emotionalitaet": float(row[4] or 0),
            }
    except Exception:
        pass
    return {}


def save_tagesplan_suggestions(date_iso: str, slot_hour: int, suggestions: list) -> None:
    """Speichert Tagesplan-Vorschläge für einen Slot in der DB."""
    now = int(time.time())
    try:
        with _push_db_lock:
            conn = sqlite3.connect(PUSH_DB_PATH)
            for i, sug in enumerate(suggestions):
                conn.execute("""INSERT OR REPLACE INTO tagesplan_suggestions
                    (date_iso, slot_hour, suggestion_num, article_title, article_link,
                     article_category, article_score, expected_or, best_cat, captured_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (date_iso, slot_hour, i,
                     sug.get("title", ""), sug.get("link", ""),
                     sug.get("cat", ""), sug.get("score", 0.0),
                     sug.get("expected_or", 0.0), sug.get("best_cat", ""),
                     now))
            conn.commit()
            conn.close()
    except Exception as e:
        log.warning("[DB] save_tagesplan_suggestions Fehler: %s", e)


def load_tagesplan_suggestions(date_iso: str | None = None, limit: int = 200) -> list:
    """Lädt gespeicherte Tagesplan-Vorschläge aus der DB."""
    with _push_db_lock:
        conn = sqlite3.connect(PUSH_DB_PATH)
        conn.row_factory = sqlite3.Row
        if date_iso:
            rows = conn.execute(
                "SELECT * FROM tagesplan_suggestions WHERE date_iso=? ORDER BY slot_hour, suggestion_num",
                (date_iso,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tagesplan_suggestions ORDER BY captured_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]
