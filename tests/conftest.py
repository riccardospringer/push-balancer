"""conftest.py — Shared fixtures für Push Balancer Tests.

Importiert den Monolith push-balancer-server.py über importlib
(Dateiname enthält Bindestriche, daher kein normaler import möglich).
"""
import importlib.util
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Monolith laden ───────────────────────────────────────────────────────────
# Der Monolith startet beim Import Snapshot-Seeding und Background-Threads.
# Wir patchen PUSH_DB_PATH auf eine leere tmp-DB BEVOR der Monolith geladen wird,
# damit keine Produktions-DB berührt wird.

_SERVER_PATH = str(
    Path(__file__).parent.parent / "push-balancer-server.py"
)


def _load_server_module():
    """Lädt push-balancer-server.py genau einmal als Modul 'pbserver'."""
    if "pbserver" in sys.modules:
        return sys.modules["pbserver"]
    spec = importlib.util.spec_from_file_location("pbserver", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pbserver"] = mod
    spec.loader.exec_module(mod)
    return mod


# Beim ersten Testlauf Modul importieren (ggf. mit leerem DB-Pfad in Env)
_server = _load_server_module()


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Erzeugt eine leere SQLite-DB, patcht PUSH_DB_PATH im Server-Modul.

    Gibt den Pfad als str zurück und stellt sicher, dass alle DB-Operationen
    in der Temp-DB landen — die Produktions-DB bleibt unberührt.
    """
    db_path = str(tmp_path / "test_pushes.db")

    # DB-Schema initialisieren (identisch zum Monolith)
    conn = sqlite3.connect(db_path)
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
    conn.commit()
    conn.close()

    with patch.object(_server, "PUSH_DB_PATH", db_path):
        yield db_path


@pytest.fixture()
def sample_pushes():
    """10 Push-Dicts mit allen Pflichtfeldern, die _push_db_upsert erwartet."""
    now = int(time.time())
    pushes = []
    categories = ["news", "sport", "politik", "unterhaltung", "geld",
                  "sport", "news", "sport", "politik", "digital"]
    titles = [
        "Eilmeldung: Terror-Anschlag in Berlin",
        "FC Bayern: Stürmer verletzt — Ausfall für Wochen",
        "Bundeskanzler tritt zurück",
        "Hollywood-Star wechselt die Seite",
        "DAX bricht ein: Rekordverlust",
        "Ronaldo wechselt — Transfer-Knaller!",
        "Warnung vor Sturm über Norddeutschland",
        "Entlassung: Trainer muss gehen",
        "Wahl-Ergebnis: Historische Mehrheit",
        "KI-Trend: Alle wollen ChatGPT",
    ]
    for i in range(10):
        pushes.append({
            "message_id": f"test_msg_{i:04d}",
            "ts_num": now - (i * 3600),
            "or": 4.5 + i * 0.3,
            "title": titles[i],
            "headline": titles[i],
            "kicker": "BILD",
            "cat": categories[i],
            "link": f"https://www.bild.de/{categories[i]}/artikel-{i}",
            "type": "editorial",
            "hour": (10 + i) % 24,
            "title_len": len(titles[i]),
            "opened": 1000 + i * 200,
            "received": 20000,
            "channel": "main",
            "channels": ["main"],
            "is_eilmeldung": i == 0,
            "n_apps": 2,
            "total_recipients": 20000,
        })
    return pushes
