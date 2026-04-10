"""conftest.py — Shared fixtures für Push Balancer Tests.

Stellt modulare Fixtures für die FastAPI-/app/-Runtime bereit.
"""
import time
from unittest.mock import patch

import pytest
from app import database


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Erzeugt eine leere SQLite-DB für die modulare Datenzugriffsschicht.

    Gibt den Pfad als str zurück und stellt sicher, dass alle DB-Operationen
    in der Temp-DB landen — die Produktions-DB bleibt unberührt.
    """
    db_path = str(tmp_path / "test_pushes.db")

    with patch.object(database, "PUSH_DB_PATH", db_path):
        database.init_db()
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
