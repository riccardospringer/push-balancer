"""Der Push-Auto-Fetch-Worker muss unabhängig von den schweren Research-/ML-
Automationen schaltbar sein.

Hintergrund: Eine Instanz im AS-Netz (Next-Pod) soll die Push-Historie 24/7
selbst von bildcms.de ziehen, OHNE die CPU-intensiven Research-Worker zu
aktivieren (die einen Core dauerhaft auslasten). Dafür gibt es das eigene Flag
PUSH_AUTO_FETCH_ENABLED, das per Default dem alten BACKGROUND_AUTOMATIONS_ENABLED
folgt (rückwärtskompatibel), sich aber getrennt überschreiben lässt.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _flag(env_overrides: dict[str, str]) -> dict[str, bool]:
    """Lädt app.config in einem frischen Interpreter mit gesetzten Env-Vars."""
    code = (
        "import app.config as c;"
        "print(int(c.PUSH_AUTO_FETCH_ENABLED), int(c.BACKGROUND_AUTOMATIONS_ENABLED),"
        " int(c.PUSH_RENDER_SYNC_ENABLED))"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": root}
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=root, timeout=60,
    )
    assert out.returncode == 0, f"config-Import fehlgeschlagen:\n{out.stderr}"
    auto, background, render_sync = out.stdout.split()
    return {
        "auto_fetch": bool(int(auto)),
        "background": bool(int(background)),
        "render_sync": bool(int(render_sync)),
    }


def test_auto_fetch_defaults_to_background_when_unset():
    """Ohne eigenes Flag folgt PUSH_AUTO_FETCH_ENABLED dem BACKGROUND-Flag."""
    on = _flag({"BACKGROUND_AUTOMATIONS_ENABLED": "true"})
    assert on["auto_fetch"] is True and on["background"] is True

    off = _flag({"BACKGROUND_AUTOMATIONS_ENABLED": "false"})
    assert off["auto_fetch"] is False and off["background"] is False


def test_auto_fetch_independent_of_background():
    """Der Next-Fall: AutoFetch AN, schwere Automationen AUS."""
    r = _flag({
        "PUSH_AUTO_FETCH_ENABLED": "true",
        "BACKGROUND_AUTOMATIONS_ENABLED": "false",
    })
    assert r["auto_fetch"] is True, "AutoFetch muss trotz BACKGROUND=false laufen"
    assert r["background"] is False, "Research-Worker müssen aus bleiben"


def test_auto_fetch_can_be_disabled_while_background_on():
    """Gegenrichtung: AutoFetch gezielt aus, Automationen an."""
    r = _flag({
        "PUSH_AUTO_FETCH_ENABLED": "false",
        "BACKGROUND_AUTOMATIONS_ENABLED": "true",
    })
    assert r["auto_fetch"] is False
    assert r["background"] is True


def test_render_sync_defaults_to_background_when_unset():
    """PUSH_RENDER_SYNC_ENABLED folgt per Default dem BACKGROUND-Flag."""
    on = _flag({"BACKGROUND_AUTOMATIONS_ENABLED": "true"})
    assert on["render_sync"] is True
    off = _flag({"BACKGROUND_AUTOMATIONS_ENABLED": "false"})
    assert off["render_sync"] is False


def test_render_sync_independent_of_background():
    """Der Next-Fall: Render-Relay AN, schwere Automationen AUS."""
    r = _flag({
        "PUSH_RENDER_SYNC_ENABLED": "true",
        "BACKGROUND_AUTOMATIONS_ENABLED": "false",
    })
    assert r["render_sync"] is True, "Render-Sync muss trotz BACKGROUND=false laufen"
    assert r["background"] is False
