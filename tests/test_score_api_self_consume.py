"""Selbst-Konsum der Score-API (Render): Loopback-Basis-URL nutzt den eigenen
Server-Key.

Auf der Instanz, die den kanonischen Score selbst berechnet und die
/api/v1/scores-Route selbst bedient (Render), zeigt
PUSH_BALANCER_SCORE_API_BASE_URL auf Loopback. Der eigene SCORE_API_KEY ist
dann per Definition der richtige Consumer-Key — ein separat gepflegter (und
potenziell veralteter) PUSH_BALANCER_SCORE_API_KEY darf den Selbstaufruf nie
brechen. Externe Basis-URLs (Next-Pod) bleiben unveraendert.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _key(env: dict[str, str]) -> str:
    code = "import app.config as c; print(c.PUSH_BALANCER_SCORE_API_KEY)"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": root}
    full.update(env)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=full, cwd=root, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_loopback_base_url_always_uses_server_key():
    assert _key({
        "SCORE_API_KEY": "serverkey",
        "PUSH_BALANCER_SCORE_API_BASE_URL": "http://127.0.0.1:8050",
        "PUSH_BALANCER_SCORE_API_KEY": "stale-dashboard-value",
    }) == "serverkey"


def test_external_base_url_keeps_consumer_key():
    assert _key({
        "SCORE_API_KEY": "serverkey",
        "PUSH_BALANCER_SCORE_API_BASE_URL": "https://push-balancer-internal.example.de",
        "PUSH_BALANCER_SCORE_API_KEY": "nextkey",
    }) == "nextkey"


def test_empty_consumer_key_falls_back_to_server_key():
    assert _key({
        "SCORE_API_KEY": "serverkey",
        "PUSH_BALANCER_SCORE_API_BASE_URL": "https://push-balancer-internal.example.de",
    }) == "serverkey"


def test_self_consume_flag_overrides_shadowed_dashboard_values():
    """Render-Dashboards koennen alte Blueprint-Werte ueberschatten; der neue
    SELF_CONSUME-Key muss sie vollstaendig ueberstimmen."""
    assert _key({
        "PUSH_BALANCER_SCORE_API_SELF_CONSUME": "true",
        "SCORE_API_KEY": "serverkey",
        "PUSH_BALANCER_SCORE_API_ENABLED": "false",
        "PUSH_BALANCER_SCORE_API_BASE_URL": "https://tot.example.invalid",
        "PUSH_BALANCER_SCORE_API_KEY": "stale",
        "PORT": "8050",
    }) == "serverkey"


def test_self_consume_without_server_key_generates_ephemeral_key():
    """Ohne gesetzten SCORE_API_KEY erzeugt der Selbstkonsum einen ephemeren
    Prozess-Key — Server-Route und Consumer-Client leben in derselben Instanz,
    daher genuegt das fuer das Loopback-Selbstgespraech (Render ohne Env-Sync)."""
    import os, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c",
         "import app.config as c; print(int(c.PUSH_BALANCER_SCORE_API_ENABLED),"
         " int(bool(c.SCORE_API_KEY)), int(c.SCORE_API_KEY==c.PUSH_BALANCER_SCORE_API_KEY))"],
        capture_output=True, text=True, cwd=root, timeout=60,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": root,
             "PUSH_BALANCER_SCORE_API_SELF_CONSUME": "true",
             "PUSH_BALANCER_SCORE_API_ENABLED": "false"},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["1", "1", "1"]


def test_render_defaults_enable_self_consume_and_disable_live_push_posts():
    """Auf Render (RENDER=true) muessen die Code-Defaults wirken, weil der
    Blueprint-Env-Sync neue Keys nicht zuverlaessig anlegt."""
    import os, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c",
         "import app.config as c; print(int(c.PUSH_BALANCER_SCORE_API_ENABLED),"
         " int(c.PUSH_TEAMS_LIVE_PUSH_POSTS_ENABLED), c.PUSH_BALANCER_SCORE_API_BASE_URL)"],
        capture_output=True, text=True, cwd=root, timeout=60,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": root,
             "RENDER": "true", "PORT": "8050"},
    )
    assert out.returncode == 0, out.stderr
    parts = out.stdout.split()
    assert parts[0] == "1", "Selbstkonsum muss auf Render aktiv sein"
    assert parts[1] == "0", "Live-Push-Posts muessen aus sein"
    assert parts[2] == "http://127.0.0.1:8050"
