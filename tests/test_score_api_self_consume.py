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
