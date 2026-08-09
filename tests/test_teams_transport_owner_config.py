"""Fail-closed configuration tests for the single Teams transport owner."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _import_config(tmp_path, **overrides: str) -> subprocess.CompletedProcess[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": root,
        "DB_PATH": str(tmp_path / "synthetic-config.db"),
        "PUSH_DB_DURABILITY_REQUIRED": "false",
        "PUSH_TEAMS_ALERTS_ENABLED": "true",
        "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED": "false",
        "POWER_AUTOMATE_API_KEY": "",
    }
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "import app.config"],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
        timeout=60,
    )


def test_config_import_rejects_two_enabled_teams_transport_owners(tmp_path):
    result = _import_config(
        tmp_path,
        PUSH_TEAMS_BACKGROUND_SENDER_ENABLED="true",
        POWER_AUTOMATE_API_KEY="synthetic-conflict-key",
    )

    assert result.returncode != 0
    assert "genau ein Transport-Eigentuemer ist erlaubt" in result.stderr
    assert "synthetic-conflict-key" not in result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"POWER_AUTOMATE_API_KEY": "synthetic-power-automate-only"},
        {"PUSH_TEAMS_BACKGROUND_SENDER_ENABLED": "true"},
        {
            "PUSH_TEAMS_ALERTS_ENABLED": "false",
            "PUSH_TEAMS_BACKGROUND_SENDER_ENABLED": "true",
            "POWER_AUTOMATE_API_KEY": "synthetic-inactive-key",
        },
    ],
)
def test_config_import_allows_at_most_one_live_teams_transport_owner(
    tmp_path,
    overrides,
):
    result = _import_config(tmp_path, **overrides)

    assert result.returncode == 0, result.stderr
