"""Verify the secret-free release-container parity fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_seed_and_heuristic_runtime_fixture():
    completed = subprocess.run(
        [sys.executable, "scripts/score-parity-fixture.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout.strip())

    assert report["seedSha256"] == (
        "63637a93f55f009b9bb6d5a5f88e3f79addcad0468b6b134714d97996777d58e"
    )
    assert report["heuristicScore"] == 79.9
    assert report["result"] == {
        "basis": "lightgbm",
        "confidence": 0.695,
        "predictedOR": 0.0535,
        "score": 77.7,
    }


def test_parity_workflow_is_secret_free_and_cross_architecture():
    path = ROOT / ".github" / "workflows" / "score-runtime-parity.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow = path.read_text(encoding="utf-8")

    assert set(document["permissions"]) == {"contents"}
    assert document["permissions"]["contents"] == "read"
    assert "pull_request_target" not in workflow
    assert "linux/amd64" in workflow
    assert "linux/arm64" in workflow
    assert "--network none" in workflow
    assert "--read-only" in workflow
    assert "retention-days: 1" in workflow
