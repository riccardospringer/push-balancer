"""Guard the exact Render blobs that define the migrated score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import score_main
from app.ml import lightgbm_model


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_render_score_files_match_source_manifest():
    manifest = json.loads((ROOT / "RENDER_SCORE_SOURCE.json").read_text(encoding="utf-8"))

    assert manifest["sourceCommit"] == "3b212fcdfb5339d8bffbfe55ae79ffb7f0ee6808"
    assert manifest["calibrationCommit"] == "8fe5472f00d9254ea0ae09647601bee2f1127b5a"
    scheduled_top5 = manifest["scheduledTop5Overlay"]
    assert scheduled_top5["approvalStatus"] == "approved"
    assert scheduled_top5["approvalRecordedAt"] == "2026-08-09"
    assert scheduled_top5["historyMode"] == "cloud_only_durable_slot_and_receipt_dedup"
    assert scheduled_top5["livePushHistoryRequired"] is False
    assert scheduled_top5["contractVersion"] == 2
    assert scheduled_top5["recommendationCount"] == 5
    assert manifest["runtimeParity"]["status"] == "pending"
    for relative_path, expected_digest in manifest["sha256"].items():
        actual_digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, f"Render score source changed: {relative_path}"


def test_frozen_seed_model_loads_in_the_test_runtime(monkeypatch):
    isolated_state = {**lightgbm_model._ml_state, "model": None}
    monkeypatch.setattr(lightgbm_model, "_ml_state", isolated_state)
    score_main._preload_local_openmp_for_development()

    assert lightgbm_model.load_seed_model() is True
    assert isolated_state["model"] is not None
    assert isolated_state["feature_names"]


def test_docker_context_excludes_runtime_models_but_keeps_frozen_seed():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "*.pkl" in dockerignore
    assert "*.joblib" in dockerignore
    assert "!app/ml/seed_model.pkl" in dockerignore
