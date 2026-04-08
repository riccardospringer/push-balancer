"""test_ml.py — Tests für die ML-Module (gbrt, features, stats, predict, lightgbm_model).

Stellt sicher, dass alle Imports und Kernfunktionen ohne Laufzeitfehler ausführbar sind.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_pushes_ml():
    """100 Push-Dicts mit validen OR-Werten für ML-Training."""
    now = int(time.time())
    cats = ["news", "sport", "politik", "unterhaltung", "geld"]
    titles = [
        "Eilmeldung: Großes Ereignis in der Welt",
        "FC Bayern gewinnt das Spiel",
        "Bundestag beschließt neues Gesetz",
        "Stars feiern große Party",
        "DAX steigt auf Rekordhoch",
    ]
    pushes = []
    for i in range(100):
        pushes.append({
            "message_id": f"ml_test_{i:04d}",
            "ts_num": now - (i * 7200),  # alle 2h
            "or": 3.0 + (i % 8) * 0.5,   # OR zwischen 3.0 und 7.0
            "title": titles[i % len(titles)],
            "cat": cats[i % len(cats)],
            "hour": (8 + i) % 24,
            "is_eilmeldung": i % 20 == 0,
            "channels": ["main"],
            "total_recipients": 20000,
        })
    return pushes


# ── Import-Tests ───────────────────────────────────────────────────────────────

def test_gbrt_imports():
    from app.ml.gbrt import (
        gbrt_load_model, gbrt_predict, gbrt_train,
        gbrt_online_update, gbrt_check_drift,
    )


def test_features_imports():
    from app.ml.features import gbrt_extract_features


def test_stats_imports():
    from app.ml.stats import _gbrt_build_history_stats, gbrt_build_history_stats


def test_lightgbm_model_imports():
    from app.ml.lightgbm_model import (
        ml_train_model, unified_train, train_stacking_model, monitoring_tick,
    )


def test_predict_imports():
    from app.ml.predict import predict_or, safety_check, safety_envelope


def test_worker_imports():
    from app.research.worker import (
        _research_state, run_autonomous_analysis, update_residual_corrector,
        _residual_corrector, _residual_corrector_lock,
    )


# ── Feature-Extraktion ─────────────────────────────────────────────────────────

def test_gbrt_extract_features_basic():
    from app.ml.features import gbrt_extract_features
    push = {"title": "Test: Großes Ereignis", "cat": "news", "hour": 10, "ts_num": 1700000000}
    feats = gbrt_extract_features(push, {}, state=None)
    assert isinstance(feats, dict)
    assert len(feats) > 10
    assert "hour_sin" in feats
    assert "title_len" in feats


def test_gbrt_extract_features_with_stats(sample_pushes_ml):
    from app.ml.stats import _gbrt_build_history_stats
    from app.ml.features import gbrt_extract_features
    stats = _gbrt_build_history_stats(sample_pushes_ml)
    push = {"title": "Bayern-Spiel heute Abend", "cat": "sport", "hour": 20,
            "ts_num": sample_pushes_ml[-1]["ts_num"] + 3600}
    feats = gbrt_extract_features(push, stats, state=None)
    assert isinstance(feats, dict)
    assert feats.get("cat_sport", 0) == 1
    assert "cat_avg_or_7d" in feats


# ── History-Stats ──────────────────────────────────────────────────────────────

def test_build_history_stats_empty():
    from app.ml.stats import _gbrt_build_history_stats
    result = _gbrt_build_history_stats([])
    assert result["global_avg"] == 4.77
    assert result["global_n"] == 0


def test_build_history_stats_with_data(sample_pushes_ml):
    from app.ml.stats import _gbrt_build_history_stats
    result = _gbrt_build_history_stats(sample_pushes_ml)
    assert result["global_n"] >= 99  # ±1 wegen ts_num-Grenzfall bei now == target_ts
    assert 3.0 < result["global_avg"] < 8.0
    assert "news" in result["cat_stats"]
    assert "sport" in result["cat_stats"]


# ── GBRT Training (kritisch: braucht import time) ─────────────────────────────

def test_gbrt_train_too_few_pushes():
    """gbrt_train soll bei < 100 Pushes False zurückgeben, aber nicht crashen."""
    from app.ml.gbrt import gbrt_train
    import app.ml.gbrt as _gbrt_mod
    import app.database as _db_mod
    from unittest.mock import patch

    few_pushes = [
        {"message_id": f"x{i}", "ts_num": int(time.time()) - i * 3600 * 48,
         "or": 4.0 + i * 0.1, "title": "Test", "cat": "news", "hour": 10}
        for i in range(10)
    ]
    with patch.object(_db_mod, "push_db_load_all", return_value=few_pushes):
        result = gbrt_train()
    assert result is False


def test_gbrt_train_with_enough_data(sample_pushes_ml):
    """gbrt_train soll mit 100+ reifen Pushes True zurückgeben."""
    from app.ml.gbrt import gbrt_train, _gbrt_model
    import app.database as _db_mod
    from unittest.mock import patch

    # Pushes über 24h alt machen (sonst werden sie als "unreif" gefiltert)
    old_pushes = []
    now = int(time.time())
    for p in sample_pushes_ml:
        p2 = dict(p)
        p2["ts_num"] = now - 2 * 86400 - p2["ts_num"] % 3600
        old_pushes.append(p2)

    with patch.object(_db_mod, "push_db_load_all", return_value=old_pushes):
        result = gbrt_train()

    assert result is True
    import app.ml.gbrt as g
    with g._gbrt_lock:
        assert g._gbrt_model is not None


# ── predict_or Fallback-Kette ──────────────────────────────────────────────────

def test_predict_or_global_avg_fallback():
    from app.ml.predict import predict_or
    push = {"title": "Test", "cat": "news", "hour": 12, "ts_num": 1700000000}
    result = predict_or(push)
    assert result is not None
    assert result["advisory_only"] is True
    assert result["action_allowed"] is False
    assert 0.5 < result["predicted_or"] < 25.0
    assert result["basis_method"] in ("global_avg", "cat_hour_heuristic", "lightgbm",
                                       "gbrt", "error_fallback")


def test_predict_or_safety_envelope():
    from app.ml.predict import predict_or, safety_check
    safety_check()  # sollte nicht werfen
    push = {"title": "Wichtige Nachricht!", "cat": "politik", "hour": 9,
            "ts_num": int(time.time()) - 3600}
    result = predict_or(push, research_state={"global_avg": 5.2})
    assert result["safety_mode"] == "ADVISORY_ONLY"
