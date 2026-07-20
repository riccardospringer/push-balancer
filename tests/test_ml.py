"""test_ml.py — Tests für die ML-Module (gbrt, features, stats, predict, lightgbm_model).

Stellt sicher, dass alle Imports und Kernfunktionen ohne Laufzeitfehler ausführbar sind.
"""
import sys
import time
import types
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


def test_predict_heuristic_skips_openai_by_default(sample_pushes_ml, monkeypatch):
    import app.config as config
    import app.ml.heuristic as heuristic

    calls = {"count": 0}

    class _DummyOpenAI:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1

    with heuristic._OPENAI_PREDICTION_CACHE_LOCK:
        heuristic._OPENAI_PREDICTION_CACHE.clear()
    heuristic._OPENAI_PREDICTION_CLIENT = None
    heuristic._OPENAI_PREDICTION_CLIENT_KEY = ""

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "PAID_EXTERNAL_APIS_ENABLED", False)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_ENABLED", False)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_DummyOpenAI))

    push = {
        "title": "Neue Nachricht zur Lage in Berlin",
        "cat": "news",
        "hour": 12,
        "ts_num": max(p["ts_num"] for p in sample_pushes_ml) + 3600,
        "is_eilmeldung": False,
    }

    result = heuristic.predict_heuristic(push, sample_pushes_ml, state={})

    assert result is not None
    assert calls["count"] == 0
    assert "gpt_content_scoring" not in result["methods"]


def test_predict_heuristic_caches_openai_results_when_enabled(sample_pushes_ml, monkeypatch):
    import app.config as config
    import app.ml.heuristic as heuristic

    calls = {"count": 0}

    class _DummyCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"or_prognose": 6.4, "reasoning": "Solider Testtitel."}'
                        )
                    )
                ]
            )

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=_DummyCompletions())

    with heuristic._OPENAI_PREDICTION_CACHE_LOCK:
        heuristic._OPENAI_PREDICTION_CACHE.clear()
    heuristic._OPENAI_PREDICTION_CLIENT = None
    heuristic._OPENAI_PREDICTION_CLIENT_KEY = ""

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "PAID_EXTERNAL_APIS_ENABLED", True)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_ENABLED", True)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MODEL", "test-mini")
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_TIMEOUT_S", 1.0)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MAX_TOKENS", 32)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_CACHE_TTL_S", 3600)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR", 10)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY", 20)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_DummyClient))

    push = {
        "title": "Neue Nachricht zur Lage in Berlin",
        "cat": "news",
        "hour": 12,
        "ts_num": max(p["ts_num"] for p in sample_pushes_ml) + 3600,
        "is_eilmeldung": False,
    }

    result1 = heuristic.predict_heuristic(push, sample_pushes_ml, state={})
    result2 = heuristic.predict_heuristic(push, sample_pushes_ml, state={})

    assert result1 is not None
    assert result2 is not None
    assert calls["count"] == 1
    assert result1["methods"]["gpt_content_scoring"] == 6.4
    assert result2["methods"]["gpt_content_scoring"] == 6.4


def test_research_external_context_uses_local_defaults_when_disabled(monkeypatch):
    import app.research.worker as worker

    monkeypatch.setattr("app.config.RESEARCH_EXTERNAL_CONTEXT_ENABLED", False)
    worker._external_context_cache = {
        "weather": {},
        "trends": [],
        "holiday": "",
        "last_fetch": 0,
    }
    state = {}

    result = worker._fetch_external_context(state)

    assert result["weather"]["weather_desc"] == "disabled"
    assert state["external_context"]["mode"] == "local-defaults"
    assert state["external_context"]["trends_count"] == 0


def test_predict_heuristic_skips_openai_when_budget_is_zero(sample_pushes_ml, monkeypatch):
    import app.config as config
    import app.ml.heuristic as heuristic

    calls = {"count": 0}

    class _DummyOpenAI:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1

    with heuristic._OPENAI_PREDICTION_CACHE_LOCK:
        heuristic._OPENAI_PREDICTION_CACHE.clear()
    heuristic._OPENAI_PREDICTION_CLIENT = None
    heuristic._OPENAI_PREDICTION_CLIENT_KEY = ""

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "PAID_EXTERNAL_APIS_ENABLED", True)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_ENABLED", True)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR", 0)
    monkeypatch.setattr(config, "OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY", 0)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_DummyOpenAI))

    push = {
        "title": "Neue Nachricht zur Lage in Berlin",
        "cat": "news",
        "hour": 12,
        "ts_num": max(p["ts_num"] for p in sample_pushes_ml) + 3600,
        "is_eilmeldung": False,
    }

    result = heuristic.predict_heuristic(push, sample_pushes_ml, state={})

    assert result is not None
    assert calls["count"] == 0
    assert "gpt_content_scoring" not in result["methods"]


def test_generate_push_title_local_fallback_without_openai(monkeypatch):
    from push_title_agent import generate_push_title

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("PAID_EXTERNAL_APIS_ENABLED", "false")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_ENABLED", "false")

    result = generate_push_title(
        article_title="Breaking Test: Wichtige Entscheidung im Bundestag",
        category="politik",
    )

    assert result["gewinner"]["titel"]
    assert result["meta"]["modus"] == "local-fallback"
    assert result["meta"]["modell"] == "local-fallback"
    assert isinstance(result["alle_kandidaten"], dict)


def test_generate_push_title_local_fallback_marks_video_context(monkeypatch):
    from push_title_agent import generate_push_title

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("PAID_EXTERNAL_APIS_ENABLED", "false")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_ENABLED", "false")

    result = generate_push_title(
        article_title="Spektakuläre Szenen aus dem Derby",
        category="sport",
        article_type="video",
    )

    assert result["meta"]["content_type"] == "video"
    assert "video" in result["gewinner"]["warum_dieser"].lower()


def test_generate_push_title_local_fallback_when_budget_is_zero(monkeypatch):
    from push_title_agent import generate_push_title

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setenv("PAID_EXTERNAL_APIS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR", "0")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY", "0")

    result = generate_push_title(
        article_title="Breaking Test: Wichtige Entscheidung im Bundestag",
        category="politik",
    )

    assert result["gewinner"]["titel"]
    assert result["meta"]["modus"] == "local-fallback"
    assert result["meta"]["modell"] == "local-fallback"


def test_generate_push_title_starts_individual_llm_call_when_enabled(monkeypatch):
    import push_title_agent
    from push_title_agent import generate_push_title

    calls = []

    def fake_llm_call(system, user, temperature=0.7, max_tokens=800):
        calls.append({"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens})
        return """
        {
          "analyse": {"kern": "Messi stellt WM-Rekord ein", "hook": "Kloses fruehe Ahnung", "emotion": "sportlich-ueberraschend"},
          "kandidaten": [
            {"titel": "Klose ahnte Messis WM-Rekord schon frueh", "ansatz": "B-zugespitzt"},
            {"titel": "Messi zieht gleich - Klose wusste es laengst", "ansatz": "D-neugier"},
            {"titel": "Messi stellt WM-Rekord ein - Klose ahnte es", "ansatz": "A-klare-news-push"},
            {"titel": "Kloses fruehe Ahnung vor Messis Rekord", "ansatz": "C-nutzwert-betroffenheit"}
          ],
          "bewertungen": [
            {"titel": "Klose ahnte Messis WM-Rekord schon frueh", "gesamt": 9.4, "schwaeche": ""}
          ],
          "gewinner": {"titel": "Klose ahnte Messis WM-Rekord schon frueh", "laenge": 39, "gesamt_score": 9.4, "warum_dieser": "konkreter Hook statt Original-Kuerzung"},
          "alternative": {"titel": "Messi zieht gleich - Klose wusste es laengst", "laenge": 45, "warum": "neugierig, aber gedeckt"},
          "warnhinweis": ""
        }
        """

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setenv("PAID_EXTERNAL_APIS_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR", "10")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY", "50")
    monkeypatch.setattr(push_title_agent, "allow_calls", lambda _limits: True)
    monkeypatch.setattr(push_title_agent, "_llm_call", fake_llm_call)

    result = generate_push_title(
        article_title="FCN - WM-Rekord von Messi eingestellt: Klose ahnte es schon früh",
        article_text="Miroslav Klose hatte schon früh mit Lionel Messis Rekordmoment gerechnet.",
        category="sport",
        kicker="Sport",
        headline="FCN - WM-Rekord von Messi eingestellt: Klose ahnte es schon früh",
        force_llm=True,
    )

    assert len(calls) == 1
    assert "FCN - WM-Rekord von Messi" in calls[0]["user"]
    assert "INDIVIDUELL" in calls[0]["system"]
    assert result["title"] == "Klose ahnte Messis WM-Rekord schon frueh"
    assert result["meta"]["modus"] == "llm-individual-headline"
    assert result["meta"]["llm_requested"] is True
    assert result["meta"]["llm_call_started"] is True


def test_title_llm_uses_gpt56_quality_parameters(monkeypatch):
    import push_title_agent

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content='{"gewinner":{"titel":"Test"}}')
                    )
                ]
            )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(push_title_agent, "MODEL", "gpt-5.6")
    monkeypatch.setattr(push_title_agent, "DEFAULT_MAX_TOKENS", 1800)
    monkeypatch.setattr(push_title_agent, "AGENT_TIMEOUT", 30.0)
    monkeypatch.setattr(push_title_agent, "REASONING_EFFORT", "medium")
    monkeypatch.setattr(push_title_agent, "_OPENAI_CLIENT", client)
    monkeypatch.setattr(push_title_agent, "_OPENAI_CLIENT_KEY", "test-key")

    result = push_title_agent._llm_call("System", "Synthetic article", max_tokens=1800)

    assert result == '{"gewinner":{"titel":"Test"}}'
    assert len(calls) == 1
    request = calls[0]
    assert request["model"] == "gpt-5.6"
    assert request["max_completion_tokens"] == 1800
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"reasoning_effort": "medium"}
    assert request["store"] is False
    assert request["timeout"] == 30.0
    assert "max_tokens" not in request
    assert "temperature" not in request


def test_stale_render_title_model_is_upgraded_to_gpt56():
    from push_title_agent import _resolve_title_model

    assert _resolve_title_model("") == "gpt-5.6"
    assert _resolve_title_model("gpt-4o-mini") == "gpt-5.6"
    assert _resolve_title_model("gpt-5.6-terra") == "gpt-5.6-terra"


def test_title_llm_keeps_legacy_parameters_for_non_reasoning_models(monkeypatch):
    import push_title_agent

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content='{"gewinner":{"titel":"Test"}}')
                    )
                ]
            )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(push_title_agent, "MODEL", "gpt-4o-mini")
    monkeypatch.setattr(push_title_agent, "DEFAULT_MAX_TOKENS", 600)
    monkeypatch.setattr(push_title_agent, "_OPENAI_CLIENT", client)
    monkeypatch.setattr(push_title_agent, "_OPENAI_CLIENT_KEY", "test-key")

    push_title_agent._llm_call("System", "Synthetic article", temperature=0.4, max_tokens=600)

    request = calls[0]
    assert request["max_tokens"] == 600
    assert request["temperature"] == 0.4
    assert "max_completion_tokens" not in request
    assert "extra_body" not in request
