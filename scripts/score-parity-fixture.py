#!/usr/bin/env python3
"""Run the frozen score fixture inside a release container.

The fixture is fully synthetic. It validates the vetted seed artefact and the
final one-decimal score without contacting UrlServer, the sitemap, or any other
external system.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

# Direct execution keeps the script directory on sys.path; container execution
# via stdin keeps the image workdir. Adding cwd supports both without packaging
# the synthetic fixture into the runtime image.
sys.path.insert(0, str(Path.cwd()))

from app import score_main
from app.ml import gbrt, lightgbm_model
from app.research import worker as research_worker
from app.routers import feed

EXPECTED_RESULT = {
    "basis": "lightgbm",
    "confidence": 0.695,
    "predictedOR": 0.0535,
    "score": 75.6,
}
EXPECTED_HEURISTIC_SCORE = 77.3
ARTICLE_URL = "https://www.bild.de/politik/synthetic-score-article"


class FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 7, 15, 12, 0, 0)
        return value if tz is None else value.replace(tzinfo=tz)


def _synthetic_sitemap() -> bytes:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
    <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'
            xmlns:news='http://www.google.com/schemas/sitemap-news/0.9'>
      <url>
        <loc>{ARTICLE_URL}</loc>
        <news:news>
          <news:title>Warnung: Synthetischer Testartikel mit neuer Entwicklung</news:title>
          <news:publication_date>2026-07-15T11:30:00</news:publication_date>
        </news:news>
      </url>
    </urlset>""".encode()


def _prepare_frozen_runtime() -> None:
    score_main._preload_local_openmp_for_development()
    if not score_main._frozen_seed_digest_is_valid():
        raise RuntimeError("Frozen score seed failed integrity validation")

    lightgbm_model._ml_state = {
        **lightgbm_model._ml_state,
        "model": None,
        "stats": None,
        "feature_names": [],
    }
    if not lightgbm_model.load_seed_model():
        raise RuntimeError("Frozen score seed could not be loaded")

    lightgbm_model._unified_state = {
        **lightgbm_model._unified_state,
        "model": None,
        "stacking_active": False,
    }
    gbrt._gbrt_model = None
    research_worker._research_state = {
        "push_data": [],
        "global_avg": 5.5,
        "cat_hour_stats": {"politik_12": {"n": 8, "avg": 7.25}},
        "cat_stats": {},
    }
    research_worker._residual_corrector = {
        "global_bias": 0.0,
        "cat_bias": {},
        "hourgroup_bias": {},
        "n_samples": 0,
        "last_update_ts": 0,
        "recent_residuals": [],
    }
    feed._fetch_url = lambda _url: _synthetic_sitemap()
    feed.ARTICLE_PREDICTION_ENRICHMENT_ENABLED = True
    feed.datetime = SimpleNamespace(datetime=FrozenDateTime)
    feed._article_prediction_cache = {}


def _run_fixture() -> dict[str, float | str | None]:
    payload = feed.build_articles_payload(offset=0, limit=120)
    article = next(item for item in payload["articles"] if item["url"] == ARTICLE_URL)
    return {
        "basis": article["predictedORBasis"],
        "confidence": article["predictedORConfidence"],
        "predictedOR": article["predictedOR"],
        "score": article["score"],
    }


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    _prepare_frozen_runtime()
    result = _run_fixture()
    if result != EXPECTED_RESULT:
        raise RuntimeError(
            f"Score parity mismatch: expected {EXPECTED_RESULT!r}, got {result!r}"
        )

    lightgbm_model._ml_state = {**lightgbm_model._ml_state, "model": None}
    feed._article_prediction_cache = {}
    heuristic_score = _run_fixture()["score"]
    if heuristic_score != EXPECTED_HEURISTIC_SCORE:
        raise RuntimeError(
            "Heuristic score parity mismatch: "
            f"expected {EXPECTED_HEURISTIC_SCORE!r}, got {heuristic_score!r}"
        )

    seed_path = Path(lightgbm_model.__file__).resolve().parent / "seed_model.pkl"
    report = {
        "architecture": platform.machine(),
        "commit": os.environ.get("SCORE_PARITY_COMMIT", "local"),
        "heuristicScore": heuristic_score,
        "packages": {
            name: version(name)
            for name in ("joblib", "lightgbm", "numpy", "scikit-learn")
        },
        "python": platform.python_version(),
        "result": result,
        "seedSha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    json.dump(report, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
