"""Guard the browser score calibration consumed by the internal score API."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "frontend_path",
    [Path("push-balancer.html"), Path("app/legacy_push_balancer.html")],
)
def test_fresh_strong_a_list_people_shape_gets_ml_and_fallback_calibration(
    frontend_path: Path,
):
    source = (ROOT / frontend_path).read_text(encoding="utf-8")

    assert "function isStrongAListPeopleDevelopment(article)" in source
    assert "articleAge <= 6" in source
    assert "isStrongAListPeopleDevelopment(article)" in source
    assert "baseScore = Math.max(baseScore, 90)" in source
    assert "if (_strongAListPeopleDevelopment) breakdown.relevanz = 30" in source
    assert "if (_strongAListPeopleDevelopment) breakdown.neugier = 25" in source
    assert "if (_strongAListPeopleDevelopment) individualBoost = Math.max(individualBoost, 15)" in source
