from app.push_titles import build_push_title_suggestions


def test_push_title_generator_prefers_editorial_depth_over_surface_length():
    result = build_push_title_suggestions(
        "WM 2026: Erdbeben nach Toren von Erling Haaland",
        category="sport",
    )

    assert result["title"] == "Haaland schießt sich Richtung WM 2026"
    assert result["title"] != "WM 2026: Erdbeben nach Toren von Erling Haaland"
    assert result["meta"]["analyse"]["akteure"] == ["Erling Haaland"]
    assert "wm" in result["meta"]["analyse"]["fallhoehe"]

    original_rating = next(
        item for item in result["bewertungen"]
        if item["titel"] == "WM 2026: Erdbeben nach Toren von Erling Haaland"
    )
    assert original_rating["gesamt"] < 8.0
    assert "metaphorischer Einstieg" in original_rating["schwaeche"]


def test_push_title_generator_penalizes_empty_sport_prefix():
    result = build_push_title_suggestions(
        "WM 2026: Erdbeben nach Toren von Erling Haaland",
        category="sport",
    )

    sport_prefix_candidates = [
        item
        for item in result["bewertungen"]
        if item["titel"].startswith("Sport:")
    ]
    assert not sport_prefix_candidates or sport_prefix_candidates[0]["gesamt"] <= 7.2


def test_agent_local_fallback_uses_deep_title_engine(monkeypatch):
    from push_title_agent import generate_push_title

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAID_EXTERNAL_APIS_ENABLED", "false")
    monkeypatch.setenv("OPENAI_TITLE_GENERATION_ENABLED", "false")

    result = generate_push_title(
        article_title="WM 2026: Erdbeben nach Toren von Erling Haaland",
        category="sport",
    )

    assert result["gewinner"]["titel"] == "Haaland schießt sich Richtung WM 2026"
    assert result["meta"]["analyse"]["akteure"] == ["Erling Haaland"]


def test_g7_soft_politics_title_is_not_copied():
    headline = "Die große Bühne der witzigen Weltpolitik: Die cringy Momente beim G7-Gipfel"

    result = build_push_title_suggestions(headline, category="politik")

    assert result["title"] == "G7-Gipfel: Die cringy Momente der Weltpolitik"
    assert result["title"] != headline
    assert result["meta"]["analyse"]["akteure"] == []
    original_rating = next(item for item in result["bewertungen"] if item["titel"] == headline)
    assert original_rating["gesamt"] < result["gewinner"]["gesamt_score"]
    assert "kopiert die Original-Headline" in original_rating["schwaeche"]
