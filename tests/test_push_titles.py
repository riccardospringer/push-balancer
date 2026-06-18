import pytest

from app.push_titles import build_push_title_suggestions


EDITORIAL_EXAMPLES = [
    (
        "Politik",
        "Merz plant neue Renten-Reform für Millionen Deutsche",
        "politik",
        "Renten-Reform: Was Merz jetzt plant",
    ),
    (
        "Sport",
        "WM 2026: Erdbeben nach Toren von Erling Haaland",
        "sport",
        "Haaland schießt sich Richtung WM 2026",
    ),
    (
        "Crime",
        "Mann nach Messerattacke am Bahnhof festgenommen",
        "news",
        "Messerattacke am Bahnhof: Mann festgenommen",
    ),
    (
        "Verbraucher",
        "Neue Regeln beim Bürgergeld: Für diese Familien ändert sich jetzt alles",
        "wirtschaft",
        "Bürgergeld: Wen die neuen Regeln treffen",
    ),
    (
        "Promi/Unterhaltung",
        "Helene Fischer spricht erstmals über ihr Familienglück",
        "unterhaltung",
        "Jetzt spricht Helene Fischer über ihr Familienglück",
    ),
    (
        "Wetter/Unwetter",
        "Unwetter-Warnung: Heftige Gewitter ziehen auf Deutschland zu",
        "news",
        "Warnung: Heftige Gewitter ziehen auf Deutschland zu",
    ),
    (
        "Breaking News",
        "Eilmeldung: Bundesregierung beschließt Milliarden-Paket für die Ukraine",
        "politik",
        "EIL: Milliarden-Paket für die Ukraine beschlossen",
    ),
    (
        "Kurios/emotional",
        "Hund läuft 20 Kilometer zurück zu seinem alten Besitzer",
        "news",
        "20 Kilometer: Hund läuft zurück zum alten Besitzer",
    ),
]


@pytest.mark.parametrize("article_type,headline,category,expected", EDITORIAL_EXAMPLES)
def test_editorial_examples_generate_push_first_titles(article_type, headline, category, expected):
    result = build_push_title_suggestions(headline, category=category)

    assert article_type
    assert result["title"] == expected
    assert result["title"] != headline
    assert 35 <= len(result["title"]) <= 65
    assert len(result["alternativeTitles"]) == 3
    assert all(title != headline for title in result["alternativeTitles"])
    assert not any(title.lower().startswith(f"{category}:") for title in result["alternativeTitles"])


def test_editorial_examples_are_visible_as_original_to_push_to_score():
    rows = []
    for article_type, headline, category, _expected in EDITORIAL_EXAMPLES:
        result = build_push_title_suggestions(headline, category=category)
        rows.append((article_type, headline, result["title"], result["gewinner"]["gesamt_score"]))

    assert len(rows) == 8
    assert all(score >= 8.0 for _article_type, _headline, _title, score in rows)


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
