import pytest

from app.push_titles import build_push_title_suggestions


EDITORIAL_EXAMPLES = [
    (
        "Politik",
        "Merz plant neue Renten-Reform für Millionen Deutsche",
        "politik",
        "Millionen Deutsche: Merz plant Renten-Reform",
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


@pytest.mark.parametrize(
    "headline,expected",
    [
        (
            "Studie zeigt: Diese Zimmerpflanzen verbessern das Raumklima",
            "Diese Zimmerpflanzen verbessern das Raumklima",
        ),
        (
            "Experten erklären, warum viele Menschen schlecht schlafen",
            "Darum schlafen viele Menschen schlecht",
        ),
    ],
)
def test_low_push_value_topics_are_not_overhyped(headline, expected):
    result = build_push_title_suggestions(headline, category="news")

    assert result["title"] == expected
    assert result["warnhinweis"]
    assert result["gewinner"]["gesamt_score"] < 8.0
    assert "Warnhinweis" in result["reasoning"]
    assert all(title != headline for title in result["alternativeTitles"])
    assert not any(title.startswith("News:") for title in result["alternativeTitles"])
    assert not any("Was jetzt wichtig ist" in title for title in result["alternativeTitles"])


def test_visible_alternative_reason_uses_strength_not_generic_weakness():
    result = build_push_title_suggestions(
        "Neue Regeln beim Bürgergeld: Für diese Familien ändert sich jetzt alles",
        category="wirtschaft",
    )

    assert "noch zu wenig Akteur-Handlung" not in result["alternative"]["warum"]
    assert "kompakten Push" in result["alternative"]["warum"]


def test_sport_record_story_rewrites_headline_into_push_hook():
    headline = "FCN - WM-Rekord von Messi eingestellt: Klose ahnte es schon früh"

    result = build_push_title_suggestions(headline, category="sport")

    assert result["title"] == "Klose ahnte Messis WM-Rekord schon früh"
    assert result["title"] != headline
    assert 35 <= len(result["title"]) <= 65
    assert result["gewinner"]["gesamt_score"] >= 8.0
    assert "FCN:" not in result["title"]
    assert all(" im Fokus:" not in title for title in result["alternativeTitles"])
    assert all(title != headline for title in result["alternativeTitles"])


def test_llm_prompt_enforces_push_first_opening_rate_rules():
    from push_title_agent import EDITORIAL_ONE_BRAIN_SYS

    assert "Opening Rate" in EDITORIAL_ONE_BRAIN_SYS
    assert "35 bis 65 Zeichen" in EDITORIAL_ONE_BRAIN_SYS
    assert "Original-Headline nicht kopieren" in EDITORIAL_ONE_BRAIN_SYS
    assert "keine Clickbait-Luege" in EDITORIAL_ONE_BRAIN_SYS
    assert "A-klare-news-push" in EDITORIAL_ONE_BRAIN_SYS
    assert "Klose ahnte Messis WM-Rekord" in EDITORIAL_ONE_BRAIN_SYS


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


def test_no_generic_filler_titles_for_political_headline():
    result = build_push_title_suggestions(
        "Briten-Premier bereitet wohl Rücktritt vor", category="politik"
    )

    all_titles = [
        result["title"],
        *result.get("alternativeTitles", []),
        (result.get("alternative") or {}).get("titel", ""),
    ]
    for title in all_titles:
        lowered = (title or "").lower()
        assert "darum geht es jetzt" not in lowered
        assert "was jetzt wichtig ist" not in lowered
