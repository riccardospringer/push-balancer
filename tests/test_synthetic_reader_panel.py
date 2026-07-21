"""Tests for the non-personal, synthetic reader-mode shadow panel."""

from app.research.synthetic_reader_panel import (
    ATTENTION_MODES,
    FOCUS_MODES,
    MOTIVATION_MODES,
    evaluate_synthetic_reader_modes,
    render_synthetic_reader_panel_markdown,
    run_synthetic_reader_panel_study,
)


def _candidate(**overrides):
    candidate = {
        "studyId": "national-warning",
        "title": "Bundesweite Warnung: Keime im Trinkwasser betreffen mehrere Staedte",
        "url": "https://example.invalid/news/inland/trinkwasser",
        "category": "wetter",
        "score": 82.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": False,
        "isBreaking": False,
        "isEilmeldung": False,
    }
    candidate.update(overrides)
    return candidate


def test_panel_has_144_situational_cells_without_demographic_personas():
    result = evaluate_synthetic_reader_modes(_candidate())

    assert result["scenarioCount"] == 144
    assert result["scenarioCount"] == (
        len(FOCUS_MODES) * len(ATTENTION_MODES) * len(MOTIVATION_MODES)
    )
    assert result["representsObservedUsers"] is False
    assert result["canEstimateOpeningRate"] is False
    assert result["notOpeningRate"] is True
    assert result["shadowOnly"] is True
    assert result["productionUseAllowed"] is False
    assert set(ATTENTION_MODES) == {
        "essential_only",
        "quick_scan",
        "curiosity_open",
        "context_seeking",
    }


def test_pure_us_domestic_people_story_is_rejected_by_every_synthetic_cell():
    result = evaluate_synthetic_reader_modes(
        _candidate(
            studyId="us-people",
            title="US-Mutter soll ihre Zwillinge getoetet haben",
            url="https://example.invalid/news/ausland/us-familienfall",
            category="news",
            score=96.0,
            recommendedHour=12,
            baselineRecommended=True,
        )
    )

    assert result["germanyRelevance"] == "usa_domestic"
    assert result["wouldOpenCells"] == 0
    assert result["wouldConsiderCells"] == 0
    assert result["wouldSkipCells"] == 144
    assert result["baselineComparison"] == "challenge_legacy_recommendation"


def test_national_warning_beats_higher_scoring_foreign_routine_story():
    national = evaluate_synthetic_reader_modes(_candidate())
    foreign = evaluate_synthetic_reader_modes(
        _candidate(
            studyId="world-routine",
            title="US-Praesident warnt vor weiterer Eskalation im Iran-Krieg",
            url="https://example.invalid/politik/ausland-und-internationales/konflikt",
            category="politik",
            score=88.0,
            recommendedHour=7,
        )
    )

    assert national["pushScore"] < foreign["pushScore"]
    assert national["syntheticInterestIndex"] > foreign["syntheticInterestIndex"]
    assert national["wouldOpenCells"] > foreign["wouldOpenCells"]


def test_isolated_tragedy_has_materially_weaker_morning_fit_than_evening_fit():
    tragedy = _candidate(
        studyId="synthetic-tragedy",
        title="Einzelner Familienfall: Kleinkind stirbt nach unterlassener Hilfe",
        url="https://example.invalid/regional/musterstadt/familienfall",
        category="regional",
        score=82.0,
    )

    morning = evaluate_synthetic_reader_modes(tragedy, hour=7)
    evening = evaluate_synthetic_reader_modes(tragedy, hour=20)

    assert morning["syntheticInterestIndex"] + 15.0 < evening["syntheticInterestIndex"]
    assert "isolierte Tragoedie ohne Morgen-Nutzwert" in morning["barriers"]


def test_verified_world_breaking_remains_a_selective_exception():
    result = evaluate_synthetic_reader_modes(
        _candidate(
            studyId="world-breaking",
            title="Eilmeldung: Waffenruhe im Iran-Krieg tritt sofort in Kraft",
            url="https://example.invalid/politik/ausland-und-internationales/waffenruhe",
            category="politik",
            score=90.0,
            recommendedHour=13,
            isBreaking=True,
            isEilmeldung=True,
        )
    )

    assert result["germanyRelevance"] == "international_breaking"
    assert result["wouldOpenCells"] > 0
    assert result["editorialBand"] in {"broad_support", "selective_support"}


def test_material_german_sport_event_reaches_the_live_sport_modes():
    event = _candidate(
        studyId="sport-live",
        title="Eilmeldung: Nationalmannschaft erreicht nach Elfmeter das Finale",
        url="https://example.invalid/sport/fussball/finale",
        category="sport",
        score=84.0,
        isBreaking=True,
        isEilmeldung=True,
        weekday=5,
    )

    evening = evaluate_synthetic_reader_modes(event, hour=21, weekday=5)

    assert "materielles Sportereignis" in evening["drivers"]
    assert "sport_live" in evening["signalTags"]
    assert any(
        item["focus"] == "sport_live" and item["wouldOpenCells"] > 0
        for item in evening["focusBreakdown"]
    )


def test_confirmed_named_german_people_milestone_gets_selective_support_only():
    public_figure = evaluate_synthetic_reader_modes(
        _candidate(
            studyId="people-parenthood",
            title="CDU-Politiker Max Beispiel und sein Partner sind Papas geworden",
            url="https://example.invalid/unterhaltung/stars-und-leute/beispiel",
            category="unterhaltung",
            score=82.0,
            recommendedHour=18,
        )
    )
    anonymous = evaluate_synthetic_reader_modes(
        _candidate(
            studyId="anonymous-parenthood",
            title="Kommunalpolitiker aus Musterstadt ist Papa geworden",
            url="https://example.invalid/unterhaltung/stars-und-leute/lokal",
            category="unterhaltung",
            score=82.0,
            recommendedHour=18,
        )
    )

    assert public_figure["germanyRelevance"] == "germany_people"
    assert public_figure["editorialBand"] == "selective_support"
    assert "people_milestone" in public_figure["signalTags"]
    assert public_figure["wouldOpenCells"] > anonymous["wouldOpenCells"] + 30
    assert public_figure["syntheticInterestIndex"] > anonymous["syntheticInterestIndex"] + 30
    assert public_figure["notOpeningRate"] is True
    assert anonymous["germanyRelevance"] == "neutral"


def test_study_is_deterministic_and_never_grants_production_use():
    cases = [
        _candidate(studyId="national"),
        _candidate(
            studyId="vague",
            title="Dieses neue Handy-Detail muessen Sie kennen",
            url="https://example.invalid/digital/detail",
            category="digital",
            score=85.0,
            recommendedHour=10,
            baselineRecommended=True,
        ),
    ]

    first = run_synthetic_reader_panel_study(cases)
    second = run_synthetic_reader_panel_study(cases)

    assert first == second
    assert first["totalScenarioDecisions"] == 288
    assert first["productionUseAllowed"] is False
    assert first["representsObservedUsers"] is False
    assert first["canEstimateOpeningRate"] is False
    assert any("keine reale OR" in lesson for lesson in first["lessons"])


def test_markdown_report_never_describes_the_index_as_opening_rate():
    study = run_synthetic_reader_panel_study([_candidate()])
    report = render_synthetic_reader_panel_markdown(study)

    assert "Keine echten BILD-Nutzer" in report
    assert "keine Opening Rate" in report
    assert "keine OR" in report
    assert "Synthetischer Fall" in report
    assert "Opening-Rate-Prognose" not in report
