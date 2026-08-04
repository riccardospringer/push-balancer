"""Sport-Zielkorridor (~1/3) im Scoring: praediktiver Ausgleich.

Der Sportanteil soll ueber den Tag ~1/3 der Pushes betragen. Die Auswahl-
Praeferenz muss deshalb — auch wenn der Tagesanteil formal im Zielbereich liegt —
das Ressort bevorzugen, das den Sportanteil am naechsten an 1/3 haelt. So werden
Sport-Cluster (mehrere Sport-Pushes hintereinander) vermieden, ohne die
Top-1-nach-Score-Garantie zu brechen (die Praeferenz wirkt nur innerhalb der
engen sport_preference_band).
"""
from __future__ import annotations

import pytest

import app.notifications.teams as teams


def _pref(sent: int, sport: int) -> str | None:
    cfg = teams.TeamsAlertConfig()
    review = teams._sport_balance_review(
        {"pushesToday": sent, "sportPushesToday": sport}, cfg
    )
    return review["selectionPreference"]


def test_no_pushes_yet_leaves_corridor_open():
    assert _pref(0, 0) is None


def test_just_pushed_sport_at_target_prefers_news():
    # 1 von 3 Pushes Sport = 33 %: ein weiterer Sport-Push (2/4 = 50 %) entfernt
    # sich von 1/3, ein News-Push (1/4 = 25 %) kommt naeher -> News bevorzugt.
    assert _pref(3, 1) == "news"


def test_sport_below_target_prefers_sport():
    # 25 % liegt unter dem Korridor -> Sport nachziehen.
    assert _pref(4, 1) == "sport"


def test_sport_over_corridor_prefers_news():
    assert _pref(3, 2) == "news"   # 67 % -> ueber
    assert _pref(2, 1) == "news"   # 50 % -> ueber


def test_predictive_balance_targets_one_third_within_corridor():
    # Formal "im" Zielbereich, aber ein weiterer Sport-Push wuerde uebersteuern.
    assert _pref(6, 2) == "news"   # 33 % -> News haelt naeher an 1/3
    assert _pref(9, 3) == "news"   # 33 %
    # Leicht unter 1/3 -> Sport bringt naeher an 1/3.
    assert _pref(10, 3) == "sport"  # 30 %


def test_preference_band_is_narrow_enough_to_protect_top1():
    # Die Praeferenz darf nur innerhalb einer engen Score-Bandbreite wirken,
    # damit ein klar staerkerer Push nie verdraengt wird.
    cfg = teams.TeamsAlertConfig()
    assert 0.0 < cfg.sport_preference_band <= 4.0


@pytest.mark.parametrize(
    "section",
    ["sport", "sports", "fussball", "fußball", "bundesliga"],
)
def test_production_sport_aliases_are_classified_as_sport(section):
    assert teams._section_key(section) == "sport"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.bild.de/sport/fussball/test", True),
        ("https://www.bild.de/fussball/test", True),
        ("https://www.bild.de/bundesliga/test", True),
        ("https://www.bild.de/politik/sportfoerderung-test", False),
        ("https://www.bild.de/news/sportstar-test", False),
    ],
)
def test_sport_url_classification_uses_path_segments(url, expected):
    assert teams._is_sport_item({"cat": "news", "link": url}) is expected
