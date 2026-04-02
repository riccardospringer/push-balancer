"""test_sport_scoring.py — Tests für Sport-Kategorisierung.

Testet die is_sport-Logik, die in _score_push_llm und _keyword_magnitude_heuristic
zur Anwendung kommt, sowie URL-basierte Kategorieerkennung.

Da die Sport-Kategorie-Logik direkt in der Inline-Bedingung lebt
(nicht als eigene Funktion), testen wir sie über:
1. _keyword_magnitude_heuristic: sport-spezifische Pfade werden nur für cat=="sport" aktiv
2. is_sport-Formel als extrahierte Hilfsfunktion (lokal nachgebaut für isolierten Test)
"""
import sys

import pytest

# Modul bereits via conftest geladen
_server = sys.modules.get("pbserver")


# ── is_sport Logik (aus _score_push_llm extrahiert) ──────────────────────────

SPORT_CATEGORIES = frozenset(
    ["sport", "fussball", "bundesliga", "formel1", "formel-1", "tennis", "boxen", "motorsport"]
)


def _is_sport_category(category: str) -> bool:
    """Repliziert die is_sport-Bedingung aus push-balancer-server.py."""
    return (category or "").lower() in SPORT_CATEGORIES


def _is_sport_url(url: str) -> bool:
    """Einfache URL-basierte Erkennung: /sport/ im Pfad."""
    return "/sport/" in (url or "").lower()


# ── Kategorie-Tests ──────────────────────────────────────────────────────────

class TestSportCategories:
    @pytest.mark.parametrize("cat", [
        "sport", "Sport", "SPORT",
        "fussball", "Fussball",
        "bundesliga", "Bundesliga",
        "tennis", "Tennis",
        "boxen", "Boxen",
        "motorsport", "Motorsport",
        "formel1", "formel-1",
    ])
    def test_sport_categories_are_recognized(self, cat):
        """Alle Sport-Kategorien müssen als Sport erkannt werden."""
        assert _is_sport_category(cat) is True

    @pytest.mark.parametrize("cat", [
        "politik", "news", "unterhaltung", "geld",
        "digital", "regional", "leben", "ratgeber",
        "", "unknown",
    ])
    def test_non_sport_categories_are_not_sport(self, cat):
        """Nicht-Sport-Kategorien dürfen nicht als Sport erkannt werden."""
        assert _is_sport_category(cat) is False

    def test_category_check_is_case_insensitive(self):
        assert _is_sport_category("SPORT") is True
        assert _is_sport_category("Fussball") is True
        assert _is_sport_category("TENNIS") is True

    def test_none_category_does_not_crash(self):
        assert _is_sport_category(None) is False


# ── URL-Tests ────────────────────────────────────────────────────────────────

class TestSportUrl:
    def test_sport_url_recognized(self):
        assert _is_sport_url("https://www.bild.de/sport/fussball/artikel") is True

    def test_fussball_subpath_in_sport(self):
        assert _is_sport_url("https://www.bild.de/sport/bundesliga/fc-bayern") is True

    def test_politik_url_not_sport(self):
        assert _is_sport_url("https://www.bild.de/politik/bundesregierung/artikel") is False

    def test_no_sport_in_url(self):
        assert _is_sport_url("https://www.bild.de/news/deutschland/artikel") is False

    def test_empty_url(self):
        assert _is_sport_url("") is False

    def test_none_url(self):
        assert _is_sport_url(None) is False


# ── Heuristic-Pfad: Sport vs. Nicht-Sport ────────────────────────────────────

class TestMagnitudeHeuristicSportPath:
    """Über _keyword_magnitude_heuristic prüfen, ob Sport-Pfad korrekt aktiviert wird."""

    def _heuristic(self, title, cat, is_eilmeldung=0):
        mod = sys.modules.get("pbserver")
        if mod is None:
            pytest.skip("Server-Modul nicht geladen")
        return mod._keyword_magnitude_heuristic(title, cat, is_eilmeldung)

    def test_verletzung_only_boosts_in_sport(self):
        """'verletzt' boost gilt für 'sport', nicht für 'politik'."""
        score_sport = self._heuristic("Star verletzt — langer Ausfall", "sport")
        score_politik = self._heuristic("Star verletzt — langer Ausfall", "politik")
        # Sport-Pfad addiert +2.0, Politik-Pfad nicht
        assert score_sport > score_politik

    def test_entlassen_boost_in_sport(self):
        """'entlassen' hat im Sport größeren Effekt als in neutraler Kategorie."""
        score_sport = self._heuristic("Trainer wird entlassen nach schlechten Ergebnissen", "sport")
        score_news = self._heuristic("Trainer wird entlassen nach schlechten Ergebnissen", "news")
        assert score_sport >= score_news

    def test_ueberblick_malus_only_in_sport(self):
        """'überblick' Malus gilt nur für Sport-Kategorie."""
        score_sport = self._heuristic("Alle Spiele im Überblick", "sport")
        score_news = self._heuristic("Alle Spiele im Überblick", "news")
        # Sport hat -1.5 für überblick, News nicht
        assert score_sport < score_news

    def test_transfer_boost_in_sport(self):
        """'transfer' boost (+0.8) greift nur im Sport."""
        score_sport = self._heuristic("Großer Transfer: Spieler wechselt nach Madrid", "sport")
        score_news = self._heuristic("Großer Transfer: Spieler wechselt nach Madrid", "news")
        assert score_sport >= score_news

    def test_sport_high_drama_gestorben(self):
        """'gestorben' im Sport → hoher dramatischer Boost."""
        score = self._heuristic("Fußball-Legende im Alter von 60 Jahren gestorben", "sport")
        assert score >= 7.0

    def test_sport_score_in_valid_range(self):
        score = self._heuristic("Bundesliga: Spieltag-Analyse", "sport")
        assert 1.0 <= score <= 10.0
