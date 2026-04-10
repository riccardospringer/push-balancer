"""test_magnitude_heuristic.py — Tests für keyword_magnitude_heuristic().

Testet das Keyword-basierte Scoring als LLM-Fallback:
  - Basis-Score, Clipping, Eilmeldung, Terror/Krieg-Keywords
  - Sport-spezifische Boni (verletzt, entlassen, transfer, überblick-Malus)
  - Kategorie-Adjustierungen (unterhaltung, politik/news)
"""
from app.scoring.magnitude import keyword_magnitude_heuristic as heuristic


# ── Basis ────────────────────────────────────────────────────────────────────

class TestBasisScore:
    def test_neutral_title_returns_near_base(self):
        """Normaler Titel ohne Keywords → Score nahe 3.0."""
        score = heuristic("Neues Café in Hamburg eröffnet", "news")
        # Kategoriebonus news: +0.5, Basis 3.0 → 3.5
        assert 3.0 <= score <= 5.0

    def test_score_never_below_one(self):
        """Score darf nie unter 1.0 fallen."""
        score = heuristic("Lifestyle-Rezept: Einfaches Horoskop für den Urlaub", "unterhaltung")
        assert score >= 1.0

    def test_score_never_above_ten(self):
        """Score darf nie über 10.0 steigen."""
        score = heuristic(
            "Eilmeldung: Terror-Anschlag Explosion Massaker Tsunami Erdbeben",
            "news",
            is_eilmeldung=1,
        )
        assert score <= 10.0

    def test_return_type_is_float(self):
        score = heuristic("Ein ganz normaler Artikel", "news")
        assert isinstance(score, float)


# ── Eilmeldung ───────────────────────────────────────────────────────────────

class TestEilmeldung:
    def test_is_eilmeldung_flag_raises_score(self):
        """is_eilmeldung=1 soll Score deutlich über Basis heben."""
        score_normal = heuristic("Bericht aus dem Bundestag", "politik")
        score_eil = heuristic("Bericht aus dem Bundestag", "politik", is_eilmeldung=1)
        assert score_eil > score_normal + 3.0

    def test_eilmeldung_in_title_raises_score(self):
        """'Eilmeldung' im Titel wird erkannt."""
        score = heuristic("Eilmeldung: Brand in Fabrik", "news")
        assert score >= 7.0

    def test_breaking_in_title_raises_score(self):
        """'Breaking' im Titel wird erkannt."""
        score = heuristic("Breaking: Explosion in der City", "news")
        assert score >= 7.0

    def test_eilmeldung_capped_at_ten(self):
        score = heuristic("Eilmeldung: Terror", "news", is_eilmeldung=1)
        assert score <= 10.0


# ── Terror / Krieg / Katastrophe ─────────────────────────────────────────────

class TestHighMagnitudeKeywords:
    def test_terror_keyword(self):
        score = heuristic("Terror-Anschlag erschüttert Europa", "news")
        assert score >= 7.0

    def test_krieg_keyword(self):
        score = heuristic("Krieg: Neue Offensive gestartet", "news")
        assert score >= 7.0

    def test_explosion_keyword(self):
        score = heuristic("Explosion im Hafen — viele Opfer", "news")
        assert score >= 7.0

    def test_multiple_high_mag_keywords_not_linear(self):
        """Mehrere High-Mag-Keywords → nur +0.8 extra, kein lineares Aufaddieren."""
        score_one = heuristic("Terror in der Stadt", "news")
        score_two = heuristic("Terror und Massaker in der Stadt", "news")
        # Beide hoch, aber Abstand begrenzt (max +0.8 für den zweiten Hit)
        assert score_two <= score_one + 1.5

    def test_low_keywords_reduce_score(self):
        """Lifestyle-Keywords (trend, beauty, horoskop) reduzieren den Score."""
        score_plain = heuristic("Neues Produkt vorgestellt", "news")
        score_low = heuristic("Neuer Beauty-Trend: Das Horoskop der Stars", "news")
        assert score_low < score_plain


# ── Sport-spezifische Keywords ────────────────────────────────────────────────

class TestSportKeywords:
    def test_verletzt_raises_score(self):
        """'verletzt' → +2.0 im Sport-Kontext."""
        score_base = heuristic("Spieler trifft für sein Team", "sport")
        score_verletzt = heuristic("Star-Stürmer verletzt — wochenlanger Ausfall", "sport")
        assert score_verletzt > score_base + 1.5

    def test_entlassen_raises_score(self):
        """'entlassen' → +2.0 im Sport-Kontext."""
        score = heuristic("Trainer entlassen nach Niederlagen-Serie", "sport")
        assert score >= 5.0

    def test_transfer_small_boost(self):
        """'transfer' → kleiner Bonus (+0.8)."""
        score_base = heuristic("Team gewinnt Sonntagsspiel", "sport")
        score_transfer = heuristic("Mega-Transfer: Spieler wechselt den Verein", "sport")
        assert score_transfer > score_base

    def test_ueberblick_malus(self):
        """'überblick' → -1.5 im Sport-Kontext."""
        score_plain = heuristic("Spannendes Spiel am Wochenende", "sport")
        score_ueberblick = heuristic("Alle Ergebnisse im Überblick", "sport")
        assert score_ueberblick < score_plain

    def test_sport_category_not_penalized_baseline(self):
        """Neutrale Sport-Meldung wird nicht durch Kategorie bestraft."""
        score = heuristic("Bundesliga-Spieltag: Ergebnisse", "sport")
        # Kein Kategorie-Malus für Sport (nur unterhaltung hat -1.0)
        assert score >= 1.0


# ── Kategorie-Adjustierungen ─────────────────────────────────────────────────

class TestCategoryAdjustments:
    def test_unterhaltung_malus(self):
        """Kategorie 'unterhaltung' → -1.0 auf Score."""
        score_news = heuristic("Schauspieler gibt Interview", "news")
        score_unt = heuristic("Schauspieler gibt Interview", "unterhaltung")
        assert score_unt < score_news

    def test_politik_bonus(self):
        """Kategorie 'politik' → +0.5 auf Score."""
        score_default = heuristic("Pressekonferenz findet statt", "geld")
        score_politik = heuristic("Pressekonferenz findet statt", "politik")
        assert score_politik > score_default

    def test_news_bonus(self):
        """Kategorie 'news' → +0.5 auf Score."""
        score_default = heuristic("Meldung des Tages", "digital")
        score_news = heuristic("Meldung des Tages", "news")
        assert score_news > score_default

    def test_unknown_category_no_crash(self):
        """Unbekannte Kategorie soll keinen Fehler werfen."""
        score = heuristic("Ein beliebiger Titel", "unbekannte_kategorie")
        assert 1.0 <= score <= 10.0


# ── Grenzfälle ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_title(self):
        """Leerer Titel → kein Fehler, Score im gültigen Bereich."""
        score = heuristic("", "news")
        assert 1.0 <= score <= 10.0

    def test_empty_category(self):
        """Leere Kategorie → kein Fehler."""
        score = heuristic("Ein normaler Titel", "")
        assert 1.0 <= score <= 10.0

    def test_very_long_title(self):
        """Sehr langer Titel → kein Fehler."""
        long_title = "Terror " * 50
        score = heuristic(long_title, "news")
        assert 1.0 <= score <= 10.0
