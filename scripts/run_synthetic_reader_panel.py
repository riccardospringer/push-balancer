#!/usr/bin/env python3
"""Run the offline synthetic reader-mode study with dummy article cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.research.synthetic_reader_panel import (
    render_synthetic_reader_panel_markdown,
    run_synthetic_reader_panel_study,
)


SYNTHETIC_CASES = [
    {
        "studyId": "national-consumer",
        "title": "Neue Speicherfrist: Was sich fuer Millionen Kunden in Deutschland aendert",
        "url": "https://example.invalid/wirtschaft/speicherfrist",
        "category": "verbraucher",
        "score": 82.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": False,
    },
    {
        "studyId": "national-warning",
        "title": "Bundesweite Warnung: Keime im Trinkwasser betreffen mehrere Staedte",
        "url": "https://example.invalid/news/inland/trinkwasser",
        "category": "wetter",
        "score": 81.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": False,
    },
    {
        "studyId": "us-people",
        "title": "US-Mutter soll ihre Zwillinge getoetet haben",
        "url": "https://example.invalid/news/ausland/us-familienfall",
        "category": "news",
        "score": 94.0,
        "recommendedHour": 12,
        "weekday": 2,
        "baselineRecommended": True,
    },
    {
        "studyId": "world-routine",
        "title": "US-Praesident warnt vor weiterer Eskalation im Iran-Krieg",
        "url": "https://example.invalid/politik/ausland-und-internationales/konflikt",
        "category": "politik",
        "score": 86.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": True,
    },
    {
        "studyId": "morning-tragedy",
        "title": "Einzelner Familienfall: Kleinkind stirbt nach unterlassener Hilfe",
        "url": "https://example.invalid/regional/musterstadt/familienfall",
        "category": "regional",
        "score": 82.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": True,
    },
    {
        "studyId": "regional-safety",
        "title": "Polizei raeumt Bahnhof nach Gefahrstoff-Fund",
        "url": "https://example.invalid/regional/musterstadt/bahnhof",
        "category": "regional",
        "score": 80.0,
        "recommendedHour": 8,
        "weekday": 2,
        "baselineRecommended": False,
    },
    {
        "studyId": "german-live-sport",
        "title": "Eilmeldung: Nationalmannschaft erreicht nach Elfmeter das Finale",
        "url": "https://example.invalid/sport/fussball/finale",
        "category": "sport",
        "score": 84.0,
        "recommendedHour": 21,
        "weekday": 5,
        "isBreaking": True,
        "isEilmeldung": True,
        "baselineRecommended": False,
    },
    {
        "studyId": "sport-rumour",
        "title": "Bundesliga-Klub prueft moeglichen Transfer eines Ersatzspielers",
        "url": "https://example.invalid/sport/fussball/transfer-pruefung",
        "category": "sport",
        "score": 82.0,
        "recommendedHour": 7,
        "weekday": 2,
        "baselineRecommended": True,
    },
    {
        "studyId": "evening-entertainment",
        "title": "TV-Star bestaetigt Trennung und spricht ueber Millionenstreit",
        "url": "https://example.invalid/unterhaltung/trennung",
        "category": "unterhaltung",
        "score": 80.0,
        "recommendedHour": 20,
        "weekday": 4,
        "baselineRecommended": False,
    },
    {
        "studyId": "german-people-parenthood",
        "title": "CDU-Politiker Max Beispiel und sein Partner sind Papas geworden",
        "url": "https://example.invalid/unterhaltung/stars-und-leute/beispiel",
        "category": "unterhaltung",
        "score": 87.0,
        "recommendedHour": 18,
        "weekday": 2,
        "baselineRecommended": False,
    },
    {
        "studyId": "vague-digital",
        "title": "Dieses neue Handy-Detail muessen Sie kennen",
        "url": "https://example.invalid/digital/handy-detail",
        "category": "digital",
        "score": 85.0,
        "recommendedHour": 10,
        "weekday": 2,
        "baselineRecommended": True,
    },
    {
        "studyId": "world-breaking",
        "title": "Eilmeldung: Waffenruhe im Iran-Krieg tritt sofort in Kraft",
        "url": "https://example.invalid/politik/ausland-und-internationales/waffenruhe",
        "category": "politik",
        "score": 90.0,
        "recommendedHour": 13,
        "weekday": 2,
        "isBreaking": True,
        "isEilmeldung": True,
        "baselineRecommended": True,
    },
    {
        "studyId": "national-politics",
        "title": "Bundestag beschliesst neue Rentenregel fuer Millionen Beschaeftigte",
        "url": "https://example.invalid/politik/inland/rentenregel",
        "category": "politik",
        "score": 86.0,
        "recommendedHour": 18,
        "weekday": 2,
        "baselineRecommended": False,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic, shadow-only editorial reader-mode panel."
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    study = run_synthetic_reader_panel_study(SYNTHETIC_CASES)
    if args.format == "json":
        print(json.dumps(study, ensure_ascii=False, indent=2))
    else:
        print(render_synthetic_reader_panel_markdown(study))


if __name__ == "__main__":
    main()
