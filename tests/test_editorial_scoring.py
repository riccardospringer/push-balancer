"""Tests for the editorial BILD push scoring layer."""
import datetime as dt
import time

from app.scoring.editorial import rebalance_push_mix, score_push_candidate


def _history(now: int) -> list[dict]:
    rows = []
    for idx in range(36):
        cat = ["politik", "verbraucher", "sport", "news"][idx % 4]
        hour = [8, 12, 20, 18][idx % 4]
        title = [
            "Bundestag beschließt neues Gesetz",
            "Warnung vor höheren Stromkosten",
            "Bayern-Star verletzt - Ausfall droht",
            "Polizei fasst Täter nach Messerattacke",
        ][idx % 4]
        rows.append(
            {
                "title": title,
                "cat": cat,
                "hour": hour,
                "ts_num": now - (idx + 8) * 7200,
                "or": [7.4, 6.8, 6.2, 5.9][idx % 4],
                "is_eilmeldung": False,
            }
        )
    return rows


def _pubdate(now: int, hours_ago: float) -> str:
    return (dt.datetime.fromtimestamp(now) - dt.timedelta(hours=hours_ago)).isoformat()


def _score(
    title: str,
    cat: str,
    *,
    now: int,
    hours_ago: float = 0.5,
    predicted_or: float = 6.0,
    history: list[dict] | None = None,
    **extra,
) -> dict:
    return score_push_candidate(
        {
            "title": title,
            "cat": cat,
            "hour": extra.pop("hour", 12),
            "ts_num": now,
            "pubDate": _pubdate(now, hours_ago),
            **extra,
        },
        history=history or _history(now),
        state={"global_avg": 5.5},
        predicted_or=predicted_or,
    )


def test_strong_bild_breaking_beats_generic_entertainment():
    now = int(time.time())
    history = _history(now)

    strong = score_push_candidate(
        {
            "title": "Eilmeldung: Bundestag beschließt Milliarden-Hilfe für Rentner",
            "cat": "politik",
            "hour": 8,
            "ts_num": now,
            "is_eilmeldung": True,
        },
        history=history,
        state={"global_avg": 5.5},
        predicted_or=8.0,
    )
    weak = score_push_candidate(
        {
            "title": "Promi zeigt neues Sommer-Outfit im Urlaub",
            "cat": "unterhaltung",
            "hour": 8,
            "ts_num": now,
        },
        history=history,
        state={"global_avg": 5.5},
        predicted_or=3.0,
    )

    assert strong["score"] >= 75
    assert strong["score"] > weak["score"] + 15
    assert strong["mixPriority"] == "hoch"
    assert strong["performanceDrivers"]
    assert strong["recommendedText"].startswith("Eilmeldung:")


def test_recent_topic_repetition_lowers_mix_balance():
    now = int(time.time())
    history = _history(now)
    repeated_history = history + [
        {
            "title": "Bayern-Star verletzt - Ausfall droht",
            "cat": "sport",
            "hour": 20,
            "ts_num": now - 1800 * (idx + 1),
            "or": 5.0,
        }
        for idx in range(3)
    ]

    clean = score_push_candidate(
        {"title": "Bayern-Star verletzt - Ausfall droht", "cat": "sport", "hour": 20, "ts_num": now},
        history=history,
        predicted_or=6.5,
    )
    repeated = score_push_candidate(
        {"title": "Bayern-Star verletzt - Ausfall droht", "cat": "sport", "hour": 20, "ts_num": now},
        history=repeated_history,
        predicted_or=6.5,
    )

    assert repeated["scoreBreakdown"]["mixBalance"] < clean["scoreBreakdown"]["mixBalance"] - 10
    assert any("Sättigung" in risk or "Ermüdung" in risk for risk in repeated["risks"])


def test_rebalance_push_mix_penalizes_duplicate_topics():
    now = int(time.time())
    history = _history(now)
    candidates = []
    for title in [
        "Trump droht mit neuen Zöllen gegen Europa",
        "Trump legt im Zoll-Streit nach",
        "Trump-Regierung verschärft Zoll-Kurs",
        "Rückruf bei beliebtem Käse: Das müssen Kunden wissen",
    ]:
        scored = {
            "title": title,
            "cat": "politik" if "Trump" in title else "verbraucher",
            "hour": 12,
            "ts_num": now,
            **score_push_candidate(
                {
                    "title": title,
                    "cat": "politik" if "Trump" in title else "verbraucher",
                    "hour": 12,
                    "ts_num": now,
                },
                history=history,
                predicted_or=7.0,
            ),
        }
        candidates.append(scored)

    before = {item["title"]: item["score"] for item in candidates}
    balanced = rebalance_push_mix(candidates, history=history, target_ts=now)
    after = {item["title"]: item["score"] for item in balanced}

    assert after["Trump-Regierung verschärft Zoll-Kurs"] < before["Trump-Regierung verschärft Zoll-Kurs"]
    assert any("Mix-Dopplung" in risk for item in balanced for risk in item["risks"])


def test_stale_abstract_politics_is_downgraded_against_bild_news():
    now = int(time.time())
    stale_politics = _score(
        "G7 verschärfen Druck auf Putin",
        "politik",
        now=now,
        hours_ago=10,
        predicted_or=6.4,
    )
    curious_news = _score(
        "Feuerwehr befreit Jungen aus Kita-Schrank - warum war er eingesperrt?",
        "news",
        now=now,
        hours_ago=0.4,
        predicted_or=5.8,
    )

    assert stale_politics["score"] < 55
    assert curious_news["score"] > stale_politics["score"] + 25
    assert any("Politik" in risk or "Aktualität" in risk for risk in stale_politics["risks"])


def test_current_trump_putin_turn_remains_high_priority():
    now = int(time.time())
    scored = _score(
        "G7-Gipfel: Plötzlich wendet sich Donald Trump von Wladimir Putin ab",
        "politik",
        now=now,
        hours_ago=0.25,
        predicted_or=8.0,
    )

    assert scored["score"] >= 75
    assert scored["mixPriority"] == "hoch"
    assert scored["scoreBreakdown"]["politicsContext"] >= 80
    assert "Politik" in " ".join(scored["performanceDrivers"])


def test_curiosity_news_beats_mediocre_policy_debate():
    now = int(time.time())
    politics = _score(
        "Startups fordern wirtschaftliche Wende von der Regierung",
        "politik",
        now=now,
        hours_ago=2.5,
        predicted_or=6.0,
    )
    weird = _score(
        "Kurioser Fund im Garten: Warum steckt dort ein Tresor?",
        "news",
        now=now,
        hours_ago=0.8,
        predicted_or=5.5,
    )

    assert weird["score"] > politics["score"]
    assert scored_breakdown_value(weird, "bildReiz") > scored_breakdown_value(politics, "bildReiz")


def test_consumer_outrage_gets_bild_push_reiz():
    now = int(time.time())
    scored = _score(
        "China-Shops tricksen Kunden aus: Diese Gebühren zahlen Millionen",
        "verbraucher",
        now=now,
        hours_ago=1,
        predicted_or=6.2,
    )

    assert scored["score"] >= 75
    assert scored["scoreBreakdown"]["bildReiz"] >= 70
    assert any("Verbraucher" in driver or "Aufreger" in driver for driver in scored["performanceDrivers"])


def test_public_money_fraud_razzia_gets_strong_push_score():
    now = int(time.time())
    scored = _score(
        "200 Polizisten im Einsatz: Großrazzia gegen Leistungsbetrüger",
        "news",
        now=now,
        hours_ago=0.25,
        predicted_or=4.9,
        hour=6,
    )

    assert scored["score"] >= 80
    assert scored["scoreBreakdown"]["bildReiz"] >= 75
    assert any("öffentlichen Leistungen" in driver for driver in scored["performanceDrivers"])


def test_evening_celebrity_relationship_money_conflict_gets_strong_push_score():
    now = int(time.time())
    scored = _score(
        "Wie bei so vielen Paaren – es geht ums Geld | Scheidungszoff bei WM-Held Schweini",
        "unterhaltung",
        now=now,
        hours_ago=0.5,
        predicted_or=4.9,
        hour=20,
    )

    assert scored["score"] >= 80
    assert scored["scoreBreakdown"]["bildReiz"] >= 85
    assert any("Beziehungs- und Geldkonflikt" in driver for driver in scored["performanceDrivers"])


def test_weak_video_is_penalized_but_live_video_can_rank_high():
    now = int(time.time())
    weak = _score(
        "Video: Zeigt dieses KI-Video die Zukunft?",
        "digital",
        now=now,
        hours_ago=4,
        predicted_or=4.5,
        video=True,
    )
    live = _score(
        "Messi knackt Klose-Rekord: JETZT Lothar legt los gucken",
        "sport",
        now=now,
        hours_ago=0.2,
        predicted_or=6.0,
        video=True,
    )

    assert weak["score"] < 65
    assert live["score"] >= 75
    assert live["scoreBreakdown"]["videoFit"] > weak["scoreBreakdown"]["videoFit"] + 30
    assert any("Video" in risk for risk in weak["risks"])


def test_previous_day_article_gets_freshness_penalty():
    now = int(time.time())
    scored = _score(
        "Journalistin behauptet bei Lanz: Deutschland soll verlieren",
        "politik",
        now=now,
        hours_ago=20,
        predicted_or=6.3,
        feedback="Artikel aus der Nacht und bezieht sich auf Abendsendung",
    )

    assert scored["scoreBreakdown"]["freshness"] < 35
    assert scored["score"] < 55
    assert any("Nacht" in risk or "zeitlich" in risk or "Aktualität" in risk for risk in scored["risks"])


def test_bild_exclusive_evergreen_keeps_a_chance_despite_age():
    now = int(time.time())
    exclusive = _score(
        "BILD-exklusiv: Was der ADHS-Patient im Folterknast erleben musste",
        "news",
        now=now,
        hours_ago=14,
        predicted_or=5.7,
        feedback="zeitloser Artikel, BILD-exklusiv geht immer, Zeile geht aber kerniger",
    )
    generic_old = _score(
        "Prozessabschluss nach Streit in der Innenstadt",
        "news",
        now=now,
        hours_ago=14,
        predicted_or=5.7,
        feedback="Prozessabschluss, aber kein neues Verbrechen",
    )

    assert exclusive["score"] > generic_old["score"]
    assert exclusive["score"] >= 60
    assert any("Exklusiv" in driver or "Exklusivität" in driver for driver in exclusive["performanceDrivers"])


def test_manual_feedback_penalizes_generic_case_and_vague_headline():
    now = int(time.time())
    generic = _score(
        "Mann randaliert in Supermarkt",
        "news",
        now=now,
        hours_ago=1,
        predicted_or=5.5,
        feedback="beliebiger Fall, passiert immer wieder",
    )
    vague = _score(
        "Dieses Detail sorgt jetzt für Rätsel",
        "news",
        now=now,
        hours_ago=1,
        predicted_or=5.5,
        headlineNote="unkonkret / verrätselt",
    )

    assert generic["score"] < 65
    assert vague["score"] < 65
    assert any("generisch" in risk or "Redaktionsfeedback" in risk for risk in generic["risks"])
    assert any("Headline" in risk or "unkonkret" in risk or "verrätselt" in risk for risk in vague["risks"])


def test_top10_rebalance_reduces_politics_dominance_when_strong_alternatives_exist():
    now = int(time.time())
    history = _history(now)
    politics_titles = [
        "G7 verschärfen Druck auf Putin",
        "Startups fordern wirtschaftliche Wende",
        "Kriminelle Kinder vor Gericht?",
        "Staatsbürgerschaft: Debatte um deutschen Pass",
        "Marine im Hormus: Regierung prüft Einsatz",
        "G7 und Trump beraten über Putin",
        "Minister fordert neue Regeln für Migranten",
    ]
    alternatives = [
        ("Messer-Alarm an Kita: Polizei nimmt Verdächtigen fest", "news"),
        ("China-Shops tricksen Kunden aus: Diese Gebühren zahlen Millionen", "verbraucher"),
        ("Messi knackt Klose-Rekord: JETZT Lothar legt los gucken", "sport"),
        ("Promi-Paar trennt sich nach TV-Skandal", "unterhaltung"),
    ]
    candidates = []
    for title in politics_titles:
        candidates.append(
            {
                "title": title,
                "cat": "politik",
                "ts_num": now,
                "pubDate": _pubdate(now, 3),
                **_score(title, "politik", now=now, hours_ago=3, predicted_or=6.7, history=history),
            }
        )
    for title, cat in alternatives:
        candidates.append(
            {
                "title": title,
                "cat": cat,
                "ts_num": now,
                "pubDate": _pubdate(now, 0.5),
                **_score(title, cat, now=now, hours_ago=0.5, predicted_or=6.2, history=history, video="gucken" in title.lower()),
            }
        )

    balanced = rebalance_push_mix(candidates, history=history, target_ts=now)
    top10 = balanced[:10]

    assert sum(1 for item in top10 if item["cat"] == "politik") <= 6
    assert any(item["cat"] != "politik" and item["score"] >= 70 for item in top10)
    assert any("Top-10-Balance" in " ".join(item.get("performanceDrivers", []) + item.get("risks", [])) for item in balanced)


def test_scoring_explanation_names_concrete_pro_and_contra_reasons():
    now = int(time.time())
    scored = _score(
        "Video: Messi knackt Klose-Rekord - JETZT live gucken",
        "sport",
        now=now,
        hours_ago=0.2,
        predicted_or=6.0,
        video=True,
        videoNote="Video, aber ok, weil aktuell und klar verständlich",
    )

    assert "hoch wegen" in scored["scoreReason"]
    assert any(word in scored["scoreReason"] for word in ("Video", "Aktualität", "Redaktionsfeedback", "Sportmoment"))
    assert scored["performanceDrivers"]
    assert scored["risks"]


def test_recommend_text_keeps_headline_without_generic_filler():
    now = int(time.time())

    politik = _score("Trump hebt neu ab", "politik", now=now)
    sport = _score("Bayern-Star vor Wechsel", "sport", now=now)
    utility = _score("Neue Strompreise ab Juli", "verbraucher", now=now)

    for result in (politik, sport, utility):
        rec = result["recommendedText"]
        assert "Was jetzt wichtig ist" not in rec
        assert "Was jetzt passiert" not in rec
        assert "Das bedeutet das für Sie" not in rec

    # Die konkrete Schlagzeile bleibt erhalten statt mit Floskeln verwaessert.
    assert "Trump hebt neu ab" in politik["recommendedText"]


def test_reuters_overload_penalises_sensationalism_but_not_breaking():
    now = int(time.time())

    sensational = _score("Wahnsinn! Unfassbarer Skandal erschüttert die Liga", "news", now=now)
    neutral = _score("Liga-Reform beschlossen: neue Regeln ab Sommer", "news", now=now)

    # Ueber-Sensationalismus wird abgewertet (Reuters DNR 2025).
    assert sensational["scoreBreakdown"]["overloadAdjustment"] < 0
    assert sensational["score"] < neutral["score"]

    # Harte Breaking-Lage ist ausgenommen, auch mit "Schock" im Titel.
    breaking = score_push_candidate(
        {
            "title": "Eilmeldung: Schock-Diagnose - Minister tritt zurück",
            "cat": "news",
            "hour": 9,
            "ts_num": now,
            "is_eilmeldung": True,
        },
        history=_history(now),
        state={"global_avg": 5.5},
        predicted_or=6.0,
    )
    assert breaking["scoreBreakdown"]["overloadAdjustment"] == 0.0


def scored_breakdown_value(result: dict, key: str) -> float:
    return float(result["scoreBreakdown"][key])
