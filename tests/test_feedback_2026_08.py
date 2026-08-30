"""Tests für die Score-Regeln aus dem Redaktions-Feedback vom 27.08.2026.

Abgedeckt: Erstpublikation (1), Spielberichte (3), Früh-Slots (4d), Top-Slots
(6), beendete Live-Formate (8), Echte-News-Erkennung und Video-Abwertung (10)
sowie die Learnings (Promi/Wirtschaft ohne Dringlichkeit, Live-Teaser,
Überraschungswissen). Punkt 9 (Sport-Rätselzeilen) und der Autounfall-Malus
werden auf Redaktionswunsch vom 27.08. bewusst NICHT gesondert abgestraft.
"""

import datetime as dt
import time

from app.scoring.editorial import score_push_candidate


def _pubdate(now: int, hours_ago: float) -> str:
    return (dt.datetime.fromtimestamp(now) - dt.timedelta(hours=hours_ago)).isoformat()


def _score(title: str, cat: str, *, now: int, hours_ago: float = 0.5, **extra) -> dict:
    return score_push_candidate(
        {
            "title": title,
            "cat": cat,
            "hour": extra.pop("hour", 11),
            "ts_num": now,
            "pubDate": _pubdate(now, hours_ago),
            **extra,
        },
        history=[],
        state={"global_avg": 5.5},
        predicted_or=6.0,
    )


def test_republished_article_is_scored_on_first_publication():
    now = int(time.time())
    republished = _score(
        "Kanzler kündigt Steuerreform an",
        "politik",
        now=now,
        hours_ago=0.3,
        firstPublishedAt=_pubdate(now, 20.0),
    )
    fresh = _score(
        "Kanzler kündigt Steuerreform an",
        "politik",
        now=now,
        hours_ago=0.3,
    )
    assert republished["score"] < fresh["score"] - 8
    assert republished["scoreBreakdown"]["freshness"] < fresh["scoreBreakdown"]["freshness"]


def test_running_live_ticker_keeps_latest_update_for_freshness():
    now = int(time.time())
    ticker = _score(
        "Unwetter über NRW: Die Lage im Live-Ticker",
        "news",
        now=now,
        hours_ago=0.4,
        firstPublishedAt=_pubdate(now, 30.0),
    )
    # Laufender Ticker mit frischem Update darf nicht als 30h alt gelten.
    assert ticker["scoreBreakdown"]["freshness"] >= 70


def test_spielberichte_get_no_special_penalty():
    """Score-Umbau 30.08.2026: Spielberichte werden nicht mehr gesondert
    abgestraft — Bewertung laeuft ueber LLM-Reader-Score und Aktualitaet."""
    now = int(time.time())
    stale = _score(
        "Spielbericht: 2:1-Sieg für den HSV gegen Hannover",
        "sport",
        now=now,
        hours_ago=2.5,
    )
    assert "isPostEventReport" not in stale
    assert not any("Ergebnis-Bericht" in risk for risk in stale["risks"])

    taxonomy = _score(
        "HSV müht sich zum Heimsieg",
        "sport",
        now=now,
        hours_ago=2.0,
        taxonomy=["Fussball", "Spielbericht"],
    )
    assert "isPostEventReport" not in taxonomy
    assert not any("Ergebnis-Bericht" in risk for risk in taxonomy["risks"])


def test_ended_livestream_is_effectively_excluded():
    now = int(time.time())
    ended = _score(
        "Pressekonferenz JETZT im Livestream",
        "news",
        now=now,
        hours_ago=6.0,
    )
    assert ended["isEndedLiveFormat"] is True
    assert any("beendet" in risk for risk in ended["risks"])

    explicitly_ended = _score(
        "Pressekonferenz JETZT im Livestream",
        "news",
        now=now,
        hours_ago=0.5,
        liveStatus="ended",
    )
    assert explicitly_ended["isEndedLiveFormat"] is True
    assert explicitly_ended["score"] < ended["score"] + 35


def test_live_teaser_without_mass_relevance_is_dampened():
    now = int(time.time())
    teaser = _score(
        "Testspiel gegen Drittligist: Jetzt live gucken im Stream",
        "sport",
        now=now,
        hours_ago=0.3,
    )
    assert any("Teaser" in risk or "Live-/Stream" in risk for risk in teaser["risks"])


def test_sport_riddle_quote_line_is_not_specially_penalized():
    now = int(time.time())
    riddle = _score(
        "„Unheimlich schnell auf die Fresse“: Koschinat hofft weiter auf die Hafenstraße",
        "sport",
        now=now,
        hours_ago=0.5,
    )
    assert not any("Rätselzeile" in risk for risk in riddle["risks"])


def test_routine_traffic_accident_is_not_specially_penalized():
    now = int(time.time())
    accident = _score(
        "Schwerer Autounfall auf der A2: Drei Tote",
        "news",
        now=now,
        hours_ago=0.5,
    )
    assert not any("Verkehrsunfälle" in risk for risk in accident["risks"])


def test_celebrity_topic_without_urgency_is_dampened():
    now = int(time.time())
    scored = _score(
        "TV-Star zeigt seine neue Villa am See",
        "unterhaltung",
        now=now,
        hours_ago=1.0,
    )
    assert any("Dringlichkeit" in risk for risk in scored["risks"])


def test_curiosity_discovery_gets_a_boost_and_no_overload_penalty():
    now = int(time.time())
    scored = _score(
        "Kurioser Dürre-Fund in der Donau: Wehrmachtssoldaten und Motorrad freigelegt",
        "news",
        now=now,
        hours_ago=0.6,
    )
    assert scored["scoreBreakdown"]["feedback2026Adjustment"] > 0
    assert not any("Overload" in risk for risk in scored["risks"])
    assert any("Überraschungswissen" in item for item in scored["performanceDrivers"])


def test_fresh_missing_person_case_ranks_high():
    now = int(time.time())
    scored = _score(
        "Letzte Spur führt in eine Klinik: Deutsche Urlauberin (24) auf Mallorca verschwunden",
        "news",
        now=now,
        hours_ago=1.5,
    )
    assert scored["score"] >= 70
    assert scored["mixPriority"] in {"hoch", "mittel"}


def test_video_has_no_special_penalty_anymore():
    """Score-Umbau 30.08.2026: Video-Sonderabwertung ist komplett gestrichen."""
    now = int(time.time())
    video = _score(
        "Video zeigt Ausflug einer Entenfamilie im Stadtpark",
        "news",
        now=now,
        hours_ago=0.5,
        isVideo=True,
    )
    assert "videoFit" not in video["scoreBreakdown"]
    assert not any("Video ohne klaren Jetzt-Anlass" in risk for risk in video["risks"])


def test_prime_slot_penalizes_routine_topics():
    now = int(time.time())
    routine_prime = _score(
        "Museum zeigt neue Ausstellung über Alltagsdesign",
        "news",
        now=now,
        hours_ago=1.0,
        hour=12,
    )
    routine_offpeak = _score(
        "Museum zeigt neue Ausstellung über Alltagsdesign",
        "news",
        now=now,
        hours_ago=1.0,
        hour=11,
    )
    assert routine_prime["score"] < routine_offpeak["score"]
    assert any("Top-Slot" in risk for risk in routine_prime["risks"])


def test_early_morning_slot_prefers_acute_impact_over_politics():
    now = int(time.time())
    weather = _score(
        "Unwetter-Warnung für NRW: Orkanböen am Morgen",
        "wetter",
        now=now,
        hours_ago=0.3,
        hour=7,
    )
    politics = _score(
        "Koalition diskutiert Rentenkonzept weiter",
        "politik",
        now=now,
        hours_ago=0.3,
        hour=7,
    )
    assert weather["scoreBreakdown"]["feedback2026Adjustment"] > politics[
        "scoreBreakdown"
    ]["feedback2026Adjustment"]
    assert any("Früh-Slot" in risk for risk in politics["risks"])
