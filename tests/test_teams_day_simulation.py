"""Ganztags-Simulation: haelt der komponierte Kanal alle Regeln ueber die Zeit ein?

Fuehrt einen ganzen Berliner Tag im 5-Minuten-Takt durch und prueft die
Produktregeln als System-Invarianten - kein Einzel-Gate-Test, sondern das
Zusammenspiel von Raster, Volumen, Mindestabstand, Ruhezeit und Sport-Korridor.
Jede angenommene Empfehlung wird als echter Push zurueckgespielt (wie die
Redaktion sie umsetzt), damit Pacing und Cooldowns real mitlaufen.
"""

import datetime as dt
from zoneinfo import ZoneInfo

from app.notifications.teams import (
    build_teams_alert_context,
    evaluate_teams_alert_candidates,
    candidate_key,
)

from tests.test_teams_notifications import _candidate, _smart_config

BERLIN = ZoneInfo("Europe/Berlin")

_TOPICS = [
    ("politik", "Regierung beschliesst neue Entlastung fuer Millionen Haushalte"),
    ("news", "Aldi ruft Hackfleisch zurueck: akute Gesundheitsgefahr"),
    ("sport", "Bayern gewinnt Topspiel gegen Leverkusen mit 3:1"),
    ("politik", "Minister tritt zurueck nach Skandal um Spendengelder"),
    ("regional", "Schwerer Unfall auf der A7: Autobahn stundenlang gesperrt"),
    ("wirtschaft", "Traditionsbaecker meldet Insolvenz an: 200 Jobs weg"),
    ("news", "Bundestag beschliesst umstrittenes Gesetz nach langer Debatte"),
    ("sport", "DFB-Team schlaegt Frankreich 2:0 im Test - Wirtz trifft"),
    ("politik", "Kanzler kuendigt Reform an: das aendert sich fuer Rentner"),
]


def _field(now_ts: int) -> list[dict]:
    """Ein realistisches Kandidatenfeld, alle frisch, mit variierenden Scores."""
    field = []
    for index, (cat, title) in enumerate(_TOPICS):
        field.append(
            _candidate(
                id=f"sim-{now_ts}-{index}",
                url=f"https://www.bild.de/{cat}/sim-{now_ts}-{index}",
                title=f"{title} (Update {now_ts % 1000})",
                category=cat,
                score=92.0 - index * 2.0,
                predictedOR=0.07,
                pubDate=dt.datetime.fromtimestamp(now_ts - 8 * 60, BERLIN).isoformat(),
                recommendedText=f"{title}",
                isBreaking=False,
            )
        )
    return field


def _push_from(candidate: dict, ts: int) -> dict:
    return {
        "message_id": f"push-{ts}",
        "ts_num": ts,
        "or": 6.0,
        "title": candidate["title"],
        "headline": candidate["title"],
        "kicker": "",
        "cat": candidate["category"],
        "link": candidate["url"],
        "type": "editorial",
        "hour": dt.datetime.fromtimestamp(ts, BERLIN).hour,
        "title_len": len(candidate["title"]),
        "opened": 100,
        "received": 1000,
        "channel": "",
        "channels": [],
        "is_eilmeldung": False,
    }


def test_full_day_respects_all_product_rules():
    config = _smart_config(
        agent_review_enabled=False,
        require_internal_score_api=False,
        min_alert_score=60.0,
        min_editorial_score=60.0,
        min_or=4.0,
        min_minutes_since_last_push=30,
        global_cooldown_minutes=30,
    )
    day = dt.date(2026, 7, 15)  # Mittwoch
    start = int(dt.datetime.combine(day, dt.time(5, 0), tzinfo=BERLIN).timestamp())
    end = int(dt.datetime.combine(day, dt.time(23, 55), tzinfo=BERLIN).timestamp())

    # Vorabend-Push: in Produktion gibt es immer gestrige Historie; ohne einen
    # letzten Push-Zeitstempel bleibt der Kanal bewusst fail-closed.
    prior = int(dt.datetime.combine(day, dt.time(4, 0), tzinfo=BERLIN).timestamp())
    live_pushes: list[dict] = [
        {
            "message_id": "seed", "ts_num": prior, "or": 6.0,
            "title": "Vorabend-Meldung", "headline": "Vorabend-Meldung",
            "kicker": "", "cat": "news", "link": "https://www.bild.de/news/seed",
            "type": "editorial", "hour": 4, "title_len": 16, "opened": 100,
            "received": 1000, "channel": "", "channels": [], "is_eilmeldung": False,
        }
    ]
    sent_times: list[int] = []
    sent_hours: list[int] = []
    sent_sport = 0

    now = start
    while now <= end:
        field = _field(now)
        context = build_teams_alert_context(
            field,
            history=list(live_pushes),
            history_authoritative=True,
            alert_state={},
            last_teams_alert_ts=(sent_times[-1] if sent_times else 0),
            teams_alerts_today=len(sent_times),
            recent_alerts=[],
            now_ts=now,
            config=config,
        )
        context["dashboardRank"] = 1
        context["pushesToday"] = len(sent_times)

        evaluation = evaluate_teams_alert_candidates(field, context, config)
        selected = evaluation.get("selectedCandidate")
        decisions = {
            item["decision"]["candidateId"]: item["decision"]
            for item in evaluation["decisions"]
        }
        if selected and decisions.get(candidate_key(selected), {}).get("shouldNotify"):
            live_pushes.append(_push_from(selected, now))
            sent_times.append(now)
            sent_hours.append(dt.datetime.fromtimestamp(now, BERLIN).hour)
            if selected["category"] == "sport":
                sent_sport += 1
        now += 5 * 60

    total = len(sent_times)

    # 1. Tagesvolumen im Zielkorridor (mindestens Minimum, nie ueber Maximum).
    assert total <= config.max_alerts_per_day, f"Ueber Maximum: {total}"
    assert total >= config.min_alerts_per_day, f"Unter Minimum: {total}"

    # 2. Nutzerfreigabe bis 23:00: Ruhezeit 23:00-06:00, davor ist Versand erlaubt.
    assert all(6 <= hour < 23 for hour in sent_hours), f"Versand in Ruhezeit: {sent_hours}"

    # 3. Mindestabstand zwischen zwei Sendungen eingehalten.
    gaps = [(b - a) // 60 for a, b in zip(sent_times, sent_times[1:])]
    assert all(gap >= config.min_minutes_since_last_push for gap in gaps), (
        f"Mindestabstand verletzt: {gaps}"
    )

    # 4. Keine Doppelung: jede gesendete Story genau einmal.
    urls = [p["link"] for p in live_pushes]
    assert len(urls) == len(set(urls)), "Doppelte Story gesendet"

    # 5. Der Sport-Korridor ist im Pflichtslot nur Diagnose: die Ressortquote
    #    darf den kanonischen API-Top-1 niemals verdraengen.
    sport_share = sent_sport / total
    assert sport_share <= 0.5, f"Sport dominiert das Feld: {sent_sport}/{total}"


def test_full_day_quiet_period_produces_fewer_but_still_valid():
    """Auch bei nur schwachen Kandidaten wird das Maximum nie ueberschritten."""
    config = _smart_config(
        agent_review_enabled=False,
        require_internal_score_api=False,
        min_alert_score=60.0,
        min_editorial_score=60.0,
        min_or=4.0,
    )
    day = dt.date(2026, 7, 15)
    start = int(dt.datetime.combine(day, dt.time(5, 0), tzinfo=BERLIN).timestamp())
    end = int(dt.datetime.combine(day, dt.time(23, 55), tzinfo=BERLIN).timestamp())

    prior = int(dt.datetime.combine(day, dt.time(4, 0), tzinfo=BERLIN).timestamp())
    live_pushes: list[dict] = [
        {
            "message_id": "seed", "ts_num": prior, "or": 6.0,
            "title": "Vorabend-Meldung", "headline": "Vorabend-Meldung",
            "kicker": "", "cat": "news", "link": "https://www.bild.de/news/seed",
            "type": "editorial", "hour": 4, "title_len": 16, "opened": 100,
            "received": 1000, "channel": "", "channels": [], "is_eilmeldung": False,
        }
    ]
    sent_times: list[int] = []
    now = start
    while now <= end:
        # Nur schwache Kandidaten (unter der Score-Schwelle).
        field = [
            _candidate(
                id=f"weak-{now}",
                url=f"https://www.bild.de/news/weak-{now}",
                title=f"Eher weiche Servicemeldung ohne klare Lage {now % 1000}",
                category="news",
                score=68.0,
                predictedOR=0.045,
                pubDate=dt.datetime.fromtimestamp(now - 8 * 60, BERLIN).isoformat(),
            )
        ]
        context = build_teams_alert_context(
            field, history=list(live_pushes), history_authoritative=True,
            alert_state={}, last_teams_alert_ts=(sent_times[-1] if sent_times else 0),
            teams_alerts_today=len(sent_times), recent_alerts=[], now_ts=now, config=config,
        )
        context["dashboardRank"] = 1
        context["pushesToday"] = len(sent_times)
        evaluation = evaluate_teams_alert_candidates(field, context, config)
        selected = evaluation.get("selectedCandidate")
        decisions = {
            item["decision"]["candidateId"]: item["decision"]
            for item in evaluation["decisions"]
        }
        if selected and decisions.get(candidate_key(selected), {}).get("shouldNotify"):
            live_pushes.append(_push_from(selected, now))
            sent_times.append(now)
        now += 5 * 60

    # Schwache Kandidaten werden nicht kuenstlich hochgepusht -> nie ueber Maximum.
    assert len(sent_times) <= config.max_alerts_per_day



def test_raster_and_quiet_hours_are_dst_safe():
    """Zeit-Robustheit: an DST-Umstellungstagen bleiben Slots wanduhrgenau.

    Berlin springt Ende Maerz vor (02->03) und Ende Oktober zurueck (03->02).
    Ein rasterbasiertes System darf dabei nicht driften: jede Slot-Zeit muss
    eine gueltige, eindeutige Berliner Wanduhrzeit sein und die Ruhezeit
    23-06 muss unveraendert gelten.
    """
    from app.notifications.teams import (
        TeamsAlertConfig,
        _daily_runtime_opportunities,
        _quiet_hours_reason,
    )

    config = TeamsAlertConfig()
    dst_days = [
        dt.date(2026, 3, 29),   # Spring-forward
        dt.date(2026, 10, 25),  # Fall-back
        dt.date(2026, 7, 15),   # Normaler Tag als Kontrolle
    ]
    for day in dst_days:
        slots = _daily_runtime_opportunities(day, config)
        assert slots, f"Keine Slots am {day}"

        # Slot-Label == tatsaechliche Berliner Wanduhrzeit (kein DST-Drift).
        for slot in slots:
            wall = dt.datetime.fromtimestamp(int(slot["ts"]), BERLIN).strftime("%H:%M")
            assert slot["label"] == wall, f"{day}: {slot['label']} != {wall}"

        # Mindestabstand bleibt gewahrt.
        gaps = [
            (int(b["ts"]) - int(a["ts"])) // 60 for a, b in zip(slots, slots[1:])
        ]
        assert min(gaps) >= 30, f"{day}: Mindestabstand verletzt {gaps}"

        # Ruhezeit 23-06 gilt unveraendert; aktives Fenster 06:00-23:00.
        q_night = int(dt.datetime.combine(day, dt.time(23, 30), tzinfo=BERLIN).timestamp())
        q_evening = int(dt.datetime.combine(day, dt.time(22, 30), tzinfo=BERLIN).timestamp())
        q_day = int(dt.datetime.combine(day, dt.time(12, 0), tzinfo=BERLIN).timestamp())
        assert _quiet_hours_reason(q_night, config), f"{day}: 23:30 nicht ruhig"
        assert not _quiet_hours_reason(q_evening, config), f"{day}: 22:30 faelschlich ruhig"
        assert not _quiet_hours_reason(q_day, config), f"{day}: 12:00 faelschlich ruhig"
