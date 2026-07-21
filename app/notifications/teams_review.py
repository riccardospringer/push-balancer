"""Fast, local consensus review for Teams push recommendations.

The reviewers are deterministic pure functions over one ephemeral snapshot.
They perform no I/O, network calls, model calls, or durable storage.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger("push-balancer")

Snapshot = dict[str, Any]
Verdict = dict[str, Any]
Reviewer = Callable[[Snapshot], Verdict]

EVIDENCE_AGENTS = frozenset(
    {
        "Prognose",
        "Response-Potenzial",
        "Slot-Timing",
        "Ressort-Fit",
        "Headline-Klarheit",
    }
)
POLICY_AGENTS = frozenset({"Tagesbalance"})


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _compact(value: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def _verdict(
    agent: str,
    verdict: str,
    reason: str,
    *,
    confidence: float,
    hard_veto: bool = False,
) -> Verdict:
    return {
        "agent": agent,
        "verdict": verdict,
        "hardVeto": bool(hard_veto),
        "confidence": int(round(_clamp(float(confidence), 0.0, 1.0) * 100)),
        "reason": _compact(reason),
    }


def _reviewer_role(agent: str) -> str:
    if agent in EVIDENCE_AGENTS:
        return "evidence"
    if agent in POLICY_AGENTS:
        return "policy"
    return "hard_gate"


def _tag_role(verdict: Verdict) -> Verdict:
    verdict["role"] = _reviewer_role(str(verdict.get("agent") or ""))
    return verdict


def _normalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").casefold()
    if host == "bild.de" or host.endswith(".bild.de"):
        host = "bild.de"
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/").casefold()
    path = re.sub(r"/(?:amp|amphtml)$", "", path).rstrip("/")
    return f"{host}{path}" if host else path


def _normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _history_url(item: dict[str, Any]) -> str:
    return _normalize_url(item.get("url") or item.get("link") or item.get("article_url"))


def _history_title(item: dict[str, Any]) -> str:
    return _normalize_title(item.get("title") or item.get("headline") or item.get("article_title"))


def _integrity_agent(snapshot: Snapshot) -> Verdict:
    if not snapshot["title"]:
        return _verdict(
            "Artikel-Integritaet", "veto", "Headline fehlt", confidence=1.0, hard_veto=True
        )
    if not snapshot["url"]:
        return _verdict(
            "Artikel-Integritaet", "veto", "Artikel-Link fehlt", confidence=1.0, hard_veto=True
        )
    if snapshot["nonArticleReason"]:
        return _verdict(
            "Artikel-Integritaet",
            "veto",
            str(snapshot["nonArticleReason"]),
            confidence=0.99,
            hard_veto=True,
        )
    section = str(snapshot["section"] or "").casefold()
    if section in snapshot["excludedSections"]:
        return _verdict(
            "Artikel-Integritaet",
            "veto",
            "Ressort ist explizit ausgeschlossen",
            confidence=0.99,
            hard_veto=True,
        )
    if snapshot["allowedSections"] and section not in snapshot["allowedSections"]:
        return _verdict(
            "Artikel-Integritaet",
            "veto",
            "Ressort ist nicht freigegeben",
            confidence=0.99,
            hard_veto=True,
        )
    return _verdict(
        "Artikel-Integritaet",
        "approve",
        "Headline, Ziel und Ressort sind versandfaehig",
        confidence=0.99,
    )


def _context_integrity_agent(snapshot: Snapshot) -> Verdict:
    availability = snapshot["contextAvailable"]
    critical_missing = [
        name
        for name in ("history", "alertState", "recentTeamsAlerts")
        if not availability.get(name, False)
    ]
    if not snapshot.get("historyAuthoritative", False):
        critical_missing.append("livePushDedup")
    if critical_missing:
        return _verdict(
            "Kontext-Integritaet",
            "veto",
            "Pflichtdaten fuer die Dublettenpruefung nicht verfuegbar: "
            + ", ".join(sorted(set(critical_missing))),
            confidence=1.0,
            hard_veto=True,
        )
    optional_missing = [
        name
        for name in ("globalCooldown", "dailyAlertCount")
        if not availability.get(name, False)
    ]
    if optional_missing:
        return _verdict(
            "Kontext-Integritaet",
            "caution",
            "Sekundaerkontext fehlt; atomarer Versand-Claim bleibt aktiv: "
            + ", ".join(sorted(optional_missing)),
            confidence=0.9,
        )
    return _verdict(
        "Kontext-Integritaet",
        "approve",
        "Live-/Teams-Dubletten, Re-Alerts, Cooldown und Tageszaehler sind verfuegbar",
        confidence=0.99,
    )


def _exact_history_agent(snapshot: Snapshot) -> Verdict:
    candidate_cms_id = str(snapshot.get("cmsId") or "").casefold()
    candidate_url = _normalize_url(snapshot["url"])
    candidate_title = _normalize_title(snapshot["title"])
    if candidate_cms_id and candidate_cms_id in (snapshot.get("historyExactCmsIds") or ()):
        return _verdict(
            "Live-Push-Vergleich",
            "veto",
            "Bereits live gepusht: identische CMS-ID",
            confidence=0.99,
            hard_veto=True,
        )
    if candidate_url and candidate_url in snapshot["historyExactUrls"]:
        return _verdict(
            "Live-Push-Vergleich",
            "veto",
            "Bereits live gepusht: identische Artikel-URL",
            confidence=0.99,
            hard_veto=True,
        )
    if candidate_title and candidate_title in snapshot["historyExactTitles"]:
        return _verdict(
            "Live-Push-Vergleich",
            "caution",
            "Identische Live-Push-Headline erkannt, aber keine identische Artikel-URL",
            confidence=0.95,
        )
    return _verdict(
        "Live-Push-Vergleich",
        "approve",
        "Keine identische bereits live gepushte Artikel-URL gefunden",
        confidence=0.99,
    )


def _story_history_agent(snapshot: Snapshot) -> Verdict:
    for reason in (
        snapshot["topicDuplicateReason"],
        snapshot["realertBlocker"],
    ):
        if reason:
            return _verdict(
                "Story-Dublette und Re-Alert",
                "veto",
                str(reason),
                confidence=0.98,
                hard_veto=True,
            )
    return _verdict(
        "Story-Dublette und Re-Alert",
        "approve",
        "Keine Story-Dublette und kein gesperrter Re-Alert",
        confidence=0.96,
    )


def _freshness_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["overtakenReason"]:
        return _verdict(
            "Aktualitaet",
            "veto",
            str(snapshot["overtakenReason"]),
            confidence=0.99,
            hard_veto=True,
        )
    publication_status = str(snapshot.get("publicationStatus") or "missing")
    if publication_status != "valid":
        reason = str(
            snapshot.get("publicationReason") or "Veroeffentlichungszeit ist nicht belastbar"
        )
        return _verdict(
            "Aktualitaet",
            "veto",
            reason,
            confidence=0.99,
            hard_veto=True,
        )
    age = snapshot["freshnessHours"]
    if (
        snapshot["isSpeculative"]
        and age is not None
        and float(age) > snapshot["speculativeMaxAgeHours"]
    ):
        return _verdict(
            "Aktualitaet",
            "veto",
            "Spekulative Lage ist fuer eine belastbare Empfehlung zu alt",
            confidence=0.97,
            hard_veto=True,
        )
    if age is not None and float(age) > snapshot["maxArticleAgeHours"]:
        return _verdict(
            "Aktualitaet",
            "veto",
            "Artikel ist ausserhalb der maximalen Aktualitaet",
            confidence=0.98,
            hard_veto=True,
        )
    if snapshot["isSpeculative"]:
        return _verdict(
            "Aktualitaet",
            "caution",
            "Frische Spekulation muss gegen den aktuellen Meldungsstand geprueft werden",
            confidence=0.9,
        )
    return _verdict(
        "Aktualitaet",
        "approve",
        f"Artikel ist aktuell ({float(age):.1f} Stunden)",
        confidence=0.96 if float(age) <= 3.0 else 0.88,
    )


def _factual_risk_agent(snapshot: Snapshot) -> Verdict:
    risks = " ".join(str(value) for value in snapshot["candidateRisks"]).casefold()
    markers = (
        "unbestaetigt",
        "unbestätigt",
        "nicht verifiziert",
        "falschmeldung",
        "widerspruechlich",
        "widersprüchlich",
        "unklar, ob",
    )
    if any(marker in risks for marker in markers):
        return _verdict(
            "Faktenrisiko",
            "veto",
            "Kandidatenmetadaten enthalten ein offenes Faktenrisiko",
            confidence=0.98,
            hard_veto=True,
        )
    if snapshot["isSpeculative"] and not snapshot.get("speculationConfirmed", False):
        return _verdict(
            "Faktenrisiko",
            "veto",
            "Spekulative Lage ist nicht durch ein strukturiertes Bestaetigungssignal abgesichert",
            confidence=0.98,
            hard_veto=True,
        )
    return _verdict(
        "Faktenrisiko",
        "approve",
        "Kein explizites Widerspruchs- oder Unbestaetigt-Signal erkannt",
        confidence=0.88,
    )


def _cooldown_agent(snapshot: Snapshot) -> Verdict:
    minutes = snapshot["minutesSinceLastPush"]
    independent = bool(snapshot.get("independentTeamsPacing"))
    if not independent and minutes is not None and float(minutes) < snapshot["minPause"]:
        return _verdict(
            "Cooldown",
            "veto",
            "Mindestpause seit dem letzten wirklichen Push ist nicht erfuellt",
            confidence=1.0,
            hard_veto=True,
        )
    teams_minutes = snapshot["minutesSinceLastTeamsAlert"]
    global_pause = snapshot["globalCooldownMinutes"]
    if (
        not snapshot["breaking"]
        and teams_minutes is not None
        and global_pause > 0
        and float(teams_minutes) < global_pause
    ):
        return _verdict(
            "Cooldown",
            "veto",
            "Globaler Teams-Cooldown ist noch aktiv",
            confidence=1.0,
            hard_veto=True,
        )
    if not independent and minutes is None and not snapshot["allowUnknownLastPush"]:
        return _verdict(
            "Cooldown",
            "veto",
            "Letzter wirklicher Push-Zeitpunkt ist unbekannt",
            confidence=0.98,
            hard_veto=True,
        )
    reason = (
        "Breaking-News wird sofort geprüft; Teams-Cooldown ist nur für normale Empfehlungen aktiv"
        if independent and snapshot["breaking"]
        else "Eigener Teams-Cooldown ist erfuellt; Live-Push-Timing ist nur Vergleichskontext"
        if independent
        else "Alle bekannten Cooldowns sind erfuellt"
    )
    return _verdict("Cooldown", "approve", reason, confidence=0.99)


def _fatigue_agent(snapshot: Snapshot) -> Verdict:
    if snapshot.get("independentTeamsPacing"):
        return _verdict(
            "Nutzerbelastung",
            "approve",
            "Teams-Hinweise folgen eigenem Tageslimit und Cooldown",
            confidence=0.96,
        )
    recent = snapshot["recentPushCount6h"]
    maximum = snapshot["maxPushesLast6h"]
    if maximum > 0 and recent > maximum and not snapshot["breaking"]:
        return _verdict(
            "Nutzerbelastung",
            "veto",
            "Push-Dichte der letzten sechs Stunden ist zu hoch",
            confidence=0.98,
            hard_veto=True,
        )
    if maximum > 0 and recent >= maximum:
        return _verdict(
            "Nutzerbelastung",
            "caution",
            "Push-Dichte liegt bereits an der Belastungsgrenze",
            confidence=0.9,
        )
    return _verdict(
        "Nutzerbelastung",
        "approve",
        "Aktuelle Push-Dichte laesst eine weitere Empfehlung zu",
        confidence=0.94,
    )


def _news_value_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["softService"] and not snapshot["breaking"] and not snapshot["urgentService"]:
        return _verdict(
            "Nachrichtenwert",
            "veto",
            "Service-, Ratgeber- oder Quiz-Stoff ist kein belastbarer Push-Anlass",
            confidence=0.98,
            hard_veto=True,
        )
    if snapshot["weakNewsShape"] and not snapshot["breaking"]:
        return _verdict(
            "Nachrichtenwert",
            "veto",
            "Kein belastbares neues Ereignis fuer eine Sofort-Empfehlung",
            confidence=0.96,
            hard_veto=True,
        )
    if not snapshot["editorialApproved"] and not snapshot["deadlineApproved"]:
        return _verdict(
            "Nachrichtenwert",
            "caution",
            "CvD-Gesamtscore liegt unter der normalen Freigabe; Einzelindikatoren entscheiden",
            confidence=0.9,
        )
    if not snapshot["newsEvent"] and not snapshot["breaking"]:
        return _verdict(
            "Nachrichtenwert",
            "veto",
            "Headline zeigt kein konkretes neues Nachrichten-Ereignis",
            confidence=0.95,
            hard_veto=True,
        )
    return _verdict(
        "Nachrichtenwert",
        "approve",
        f"Konkrete Lage mit CvD-Nachrichtenwert {snapshot['editorialNewsValue']:.1f}",
        confidence=0.94,
    )


def _forecast_agent(snapshot: Snapshot) -> Verdict:
    reliable = snapshot["forecastSource"] == "article_model" and snapshot["predictedOR"] is not None
    if reliable:
        if snapshot["predictedOR"] < snapshot["minOR"]:
            if snapshot["forecastNearMissAccepted"]:
                return _verdict(
                    "Prognose",
                    "caution",
                    "OR liegt knapp unter der Norm, wird aber durch ein starkes Public-Need-Signal gestuetzt",
                    confidence=0.92,
                )
            if snapshot.get("highScoreOverrideApproved"):
                return _verdict(
                    "Prognose",
                    "caution",
                    "Artikelprognose liegt unter der OR-Norm; der kanonische "
                    "Push Score ueber 80 ist hier das staerkere Artikelsignal",
                    confidence=0.9,
                )
            return _verdict(
                "Prognose",
                "caution" if snapshot["deadlineApproved"] else "veto",
                "Artikelprognose liegt unter der normalen OR-Schwelle",
                confidence=0.94,
                hard_veto=not snapshot["deadlineApproved"],
            )
        return _verdict(
            "Prognose",
            "approve",
            "Belastbare artikelspezifische OR-Prognose liegt vor",
            confidence=max(0.75, snapshot["forecastConfidence"]),
        )
    allowed_exception = bool(
        snapshot["deadlineApproved"]
        or snapshot["breaking"]
        or snapshot["hardPublicNeed"]
        or snapshot.get("verifiedPeopleMilestone")
    )
    if snapshot["requireArticleForecast"] and not allowed_exception:
        if snapshot.get("highScoreOverrideApproved"):
            return _verdict(
                "Prognose",
                "caution",
                "Kanonischer Push Score ueber 80 ersetzt keine OR-Prognose, "
                "ist hier aber das staerkere Artikelsignal",
                confidence=0.9,
            )
        return _verdict(
            "Prognose",
            "veto",
            "Artikelspezifische Prognose fehlt und keine erlaubte Ausnahme greift",
            confidence=0.96,
            hard_veto=True,
        )
    return _verdict(
        "Prognose",
        "caution",
        "Nur historische Slot-Prognose; die OR-Schätzung ist mit Unsicherheit zu lesen",
        confidence=0.85,
    )


def _visit_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["expectedOpens"] <= 0 or snapshot["qualityAdjustedOpens"] <= 0:
        return _verdict(
            "Response-Potenzial",
            "caution",
            "Erwartete Push-Öffnungen sind nicht belastbar quantifizierbar",
            confidence=0.88,
        )
    if snapshot["reachConfidence"] < 0.25:
        return _verdict(
            "Response-Potenzial",
            "caution",
            "Positive OR-Schätzung, aber geringe Reichweitenkonfidenz",
            confidence=0.82,
        )
    return _verdict(
        "Response-Potenzial",
        "approve",
        "Positives qualitaetsbereinigtes Response-Potenzial "
        f"({snapshot['qualityAdjustedOpens']} erwartete Öffnungen)",
        confidence=min(0.96, 0.75 + snapshot["reachConfidence"] * 0.25),
    )


def _timing_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["slotGateEnabled"] and not snapshot["slotApproved"]:
        return _verdict(
            "Slot-Timing",
            "veto",
            "Aktuelles Tagesfenster ist noch nicht zur Entscheidung freigegeben",
            confidence=0.99,
            hard_veto=True,
        )
    if snapshot["slotMode"] in {
        "deadline_fallback",
        "projected_shortfall_catchup",
    }:
        reason = (
            "Shortfall-Recovery ist fuer das Tagesminimum mathematisch erforderlich"
            if snapshot["slotMode"] == "projected_shortfall_catchup"
            else ":45-Fenster ist faellig; das vollstaendige Kandidatenfeld wurde gesammelt"
        )
        return _verdict(
            "Slot-Timing",
            "approve",
            reason,
            confidence=0.96,
        )
    if snapshot["slotMode"] == "sport_event_override":
        return _verdict(
            "Slot-Timing",
            "approve",
            "Frischer materieller Sportzustand ist zeitkritischer als der Standardslot",
            confidence=0.97,
        )
    if snapshot["timeFit"] < snapshot["minTimeFit"] and not snapshot["breaking"]:
        return _verdict(
            "Slot-Timing",
            "veto",
            "Zeitfenster liegt unter dem Mindestfit",
            confidence=0.96,
            hard_veto=True,
        )
    return _verdict(
        "Slot-Timing",
        "approve",
        f"Zeitfenster ist freigegeben (Fit {snapshot['timeFit']:.1f}/10)",
        confidence=0.95,
    )


def _section_fit_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["breaking"] or not snapshot["hasCurrentSlot"]:
        return _verdict(
            "Ressort-Fit",
            "approve",
            "Breaking oder freies Fenster: Ressortfit ist kein Wartegrund",
            confidence=0.9,
        )
    section = snapshot["section"]
    if section in snapshot["slotPreferredSections"] or section == snapshot["slotTopCategory"]:
        return _verdict(
            "Ressort-Fit",
            "approve",
            "Ressort passt zum historisch starken Wochentags-Slot",
            confidence=0.94,
        )
    return _verdict(
        "Ressort-Fit",
        "caution",
        "Ressort ist nicht der historische Top-Fit dieses Slots",
        confidence=0.86,
    )


def _sport_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["section"] != "sport":
        return _verdict("Sport-Ereignis", "abstain", "Nicht anwendbar", confidence=0.99)
    if not snapshot["sportEventful"]:
        return _verdict(
            "Sport-Ereignis",
            "veto",
            "Keine bestaetigte Ergebnis-, Transfer-, Ausfall- oder Live-Lage erkannt",
            confidence=0.96,
            hard_veto=True,
        )
    return _verdict(
        "Sport-Ereignis",
        "approve",
        f"Bestaetigte Sportlage: {snapshot['sportEventType'] or 'neues Ereignis'}",
        confidence=0.96,
    )


def _headline_agent(snapshot: Snapshot) -> Verdict:
    if len(snapshot["title"]) < 16:
        return _verdict(
            "Headline-Klarheit",
            "veto",
            "Headline ist zu kurz fuer eine belastbare Handlungsempfehlung",
            confidence=0.97,
            hard_veto=True,
        )
    if snapshot["isSpeculative"]:
        return _verdict(
            "Headline-Klarheit",
            "caution",
            "Headline signalisiert Unsicherheit statt bestaetigter Entwicklung",
            confidence=0.93,
        )
    if snapshot["headlineClarity"] < 5.0 or (
        snapshot["genericHeadline"] and not snapshot["newsEvent"]
    ):
        return _verdict(
            "Headline-Klarheit",
            "caution",
            "Headline ist verstaendlich, aber noch nicht maximal konkret",
            confidence=0.86,
        )
    return _verdict(
        "Headline-Klarheit",
        "approve",
        f"Headline ist konkret und schnell erfassbar (Klarheit {snapshot['headlineClarity']:.1f}/10)",
        confidence=0.92,
    )


def _daily_balance_agent(snapshot: Snapshot) -> Verdict:
    if (
        snapshot["maxAlertsPerDay"] > 0
        and snapshot["teamsAlertsToday"] >= snapshot["maxAlertsPerDay"]
        and not snapshot["breaking"]
    ):
        return _verdict(
            "Tagesbalance",
            "veto",
            "Maximale Zahl der Teams-Empfehlungen ist erreicht",
            confidence=1.0,
            hard_veto=True,
        )
    if snapshot["minimumPressureActive"]:
        return _verdict(
            "Tagesbalance",
            "approve",
            "Tagesplan liegt zurueck; Kandidat hilft das 15er-Minimum kontrolliert zu erreichen",
            confidence=0.94,
        )
    if snapshot["pushSurplus"] >= 2.0:
        return _verdict(
            "Tagesbalance",
            "caution",
            "Tagespacing liegt bereits vor dem Plan; Zusatzbelastung kritisch abwaegen",
            confidence=0.9,
        )
    return _verdict(
        "Tagesbalance", "approve", "Empfehlung passt in das aktuelle Tagespacing", confidence=0.93
    )


def _skeptic_agent(snapshot: Snapshot) -> Verdict:
    if snapshot["remainingBlockers"]:
        return _verdict(
            "Adversarialer Gegenpruefer",
            "veto",
            f"Unaufgeloester Einwand: {snapshot['remainingBlockers'][0]}",
            confidence=0.99,
            hard_veto=True,
        )
    if snapshot["waivedBlockers"]:
        return _verdict(
            "Adversarialer Gegenpruefer",
            "caution",
            f"Starkstes Gegenargument: Die Sonderregel lockerte {len(snapshot['waivedBlockers'])} weiche Gate(s)",
            confidence=0.95,
        )
    if snapshot["isSpeculative"]:
        return _verdict(
            "Adversarialer Gegenpruefer",
            "caution",
            "Starkstes Gegenargument: Tatsachenstand kann sich bei Spekulation schnell aendern",
            confidence=0.94,
        )
    return _verdict(
        "Adversarialer Gegenpruefer",
        "approve",
        "Aktive Gegenpruefung findet kein unaufgeloestes Ausschlusskriterium",
        confidence=0.93,
    )


REVIEWERS: tuple[Reviewer, ...] = (
    _context_integrity_agent,
    _integrity_agent,
    _exact_history_agent,
    _story_history_agent,
    _freshness_agent,
    _factual_risk_agent,
    _cooldown_agent,
    _fatigue_agent,
    _news_value_agent,
    _forecast_agent,
    _visit_agent,
    _timing_agent,
    _section_fit_agent,
    _sport_agent,
    _headline_agent,
    _daily_balance_agent,
    _skeptic_agent,
)


def run_agent_review_network(
    snapshot: Snapshot,
    *,
    enabled: bool,
    min_evidence_approvals: int,
    min_consensus_score: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """Return a fail-closed verdict with hard gates and independent evidence."""
    if not enabled:
        return {
            "enabled": False,
            "approved": True,
            "mode": "disabled",
            "reviewerSetVersion": "teams-review-v3",
            "agentCount": 0,
            "approvalCount": 0,
            "evidenceApprovalCount": 0,
            "evidenceReviewerCount": len(EVIDENCE_AGENTS),
            "cautionCount": 0,
            "abstainCount": 0,
            "vetoCount": 0,
            "hardVetoCount": 0,
            "requiredApprovals": 0,
            "requiredEvidenceApprovals": 0,
            "consensusScore": 100.0,
            "minConsensusScore": 0.0,
            "latencyMs": 0.0,
            "latencyBudgetMs": 0,
            "latencyBreached": False,
            "mainCounterargument": "",
            "blockingReason": "",
            "summary": "Lokales Pruefkollegium deaktiviert",
            "verdicts": [],
        }

    started_ns = time.perf_counter_ns()
    verdicts: list[Verdict] = []
    for reviewer in REVIEWERS:
        try:
            verdicts.append(_tag_role(reviewer(snapshot)))
        except Exception as exc:
            log.exception("[TeamsAlert] local reviewer failed reviewer=%s", reviewer.__name__)
            verdicts.append(
                _tag_role(
                    _verdict(
                        reviewer.__name__,
                        "veto",
                        f"Interner Prueferfehler ({type(exc).__name__}); Freigabe sicherheitshalber verweigert",
                        confidence=1.0,
                        hard_veto=True,
                    )
                )
            )

    latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    latency_budget_ms = max(1, int(max_latency_ms or 1))
    latency_breached = latency_ms > latency_budget_ms
    if latency_breached:
        log.warning(
            "[TeamsAlert] local reviewer latency budget exceeded latency_ms=%.3f budget_ms=%s",
            latency_ms,
            latency_budget_ms,
        )
        latency_veto = _verdict(
            "Pruef-Latenz",
            "veto",
            (
                "Lokale Pruefung ueberschritt das Zeitbudget: "
                f"{latency_ms:.1f} > {latency_budget_ms} ms"
            ),
            confidence=1.0,
            hard_veto=True,
        )
        latency_veto["role"] = "hard_gate"
        verdicts.append(latency_veto)

    approvals = [item for item in verdicts if item["verdict"] == "approve"]
    cautions = [item for item in verdicts if item["verdict"] == "caution"]
    abstentions = [item for item in verdicts if item["verdict"] == "abstain"]
    vetoes = [item for item in verdicts if item["verdict"] == "veto"]
    hard_vetoes = [item for item in vetoes if item.get("hardVeto")]
    evidence = [item for item in verdicts if item.get("role") == "evidence"]
    evidence_approvals = [item for item in evidence if item["verdict"] == "approve"]
    count = len(verdicts)
    required = min(
        len(EVIDENCE_AGENTS),
        max(1, int(min_evidence_approvals or 1)),
    )
    consensus_score = round(
        100.0 * len(evidence_approvals) / max(1, len(evidence)),
        1,
    )
    minimum_consensus = _clamp(float(min_consensus_score), 0.0, 100.0)
    approved = bool(
        not hard_vetoes
        and len(evidence_approvals) >= required
        and consensus_score >= minimum_consensus
    )
    main_counterargument = ""
    if hard_vetoes:
        main_counterargument = str(hard_vetoes[0]["reason"])
    elif cautions:
        main_counterargument = str(cautions[0]["reason"])

    if hard_vetoes:
        first = hard_vetoes[0]
        blocking_reason = f"Agenten-Veto ({first['agent']}): {first['reason']}"
    elif len(evidence_approvals) < required:
        blocking_reason = (
            "Agenten-Evidenz zu schwach: "
            f"{len(evidence_approvals)} von {required} noetigen Evidenz-Freigaben"
        )
    elif consensus_score < minimum_consensus:
        blocking_reason = (
            f"Agenten-Evidenz zu schwach: {consensus_score:.1f} < {minimum_consensus:.1f}"
        )
    else:
        blocking_reason = ""

    summary = (
        f"Evidenz {len(evidence_approvals)}/{len(evidence)}, "
        f"{len(hard_vetoes)} harte Vetos, "
        f"Konsens {consensus_score:.0f}/100"
    )
    return {
        "enabled": True,
        "approved": approved,
        "mode": "local_deterministic_evidence_gates",
        "reviewerSetVersion": "teams-review-v3",
        "agentCount": count,
        "approvalCount": len(approvals),
        "evidenceApprovalCount": len(evidence_approvals),
        "evidenceReviewerCount": len(evidence),
        "cautionCount": len(cautions),
        "abstainCount": len(abstentions),
        "vetoCount": len(vetoes),
        "hardVetoCount": len(hard_vetoes),
        "requiredApprovals": required,
        "requiredEvidenceApprovals": required,
        "consensusScore": consensus_score,
        "minConsensusScore": round(minimum_consensus, 1),
        "latencyMs": round(latency_ms, 3),
        "latencyBudgetMs": latency_budget_ms,
        "latencyBreached": latency_breached,
        "mainCounterargument": main_counterargument,
        "blockingReason": blocking_reason,
        "summary": summary,
        "verdicts": verdicts,
    }


def add_agent_review_veto(
    review: dict[str, Any] | None,
    *,
    agent: str,
    reason: str,
) -> dict[str, Any]:
    """Attach a fail-closed batch/dispatch veto to an existing local review."""
    result = dict(review or {})
    verdicts = [dict(item) for item in (result.get("verdicts") or [])]
    veto = _verdict(agent, "veto", reason, confidence=1.0, hard_veto=True)
    veto["role"] = "hard_gate"
    verdicts.append(veto)
    for item in verdicts:
        item.setdefault("role", _reviewer_role(str(item.get("agent") or "")))
    approvals = sum(item.get("verdict") == "approve" for item in verdicts)
    cautions = sum(item.get("verdict") == "caution" for item in verdicts)
    abstentions = sum(item.get("verdict") == "abstain" for item in verdicts)
    vetoes = sum(item.get("verdict") == "veto" for item in verdicts)
    hard_vetoes = sum(
        item.get("verdict") == "veto" and bool(item.get("hardVeto")) for item in verdicts
    )
    count = len(verdicts)
    evidence = [item for item in verdicts if item.get("role") == "evidence"]
    evidence_approvals = sum(item.get("verdict") == "approve" for item in evidence)
    consensus_score = round(100.0 * evidence_approvals / max(1, len(evidence)), 1)
    result.update(
        {
            "enabled": True,
            "approved": False,
            "agentCount": count,
            "approvalCount": approvals,
            "evidenceApprovalCount": evidence_approvals,
            "evidenceReviewerCount": len(evidence),
            "cautionCount": cautions,
            "abstainCount": abstentions,
            "vetoCount": vetoes,
            "hardVetoCount": hard_vetoes,
            "consensusScore": consensus_score,
            "mainCounterargument": _compact(reason),
            "blockingReason": f"Agenten-Veto ({agent}): {_compact(reason)}",
            "summary": (
                f"Evidenz {evidence_approvals}/{len(evidence)}, "
                f"{hard_vetoes} harte Vetos, "
                f"Konsens {consensus_score:.0f}/100"
            ),
            "verdicts": verdicts,
        }
    )
    return result
