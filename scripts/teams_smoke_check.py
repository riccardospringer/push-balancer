#!/usr/bin/env python3
"""Produktions-Smoke-Check fuer den Teams-Kanal des Push Balancers.

Prueft die laufende Instanz (Default: lokal auf dem Mac) end-to-end:
Score-API-Kette, Push-Historie, Slot-Planung, Webhook-Konfiguration.

Aufruf:
    python3 scripts/teams_smoke_check.py [--base-url http://localhost:8050]

Exit-Code 0 = alle Pruefungen gruen, 1 = mindestens eine rot.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _fetch(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8050")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        health = _fetch(base_url, "/api/health")
    except Exception as exc:
        print(f"❌ Instanz nicht erreichbar unter {base_url}: {exc}")
        return 1
    print(f"✅ Instanz erreichbar ({base_url}), Status: {health.get('status')}")

    try:
        readiness = _fetch(base_url, "/api/teams-readiness")
    except Exception as exc:
        print(f"❌ /api/teams-readiness nicht abrufbar: {exc}")
        return 1

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "Teams-Kanal aktiviert",
        bool(readiness.get("teamsAlertsEnabled")),
        "PUSH_TEAMS_ALERTS_ENABLED=1 setzen",
    ))
    checks.append((
        "Webhook konfiguriert",
        bool(readiness.get("webhookConfigured")),
        "PUSH_TEAMS_WEBHOOK_URL als Secret setzen",
    ))

    score = readiness.get("scoreApi") or {}
    if score.get("enabled"):
        detail = (
            f"{score.get('freshCanonicalScores', 0)}/{score.get('checkedCandidates', 0)} "
            f"Kandidaten mit frischem kanonischem Score; Quellen: {score.get('sources')}"
        )
        checks.append((
            f"Score-API liefert kanonische Scores ({detail})",
            bool(score.get("ok")),
            "Score-API-Erreichbarkeit/Key/Frische pruefen (fail-closed: ohne Score keine Empfehlung)",
        ))
    else:
        checks.append((
            "Score-API deaktiviert - Capture/Fallback-Modus",
            True,
            "",
        ))

    history = readiness.get("pushHistory") or {}
    age = history.get("lastPushAgeMinutes")
    checks.append((
        f"Push-Historie vorhanden (letzter Push vor {age} Min, heute: "
        f"{history.get('pushesToday')}, davon Sport: {history.get('sportPushesToday')})",
        bool(history.get("ok")),
        "BILD-Push-API-Erreichbarkeit pruefen (Pacing ist ohne Historie fail-closed)",
    ))

    slots = readiness.get("slots") or {}
    next_slot = slots.get("nextSlot") or {}
    checks.append((
        f"Tagesplan korrekt ({slots.get('plannedToday')} verbindliche Slots, "
        f"naechster: {next_slot.get('label')} [{next_slot.get('role')}])",
        bool(slots.get("ok")),
        "Slot-Planung pruefen",
    ))

    runtime = readiness.get("runtime") or {}
    if runtime:
        checks.append((
            f"Worker laeuft (Status {runtime.get('status')}, letzter Zyklus vor "
            f"{runtime.get('cycleAgeSeconds')} s, {runtime.get('cycleCount')} Zyklen, "
            f"{runtime.get('workerRestarts')} Neustarts)",
            runtime.get("status") in {"ok", "starting", "disabled"},
            "Worker-Logs pruefen; Watchdog startet automatisch neu",
        ))

    if readiness.get("quietHoursActive"):
        print(f"ℹ️  Ruhezeit aktiv: {readiness.get('quietHoursReason')}")

    failures = 0
    for label, ok, hint in checks:
        print(f"{'✅' if ok else '❌'} {label}" + (f" -> {hint}" if not ok and hint else ""))
        if not ok:
            failures += 1

    volume = readiness.get("volume") or {}
    print(
        f"ℹ️  Volumen {volume.get('min')}-{volume.get('max')} Pushes/Tag, "
        f"Sportkorridor {volume.get('sportMin')}-{volume.get('sportMax')} | "
        f"Berliner Zeit: {readiness.get('berlinTime')}"
    )

    if failures:
        print(f"\n❌ {failures} Pruefung(en) rot - Kanal laeuft NICHT vollstaendig.")
        return 1
    print("\n✅ Alle Pruefungen gruen - der Teams-Kanal ist einsatzbereit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
