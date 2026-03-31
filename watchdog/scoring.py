#!/usr/bin/env python3
"""
Einheitliches Multi-Faktor-Scoring für alle Watchdog-Scraper.

scoop_score = severity + exclusivity + freshness + media   (0-10)

Severity: MAX(keyword-matches), nicht Summe!
Exclusivity: BILD-Overlap-Check (reduziert, max +1)
Freshness: HOT/WARM/COOLING
Media: Foto/Video vorhanden

Zielverteilung:
  9-10  ALARM — Mord, Terror, Geiselnahme + frisch + boulevard-relevant
  7-8   TOP SCOOP — Schwere Gewalt, Messer, Schüsse + frisch
  5-6   MELDUNG — Großeinsatz, Brand mit Verletzten
  3-4   MONITOR — Normaler Brand, Überfall, Festnahme
  0-2   ROUTINE — Verkehrsunfall, Diebstahl, Sachbeschädigung
"""

import re
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Severity-Tiers (MAX-basiert, nicht additiv!)
# ---------------------------------------------------------------------------

# Blaulicht-Tiers (Default — Polizei, Feuer, Unfall, Rettung, Unwetter)
SEVERITY_TIERS: list[tuple[float, list[str]]] = [
    # (Score, Keywords) — höchster Tier gewinnt
    (5.0, [
        # Tod / Todesopfer — alle Polizei-Formulierungen
        "tod", "tot", "tödlich", "getötet", "todesopfer", "leiche",
        "verstirbt", "verstorben", "gestorben", "ums leben",
        "erlag", "tödlich verletzt", "leblos", "obduktion",
        "leichenfund", "skelett",
        # Mord / Tötung
        "mord", "mordkommission", "tötungsdelikt", "totschlag",
        "erschlagen", "erstochen", "erschossen",
        # Extremlagen
        "amoklauf", "amokfahrt",
        "terror", "terrorverdacht", "terroranschlag",
        "geiselnahme", "geisel",
        "flugzeugabsturz", "zugunglück",
    ]),
    (4.0, [
        # Waffen
        "schüsse", "schusswaffe", "geschossen", "pistole", "revolver",
        "messer", "messerangriff", "messerattacke", "messerstecher",
        "machete", "axt", "schwert",
        # Sprengstoff
        "explosion", "bombe", "sprengstoff", "sprengstoffverdacht",
        # Lebensgefahr
        "lebensgefahr", "lebensgefährlich", "reanimation", "reanimiert",
        "intensivstation",
        # Entführung / Kinder
        "entführung", "entführt",
        "vermisstes kind", "kindesmisshandlung",
        "vergewaltigung", "sexualdelikt",
    ]),
    (3.0, [
        # Großlagen
        "großeinsatz", "großlage", "großalarm",
        "sek", "spezialeinsatzkommando", "mek",
        "schwerverletzt", "schwer verletzt",
        "großbrand", "vollbrand",
        "evakuierung", "evakuiert",
        "massenkarambolage",
        "hochwasser", "überschwemmung",
        "chemieunfall", "gefahrgut",
        # Flucht / Verfolgung
        "verfolgungsjagd", "flucht vor polizei",
    ]),
    (1.5, [
        "brand", "feuer", "dachstuhlbrand",
        "überfall", "raubüberfall", "bankraub",
        "fahndung", "festnahme", "festgenommen",
        "vermisst",
        "razzia", "durchsuchung",
        "wohnungsbrand", "kellerbrand",
    ]),
    (0.5, [
        "unfall", "verkehrsunfall", "auffahrunfall",
        "raub", "diebstahl",
        "vollsperrung",
        "schlägerei", "körperverletzung",
    ]),
    (0.0, [
        "demo", "demonstration", "protest",
        "sachbeschädigung",
    ]),
]

def compute_severity(title: str, description: str = "", category: str = "") -> float:
    """MAX-basierte Severity (0-5). Höchster Tier-Match gewinnt.

    Nutzt ausschließlich Blaulicht-Tiers (Crime-Radar).
    category-Parameter bleibt für Backward-Kompatibilität, wird ignoriert.
    """
    text = f"{title} {description}".lower()

    for tier_score, keywords in SEVERITY_TIERS:
        for kw in keywords:
            if kw in text:
                return tier_score

    return 0.0


def compute_exclusivity(bild_overlap_type: str) -> float:
    """Exclusivity-Score basierend auf BILD-Overlap (0-1).

    Reduziert auf max 1.0 — der Haupttreiber ist Severity, nicht ob
    BILD es schon hat oder nicht. Sonst werden Routine-Meldungen
    künstlich auf 5+ gepusht nur weil BILD sie nicht gebracht hat.
    """
    if bild_overlap_type == "none":
        return 1.0  # Exklusiv — kleiner Bonus
    elif bild_overlap_type == "topic":
        return 0.5  # Thema bekannt
    else:  # "exact"
        return 0.0  # BILD hat bereits berichtet


def compute_freshness_score(zeitpunkt: str) -> float:
    """Freshness-Score (0-2): HOT=2, WARM=1, COOLING=0."""
    try:
        if zeitpunkt.endswith("Z"):
            zeitpunkt = zeitpunkt.replace("Z", "+00:00")
        dt = datetime.fromisoformat(zeitpunkt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600

        if age_hours < 0.5:
            return 2.0  # HOT
        elif age_hours < 2.0:
            return 1.0  # WARM
        else:
            return 0.0  # COOLING/COLD
    except Exception:
        return 0.0


def compute_media_score(media_urls: list[str]) -> float:
    """Media-Score (0-0.5): Hat Foto/Video = +0.5."""
    if not media_urls:
        return 0.0
    real_media = [u for u in media_urls if u and len(u) > 10]
    return 0.5 if real_media else 0.0


def compute_scoop_score(
    title: str,
    description: str = "",
    bild_overlap_type: str = "none",
    zeitpunkt: str = "",
    media_urls: Optional[list[str]] = None,
    category: str = "",
) -> float:
    """
    Einheitlicher Multi-Faktor Scoop-Score (0-10).

    severity (0-5) + exclusivity (0-1) + freshness (0-2) + media (0-0.5)
    = max 8.5 Basis, mit Boulevard-Boost bis 10.
    """
    severity = compute_severity(title, description, category=category)
    exclusivity = compute_exclusivity(bild_overlap_type)
    freshness = compute_freshness_score(zeitpunkt) if zeitpunkt else 0.5
    media = compute_media_score(media_urls or [])

    total = severity + exclusivity + freshness + media
    return min(10.0, round(total, 1))
