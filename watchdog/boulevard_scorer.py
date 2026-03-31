#!/usr/bin/env python3
"""
Boulevard-Scorer: Bewertet Meldungen nach BILD-Relevanz.

Zwei Stufen:
  1. Regelbasiert (compute_boulevard_boost): Sofort, kein API nötig.
     Erkennt BILD-typische Muster und gibt einen Boost/Malus (-2 bis +3).
  2. LLM-Prompt (BOULEVARD_RATING_PROMPT): Für optionale Nachbewertung
     der Top-Kandidaten via GPT/Claude API.

Boulevard-DNA (was BILD druckt):
  - Tod, Gewalt, Sex, Kinder, Tiere
  - "Das könnte mir auch passieren" (Alltagssituationen die eskalieren)
  - Bizarre/ungewöhnliche Umstände
  - Prominente, Fußball, Schlagersänger
  - Empörung (Justizversagen, Behördenversagen, Wiederholungstäter)
  - Visuell stark (Tatortfotos, Fahndungsbilder, Dashcam)
  - Emotionale Opfer (Kinder, Senioren, Schwangere)

Was BILD NICHT druckt:
  - Routine-Polizeimeldungen ohne Drama
  - Sachbeschädigung, Ladendiebstahl
  - Reine Verkehrsbehinderungen
  - Politische Demos (außer Gewalt)
  - Behörden-Pressemitteilungen ohne Nachrichtenwert
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# LLM-Prompt für BILD-Relevanz-Bewertung
# ---------------------------------------------------------------------------

BOULEVARD_RATING_PROMPT = """Du bist ein erfahrener BILD-Redakteur im Ressort Blaulicht/Regional.
Bewerte die folgende Polizeimeldung auf einer Skala von 0-10 nach BILD-Relevanz.

BEWERTUNGSKRITERIEN:

HOHE RELEVANZ (8-10):
- Todesopfer, besonders bei ungewöhnlichen Umständen
- Kinder als Opfer oder Täter
- Gewaltverbrechen mit emotionalem Faktor (Familie, Beziehungstat)
- Bizarre Umstände die Leser zum Staunen/Entsetzen bringen
- "Mitten unter uns"-Faktor (Supermarkt, Spielplatz, Schule, ÖPNV)
- Serientäter, Intensivtäter, Wiederholungstäter
- Polizei-Großeinsatz mit Hubschrauber/SEK in belebter Gegend
- Fahndung mit Foto/Täterbeschreibung
- Prominente als Opfer oder Täter

MITTLERE RELEVANZ (4-7):
- Schwere Verletzungen ohne Todesfolge
- Raubüberfälle auf Geschäfte/Tankstellen
- Verfolgungsjagden, spektakuläre Fluchten
- Wohnungsbrände mit geretteten Bewohnern
- Vermisste Personen (besonders Kinder/Senioren)
- Drogenrazzia mit großem Fundvolumen
- Ungewöhnliche Tatmittel oder Tatorte

NIEDRIGE RELEVANZ (0-3):
- Routine-Verkehrsunfälle ohne Besonderheit
- Fahrraddiebstahl, Sachbeschädigung, Graffiti
- Trunkenheit im Verkehr (ohne Unfall)
- Ruhestörung, Hausfriedensbruch
- Politische Demos ohne Gewalt
- Behördliche Ankündigungen, Verkehrshinweise
- Wiederholte Wetterwarnungen

ZUSÄTZLICHE BILD-FAKTOREN (je +1):
- Konkretes Alter des Opfers im Titel ("63-Jähriger", "5-jähriges Mädchen")
- Ort ist bekannt/belebt ("am Alexanderplatz", "vor der Schule")
- Tatzeit ist ungewöhnlich ("um 3 Uhr nachts", "am hellichten Tag")
- Dramatische Verben ("rast", "stürzt", "flieht", "prügelt")
- Emotionale Details ("vor den Augen seiner Kinder")

MELDUNG:
Titel: {title}
Zusammenfassung: {description}
Ort: {location}
Quelle: {source}

Antworte NUR mit diesem JSON-Format:
{{
  "score": <0-10>,
  "grund": "<1 Satz warum>",
  "headline_vorschlag": "<BILD-typische Überschrift, max 60 Zeichen>",
  "kategorie": "<tot|gewalt|sex|kinder|brand|fahndung|kurios|routine>"
}}"""

# ---------------------------------------------------------------------------
# Regelbasierter Boulevard-Boost (kein LLM nötig)
# ---------------------------------------------------------------------------

# Opfer-Kategorien die BILD-Leser berühren
_VULNERABLE_VICTIMS = re.compile(
    r'(?:(\d{1,2})\s*-?\s*jährig)|'       # Altersangabe ("63-jähriger", "5-jährige")
    r'kind|mädchen|junge|baby|säugling|'
    r'schüler|jugendlich|teenager|'
    r'senior|rentn|oma|opa|greis|'
    r'schwanger|mutter|vater mit|'
    r'rollstuhl|blind|gehörlos',
    re.IGNORECASE,
)

# Orte die Nähe/Betroffenheit erzeugen
_SCARY_LOCATIONS = re.compile(
    r'schule|kita|kindergarten|spielplatz|'
    r'supermarkt|einkaufszentrum|bahnhof|haltestelle|'
    r'wohngebiet|wohnhaus|mehrfamilienhaus|'
    r'autobahn|a\s?\d{1,3}\b|'
    r'innenstadt|fußgängerzone|alexanderplatz|'
    r'krankenhaus|klinik|'
    r'kirche|friedhof|'
    r'park\b|freibad|schwimmbad|see\b',
    re.IGNORECASE,
)

# Bizarre / ungewöhnliche Umstände (BILD liebt das)
_BIZARRE_PATTERNS = re.compile(
    r'nackt|unbekleidet|'
    r'clown|verkleid|kostüm|'
    r'e-?scooter|segway|tretroller|'
    r'drohne|'
    r'tiktok|instagram|selfie|'
    r'betrunken.*kind|kind.*betrunken|'
    r'falsch.*autobahn|geisterfahrer|'
    r'hund|katze|schlange|krokodil|wolf|wildschwein|'
    r'flugzeug.*notland|hubschrauber.*land|'
    r'millionen|goldbarren|tresor|'
    r'waffen.*lager|arsenal',
    re.IGNORECASE,
)

# Dramatik-Verben die BILD-Headlines ausmachen
_DRAMATIC_VERBS = re.compile(
    r'rast|stürzt|flieht|prügel|würgt|'
    r'rammt|schleift|zertrümmert|'
    r'jagt|verfolg|'
    r'sticht|schlägt.*nieder|tritt.*ein|'
    r'fällt.*vom|springt.*von|'
    r'attackier|bedroht|terrorisier|'
    r'entkomm|verschwind|'
    r'rettet|befreit|überlebt',
    re.IGNORECASE,
)

# Empörungsfaktoren ("Das darf doch nicht sein!")
_OUTRAGE_PATTERNS = re.compile(
    r'bewähr|freispruch|milde.*strafe|'
    r'vorbestraft|intensivtäter|wiederholungstäter|'
    r'trotz.*verbot|trotz.*auflage|'
    r'polizei.*angegriff|rettungskräfte.*attackier|'
    r'geflohen.*unfallstelle|fahrerflucht|unfallflucht|'
    r'illegal|ohne.*führerschein|ohne.*versicherung|'
    r'abschieb|haftbefehl.*offen',
    re.IGNORECASE,
)

# Explizite Nicht-Boulevard-Meldungen (Score-Malus)
_BORING_PATTERNS = re.compile(
    r'sachbeschädigung|graffiti|schmiererei|'
    r'ruhestörung|lärm|nachbarschaftsstreit|'
    r'ladendiebstahl|taschendieb|'
    r'verkehrsbehinderung|baustelle|'
    r'präventionsveranstaltung|informationsveranstaltung|'
    r'kontrollaktion|geschwindigkeitsmessung|blitzer|'
    r'aktionswoche|prävention|sicherheitstipp|'
    r'pressemitteilung.*polizeipräsident|'
    r'bilanz\b|statistik|jahresbericht',
    re.IGNORECASE,
)


def compute_boulevard_boost(title: str, description: str = "") -> float:
    """
    Boulevard-Relevanz-Boost (-2 bis +3).

    Wird auf den bestehenden scoop_score addiert.
    Erkennt BILD-typische Muster die das reine Keyword-Scoring nicht erfasst.
    """
    text = f"{title} {description}"
    boost = 0.0

    # --- POSITIVE BOOSTS ---

    # Verletzliche Opfer (+0.5 bis +1.5)
    victim_match = _VULNERABLE_VICTIMS.search(text)
    if victim_match:
        boost += 0.5
        # Altersangabe im Titel? Extra-Boost — BILD liebt "63-Jähriger..."
        age_match = re.search(r'(\d{1,3})\s*-?\s*jährig', text, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
            if age <= 14:
                boost += 1.0  # Kind
            elif age >= 70:
                boost += 0.5  # Senior

    # Bedrohliche Orte (+0.5)
    if _SCARY_LOCATIONS.search(text):
        boost += 0.5

    # Bizarre Umstände (+1.0) — BILD-Gold
    if _BIZARRE_PATTERNS.search(text):
        boost += 1.0

    # Dramatische Sprache (+0.5)
    if _DRAMATIC_VERBS.search(text):
        boost += 0.5

    # Empörungsfaktor (+0.5)
    if _OUTRAGE_PATTERNS.search(text):
        boost += 0.5

    # Beziehungstat / Familientragödie (+0.5)
    if re.search(r'ehefrau|ehemann|ex-?freund|lebensgefährt|partner|famili|beziehungstat', text, re.IGNORECASE):
        boost += 0.5

    # --- NEGATIVE MALUS ---

    # Langweilige Routinemeldungen (-1 bis -2)
    if _BORING_PATTERNS.search(text):
        boost -= 1.5

    # Reine Demo/Protest ohne Gewalt (-1)
    if re.search(r'demo|kundgebung|protest|versammlung', text, re.IGNORECASE):
        if not re.search(r'gewalt|angriff|verletz|brand|randale', text, re.IGNORECASE):
            boost -= 1.0

    # Cap: -2 bis +3
    return max(-2.0, min(3.0, round(boost, 1)))


def compute_boulevard_score(
    title: str,
    description: str = "",
    bild_overlap_type: str = "none",
    zeitpunkt: str = "",
    media_urls: Optional[list[str]] = None,
) -> float:
    """
    Vollständiger Boulevard-Score (0-10).

    Basis: severity + exclusivity + freshness + media (wie bisher)
    Plus: boulevard_boost (-2 bis +3)

    Formel: base_score + boulevard_boost, geclampt auf 0-10.
    """
    import scoring

    base = scoring.compute_scoop_score(
        title=title,
        description=description,
        bild_overlap_type=bild_overlap_type,
        zeitpunkt=zeitpunkt,
        media_urls=media_urls,
    )
    boost = compute_boulevard_boost(title, description)

    return max(0.0, min(10.0, round(base + boost, 1)))
