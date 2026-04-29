#!/usr/bin/env python3
"""Export a concise Story Radar ML model explanation as PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


OUTPUT_PATH = Path("docs/story_radar_ml_model_explanation.pdf")


TITLE = "Story Radar - ML Modell Erklaerung"

SECTIONS = [
    (
        "1. Ziel",
        [
            "Das ML-Modell bewertet Story-/Topic-Cluster nicht nach allgemeinem Nachrichtenwert, sondern nach der Frage: "
            "Wie relevant ist dieses Thema fuer BILD genau jetzt?",
            "Es soll nur Cluster nach vorne ziehen, die fuer BILD-Leser voraussichtlich interessant sind, aktuell passen, "
            "bei BILD fehlen oder zu schwach abgedeckt sind und sich in eine starke Geschichte uebersetzen lassen.",
        ],
    ),
    (
        "2. Input",
        [
            "Primarer Input sind Story-Cluster mit Titel, Summary, Topics, Entities, Laendern, Quellenbreite, Dokumentanzahl und Freshness.",
            "Dazu kommen BILD-Coverage-Daten, aktuelle Performance-Snapshots aus BILD, CMS-/Topic-Metadaten und redaktionelles Feedback.",
        ],
    ),
    (
        "3. Feature-Gruppen",
        [
            "Topic-Fit: Passt das Cluster zu Crime, Consumer, Promi, Sport, Politik oder anderen BILD-Kernfeldern?",
            "Freshness und Urgency: Wie frisch ist das Thema, wie schnell waechst es, ist es Breaking oder Follow-up?",
            "Gap und Coverage: Hat BILD das Thema schon, fehlt ein Angle, fehlt ein Update oder fehlt die Story komplett?",
            "Storyability: Gibt es Personen, Konflikt, Konsequenzen, Nutzwert oder klare Emotionalitaet?",
            "Aktuelle BILD-Nachfrage: Welche Topics, Entities und Ressorts funktionieren in den letzten Stunden besonders gut?",
            "Noise-Erkennung: Ist das Thema Standard-Agenturrauschen, Buerokratie oder nur Pflichtstoff ohne BILD-Winkel?",
        ],
    ),
    (
        "4. Modelltyp",
        [
            "Empfohlen ist ein Learning-to-Rank-Modell mit LightGBM LambdaRank als Primaermodell.",
            "Warum Ranking statt nur Klassifikation? Weil die Redaktion am Ende eine Top-Liste braucht und die relative Reihenfolge innerhalb "
            "eines Refresh-Zyklus entscheidend ist.",
            "Optional kann daneben ein Binary-Suppressor laufen, der 'already covered', 'noise', 'duplicate' und 'weak story' frueh abraeumt.",
        ],
    ),
    (
        "5. Zielvariable",
        [
            "Positive Trainingsbeispiele sind Cluster, die redaktionell akzeptiert, als echte Luecke bestaetigt, publiziert "
            "oder im Blind-Rating hoch priorisiert wurden.",
            "Negative Beispiele sind Cluster, die als 'already covered', 'noise', 'duplicate' oder 'nicht brauchbar' verworfen wurden.",
            "Post-Publish-Performance fliesst nicht roh ein, sondern relativ zur aktuellen Ressort-/Topic-Baseline, damit das Modell "
            "nicht nur alte Klickmuster einfriert.",
        ],
    ),
    (
        "6. Dynamik",
        [
            "Das Modell darf nicht statisch sein. Re-Scoring laeuft eng getaktet, Trainingsdaten werden mit Rolling Windows und Recency-Weighting "
            "frisch gehalten.",
            "Empfohlen: taegliches leichtes Retraining, woechentliches Volltraining, 28-Tage-Hauptfenster plus abgeschwaechtes 90-Tage-Gedaechtnis.",
            "Drift wird ueber Precision@K, Suppression Precision, PSI auf Kernfeatures und Topic-/Ressort-Verschiebungen erkannt.",
        ],
    ),
    (
        "7. Gap Detection",
        [
            "Gap Detection ist vorgeschaltet und fuer das Modell zentral.",
            "Der Status pro Cluster ist: already_covered, partially_covered, not_covered, angle_gap oder follow_up.",
            "Diese Signale gehen als Features in das ML-Ranking ein und verhindern, dass sauber abgedeckte Themen erneut hochgerankt werden.",
        ],
    ),
    (
        "8. graphD-Entfernung",
        [
            "GraphD bzw. graph-basierte Entity-Context-Signale sind nicht mehr Teil des produktiven Story-Radar-Scorings.",
            "Im aktuellen Code wurden graph-nahe Features fuer die Inferenz auf 0 gesetzt, damit bestehende Modell-Schnittstellen stabil bleiben, "
            "ohne dass das Signal noch Einfluss hat.",
            "In der UI wurde die kombinierte modellgetriebene Sortieroption als sichtbare Auswahl entfernt, damit kein graph-naher Sortierpfad "
            "mehr prominent angeboten wird.",
        ],
    ),
    (
        "9. Evaluation",
        [
            "Verglichen werden ML, LLM und optional Hybrid ueber precision@10, precision@20, nDCG, Suppression Precision, redaktionelle Akzeptanz, "
            "Latenz und Kosten.",
            "Der faire Vergleich laeuft offline, im Shadow-Mode und anschliessend per Blind-Rating oder Interleaving.",
        ],
    ),
    (
        "10. Ergebnis fuer die Redaktion",
        [
            "Die Redaktion sieht keine Modell-Spielerei, sondern eine kurze priorisierte Liste mit echten Chancen: warum relevant, warum jetzt, "
            "warum Luecke, empfohlener Angle, Confidence und Suppression-Reason falls unterdrueckt.",
        ],
    ),
]


def build_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "StoryRadarTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=HexColor("#111111"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "StoryRadarSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=HexColor("#991B1B"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "StoryRadarBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=HexColor("#222222"),
        spaceAfter=5,
    )

    story = [
        Paragraph(TITLE, title_style),
        Paragraph(
            "Exportierte Kurzfassung des empfohlenen ML-Ansatzes fuer Story Radar. "
            "Stand: April 2026.",
            body_style,
        ),
        Spacer(1, 4 * mm),
    ]

    for heading, paragraphs in SECTIONS:
        story.append(Paragraph(heading, section_style))
        for text in paragraphs:
            story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 2 * mm))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=TITLE,
        author="Codex",
    )
    doc.build(story)
    return output_path


if __name__ == "__main__":
    path = build_pdf(OUTPUT_PATH)
    print(path)
