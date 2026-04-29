"""
Generiert ein PDF: 'Story Radar ML-Modell für Einsteiger'
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import date


OUTPUT = "/Users/riccardo.longo/editorial-intel/story_radar_ml_erklaerung.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── Farben ───────────────────────────────────────────────────────────────────
BILD_RED     = colors.HexColor("#E3001B")
DARK         = colors.HexColor("#1A1A1A")
MID          = colors.HexColor("#444444")
LIGHT        = colors.HexColor("#F5F5F5")
ACCENT_BLUE  = colors.HexColor("#1A73E8")
ACCENT_GREEN = colors.HexColor("#0F9D58")
ACCENT_AMBER = colors.HexColor("#F4B400")
BORDER       = colors.HexColor("#DDDDDD")
WHITE        = colors.white

# ── Stile ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

H1 = S("H1", fontSize=22, textColor=DARK, leading=28, spaceAfter=6, fontName="Helvetica-Bold")
H2 = S("H2", fontSize=14, textColor=BILD_RED, leading=20, spaceBefore=18, spaceAfter=4, fontName="Helvetica-Bold")
H3 = S("H3", fontSize=11, textColor=DARK, leading=16, spaceBefore=10, spaceAfter=3, fontName="Helvetica-Bold")
BODY = S("BODY", fontSize=9.5, textColor=MID, leading=15, spaceAfter=6,
          fontName="Helvetica", alignment=TA_JUSTIFY)
SMALL = S("SMALL", fontSize=8.5, textColor=MID, leading=13, fontName="Helvetica")
MONO = S("MONO", fontSize=8.5, textColor=DARK, leading=13, fontName="Courier",
          backColor=LIGHT, borderPadding=(3, 5, 3, 5))
CAPTION = S("CAPTION", fontSize=8, textColor=colors.HexColor("#888888"),
             leading=11, fontName="Helvetica-Oblique", alignment=TA_CENTER)
BULLET = S("BULLET", fontSize=9.5, textColor=MID, leading=15, spaceAfter=3,
            fontName="Helvetica", leftIndent=14, firstLineIndent=-10)
CENTER = S("CENTER", fontSize=9.5, textColor=MID, leading=15, fontName="Helvetica",
            alignment=TA_CENTER)
BADGE = S("BADGE", fontSize=8, textColor=WHITE, leading=12, fontName="Helvetica-Bold",
           alignment=TA_CENTER)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6, spaceBefore=6)


def section(title):
    return [
        Spacer(1, 0.3 * cm),
        Paragraph(title, H2),
        HRFlowable(width="100%", thickness=1.5, color=BILD_RED, spaceAfter=6),
    ]


def box(text, bg=LIGHT, text_color=DARK, pad=8):
    tbl = Table([[Paragraph(text, S("bt", fontSize=9.5, textColor=text_color,
                                     leading=14, fontName="Helvetica",
                                     alignment=TA_JUSTIFY))]], colWidths=[PAGE_W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING",   (0, 0), (-1, -1), pad + 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), pad + 4),
    ]))
    return tbl


def two_col(left_items, right_items, left_w=0.52, right_w=0.44):
    w = PAGE_W - 2 * MARGIN
    lw = w * left_w
    rw = w * right_w
    left_tbl  = Table([[item] for item in left_items],  colWidths=[lw])
    right_tbl = Table([[item] for item in right_items], colWidths=[rw])
    outer = Table([[left_tbl, Spacer(0.04 * w, 1), right_tbl]], colWidths=[lw, 0.04 * w, rw])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return outer


def feature_table(rows):
    col_w = [5.5 * cm, 4.0 * cm, PAGE_W - 2 * MARGIN - 5.5 * cm - 4.0 * cm]
    header = [Paragraph(h, S("th", fontSize=8.5, fontName="Helvetica-Bold",
                               textColor=WHITE, leading=12))
              for h in ["Feature", "Wertebereich", "Was misst es?"]]
    data = [header]
    for name, rng, desc in rows:
        data.append([
            Paragraph(name, S("td1", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
            Paragraph(rng,  S("td2", fontSize=8,   fontName="Helvetica", textColor=MID, leading=11)),
            Paragraph(desc, S("td3", fontSize=8.5, fontName="Helvetica", textColor=MID, leading=12)),
        ])
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), DARK),
        ("BACKGROUND",    (0, 2), (-1,  2), LIGHT),
        ("BACKGROUND",    (0, 4), (-1,  4), LIGHT),
        ("BACKGROUND",    (0, 6), (-1,  6), LIGHT),
        ("BACKGROUND",    (0, 8), (-1,  8), LIGHT),
        ("BACKGROUND",    (0,10), (-1, 10), LIGHT),
        ("BACKGROUND",    (0,12), (-1, 12), LIGHT),
        ("BACKGROUND",    (0,14), (-1, 14), LIGHT),
        ("BACKGROUND",    (0,16), (-1, 16), LIGHT),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return tbl


def pipeline_table():
    steps = [
        ("1", "Ingestion",         ACCENT_BLUE,  "Nachrichten-Cluster kommen rein (Titel, Quellen, Entitäten, Themen)"),
        ("2", "Coverage-Check",    ACCENT_AMBER, "Hat BILD das Thema schon? Jaccard/Entity-Overlap mit aktuellen Artikeln"),
        ("3", "Feature-Builder",   ACCENT_GREEN, "17 numerische Signale werden berechnet (topic_fit, freshness, ...)"),
        ("4", "ML-Scorer",         BILD_RED,     "LightGBM LambdaRank bewertet die Relevanz (0–1)"),
        ("5", "LLM-Scorer",        ACCENT_BLUE,  "GPT bewertet redaktionellen Wert, Lücke, Dringlichkeit (0–1)"),
        ("6", "Ranking Engine",    DARK,         "3 Varianten: ML / LLM / Hybrid (55% ML + 45% LLM)"),
        ("7", "Suppression",       colors.HexColor("#888888"), "Schwache Stories werden unterdrückt (noise, already_covered, ...)"),
        ("8", "Dashboard",         ACCENT_GREEN, "Editor sieht sortierte Top-Stories mit Erklärung"),
    ]
    data = []
    for num, name, col, desc in steps:
        badge = Table([[Paragraph(num, BADGE)]], colWidths=[0.65 * cm])
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), col),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 2),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ]))
        data.append([
            badge,
            Paragraph(f"<b>{name}</b>", S("ps", fontSize=9, fontName="Helvetica-Bold",
                                           textColor=DARK, leading=13)),
            Paragraph(desc, S("pd", fontSize=8.5, fontName="Helvetica",
                               textColor=MID, leading=13)),
        ])
    tbl = Table(data, colWidths=[0.9 * cm, 4.0 * cm, PAGE_W - 2 * MARGIN - 4.9 * cm])
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, BORDER),
    ]))
    return tbl


def weights_table(rows, title, avail_w=None):
    if avail_w is None:
        avail_w = PAGE_W - 2 * MARGIN
    data = [[Paragraph(h, S("wh", fontSize=8.5, fontName="Helvetica-Bold",
                             textColor=WHITE, leading=12))
             for h in ["Komponente", "Gewicht"]]]
    for comp, w in rows:
        data.append([
            Paragraph(comp, S("wc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
            Paragraph(w,    S("ww", fontSize=9, fontName="Helvetica-Bold",
                               textColor=BILD_RED, leading=12)),
        ])
    col_w = [avail_w * 0.74, avail_w * 0.26]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), DARK),
        ("BACKGROUND",    (0, 2), (-1,  2), LIGHT),
        ("BACKGROUND",    (0, 4), (-1,  4), LIGHT),
        ("BACKGROUND",    (0, 6), (-1,  6), LIGHT),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([Paragraph(title, H3), tbl])


# ── Dokument aufbauen ─────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    title="Story Radar – ML-Modell für Einsteiger",
    author="Editorial Intelligence",
)

story = []

# ── Titelseite ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 1.5 * cm))

title_tbl = Table(
    [[Paragraph("Story Radar", S("T", fontSize=32, textColor=WHITE, fontName="Helvetica-Bold",
                                  leading=38, alignment=TA_CENTER)),
      Paragraph("ML-Modell", S("T2", fontSize=18, textColor=WHITE, fontName="Helvetica",
                                 leading=24, alignment=TA_CENTER))]],
    colWidths=[PAGE_W - 2 * MARGIN]
)
cover = Table(
    [[Paragraph("Story Radar", S("T", fontSize=36, textColor=WHITE, fontName="Helvetica-Bold",
                                  leading=42, alignment=TA_CENTER))],
     [Paragraph("ML-Modell – Für Einsteiger erklärt", S("T2", fontSize=16, textColor=colors.HexColor("#FFDDDD"),
                                 fontName="Helvetica", leading=22, alignment=TA_CENTER))]],
    colWidths=[PAGE_W - 2 * MARGIN],
)
cover.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), BILD_RED),
    ("TOPPADDING",    (0, 0), (-1, -1), 20),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
    ("LEFTPADDING",   (0, 0), (-1, -1), 20),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
]))
story.append(cover)
story.append(Spacer(1, 0.6 * cm))

subtitle_tbl = Table(
    [[Paragraph(
        f"Editorial Intelligence · Stand {date.today().strftime('%d.%m.%Y')}",
        S("sub", fontSize=10, textColor=colors.HexColor("#888888"), fontName="Helvetica",
          alignment=TA_CENTER, leading=14)
    )]],
    colWidths=[PAGE_W - 2 * MARGIN]
)
story.append(subtitle_tbl)
story.append(Spacer(1, 0.8 * cm))

story.append(box(
    "<b>Was ist das?</b> Story Radar ist ein KI-System, das täglich tausende Nachrichtencluster "
    "aus dem Internet analysiert und bewertet – vollautomatisch. Redakteure sehen am Ende eine "
    "sortierte Liste der relevantesten Geschichten für BILD, mit Erklärung warum jede Story empfohlen wird.",
    bg=colors.HexColor("#FFF3F3"), text_color=DARK
))
story.append(Spacer(1, 0.5 * cm))

# ── 1. Das Problem ─────────────────────────────────────────────────────────────
story += section("1. Das Problem: Welche Story soll BILD als nächstes bringen?")

story.append(Paragraph(
    "Jeden Tag entstehen tausende Nachrichten. Ein Redakteur kann unmöglich alle lesen und entscheiden, "
    "welche davon wirklich relevant für BILD-Leser sind. Das Problem hat drei Teile:",
    BODY
))

prob_data = [
    [Paragraph("Menge", S("ph", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Duplication", S("ph", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Relevanz", S("ph", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))],
    [Paragraph("Hunderte neue Cluster pro Stunde müssen bewertet werden.", SMALL),
     Paragraph("Viele Meldungen sind Duplikate – BILD hat sie schon.", SMALL),
     Paragraph("Was interessiert BILD-Leser wirklich? Das variiert je nach Tageszeit.", SMALL)],
]
prob_tbl = Table(prob_data, colWidths=[(PAGE_W - 2 * MARGIN) / 3] * 3)
prob_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 1), (0, 1), colors.HexColor("#FFF3F3")),
    ("BACKGROUND",    (1, 1), (1, 1), colors.HexColor("#FFF8E1")),
    ("BACKGROUND",    (2, 1), (2, 1), colors.HexColor("#E8F5E9")),
    ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
    ("INNERGRID",     (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
]))
story.append(KeepTogether([prob_tbl]))
story.append(Spacer(1, 0.4 * cm))

story.append(box(
    "<b>Lösung:</b> Ein ML-Ranking-System bewertet jede Story auf einer Skala von 0 bis 1 und sortiert "
    "die Liste automatisch. Redakteure sehen sofort, welche Stories am vielversprechendsten sind.",
    bg=colors.HexColor("#E8F5E9"), text_color=DARK
))

# ── 2. Die Pipeline ────────────────────────────────────────────────────────────
story += section("2. Die Pipeline – von der Nachricht zur Empfehlung")

story.append(Paragraph(
    "Story Radar verarbeitet jeden Cluster in 8 Schritten. Jeder Schritt fügt Informationen hinzu:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))
story.append(KeepTogether([pipeline_table()]))

# ── 3. Features ────────────────────────────────────────────────────────────────
story += section("3. Features – die 'Sinne' des Modells")

story.append(Paragraph(
    "Ein ML-Modell versteht keinen Text direkt. Stattdessen bekommt es eine Liste von <b>Zahlen</b> – "
    "die sogenannten <i>Features</i>. Jedes Feature beschreibt einen bestimmten Aspekt einer Story. "
    "Alle Werte liegen zwischen 0.0 (schwach) und 1.0 (stark).",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

features = [
    ("topic_fit",           "0.0 – 1.0", "Wie gut passt das Thema zu BILD? (Crime=1.0, Regulation=0.28)"),
    ("emotion_score",       "0.0 – 1.0", "Emotionale Intensität (Angriff, Drama, Tote, Festnahme ...)"),
    ("service_score",       "0.0 – 1.0", "Nutzwert für Leser (Bahn, Preise, Wetter, Steuer ...)"),
    ("source_strength",     "0.0 – 1.0", "Wie viele unabhängige Quellen berichten? (6 Quellen = 1.0)"),
    ("document_strength",   "0.0 – 1.0", "Wie viele Artikel gibt es zum Cluster? (10 = 1.0)"),
    ("entity_heat",         "0.0 – 1.0", "Wie 'heiß' sind die erwähnten Personen/Orte gerade?"),
    ("topic_heat",          "0.0 – 1.0", "Wie viel Aufmerksamkeit bekommt das Thema gerade?"),
    ("section_heat",        "0.0 – 1.0", "Wie performt der BILD-Bereich gerade (Sport, Politik, ...)?"),
    ("freshness_score",     "0.0 – 1.0", "Wie aktuell ist die Story? (Exponentieller Zerfall, Halbzeit 150 min)"),
    ("novelty_score",       "0.0 – 1.0", "Wie neu ist das für BILD? (65% Gap-Score + 35% Unsicherheit)"),
    ("gap_score",           "0.0 – 1.0", "Coverage-Lücke: Hat BILD das Thema schon? (0.08=ja, 0.92=nein)"),
    ("follow_up_potential", "0.0 – 1.0", "Lohnt sich ein Update zu einer bereits gedeckten Story?"),
    ("expected_bild_interest","0.0–1.0", "Gewichtete Schätzung des redaktionellen Interesses"),
    ("storyability_score",  "0.0 – 1.0", "Kann man daraus einen guten BILD-Artikel schreiben?"),
    ("urgency_score",       "0.0 – 1.0", "Wie dringend muss die Story jetzt gebracht werden?"),
    ("actionability_score", "0.0 – 1.0", "Hat die Story genug Substanz für einen vollständigen Artikel?"),
    ("standard_noise_score","0.0 – 1.0", "Bürokratisches Rauschen (Ausschüsse, Berichte) – NEGATIVSIGNAL"),
]
story.append(KeepTogether([feature_table(features)]))

story.append(Spacer(1, 0.3 * cm))
story.append(box(
    "<b>Analogie:</b> Stell dir vor, du bewertest einen Film. Anstatt den Film selbst zu schauen, "
    "bekommst du eine Tabelle: Regisseur-Score=0.9, Budget=0.7, Kritiken=0.85, Altersfreigabe=0.3 ... "
    "Genau so sieht das ML-Modell jede Story – als Zahlenvektor.",
    bg=colors.HexColor("#E3F2FD"), text_color=DARK
))

# ── 4. Coverage Check ──────────────────────────────────────────────────────────
story += section("4. Coverage-Check – Hat BILD das schon?")

story.append(Paragraph(
    "Bevor das ML-Modell eine Story bewertet, prüft der <b>CoverageMatcher</b>, ob BILD die Geschichte "
    "bereits abdeckt. Dafür vergleicht er den neuen Cluster mit allen aktuellen BILD-Artikeln.",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

cov_data = [
    [Paragraph("Status", S("ch", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Gap Score", S("ch", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Bedeutung", S("ch", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))],
    [Paragraph("not_covered", S("cm", fontSize=9, fontName="Courier", textColor=ACCENT_GREEN, leading=12)),
     Paragraph("0.92", SMALL),
     Paragraph("BILD hat keinen belastbaren Treffer – sehr empfehlenswert!", SMALL)],
    [Paragraph("angle_gap", S("cm", fontSize=9, fontName="Courier", textColor=ACCENT_AMBER, leading=12)),
     Paragraph("0.76", SMALL),
     Paragraph("BILD berichtet, aber ein neuer Winkel fehlt", SMALL)],
    [Paragraph("follow_up", S("cm", fontSize=9, fontName="Courier", textColor=ACCENT_BLUE, leading=12)),
     Paragraph("0.68", SMALL),
     Paragraph("BILD hat Vorab-Coverage, aber ein frisches Update lohnt sich", SMALL)],
    [Paragraph("partially_covered", S("cm", fontSize=9, fontName="Courier", textColor=MID, leading=12)),
     Paragraph("0.42", SMALL),
     Paragraph("BILD hat es, aber nicht vollständig ausgebaut", SMALL)],
    [Paragraph("already_covered", S("cm", fontSize=9, fontName="Courier", textColor=BILD_RED, leading=12)),
     Paragraph("0.08", SMALL),
     Paragraph("BILD hat das Thema aktuell und konkret – Story wird unterdrückt", SMALL)],
]
cov_tbl = Table(cov_data, colWidths=[4.0 * cm, 2.5 * cm, PAGE_W - 2 * MARGIN - 6.5 * cm])
cov_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
]))
story.append(KeepTogether([cov_tbl]))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "Der Vergleich verwendet <b>Jaccard-Ähnlichkeit</b> (Überlappung von Wörtern) und "
    "<b>Entity-Overlap</b> (gemeinsame Personen/Orte). Ab einer Überlappung von 22% wird ein "
    "Match erkannt und analysiert.",
    BODY
))

# ── 5. Das ML-Modell ───────────────────────────────────────────────────────────
story += section("5. Das ML-Modell: LightGBM mit LambdaRank")

story.append(Paragraph(
    "Das Herzstück von Story Radar ist ein <b>LightGBM-Modell</b> mit dem "
    "<b>LambdaRank-Ziel</b>. Klingt kompliziert – ist es aber nicht:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

lgbm_data = [
    [Paragraph("Begriff", S("lh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Einfach erklärt", S("lh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))],
    [Paragraph("LightGBM", S("lm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Ein sehr schneller Entscheidungsbaum-Algorithmus. Stell dir vor: "
         "ein Baum fragt 'Ist topic_fit > 0.8? Wenn ja, dann ...' – "
         "LightGBM baut 150 solcher Bäume und kombiniert ihre Ergebnisse.", SMALL)],
    [Paragraph("LambdaRank", S("lm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Ein spezielles Lernziel für Sortier-Probleme. Das Modell lernt nicht nur "
         "'ist Story A gut?', sondern 'ist Story A besser als Story B?'. "
         "Ziel ist NDCG@10 – die Top-10 sollen möglichst die besten 10 Stories sein.", SMALL)],
    [Paragraph("NDCG@10", S("lm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Normalized Discounted Cumulative Gain: Eine Metrik die misst, wie gut die "
         "Top-10 Rankings sind. Stories ganz oben bekommen mehr Gewicht als Stories weiter unten. "
         "1.0 = perfekte Reihenfolge.", SMALL)],
    [Paragraph("Fallback-Formel", S("lm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Falls noch kein trainiertes Modell vorliegt, berechnet das System den Score "
         "als gewichtete Summe der Features (0.24×topic_fit + 0.18×interest + ...). "
         "Das ist der 'dynamic_warm_start'-Modus.", SMALL)],
]
lgbm_tbl = Table(lgbm_data, colWidths=[3.8 * cm, PAGE_W - 2 * MARGIN - 3.8 * cm])
lgbm_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(KeepTogether([lgbm_tbl]))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "<b>Training:</b> Das Modell wird mit historischen Daten trainiert. "
    "Jede Trainings-Zeile hat: eine Query-ID (Zeitfenster), die 17 Feature-Werte, und ein Label "
    "(0 = nicht relevant, 1 = BILD hat Story gebracht). LightGBM lernt dann, welche Feature-Kombination "
    "auf gute Stories hinweist.",
    BODY
))

story.append(Spacer(1, 0.3 * cm))
story.append(box(
    "<b>Hyperparameter des Modells:</b>  "
    "objective=lambdarank  |  metric=ndcg@[10,20]  |  learning_rate=0.05  |  "
    "num_leaves=31  |  min_data_in_leaf=20  |  feature_fraction=0.85  |  num_boost_round=150",
    bg=LIGHT, text_color=DARK
))

# ── 6. LLM-Scoring ─────────────────────────────────────────────────────────────
story += section("6. LLM-Scoring – der KI-Redakteur")

story.append(Paragraph(
    "Neben dem ML-Modell gibt es einen zweiten Bewerter: ein <b>Large Language Model (GPT)</b>. "
    "Es liest den Cluster-Titel und die Zusammenfassung und gibt eine strukturierte Bewertung ab:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

llm_items = [
    ("relevance_score", "Wie relevant ist die Story für BILD? (0–1)"),
    ("expected_interest", "Erwartetes Leser-Interesse (0–1)"),
    ("gap_score", "Gibt es eine Coverage-Lücke? (0–1)"),
    ("urgency_score", "Wie dringend ist die Story? (0–1)"),
    ("confidence", "Wie sicher ist das LLM in seiner Bewertung? (0–1)"),
    ("why_relevant", "Text-Erklärung: Warum ist die Story relevant?"),
    ("why_now", "Text-Erklärung: Warum gerade jetzt?"),
    ("why_gap", "Text-Erklärung: Welche Lücke gibt es?"),
    ("recommended_angle", "Empfohlener Artikel-Winkel für BILD"),
    ("suppressed", "Soll die Story unterdrückt werden? (True/False)"),
]
llm_data = [[Paragraph("Ausgabe-Feld", S("lh2", fontSize=9, fontName="Helvetica-Bold",
                                          textColor=WHITE, leading=12)),
             Paragraph("Bedeutung", S("lh2", fontSize=9, fontName="Helvetica-Bold",
                                       textColor=WHITE, leading=12))]]
for field, desc in llm_items:
    llm_data.append([
        Paragraph(field, S("lm2", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
        Paragraph(desc, SMALL),
    ])
llm_tbl = Table(llm_data, colWidths=[4.8 * cm, PAGE_W - 2 * MARGIN - 4.8 * cm])
llm_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("BACKGROUND",    (0, 6), (-1, 6), LIGHT),
    ("BACKGROUND",    (0, 8), (-1, 8), LIGHT),
    ("BACKGROUND",    (0,10), (-1,10), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
]))
story.append(KeepTogether([llm_tbl]))

# ── 7. Ranking Engine ──────────────────────────────────────────────────────────
story += section("7. Ranking Engine – drei Varianten, ein Ergebnis")

story.append(Paragraph(
    "Die Ranking Engine kombiniert ML-Score und LLM-Score zu drei verschiedenen Sortierungen. "
    "Jede Variante gewichtet die Signale anders:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

rank_data = [
    [Paragraph(h, S("rh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))
     for h in ["Variante", "Komponente", "Gewicht"]],
    # ML
    [Paragraph("ML",   S("rv", fontSize=9, fontName="Helvetica-Bold", textColor=ACCENT_BLUE,  leading=12)),
     Paragraph("ml_score.relevance_score", S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("42%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("gap.gap_score",            S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("18%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("freshness_score",          S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("14%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("novelty_score + actionability + storyability", S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("26%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    # LLM
    [Paragraph("LLM",  S("rv", fontSize=9, fontName="Helvetica-Bold", textColor=ACCENT_GREEN, leading=12)),
     Paragraph("llm_score.relevance_score", S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("42%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("llm_score.gap_score",       S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("22%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("llm_score.urgency + storyability + expected_interest", S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("36%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    # Hybrid
    [Paragraph("Hybrid\n(Standard)", S("rv", fontSize=9, fontName="Helvetica-Bold", textColor=DARK, leading=12)),
     Paragraph("ML-Variante",  S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("55%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
    [Paragraph("",     SMALL),
     Paragraph("LLM-Variante", S("rc", fontSize=8.5, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph("45%",  S("rw", fontSize=9, fontName="Helvetica-Bold", textColor=BILD_RED, leading=12))],
]
avail = PAGE_W - 2 * MARGIN
rank_tbl = Table(rank_data, colWidths=[2.5 * cm, avail - 2.5 * cm - 1.8 * cm, 1.8 * cm], repeatRows=1)
rank_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 1), (-1, 4), colors.HexColor("#E8F0FE")),
    ("BACKGROUND",    (0, 5), (-1, 7), colors.HexColor("#E8F5E9")),
    ("BACKGROUND",    (0, 8), (-1, 9), LIGHT),
    ("SPAN",          (0, 1), (0, 4)),
    ("SPAN",          (0, 5), (0, 7)),
    ("SPAN",          (0, 8), (0, 9)),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
]))
story.append(KeepTogether([rank_tbl]))

story.append(Spacer(1, 0.3 * cm))
story.append(box(
    "<b>Tie-Breaker:</b> Bei gleichem Score entscheidet: 45% Gap-Score + 25% Urgency + "
    "15% Quellen-Anzahl + 15% Storyability. "
    "<b>Supprimierte Stories</b> werden auf 10% ihres Scores reduziert und ans Ende verschoben.",
    bg=LIGHT, text_color=DARK
))

# ── 8. Suppression ─────────────────────────────────────────────────────────────
story += section("8. Suppression – wann eine Story unterdrückt wird")

story.append(Paragraph(
    "Manche Stories sind technisch vorhanden, sollen aber nicht empfohlen werden. "
    "Die Suppression-Logik prüft 4 Bedingungen:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

supp_data = [
    [Paragraph("Grund", S("sh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Bedingung", S("sh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))],
    [Paragraph("already_covered", S("sc", fontSize=9, fontName="Courier", textColor=BILD_RED, leading=12)),
     Paragraph("Coverage-Status = 'already_covered' UND follow_up_potential < 0.55", SMALL)],
    [Paragraph("standard_noise", S("sc", fontSize=9, fontName="Courier", textColor=MID, leading=12)),
     Paragraph("standard_noise_score >= 0.70 UND max(ml_score, llm_score) < 0.60", SMALL)],
    [Paragraph("low_confidence", S("sc", fontSize=9, fontName="Courier", textColor=ACCENT_AMBER, leading=12)),
     Paragraph("max(ml_confidence, llm_confidence) < 0.42", SMALL)],
    [Paragraph("weak_story", S("sc", fontSize=9, fontName="Courier", textColor=ACCENT_BLUE, leading=12)),
     Paragraph("min(storyability_score, actionability_score) < 0.30", SMALL)],
]
supp_tbl = Table(supp_data, colWidths=[4.0 * cm, PAGE_W - 2 * MARGIN - 4.0 * cm])
supp_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(KeepTogether([supp_tbl]))

# ── 9. Confidence ──────────────────────────────────────────────────────────────
story += section("9. Confidence – wie sicher ist das Modell?")

story.append(Paragraph(
    "Jede Bewertung enthält einen <b>Confidence-Wert</b> (0–1), der anzeigt, wie zuverlässig "
    "der Score ist. Er wird separat berechnet:",
    BODY
))
story.append(Spacer(1, 0.2 * cm))

conf_data = [
    [Paragraph("Quelle", S("cfh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12)),
     Paragraph("Formel / Logik", S("cfh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))],
    [Paragraph("ML Confidence", S("cfm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Basiswert 0.45 + 0.20 × min(doc_count/8, 1) + 0.15 × min(src_count/5, 1) "
         "+ 0.20 × (1 – noise_score). Mehr Quellen und Dokumente = höhere Sicherheit.", SMALL)],
    [Paragraph("LLM Confidence", S("cfm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "Das LLM gibt selbst einen Konfidenzwert ab. Niedriger Wert = das LLM ist sich "
         "bei dieser Story unsicher (z.B. wenig Kontext, widersprüchliche Signale).", SMALL)],
    [Paragraph("Final Confidence", S("cfm", fontSize=9, fontName="Courier", textColor=DARK, leading=12)),
     Paragraph(
         "50% ML Confidence + 50% LLM Confidence = kombinierter Endwert. "
         "Unter 0.42 wird die Story supprimiert ('low_confidence').", SMALL)],
]
conf_tbl = Table(conf_data, colWidths=[3.8 * cm, PAGE_W - 2 * MARGIN - 3.8 * cm])
conf_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(KeepTogether([conf_tbl]))

# ── 10. Beispiel ───────────────────────────────────────────────────────────────
story += section("10. Beispiel: Eine Story von A bis Z")

story.append(box(
    '<b>Beispiel-Story:</b> "Messerangriff in Hamburg – drei Verletzte"  |  '
    '3 Quellen, 7 Artikel, Entitäten: [Hamburg]  |  BILD hat noch nichts dazu',
    bg=colors.HexColor("#FFF3F3"), text_color=DARK
))
story.append(Spacer(1, 0.3 * cm))

ex_items = [
    ("Coverage-Check", "not_covered → gap_score = 0.92  (BILD hat nichts dazu)"),
    ("topic_fit",      "0.93  (crime-Topic → BILD-Prior = 1.0, mit Quellen-Gewichtung)"),
    ("emotion_score",  "0.67  (Wort 'messer' im Text → 1/3 = 0.33, aufgewertet durch Entitäten)"),
    ("freshness_score","0.88  (Story ist 20 Minuten alt → kaum Zerfall)"),
    ("source_strength","0.50  (3 von 6 Quellen = 50%)"),
    ("ML-Score",       "≈ 0.81  (LambdaRank-Modell oder Fallback-Formel)"),
    ("LLM-Score",      "≈ 0.88  (GPT erkennt: breaking crime, klare Coverage-Lücke, Hamburger Lokalrelevanz)"),
    ("Hybrid-Score",   "≈ 0.84  (0.55 × 0.81 + 0.45 × 0.88)"),
    ("Suppression",    "NEIN  (kein noise, kein already_covered, confidence hoch)"),
    ("Ergebnis",       "Story erscheint auf Rank #1 oder #2 im Dashboard"),
]
ex_data = [[Paragraph(h, S("exh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))
            for h in ["Schritt", "Wert"]]]
for step, val in ex_items:
    ex_data.append([
        Paragraph(step, S("exs", fontSize=9, fontName="Helvetica-Bold", textColor=DARK, leading=13)),
        Paragraph(val,  SMALL),
    ])
ex_tbl = Table(ex_data, colWidths=[3.5 * cm, PAGE_W - 2 * MARGIN - 3.5 * cm])
ex_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("BACKGROUND",    (0, 6), (-1, 6), LIGHT),
    ("BACKGROUND",    (0, 8), (-1, 8), LIGHT),
    ("BACKGROUND",    (0,10), (-1,10), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(KeepTogether([ex_tbl]))

# ── 11. Glossar ────────────────────────────────────────────────────────────────
story += section("11. Kurzglossar")

glossar = [
    ("Cluster",        "Gruppe von Artikeln verschiedener Quellen zum gleichen Thema"),
    ("Feature",        "Eine Zahl die einen Aspekt der Story beschreibt (0–1)"),
    ("LightGBM",       "Schnelles Gradient-Boosting-Verfahren (Entscheidungsbäume)"),
    ("LambdaRank",     "Lernziel speziell für Ranking-Probleme (besser als einfache Regression)"),
    ("NDCG",           "Bewertungsmetrik für Rankings – prämiert gute Stories weit oben"),
    ("LLM",            "Large Language Model (z.B. GPT) – versteht Texte wie ein Mensch"),
    ("Jaccard",        "Ähnlichkeitsmaß: Anteil gemeinsamer Wörter (0=nichts gemeinsam, 1=identisch)"),
    ("Gap Score",      "Wie groß ist die Coverage-Lücke für BILD? (0=schon da, 1=völlig neu)"),
    ("Suppression",    "Story wird aus der Empfehlungsliste entfernt (aber nicht gelöscht)"),
    ("Confidence",     "Wie sicher ist das Modell in seiner Bewertung? (unter 0.42 = unterdrückt)"),
    ("Hybrid",         "Kombination aus ML-Score (55%) und LLM-Score (45%)"),
    ("entity_heat",    "Wie viel Aufmerksamkeit bekommt eine Person/Ort gerade auf BILD?"),
]
glos_data = [[Paragraph(h, S("gh", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, leading=12))
              for h in ["Begriff", "Bedeutung"]]]
for term, defn in glossar:
    glos_data.append([
        Paragraph(term, S("gm", fontSize=9, fontName="Helvetica-Bold", textColor=DARK, leading=13)),
        Paragraph(defn, SMALL),
    ])
glos_tbl = Table(glos_data, colWidths=[3.5 * cm, PAGE_W - 2 * MARGIN - 3.5 * cm])
glos_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), DARK),
    ("BACKGROUND",    (0, 2), (-1, 2), LIGHT),
    ("BACKGROUND",    (0, 4), (-1, 4), LIGHT),
    ("BACKGROUND",    (0, 6), (-1, 6), LIGHT),
    ("BACKGROUND",    (0, 8), (-1, 8), LIGHT),
    ("BACKGROUND",    (0,10), (-1,10), LIGHT),
    ("BACKGROUND",    (0,12), (-1,12), LIGHT),
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(KeepTogether([glos_tbl]))

# ── Fußzeile-Funktion ─────────────────────────────────────────────────────────
def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(MARGIN, 1.4 * cm, "Story Radar – ML-Modell Erklärung | Editorial Intelligence")
    canvas.drawRightString(PAGE_W - MARGIN, 1.4 * cm, f"Seite {doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 1.6 * cm, PAGE_W - MARGIN, 1.6 * cm)
    canvas.restoreState()

doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
print(f"PDF erstellt: {OUTPUT}")
