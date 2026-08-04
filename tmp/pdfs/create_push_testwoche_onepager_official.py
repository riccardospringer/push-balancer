from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "push-testwoche-03-08-bis-09-08-2026.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A4
RED = colors.HexColor("#D40000")
BLACK = colors.HexColor("#161616")
TEXT = colors.HexColor("#262626")
MID = colors.HexColor("#5C5C5C")
LINE = colors.HexColor("#B8B8B8")
LIGHT = colors.HexColor("#F2F2F2")
WHITE = colors.white


def fonts():
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if Path(regular).exists() and Path(bold).exists():
        pdfmetrics.registerFont(TTFont("Official-Regular", regular))
        pdfmetrics.registerFont(TTFont("Official-Bold", bold))
        return "Official-Regular", "Official-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, BOLD = fonts()


def style(name, **kwargs):
    defaults = dict(fontName=FONT, fontSize=8.6, leading=10.8, textColor=TEXT)
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


S = {
    "kicker": style("kicker", fontName=BOLD, fontSize=8, leading=9, textColor=RED, spaceAfter=3),
    "title": style("title", fontName=BOLD, fontSize=22, leading=24, textColor=BLACK),
    "subtitle": style("subtitle", fontSize=9.5, leading=12, textColor=MID),
    "section": style("section", fontName=BOLD, fontSize=10.3, leading=12, textColor=BLACK),
    "label": style("label", fontName=BOLD, fontSize=7.7, leading=9, textColor=MID),
    "body": style("body"),
    "body_bold": style("body_bold", fontName=BOLD),
    "small": style("small", fontSize=7.5, leading=9.2, textColor=MID),
    "white": style("white", fontName=BOLD, fontSize=8, leading=9.5, textColor=WHITE),
    "number": style("number", fontName=BOLD, fontSize=11, leading=12, textColor=RED),
}


def P(text, key="body"):
    return Paragraph(text, S[key])


def draw_page(canvas, _doc):
    canvas.saveState()
    canvas.setFillColor(BLACK)
    canvas.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(RED)
    canvas.rect(0, PAGE_H - 9.4 * mm, PAGE_W, 1.4 * mm, fill=1, stroke=0)
    canvas.setFillColor(LIGHT)
    canvas.rect(0, 0, PAGE_W, 9 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(15 * mm, 9 * mm, PAGE_W - 15 * mm, 9 * mm)
    canvas.setFillColor(MID)
    canvas.setFont(FONT, 7)
    canvas.drawString(15 * mm, 4.4 * mm, "Interne Arbeitsanweisung | Push Balancer")
    canvas.drawRightString(PAGE_W - 15 * mm, 4.4 * mm, "Stand: 28.07.2026 | Seite 1 von 1")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=15 * mm,
    rightMargin=15 * mm,
    topMargin=15 * mm,
    bottomMargin=14 * mm,
    title="Arbeitsanweisung Push-Testwoche 03.08.-09.08.2026",
    author="Push Balancer",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, 0, 0, 0, 0)
doc.addPageTemplates([PageTemplate(id="official", frames=[frame], onPage=draw_page)])


def section_bar(number, title):
    t = Table([[P(number, "white"), P(title.upper(), "white")]], colWidths=[12 * mm, doc.width - 12 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BLACK),
                ("BACKGROUND", (0, 0), (0, 0), RED),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def grid(rows, widths, header=False):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.55, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for row in range(len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLACK),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


story = [
    Spacer(1, 2 * mm),
    P("ARBEITSANWEISUNG · TESTBETRIEB", "kicker"),
    P("Push-Testwoche", "title"),
    P("Verbindlicher Ablauf für Auswahl, Freigabe und Versand von Push-Mitteilungen", "subtitle"),
    Spacer(1, 4 * mm),
]

meta = grid(
    [
        [P("GELTUNGSZEITRAUM", "label"), P("BETRIEBSZEIT", "label"), P("TAGESZIEL", "label"), P("ERFOLGSMESSUNG", "label")],
        [P("<b>03.08.-09.08.2026</b>"), P("<b>06:00-22:00 Uhr</b>"), P("<b>11-15 Pushes</b><br/>inkl. Sport"), P("<b>OR · Visits</b><br/>Redaktionelle Qualität")],
    ],
    [doc.width / 4] * 4,
    header=True,
)
story.extend([meta, Spacer(1, 4 * mm), section_bar("01", "Grundsatz und Systeme"), Spacer(1, 1.5 * mm)])

systems = grid(
    [
        [P("Vorgabe", "label"), P("Push-Auswahl und Timing erfolgen grundsätzlich nach Empfehlung des Push Balancers.")],
        [P("Empfehlungen", "label"), P('Teams-Kanal <b>„Push Empfehlungen“</b>; Meldungen zum vorberechneten Zeitpunkt umsetzen.')],
        [P("Auswahl", "label"), P("<link href='https://editorial.one/push-balancer/bild/kandidaten' color='#D40000'>editorial.one/push-balancer/bild/kandidaten</link>")],
        [P("Versand", "label"), P("<link href='https://push-frontend.bildcms.de/frontend/' color='#D40000'>push-frontend.bildcms.de/frontend/</link> · ausschließlich Eilmeldungs-Channel")],
    ],
    [36 * mm, doc.width - 36 * mm],
)
story.extend([systems, Spacer(1, 4 * mm), section_bar("02", "Zuständigkeiten"), Spacer(1, 1.5 * mm)])

roles = grid(
    [
        [P("ZEIT / FALL", "label"), P("VERANTWORTLICH", "label"), P("AUFGABE", "label")],
        [P("06:00 Uhr"), P("<b>Ticker</b>"), P("Ersten Push des Tages versenden.")],
        [P("Bis 22:00 Uhr"), P("<b>Elisabeth · Knuth · René</b>"), P("Empfehlungen in den jeweiligen CvD-Diensten umsetzen.")],
        [P("Sport-Empfehlung"), P("<b>Newsroom-CvD</b>"), P("Sport-Push versenden; der Sport pusht in der Testwoche nicht selbst.")],
        [P("Sport pusht trotzdem"), P("<b>Flo Witte</b>"), P("Direkt informieren.")],
        [P("Technische Störung"), P("<b>Riccardo</b>"), P("Direkt kontaktieren.")],
    ],
    [37 * mm, 52 * mm, doc.width - 89 * mm],
    header=True,
)
story.extend([roles, Spacer(1, 4 * mm), section_bar("03", "Standardablauf"), Spacer(1, 1.5 * mm)])

steps = Table(
    [
        [P("1", "number"), P("2", "number"), P("3", "number"), P("4", "number")],
        [P("<b>BEOBACHTEN</b>", "label"), P("<b>PRÜFEN</b>", "label"), P("<b>SENDEN</b>", "label"), P("<b>AUSWERTEN</b>", "label")],
        [
            P("Empfehlung im Teams-Kanal empfangen."),
            P("CvD prüft auf klare Fehlentscheidung."),
            P("Zum empfohlenen Zeitpunkt versenden."),
            P("OR, Visits und redaktionelle Qualität bewerten."),
        ],
    ],
    colWidths=[doc.width / 4] * 4,
)
steps.setStyle(
    TableStyle(
        [
            ("BOX", (0, 0), (-1, -1), 0.55, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
            ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story.extend([steps, Spacer(1, 4 * mm), section_bar("04", "Override-Regel"), Spacer(1, 1.5 * mm)])

override = grid(
    [
        [P("Berechtigt", "label"), P("<b>Ausschließlich Elisabeth, Knuth und René</b> als diensthabende CvDs.")],
        [P("Voraussetzung", "label"), P("Klare Fehlentscheidung der ML-Empfehlung.")],
        [P("Alternative", "label"), P("Top-2- oder Top-3-Meldung wählen oder keinen Push senden, wenn keine Meldung geeignet ist.")],
        [P("Dokumentation", "label"), P("Override und kurzen Grund im Kommunikationskanal festhalten.")],
    ],
    [36 * mm, doc.width - 36 * mm],
)
story.extend([override, Spacer(1, 4 * mm), section_bar("05", "Tagesabschluss"), Spacer(1, 1.5 * mm)])

closing = grid(
    [
        [P("Austausch", "label"), P("Jeden Abend kurze gemeinsame Rückmeldung zum Tagesverlauf.")],
        [P("Prüfpunkte", "label"), P("Umsetzung · Overrides · technische Auffälligkeiten · OR · Visits · redaktionelle Qualität")],
    ],
    [36 * mm, doc.width - 36 * mm],
)
story.extend([closing])

doc.build(story)
print(OUTPUT)
