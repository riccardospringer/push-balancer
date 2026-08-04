from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "push-testwoche-03-08-bis-09-08-2026.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A4
RED = colors.HexColor("#E30613")
BLACK = colors.HexColor("#111111")
INK = colors.HexColor("#242424")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F3F3F3")
PALE_RED = colors.HexColor("#FFF0F1")
WHITE = colors.white


def register_fonts():
    regular_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    regular = next((p for p in regular_candidates if Path(p).exists()), None)
    bold = next((p for p in bold_candidates if Path(p).exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("PB-Regular", regular))
        pdfmetrics.registerFont(TTFont("PB-Bold", bold))
        return "PB-Regular", "PB-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()

styles = {
    "eyebrow": ParagraphStyle(
        "eyebrow",
        fontName=FONT_BOLD,
        fontSize=8.4,
        leading=10,
        textColor=RED,
        spaceAfter=3,
        uppercase=True,
    ),
    "title": ParagraphStyle(
        "title",
        fontName=FONT_BOLD,
        fontSize=23,
        leading=25,
        textColor=BLACK,
        spaceAfter=5,
    ),
    "subtitle": ParagraphStyle(
        "subtitle",
        fontName=FONT,
        fontSize=10,
        leading=13,
        textColor=MUTED,
    ),
    "section": ParagraphStyle(
        "section",
        fontName=FONT_BOLD,
        fontSize=11.5,
        leading=13,
        textColor=BLACK,
        spaceAfter=5,
    ),
    "body": ParagraphStyle(
        "body",
        fontName=FONT,
        fontSize=9.1,
        leading=12.1,
        textColor=INK,
    ),
    "body_small": ParagraphStyle(
        "body_small",
        fontName=FONT,
        fontSize=8.2,
        leading=10.5,
        textColor=INK,
    ),
    "label": ParagraphStyle(
        "label",
        fontName=FONT_BOLD,
        fontSize=8,
        leading=9.5,
        textColor=MUTED,
    ),
    "metric": ParagraphStyle(
        "metric",
        fontName=FONT_BOLD,
        fontSize=12,
        leading=14,
        textColor=BLACK,
        alignment=TA_CENTER,
    ),
    "metric_label": ParagraphStyle(
        "metric_label",
        fontName=FONT,
        fontSize=7.8,
        leading=9.5,
        textColor=MUTED,
        alignment=TA_CENTER,
    ),
    "footer": ParagraphStyle(
        "footer",
        fontName=FONT,
        fontSize=7.3,
        leading=9,
        textColor=MUTED,
        alignment=TA_LEFT,
    ),
}


def P(text, style="body"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph(
        f'<font color="#E30613">●</font>&nbsp;&nbsp;{text}',
        ParagraphStyle(
            f"bullet-{hash(text)}",
            parent=styles["body"],
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=3.5,
        ),
    )


def card(title, items, width):
    content = [P(title, "section")]
    content.extend(bullet(item) for item in items)
    table = Table([[content]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(RED)
    canvas.rect(0, PAGE_H - 7 * mm, PAGE_W, 7 * mm, stroke=0, fill=1)
    canvas.setFillColor(BLACK)
    canvas.rect(0, 0, PAGE_W, 8 * mm, stroke=0, fill=1)
    canvas.setFont(FONT, 7.3)
    canvas.setFillColor(colors.HexColor("#D4D4D4"))
    canvas.drawString(16 * mm, 3.1 * mm, "Push-Testwoche | Arbeitsgrundlage | Stand: 28.07.2026")
    canvas.drawRightString(PAGE_W - 16 * mm, 3.1 * mm, "Push Balancer")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=16 * mm,
    rightMargin=16 * mm,
    topMargin=14 * mm,
    bottomMargin=14 * mm,
    title="Push-Testwoche 03.08.-09.08.2026",
    author="Push Balancer",
    subject="Onepager zur operativen Durchführung der Push-Testwoche",
)
frame = Frame(
    doc.leftMargin,
    doc.bottomMargin,
    doc.width,
    doc.height,
    leftPadding=0,
    rightPadding=0,
    topPadding=0,
    bottomPadding=0,
)
doc.addPageTemplates([PageTemplate(id="onepager", frames=[frame], onPage=draw_page)])

story = [
    Spacer(1, 2 * mm),
    P("PUSH BALANCER · TESTBETRIEB", "eyebrow"),
    P("Push-Testwoche", "title"),
    P(
        "<b>Montag, 03.08. bis Sonntag, 09.08.2026</b><br/>"
        "Eine Woche lang steuern wir die Push-Auswahl und das Timing konsequent "
        "entlang der ML-basierten Empfehlungen des Push Balancers.",
        "subtitle",
    ),
    Spacer(1, 5 * mm),
]

metrics = Table(
    [
        [
            [P("11-15", "metric"), P("Pushes pro Tag, inkl. Sport", "metric_label")],
            [P("06-22 Uhr", "metric"), P("Empfehlungskanal aktiv", "metric_label")],
            [P("OR · Visits", "metric"), P("plus redaktionelle Qualität", "metric_label")],
        ]
    ],
    colWidths=[doc.width / 3] * 3,
)
metrics.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), PALE_RED),
            ("BOX", (0, 0), (-1, -1), 0.6, RED),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#F3B9BD")),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )
)
story.extend([metrics, Spacer(1, 4.5 * mm)])

gap = 4 * mm
col = (doc.width - gap) / 2
left = card(
    "So arbeiten wir",
    [
        "Alle Beteiligten werden in den Teams-Kanal <b>„Push Empfehlungen“</b> eingeladen.",
        "Dort erscheinen vorberechnete Empfehlungen. Diese werden zum empfohlenen Zeitpunkt gepusht.",
        "Bitte ausschließlich den <b>Eilmeldungs-Channel</b> nutzen.",
        "Versand weiterhin über das bisherige Push-Frontend.",
    ],
    col,
)
right = card(
    "Dienste & Verantwortung",
    [
        "<b>06:00 Uhr:</b> Versand durch den Ticker.",
        "<b>Danach bis 22:00 Uhr:</b> René, Knuth und Elisabeth covern die Testwoche in ihren Diensten.",
        "Nur diese drei <b>CvDs dürfen Empfehlungen overrulen</b>.",
        "Auch Sport-Empfehlungen werden durch den <b>Newsroom-CvD</b> gepusht.",
    ],
    col,
)
cards = Table([[left, right]], colWidths=[col, col], hAlign="LEFT")
cards.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]
    )
)
story.extend([cards, Spacer(1, 4.5 * mm)])

override = Table(
    [
        [
            [
                P("Override nur bei klarer Fehlentscheidung", "section"),
                P(
                    "Grundsatz: <b>Der Empfehlung des Push Balancers vertrauen und sie umsetzen.</b> "
                    "Nur die diensthabenden CvDs Elisabeth, Knuth und René dürfen eine eindeutig "
                    "unpassende Empfehlung überstimmen. Der Override wird im Kommunikationskanal "
                    "kurz mit Grund dokumentiert. Danach die Top-2- oder Top-3-Meldung des Push Balancers wählen - oder auf einen Push "
                    "verzichten, wenn keine Meldung sinnvoll ist. Die Tageszielgröße von 11 bis 15 "
                    "Pushes inklusive Sport bleibt dabei im Blick.",
                    "body",
                ),
            ]
        ]
    ],
    colWidths=[doc.width],
)
override.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), PALE_RED),
            ("LINEBEFORE", (0, 0), (0, -1), 3, RED),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
    )
)
story.extend([override, Spacer(1, 4.5 * mm)])

workflow_data = [
    [P("1 · EMPFEHLUNG", "label"), P("2 · PRÜFEN", "label"), P("3 · SENDEN", "label"), P("4 · LERNEN", "label")],
    [
        P("Teams-Kanal beobachten", "body_small"),
        P("CvD prüft und dokumentiert Override", "body_small"),
        P("Zum empfohlenen Zeitpunkt pushen", "body_small"),
        P("OR, Visits und Qualität auswerten", "body_small"),
    ],
]
workflow = Table(workflow_data, colWidths=[doc.width / 4] * 4)
workflow.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BLACK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("BACKGROUND", (0, 1), (-1, 1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
)
story.extend([P("Tagesablauf", "section"), workflow, Spacer(1, 4.5 * mm)])

links = Table(
    [
        [
            P("<b>Auswahl:</b><br/><link href='https://editorial.one/push-balancer/bild/kandidaten' color='#E30613'>editorial.one/push-balancer/bild/kandidaten</link>", "body_small"),
            P("<b>Versand:</b><br/><link href='https://push-frontend.bildcms.de/frontend/' color='#E30613'>push-frontend.bildcms.de/frontend/</link>", "body_small"),
        ]
    ],
    colWidths=[doc.width / 2] * 2,
)
links.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]
    )
)
story.extend(
    [
        links,
        Spacer(1, 3.5 * mm),
        P(
            "<b>Wichtig:</b> Sollten trotz des vereinbarten Push-Stopps Pushes aus dem Sport einlaufen, "
            "bitte direkt Flo Witte informieren. Ein zusätzlicher Teams-Kanal dient dem laufenden "
            "Austausch während der Testwoche. Bei technischen Störungen bitte direkt Riccardo kontaktieren. "
            "Danke an René, Knuth, Elisabeth und den Ticker für die Unterstützung.",
            "footer",
        ),
    ]
)

doc.build(story)
print(OUTPUT)
