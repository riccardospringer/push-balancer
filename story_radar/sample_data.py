"""Seed data so the Story Radar API can run in this lightweight repo."""

from __future__ import annotations

from .models import BildArticle, EvaluationLabel, FeedbackEvent, PerformanceSnapshot, parse_datetime


def seed_clusters() -> list[dict]:
    return [
        {
            "cluster_id": "crime-hamburg-shooting-01",
            "title": "Schüsse in Hamburg-Altona: Polizei sperrt Bereich ab",
            "summary": "Mehrere Schüsse in einem Wohngebiet. Großeinsatz der Polizei, Hubschrauber im Einsatz. Täter flüchtig, ein Verletzter.",
            "entities": ["Hamburg", "Polizei Hamburg", "Altona"],
            "topics": ["crime", "breaking", "public safety"],
            "countries": ["DE"],
            "source_count": 7,
            "document_count": 12,
            "first_seen_at": "2026-04-29T07:40:00Z",
            "last_seen_at": "2026-04-29T08:05:00Z",
        },
        {
            "cluster_id": "promi-royals-kate-01",
            "title": "Prinzessin Kate überraschend beim Wimbledon-Training gesichtet",
            "summary": "Erste öffentliche Sichtung nach wochenlanger Abwesenheit. Fotos zeigen sie lächelnd am Rande des Courts.",
            "entities": ["Kate Middleton", "Wimbledon", "Königshaus"],
            "topics": ["royals", "promi", "entertainment"],
            "countries": ["GB"],
            "source_count": 9,
            "document_count": 17,
            "first_seen_at": "2026-04-29T09:10:00Z",
            "last_seen_at": "2026-04-29T09:45:00Z",
        },
        {
            "cluster_id": "consumer-strom-preis-01",
            "title": "Strom wird im Mai wieder teurer: diese Anbieter erhöhen zuerst",
            "summary": "Mehrere große Stromanbieter kündigen Preiserhöhungen für Mai an. Verbraucherzentrale warnt und gibt Tipps zum Tarifwechsel.",
            "entities": ["Bundesnetzagentur", "Verbraucherzentrale"],
            "topics": ["consumer", "service", "energy"],
            "countries": ["DE"],
            "source_count": 5,
            "document_count": 8,
            "first_seen_at": "2026-04-29T06:00:00Z",
            "last_seen_at": "2026-04-29T08:30:00Z",
        },
        {
            "cluster_id": "sport-bvb-champions-01",
            "title": "BVB im Champions-League-Halbfinale: Watzke kündigt überraschenden Transfer an",
            "summary": "Nach dem Einzug ins Halbfinale gibt es Gerüchte über einen namhaften Neuzugang. Sportdirektor Kehl dementiert – Watzke nicht.",
            "entities": ["BVB", "Hans-Joachim Watzke", "Champions League"],
            "topics": ["sport", "football", "transfer"],
            "countries": ["DE"],
            "source_count": 6,
            "document_count": 14,
            "first_seen_at": "2026-04-29T08:00:00Z",
            "last_seen_at": "2026-04-29T09:20:00Z",
        },
        {
            "cluster_id": "crime-berlin-knife-01",
            "title": "Messer-Angriff in Berliner U-Bahn: vier Verletzte, Täter flüchtig",
            "summary": "Mehrere große deutsche Outlets berichten über einen Angriff in einer Berliner U-Bahn. Polizei sucht nach dem Täter, Augenzeugen berichten von Panik.",
            "entities": ["Berlin", "Polizei", "U-Bahn"],
            "topics": ["crime", "breaking", "public safety"],
            "countries": ["DE"],
            "source_count": 5,
            "document_count": 9,
            "first_seen_at": "2026-04-24T08:12:00Z",
            "last_seen_at": "2026-04-24T08:31:00Z",
            "documents": [
                {
                    "document_id": "wire-1",
                    "source": "tagesspiegel",
                    "title": "Messer-Angriff in Berliner U-Bahn: vier Verletzte",
                    "summary": "Die Polizei fahndet, mehrere Menschen wurden verletzt.",
                    "published_at": "2026-04-24T08:12:00Z",
                    "url": "https://example.com/wire-1",
                },
                {
                    "document_id": "wire-2",
                    "source": "welt",
                    "title": "Polizei sucht Täter nach Angriff in U-Bahn",
                    "summary": "Augenzeugen berichten von Panik und großer Polizeipräsenz.",
                    "published_at": "2026-04-24T08:26:00Z",
                    "url": "https://example.com/wire-2",
                },
            ],
        },
        {
            "cluster_id": "politik-merz-energie-01",
            "title": "Merz fordert neuen Energie-Plan nach Koalitionsstreit",
            "summary": "Politische Reaktion auf einen aktuellen Koalitionsstreit. Große Reichweite im politischen Berlin, aber bereits umfangreich bei BILD abgebildet.",
            "entities": ["Friedrich Merz", "Bundesregierung"],
            "topics": ["politics", "energy"],
            "countries": ["DE"],
            "source_count": 4,
            "document_count": 6,
            "first_seen_at": "2026-04-24T06:40:00Z",
            "last_seen_at": "2026-04-24T07:55:00Z",
        },
        {
            "cluster_id": "consumer-bahn-strike-01",
            "title": "Bahn-Warnstreik droht am Montag: diese Pendler-Strecken wären zuerst betroffen",
            "summary": "Mehrere Quellen berichten über einen drohenden Warnstreik. Besonders relevant für Pendler in NRW, Hessen und Berlin.",
            "entities": ["Deutsche Bahn", "EVG"],
            "topics": ["consumer", "transport", "service"],
            "countries": ["DE"],
            "source_count": 6,
            "document_count": 11,
            "first_seen_at": "2026-04-24T05:10:00Z",
            "last_seen_at": "2026-04-24T08:18:00Z",
        },
        {
            "cluster_id": "noise-eu-committee-01",
            "title": "EU-Ausschuss vertagt technischen Bericht zu Verpackungsnormen",
            "summary": "Sachliche Ausschussmeldung ohne klare Auswirkungen für BILD-Leser.",
            "entities": ["EU", "Brüssel"],
            "topics": ["policy", "regulation"],
            "countries": ["EU"],
            "source_count": 3,
            "document_count": 4,
            "first_seen_at": "2026-04-24T04:00:00Z",
            "last_seen_at": "2026-04-24T04:45:00Z",
        },
    ]


def seed_coverage() -> list[BildArticle]:
    return [
        BildArticle(
            article_id="bild-100",
            title="Merz greift Regierung im Energie-Chaos frontal an",
            summary="BILD hat das politische Thema bereits prominent auf der Startseite.",
            section="politik",
            tags=["politik", "energie", "merz"],
            entities=["Friedrich Merz", "Bundesregierung"],
            published_at=parse_datetime("2026-04-24T07:02:00Z"),
            updated_at=parse_datetime("2026-04-24T07:54:00Z"),
            click_index=0.88,
            url="https://bild.example.com/100",
        ),
        BildArticle(
            article_id="bild-101",
            title="Bahn-Warnstreik: Das müssen Pendler jetzt wissen",
            summary="BILD berichtet bereits, aber nur als generische Servicemeldung ohne regionale Streckenliste.",
            section="ratgeber",
            tags=["bahn", "warnstreik", "pendler"],
            entities=["Deutsche Bahn"],
            published_at=parse_datetime("2026-04-24T06:40:00Z"),
            updated_at=parse_datetime("2026-04-24T06:40:00Z"),
            click_index=0.62,
            url="https://bild.example.com/101",
        ),
    ]


def seed_performance_snapshot() -> PerformanceSnapshot:
    return PerformanceSnapshot(
        snapshot_id="perf-2026-04-24T08:30Z",
        captured_at=parse_datetime("2026-04-24T08:30:00Z"),
        section_heat={
            "crime": 1.35,
            "politik": 1.05,
            "consumer": 1.22,
            "sport": 1.18,
            "promi": 1.11,
            "regulation": 0.72,
        },
        entity_heat={
            "Berlin": 1.20,
            "Polizei": 1.18,
            "Deutsche Bahn": 1.26,
            "EVG": 1.10,
            "Friedrich Merz": 0.96,
        },
        topic_heat={
            "crime": 1.34,
            "breaking": 1.38,
            "public safety": 1.29,
            "consumer": 1.21,
            "service": 1.19,
            "energy": 0.95,
            "regulation": 0.70,
        },
        breaking_mode=True,
        consumer_alert_mode=True,
        rolling_ctr_index=1.11,
        rolling_subscription_index=1.03,
    )


def seed_evaluation_labels() -> list[EvaluationLabel]:
    return [
        EvaluationLabel(
            cluster_id="crime-berlin-knife-01",
            label_source="editor_blind_rating",
            editorial_decision="show_top_5",
            outcome_label="gap_confirmed",
            created_at=parse_datetime("2026-04-24T09:10:00Z"),
            notes="Hohes Breaking- und BILD-Potenzial.",
        ),
        EvaluationLabel(
            cluster_id="politik-merz-energie-01",
            label_source="editor_blind_rating",
            editorial_decision="suppress",
            outcome_label="already_covered",
            created_at=parse_datetime("2026-04-24T09:10:00Z"),
        ),
    ]


def seed_feedback_events() -> list[FeedbackEvent]:
    return [
        FeedbackEvent(
            cluster_id="consumer-bahn-strike-01",
            editor_id="desk-ratgeber",
            action="angle_requested",
            created_at=parse_datetime("2026-04-24T08:45:00Z"),
            notes="Regionale Streckenliste fehlt noch.",
        )
    ]
