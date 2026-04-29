"""Feature builder for the ML and heuristic rankers."""

from __future__ import annotations

from .models import FeatureBundle, GapAssessment, PerformanceSnapshot, StoryCluster
from .text import clamp, recency_decay, tokenize


BILD_TOPIC_PRIORS = {
    "crime": 1.00,
    "breaking": 0.98,
    "public safety": 0.93,
    "consumer": 0.90,
    "service": 0.88,
    "promi": 0.92,
    "royals": 0.89,
    "sport": 0.87,
    "transfer": 0.90,
    "politics": 0.75,
    "energy": 0.58,
    "regulation": 0.28,
    "policy": 0.32,
    "finance": 0.25,
}

EMOTION_TERMS = {
    "angriff",
    "messer",
    "drama",
    "schock",
    "tod",
    "tote",
    "krise",
    "warnstreik",
    "warnung",
    "festnahme",
    "panik",
}

SERVICE_TERMS = {
    "preise",
    "pendler",
    "rentner",
    "strecken",
    "steuer",
    "krankenkasse",
    "streik",
    "bahn",
    "wetter",
}

BUREAUCRACY_TERMS = {
    "ausschuss",
    "bericht",
    "verpackungsnormen",
    "konsultation",
    "entwurf",
    "komitee",
    "technisch",
}


class FeatureBuilder:
    """Combines cluster, coverage and performance context into dynamic features."""

    def build(
        self,
        cluster: StoryCluster,
        gap: GapAssessment,
        performance: PerformanceSnapshot | None,
    ) -> FeatureBundle:
        cluster_tokens = tokenize(f"{cluster.title} {cluster.summary}")
        age_minutes = int((cluster.last_seen_at - cluster.first_seen_at).total_seconds() // 60)
        freshness_score = recency_decay(max(age_minutes, 0), 150.0)
        prior_scores = [BILD_TOPIC_PRIORS.get(topic.lower(), 0.35) for topic in cluster.topics] or [0.35]
        topic_fit = max(prior_scores)
        emotion_score = clamp(sum(1 for token in cluster_tokens if token in EMOTION_TERMS) / 3.0)
        service_score = clamp(sum(1 for token in cluster_tokens if token in SERVICE_TERMS) / 3.0)
        bureaucracy_score = clamp(sum(1 for token in cluster_tokens if token in BUREAUCRACY_TERMS) / 3.0)
        source_strength = clamp(cluster.source_count / 6.0)
        document_strength = clamp(cluster.document_count / 10.0)
        entity_heat = 0.0
        topic_heat = 0.0
        section_heat = 0.0
        breaking_bonus = 0.0
        if performance:
            entity_heat = max([performance.entity_heat.get(entity, 1.0) for entity in cluster.entities] or [1.0]) - 1.0
            topic_heat = max([performance.topic_heat.get(topic, 1.0) for topic in cluster.topics] or [1.0]) - 1.0
            section_heat = max([performance.section_heat.get(topic, 1.0) for topic in cluster.topics] or [1.0]) - 1.0
            breaking_bonus = 0.12 if performance.breaking_mode and "breaking" in [t.lower() for t in cluster.topics] else 0.0

        expected_interest = clamp(
            0.28 * topic_fit
            + 0.16 * emotion_score
            + 0.16 * service_score
            + 0.12 * source_strength
            + 0.10 * document_strength
            + 0.08 * clamp(entity_heat + 0.5)
            + 0.10 * clamp(topic_heat + section_heat + 0.4)
        )
        novelty_score = clamp(0.65 * gap.gap_score + 0.35 * (1.0 - gap.coverage_confidence))
        storyability_score = clamp(
            0.30 * max(emotion_score, service_score, topic_fit)
            + 0.20 * source_strength
            + 0.20 * document_strength
            + 0.15 * (1.0 if len(cluster.title) <= 110 and len(cluster.summary) >= 60 else 0.45)
            + 0.15 * (1.0 if cluster.entities else 0.4)
        )
        urgency_score = clamp(
            0.45 * freshness_score
            + 0.20 * (1.0 if "breaking" in [t.lower() for t in cluster.topics] else 0.0)
            + 0.15 * source_strength
            + 0.08 * document_strength
            + 0.12 * (breaking_bonus + clamp(entity_heat + 0.25))
        )
        actionability_score = clamp(
            0.30 * (1.0 if len(cluster.summary) >= 50 else 0.4)
            + 0.20 * (1.0 if cluster.entities else 0.45)
            + 0.20 * source_strength
            + 0.20 * gap.gap_score
            + 0.10 * (1.0 if gap.coverage_status != "already_covered" else 0.1)
        )
        standard_noise_score = clamp(
            0.40 * bureaucracy_score
            + 0.20 * (1.0 - topic_fit)
            + 0.20 * (1.0 - max(emotion_score, service_score))
            + 0.20 * (1.0 - source_strength)
        )
        entity_heat_norm = clamp(entity_heat)
        topic_heat_norm = clamp(topic_heat)
        section_heat_norm = clamp(section_heat)
        trend_score = clamp(
            0.45 * entity_heat_norm
            + 0.35 * topic_heat_norm
            + 0.15 * section_heat_norm
            + 0.05 * (breaking_bonus / 0.12 if breaking_bonus else 0.0)
        )
        feature_values = {
            "topic_fit": round(topic_fit, 4),
            "emotion_score": round(emotion_score, 4),
            "service_score": round(service_score, 4),
            "bureaucracy_score": round(bureaucracy_score, 4),
            "source_strength": round(source_strength, 4),
            "document_strength": round(document_strength, 4),
            "entity_heat": round(entity_heat_norm, 4),
            "topic_heat": round(topic_heat_norm, 4),
            "section_heat": round(section_heat_norm, 4),
            "trend_score": round(trend_score, 4),
            "freshness_score": round(freshness_score, 4),
            "novelty_score": round(novelty_score, 4),
            "gap_score": round(gap.gap_score, 4),
            "follow_up_potential": round(gap.follow_up_potential, 4),
            "expected_bild_interest": round(expected_interest, 4),
            "storyability_score": round(storyability_score, 4),
            "urgency_score": round(urgency_score, 4),
            "actionability_score": round(actionability_score, 4),
            "standard_noise_score": round(standard_noise_score, 4),
        }
        reasons = []
        if topic_fit >= 0.85:
            reasons.append("starker BILD-Topic-Fit")
        if emotion_score >= 0.66:
            reasons.append("hohe emotionale Fallhöhe")
        if service_score >= 0.66:
            reasons.append("klarer Nutzwert für Leser")
        if gap.gap_score >= 0.70:
            reasons.append("sichtbare Coverage-Lücke")
        if standard_noise_score >= 0.70:
            reasons.append("Gefahr von Standardrauschen")
        if trend_score >= 0.60:
            reasons.append("hohes Trend-Signal")
        return FeatureBundle(
            cluster_id=cluster.cluster_id,
            feature_values=feature_values,
            freshness_score=round(freshness_score, 4),
            novelty_score=round(novelty_score, 4),
            expected_bild_interest=round(expected_interest, 4),
            storyability_score=round(storyability_score, 4),
            urgency_score=round(urgency_score, 4),
            actionability_score=round(actionability_score, 4),
            standard_noise_score=round(standard_noise_score, 4),
            trend_score=round(trend_score, 4),
            reasons=reasons,
        )
