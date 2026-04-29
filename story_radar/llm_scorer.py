"""Heuristic scorer for Story Radar with legacy LLM request compatibility."""

from __future__ import annotations

import json

from .models import FeatureBundle, GapAssessment, LLMScore, PerformanceSnapshot, StoryCluster
from .text import clamp


LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevance_score": {"type": "number"},
        "expected_interest": {"type": "number"},
        "gap_score": {"type": "number"},
        "urgency_score": {"type": "number"},
        "confidence": {"type": "number"},
        "suppressed": {"type": "boolean"},
        "suppressed_reason": {"type": "string"},
        "why_relevant": {"type": "string"},
        "why_now": {"type": "string"},
        "why_gap": {"type": "string"},
        "recommended_angle": {"type": "string"},
    },
    "required": [
        "relevance_score",
        "expected_interest",
        "gap_score",
        "urgency_score",
        "confidence",
        "suppressed",
        "suppressed_reason",
        "why_relevant",
        "why_now",
        "why_gap",
        "recommended_angle",
    ],
}


class StoryRadarLLMScorer:
    """Feature-based scorer that preserves the former LLM scorer interface."""

    def __init__(self, client: object | None = None, model_name: str = "heuristic"):
        self.client = client
        self.model_name = model_name

    def build_request(
        self,
        cluster: StoryCluster,
        gap: GapAssessment,
        features: FeatureBundle,
        performance: PerformanceSnapshot | None,
    ) -> dict[str, object]:
        payload = {
            "cluster": cluster.to_dict(),
            "gap_assessment": gap.to_dict(),
            "features": features.to_dict(),
            "performance": performance.to_dict() if performance else None,
        }
        return {
            "model": self.model_name,
            "input": [
                {
                    "role": "system",
                    "content": "Return strict JSON matching the provided schema.",
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=True),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "story_radar_llm_score",
                    "strict": True,
                    "schema": LLM_OUTPUT_SCHEMA,
                }
            },
        }

    def score(
        self,
        cluster: StoryCluster,
        gap: GapAssessment,
        features: FeatureBundle,
        performance: PerformanceSnapshot | None,
    ) -> LLMScore:
        suppressed = False
        suppressed_reason = ""
        if gap.coverage_status == "already_covered" and gap.follow_up_potential < 0.55:
            suppressed = True
            suppressed_reason = "already_covered"
        elif features.standard_noise_score >= 0.72:
            suppressed = True
            suppressed_reason = "standard_noise"

        relevance = clamp(
            0.40 * features.expected_bild_interest
            + 0.22 * gap.gap_score
            + 0.18 * features.storyability_score
            + 0.10 * features.urgency_score
            + 0.10 * features.novelty_score
            - 0.18 * features.standard_noise_score
        )
        confidence = clamp(
            0.52
            + 0.18 * min(cluster.source_count / 5.0, 1.0)
            + 0.10 * features.actionability_score
        )

        if relevance >= 0.65:
            why_relevant = "BILD-Thema mit klarer Leserresonanz."
        else:
            why_relevant = "Nur bedingt massentauglich für BILD."

        if features.urgency_score >= 0.60:
            why_now = "Mehrere frische Quellen und hohe Aktualität sprechen für sofortige Bewertung."
        else:
            why_now = "Der Zeitdruck ist moderat, aber noch aktuell genug."

        why_gap = gap.reason

        angle_bits = []
        if gap.missing_angle:
            angle_bits.append("fehlender Winkel: " + ", ".join(gap.missing_angle[:3]))
        if "crime" in [t.lower() for t in cluster.topics]:
            angle_bits.append("konkretisieren: Täter, Ort, Opfer, Fahndung")
        if "consumer" in [t.lower() for t in cluster.topics]:
            angle_bits.append("Service-Angle mit konkreten Folgen für Leser")
        recommended_angle = "; ".join(angle_bits) or "konkreten BILD-Winkel auf Leserfolgen zuspitzen"

        return LLMScore(
            cluster_id=cluster.cluster_id,
            relevance_score=round(relevance, 4),
            expected_interest=round(features.expected_bild_interest, 4),
            gap_score=round(gap.gap_score, 4),
            urgency_score=round(features.urgency_score, 4),
            confidence=round(confidence, 4),
            suppressed=suppressed,
            suppressed_reason=suppressed_reason,
            why_relevant=why_relevant,
            why_now=why_now,
            why_gap=why_gap,
            recommended_angle=recommended_angle,
            model_name=self.model_name,
            raw_payload={},
        )
