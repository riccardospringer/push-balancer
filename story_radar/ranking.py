"""Final ranking logic — ML-only."""

from __future__ import annotations

from .models import ClusterScorecard, FinalRanking
from .text import clamp


class RankingEngine:
    variants = ("ml",)

    def rank(self, scorecards: list[ClusterScorecard]) -> list[ClusterScorecard]:
        for scorecard in scorecards:
            scorecard.rankings = self._build_variant_rankings(scorecard)

        for variant in self.variants:
            ordered = sorted(
                scorecards,
                key=lambda item: (
                    item.rankings[variant].suppressed,
                    -item.rankings[variant].final_score,
                    -item.rankings[variant].tie_breaker_score,
                ),
            )
            for rank, scorecard in enumerate(ordered, start=1):
                scorecard.rankings[variant].final_rank = rank
        return scorecards

    def _build_variant_rankings(self, scorecard: ClusterScorecard) -> dict[str, FinalRanking]:
        ml_score = self._ml_variant(scorecard)
        return {"ml": self._make_ranking(scorecard, "ml", ml_score)}

    def _ml_variant(self, scorecard: ClusterScorecard) -> float:
        return clamp(
            0.42 * scorecard.ml_score.relevance_score
            + 0.18 * scorecard.gap.gap_score
            + 0.14 * scorecard.features.freshness_score
            + 0.12 * scorecard.features.novelty_score
            + 0.08 * scorecard.features.actionability_score
            + 0.06 * scorecard.features.storyability_score
        )

    def _make_ranking(self, scorecard: ClusterScorecard, variant: str, score: float) -> FinalRanking:
        suppressed, suppression_reason = self._suppression(scorecard)
        tie_breaker = clamp(
            0.45 * scorecard.gap.gap_score
            + 0.25 * scorecard.features.urgency_score
            + 0.15 * min(scorecard.cluster.source_count / 5.0, 1.0)
            + 0.15 * scorecard.features.storyability_score
        )
        explainability = [
            scorecard.gap.reason,
            *scorecard.features.reasons[:2],
            scorecard.ml_score.score_components.get("scored_by", ""),
        ]
        ranking_reason = " | ".join(dict.fromkeys([item for item in explainability if item][:3]))
        confidence = scorecard.ml_score.confidence
        adjusted_score = score if not suppressed else score * 0.10
        return FinalRanking(
            cluster_id=scorecard.cluster.cluster_id,
            model_variant=variant,
            final_score=round(adjusted_score, 4),
            final_rank=0,
            confidence=round(confidence, 4),
            suppressed=suppressed,
            suppression_reason=suppression_reason,
            ranking_reason=ranking_reason,
            explainability=explainability[:3],
            tie_breaker_score=round(tie_breaker, 4),
        )

    def _suppression(self, scorecard: ClusterScorecard) -> tuple[bool, str]:
        if scorecard.gap.coverage_status == "already_covered" and scorecard.gap.follow_up_potential < 0.55:
            return True, "already_covered"
        if scorecard.features.standard_noise_score >= 0.70 and scorecard.ml_score.relevance_score < 0.60:
            return True, "standard_noise"
        if scorecard.ml_score.confidence < 0.42:
            return True, "low_confidence"
        if min(scorecard.features.storyability_score, scorecard.features.actionability_score) < 0.30:
            return True, "weak_story"
        return False, ""
