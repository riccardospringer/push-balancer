"""ML scorer and ranker training skeleton."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional runtime dependency
    lgb = None

from .models import FeatureBundle, GapAssessment, MLScore, StoryCluster
from .text import clamp


FEATURE_ORDER = [
    "topic_fit",
    "emotion_score",
    "service_score",
    "source_strength",
    "document_strength",
    "entity_heat",
    "topic_heat",
    "section_heat",
    "trend_score",
    "freshness_score",
    "novelty_score",
    "gap_score",
    "follow_up_potential",
    "expected_bild_interest",
    "storyability_score",
    "urgency_score",
    "actionability_score",
    "standard_noise_score",
]


@dataclass(slots=True)
class TrainingRow:
    query_id: str
    cluster_id: str
    features: dict[str, float]
    label: float


class StoryRadarMLScorer:
    """Warm-start ML scorer with an optional LightGBM lambdarank model."""

    def __init__(self, model=None, ranker_version: str = "story-radar-ltr-v1"):
        self.model = model
        self.ranker_version = ranker_version

    def score(self, cluster: StoryCluster, features: FeatureBundle, gap: GapAssessment) -> MLScore:
        if self.model is not None:
            score = self._model_score(features.feature_values)
            scorer = "lightgbm_lambdarank"
        else:
            score = self._fallback_score(features, gap)
            scorer = "dynamic_warm_start"

        expected_interest = clamp(
            0.60 * features.expected_bild_interest
            + 0.20 * features.storyability_score
            + 0.20 * features.urgency_score
        )
        confidence = clamp(
            0.45
            + 0.20 * min(cluster.document_count / 8.0, 1.0)
            + 0.15 * min(cluster.source_count / 5.0, 1.0)
            + 0.20 * (1.0 - features.standard_noise_score)
        )
        score_components = {
            "predicted_relevance": round(score, 4),
            "expected_interest": round(expected_interest, 4),
            "gap_score": round(gap.gap_score, 4),
            "novelty_score": round(features.novelty_score, 4),
            "urgency_score": round(features.urgency_score, 4),
            "standard_noise_penalty": round(features.standard_noise_score, 4),
        }
        reasons = list(features.reasons)
        reasons.append(f"scored_by={scorer}")
        return MLScore(
            cluster_id=cluster.cluster_id,
            relevance_score=round(score, 4),
            expected_interest=round(expected_interest, 4),
            confidence=round(confidence, 4),
            ranker_version=f"{self.ranker_version}:{scorer}",
            reasons=reasons,
            score_components=score_components,
        )

    def _model_score(self, feature_values: dict[str, float]) -> float:
        vector = [[feature_values.get(name, 0.0) for name in FEATURE_ORDER]]
        prediction = self.model.predict(vector)[0]
        if prediction > 1.0 or prediction < 0.0:
            prediction = 1.0 / (1.0 + math.exp(-prediction))
        return clamp(float(prediction))

    def _fallback_score(self, features: FeatureBundle, gap: GapAssessment) -> float:
        raw = (
            0.24 * features.feature_values["topic_fit"]
            + 0.18 * features.expected_bild_interest
            + 0.18 * gap.gap_score
            + 0.12 * features.novelty_score
            + 0.10 * features.freshness_score
            + 0.10 * features.urgency_score
            + 0.08 * features.actionability_score
            + 0.08 * features.storyability_score
            - 0.20 * features.standard_noise_score
        )
        if gap.coverage_status == "already_covered":
            raw -= 0.32
        return clamp(raw)

    def fit(self, rows: Iterable[TrainingRow]):
        """Train a LightGBM ranker once labeled offline data is available."""
        row_list = list(rows)
        if not row_list or lgb is None:
            return None

        query_groups: list[int] = []
        current_query = None
        for row in row_list:
            if row.query_id != current_query:
                query_groups.append(0)
                current_query = row.query_id
            query_groups[-1] += 1

        dataset = lgb.Dataset(
            data=[[row.features.get(name, 0.0) for name in FEATURE_ORDER] for row in row_list],
            label=[row.label for row in row_list],
            group=query_groups,
            feature_name=FEATURE_ORDER,
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 20],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.85,
            "verbosity": -1,
        }
        self.model = lgb.train(params, dataset, num_boost_round=150)
        return self.model
