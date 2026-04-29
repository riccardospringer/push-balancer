"""Story Radar application service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .coverage import CoverageMatcher
from .evaluation import build_evaluation_snapshot
from .features import FeatureBuilder
from .ingestion import ClusterIngestionService
from .llm_scorer import StoryRadarLLMScorer
from .ml_scorer import StoryRadarMLScorer
from .models import ClusterScorecard, FeedbackEvent, json_ready, parse_datetime
from .ranking import RankingEngine
from .repositories import InMemoryStoryRadarRepository


class StoryRadarService:
    def __init__(
        self,
        repository: InMemoryStoryRadarRepository | None = None,
        ingestion: ClusterIngestionService | None = None,
        coverage_matcher: CoverageMatcher | None = None,
        feature_builder: FeatureBuilder | None = None,
        ml_scorer: StoryRadarMLScorer | None = None,
        llm_scorer: StoryRadarLLMScorer | None = None,
        ranking_engine: RankingEngine | None = None,
    ):
        self.repository = repository or InMemoryStoryRadarRepository.seeded()
        self.ingestion = ingestion or ClusterIngestionService()
        self.coverage_matcher = coverage_matcher or CoverageMatcher()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.ml_scorer = ml_scorer or StoryRadarMLScorer()
        self.llm_scorer = llm_scorer or StoryRadarLLMScorer()
        self.ranking_engine = ranking_engine or RankingEngine()
        self._last_scorecards: list[ClusterScorecard] = []
        self._rescore()

    def list_clusters(self) -> list[dict[str, Any]]:
        if not self._last_scorecards:
            self._rescore()
        return [card.to_dict() for card in self._last_scorecards]

    def get_ranked(self, model_variant: str = "ml", include_suppressed: bool = False) -> dict[str, Any]:
        # Only "ml" variant exists now; accept legacy names for backwards compat
        model_variant = "ml"
        if not self._last_scorecards:
            self._rescore()
        ranked = sorted(self._last_scorecards, key=lambda card: card.rankings[model_variant].final_rank)
        items = []
        for card in ranked:
            ranking = card.rankings[model_variant]
            if ranking.suppressed and not include_suppressed:
                continue
            items.append(
                {
                    "cluster": card.cluster.to_dict(),
                    "gap_assessment": card.gap.to_dict(),
                    "features": card.features.to_dict(),
                    "ml_score": card.ml_score.to_dict(),
                    "explanations": card.llm_score.to_dict(),
                    "ranking": ranking.to_dict(),
                    "variants": {name: r.to_dict() for name, r in card.rankings.items()},
                }
            )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_variant": model_variant,
            "count": len(items),
            "items": items,
        }

    def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        for card in self._last_scorecards:
            if card.cluster.cluster_id == cluster_id:
                return card.to_dict()
        return None

    def get_explanation(self, cluster_id: str) -> dict[str, Any] | None:
        cluster = self.get_cluster(cluster_id)
        if not cluster:
            return None
        expl = cluster.get("explanations") or cluster.get("llm_score", {})
        return {
            "cluster_id": cluster_id,
            "why_relevant": expl.get("why_relevant", ""),
            "why_now": expl.get("why_now", ""),
            "why_gap": expl.get("why_gap", ""),
            "recommended_angle": expl.get("recommended_angle", ""),
            "ranking_reasons": {"ml": cluster["rankings"]["ml"]["ranking_reason"]},
        }

    def get_evaluation(self) -> dict[str, Any]:
        return build_evaluation_snapshot(
            self._last_scorecards,
            self.repository.list_evaluation_labels(),
            self.repository.list_feedback_events(),
        )

    def get_debug_coverage(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": [
                {
                    "cluster_id": card.cluster.cluster_id,
                    "coverage_status": card.gap.coverage_status,
                    "gap_score": card.gap.gap_score,
                    "coverage_confidence": card.gap.coverage_confidence,
                    "matches": [match.to_dict() for match in card.gap.coverage_matches],
                }
                for card in self._last_scorecards
            ],
        }

    def get_suppressed(self) -> dict[str, Any]:
        items = []
        for card in self._last_scorecards:
            if card.rankings["hybrid"].suppressed:
                items.append(
                    {
                        "cluster_id": card.cluster.cluster_id,
                        "title": card.cluster.title,
                        "suppression_reason": card.rankings["hybrid"].suppression_reason,
                        "ml_score": card.ml_score.relevance_score,
                    }
                )
        return {"count": len(items), "items": items}

    def train(self) -> dict[str, Any]:
        labels: dict[str, float] = {}
        for event in self.repository.list_feedback_events():
            if event.action in {"picked_up", "gap_confirmed", "assigned", "angle_requested"}:
                labels[event.cluster_id] = 1.0
            elif event.action in {"dismissed", "suppressed", "noise"}:
                labels[event.cluster_id] = 0.0
        for label in self.repository.list_evaluation_labels():
            if label.outcome_label in {"picked_up", "gap_confirmed", "assigned"}:
                labels[label.cluster_id] = 1.0
            elif label.outcome_label in {"noise", "duplicate", "already_covered"}:
                labels[label.cluster_id] = 0.0

        if not labels:
            return {"ok": False, "error": "No labeled data available for training.", "trained_rows": 0}

        from .ml_scorer import TrainingRow
        rows: list[TrainingRow] = []
        for card in self._last_scorecards:
            cid = card.cluster.cluster_id
            if cid in labels:
                rows.append(TrainingRow(
                    query_id="default",
                    cluster_id=cid,
                    features=card.features.feature_values,
                    label=labels[cid],
                ))

        if not rows:
            return {"ok": False, "error": "Labeled clusters not found in current scorecards.", "trained_rows": 0}

        model = self.ml_scorer.fit(rows)
        if model is None:
            return {"ok": False, "error": "Training failed — LightGBM not available or too few rows.", "trained_rows": len(rows)}

        self._rescore()
        return {"ok": True, "trained_rows": len(rows), "scorer": "lightgbm_lambdarank"}

    def rescore(self, cluster_payloads: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if cluster_payloads is not None:
            clusters = self.ingestion.ingest(cluster_payloads)
            self.repository.replace_clusters(clusters)
        self._rescore()
        return self.get_ranked("ml", include_suppressed=True)

    def submit_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = FeedbackEvent(
            cluster_id=str(payload["cluster_id"]),
            editor_id=str(payload["editor_id"]),
            action=str(payload["action"]),
            created_at=parse_datetime(payload.get("created_at")),
            notes=str(payload.get("notes") or ""),
            payload={k: v for k, v in payload.items() if k not in {"cluster_id", "editor_id", "action", "created_at", "notes"}},
        )
        self.repository.append_feedback(event)
        return event.to_dict()

    def _rescore(self) -> None:
        clusters = self.repository.list_clusters()
        coverage_articles = self.repository.list_coverage_articles()
        performance = self.repository.get_performance_snapshot()
        scorecards: list[ClusterScorecard] = []

        for cluster in clusters:
            gap = self.coverage_matcher.assess(cluster, coverage_articles, performance)
            features = self.feature_builder.build(cluster, gap, performance)
            ml_score = self.ml_scorer.score(cluster, features, gap)
            llm_score = self.llm_scorer.score(cluster, gap, features, performance)
            scorecards.append(
                ClusterScorecard(
                    cluster=cluster,
                    gap=gap,
                    features=features,
                    ml_score=ml_score,
                    llm_score=llm_score,
                    rankings={},
                )
            )
        self._last_scorecards = self.ranking_engine.rank(scorecards)

    def export_state(self) -> dict[str, Any]:
        return json_ready({"scorecards": [card.to_dict() for card in self._last_scorecards]})
