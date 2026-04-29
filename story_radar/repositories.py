"""Repository abstractions for Story Radar."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ingestion import ClusterIngestionService
from .models import EvaluationLabel, FeedbackEvent, StoryCluster
from .sample_data import (
    seed_clusters,
    seed_coverage,
    seed_evaluation_labels,
    seed_feedback_events,
    seed_performance_snapshot,
)


@dataclass
class InMemoryStoryRadarRepository:
    clusters: list[StoryCluster] = field(default_factory=list)
    coverage_articles: list = field(default_factory=list)
    performance_snapshot: object | None = None
    evaluation_labels: list[EvaluationLabel] = field(default_factory=list)
    feedback_events: list[FeedbackEvent] = field(default_factory=list)

    @classmethod
    def seeded(cls) -> "InMemoryStoryRadarRepository":
        ingestion = ClusterIngestionService()
        return cls(
            clusters=ingestion.ingest(seed_clusters()),
            coverage_articles=seed_coverage(),
            performance_snapshot=seed_performance_snapshot(),
            evaluation_labels=seed_evaluation_labels(),
            feedback_events=seed_feedback_events(),
        )

    def list_clusters(self) -> list[StoryCluster]:
        return list(self.clusters)

    def replace_clusters(self, clusters: list[StoryCluster]) -> None:
        self.clusters = list(clusters)

    def list_coverage_articles(self) -> list:
        return list(self.coverage_articles)

    def get_performance_snapshot(self):
        return self.performance_snapshot

    def list_evaluation_labels(self) -> list[EvaluationLabel]:
        return list(self.evaluation_labels)

    def list_feedback_events(self) -> list[FeedbackEvent]:
        return list(self.feedback_events)

    def append_feedback(self, event: FeedbackEvent) -> FeedbackEvent:
        self.feedback_events.append(event)
        return event
