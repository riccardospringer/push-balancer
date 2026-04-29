"""Typed data models for the Story Radar relevance pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal


CoverageStatus = Literal[
    "already_covered",
    "partially_covered",
    "not_covered",
    "angle_gap",
    "follow_up",
]

SuppressionReason = Literal[
    "already_covered",
    "standard_noise",
    "duplicate_cluster",
    "low_confidence",
    "weak_story",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return utcnow()
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return utcnow()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {k: json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


@dataclass(slots=True)
class ClusterDocument:
    document_id: str
    source: str
    title: str
    summary: str = ""
    published_at: datetime = field(default_factory=utcnow)
    url: str = ""


@dataclass(slots=True)
class StoryCluster:
    cluster_id: str
    title: str
    summary: str
    entities: list[str]
    topics: list[str]
    countries: list[str]
    source_count: int
    document_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    documents: list[ClusterDocument] = field(default_factory=list)
    newsroom_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class BildArticle:
    article_id: str
    title: str
    summary: str
    section: str
    tags: list[str]
    entities: list[str]
    published_at: datetime
    updated_at: datetime
    click_index: float
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class PerformanceSnapshot:
    snapshot_id: str
    captured_at: datetime
    section_heat: dict[str, float]
    entity_heat: dict[str, float]
    topic_heat: dict[str, float]
    breaking_mode: bool = False
    consumer_alert_mode: bool = False
    rolling_ctr_index: float = 1.0
    rolling_subscription_index: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class CoverageMatch:
    cluster_id: str
    article_id: str
    coverage_type: str
    title_overlap: float
    entity_overlap: float
    topic_overlap: float
    freshness_delta_minutes: int
    confidence: float
    matched_title: str
    missing_angle_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class GapAssessment:
    cluster_id: str
    coverage_status: CoverageStatus
    gap_score: float
    coverage_confidence: float
    best_match_article_id: str | None
    missing_angle: list[str]
    follow_up_potential: float
    reason: str
    coverage_matches: list[CoverageMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class FeatureBundle:
    cluster_id: str
    feature_values: dict[str, float]
    freshness_score: float
    novelty_score: float
    expected_bild_interest: float
    storyability_score: float
    urgency_score: float
    actionability_score: float
    standard_noise_score: float
    trend_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class MLScore:
    cluster_id: str
    relevance_score: float
    expected_interest: float
    confidence: float
    ranker_version: str
    reasons: list[str]
    score_components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class LLMScore:
    cluster_id: str
    relevance_score: float
    expected_interest: float
    gap_score: float
    urgency_score: float
    confidence: float
    suppressed: bool
    suppressed_reason: str
    why_relevant: str
    why_now: str
    why_gap: str
    recommended_angle: str
    model_name: str
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class FinalRanking:
    cluster_id: str
    model_variant: str
    final_score: float
    final_rank: int
    confidence: float
    suppressed: bool
    suppression_reason: str
    ranking_reason: str
    explainability: list[str]
    tie_breaker_score: float

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class EvaluationLabel:
    cluster_id: str
    label_source: str
    editorial_decision: str
    outcome_label: str
    created_at: datetime
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class FeedbackEvent:
    cluster_id: str
    editor_id: str
    action: str
    created_at: datetime
    notes: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)


@dataclass(slots=True)
class ClusterScorecard:
    cluster: StoryCluster
    gap: GapAssessment
    features: FeatureBundle
    ml_score: MLScore
    llm_score: LLMScore
    rankings: dict[str, FinalRanking]

    def to_dict(self) -> dict[str, Any]:
        return json_ready(self)

