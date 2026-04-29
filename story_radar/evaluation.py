"""Offline and shadow-mode evaluation helpers."""

from __future__ import annotations

import math
from statistics import median

from .models import ClusterScorecard, EvaluationLabel, FeedbackEvent


POSITIVE_OUTCOMES = {"gap_confirmed", "picked_up", "angle_requested", "assigned"}
SUPPRESSION_GOOD_OUTCOMES = {"already_covered", "noise", "duplicate"}


def precision_at_k(scorecards: list[ClusterScorecard], labels: dict[str, str], variant: str, k: int) -> float:
    ranked = [
        card
        for card in sorted(scorecards, key=lambda item: item.rankings[variant].final_rank)
        if not card.rankings[variant].suppressed
    ][:k]
    if not ranked:
        return 0.0
    hits = sum(1 for card in ranked if labels.get(card.cluster.cluster_id) in POSITIVE_OUTCOMES)
    return round(hits / len(ranked), 4)


def ndcg(scorecards: list[ClusterScorecard], labels: dict[str, str], variant: str, k: int) -> float:
    relevance_map = {
        "gap_confirmed": 3,
        "picked_up": 3,
        "angle_requested": 2,
        "assigned": 2,
        "already_covered": 0,
        "noise": 0,
        "duplicate": 0,
    }
    ranked = sorted(scorecards, key=lambda item: item.rankings[variant].final_rank)[:k]
    if not ranked:
        return 0.0

    def dcg(items):
        score = 0.0
        for idx, card in enumerate(items, start=1):
            rel = relevance_map.get(labels.get(card.cluster.cluster_id, ""), 0)
            score += (2**rel - 1) / math.log2(idx + 1)
        return score

    actual = dcg(ranked)
    ideal = dcg(sorted(ranked, key=lambda card: relevance_map.get(labels.get(card.cluster.cluster_id, ""), 0), reverse=True))
    if ideal == 0:
        return 0.0
    return round(actual / ideal, 4)


def suppression_precision(scorecards: list[ClusterScorecard], labels: dict[str, str], variant: str) -> float:
    suppressed = [card for card in scorecards if card.rankings[variant].suppressed]
    if not suppressed:
        return 0.0
    hits = sum(1 for card in suppressed if labels.get(card.cluster.cluster_id) in SUPPRESSION_GOOD_OUTCOMES)
    return round(hits / len(suppressed), 4)


def build_evaluation_snapshot(
    scorecards: list[ClusterScorecard],
    labels: list[EvaluationLabel],
    feedback: list[FeedbackEvent],
) -> dict:
    label_lookup = {label.cluster_id: label.outcome_label for label in labels}
    feedback_positive = [event for event in feedback if event.action in {"picked", "angle_requested", "assigned"}]
    editorial_accept_rate = round(
        len(feedback_positive) / len(feedback), 4
    ) if feedback else 0.0

    summary = {"variants": {}, "feedback": {"editorial_accept_rate": editorial_accept_rate, "feedback_count": len(feedback)}}
    for variant in ("ml", "llm", "hybrid"):
        summary["variants"][variant] = {
            "precision_at_10": precision_at_k(scorecards, label_lookup, variant, 10),
            "precision_at_20": precision_at_k(scorecards, label_lookup, variant, 20),
            "ndcg_at_10": ndcg(scorecards, label_lookup, variant, 10),
            "suppression_precision": suppression_precision(scorecards, label_lookup, variant),
            "median_rank_confidence": round(
                median([card.rankings[variant].confidence for card in scorecards]) if scorecards else 0.0,
                4,
            ),
        }
    return summary
