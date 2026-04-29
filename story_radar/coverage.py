"""Coverage matching and gap detection."""

from __future__ import annotations

from datetime import timedelta

from .models import BildArticle, CoverageMatch, GapAssessment, PerformanceSnapshot, StoryCluster
from .text import clamp, jaccard, overlap_ratio, tokenize


class CoverageMatcher:
    """Determine whether BILD already covers a cluster and if a real gap remains."""

    def assess(
        self,
        cluster: StoryCluster,
        coverage_articles: list[BildArticle],
        performance: PerformanceSnapshot | None,
    ) -> GapAssessment:
        matches: list[CoverageMatch] = []
        cluster_title_tokens = tokenize(cluster.title)
        cluster_summary_tokens = tokenize(cluster.summary)
        cluster_topics = [topic.lower() for topic in cluster.topics]
        cluster_entities = [entity.lower() for entity in cluster.entities]
        nowish = cluster.last_seen_at

        for article in coverage_articles:
            article_tokens = tokenize(article.title + " " + article.summary)
            article_topics = [article.section.lower(), *[tag.lower() for tag in article.tags]]
            article_entities = [entity.lower() for entity in article.entities]
            title_overlap = jaccard(cluster_title_tokens, tokenize(article.title))
            entity_overlap = overlap_ratio(cluster_entities, article_entities)
            topic_overlap = max(
                overlap_ratio(cluster_topics, article_topics),
                jaccard(cluster_summary_tokens, article_tokens),
            )
            if max(title_overlap, entity_overlap, topic_overlap) < 0.22:
                continue

            missing_angle_tokens = [
                token
                for token in cluster_title_tokens
                if token not in set(article_tokens) and token not in set(article_entities)
            ][:4]
            freshness_delta = int((nowish - article.updated_at).total_seconds() // 60)
            confidence = clamp(title_overlap * 0.45 + entity_overlap * 0.30 + topic_overlap * 0.25)
            coverage_type = "semantic"
            if title_overlap >= 0.88:
                coverage_type = "exact"
            elif confidence >= 0.72 and not missing_angle_tokens:
                coverage_type = "strong_semantic"
            elif missing_angle_tokens:
                coverage_type = "angle_gap"
            matches.append(
                CoverageMatch(
                    cluster_id=cluster.cluster_id,
                    article_id=article.article_id,
                    coverage_type=coverage_type,
                    title_overlap=round(title_overlap, 4),
                    entity_overlap=round(entity_overlap, 4),
                    topic_overlap=round(topic_overlap, 4),
                    freshness_delta_minutes=freshness_delta,
                    confidence=round(confidence, 4),
                    matched_title=article.title,
                    missing_angle_tokens=missing_angle_tokens,
                )
            )

        matches.sort(key=lambda match: (match.confidence, -match.freshness_delta_minutes), reverse=True)
        best_match = matches[0] if matches else None

        if not best_match:
            return GapAssessment(
                cluster_id=cluster.cluster_id,
                coverage_status="not_covered",
                gap_score=0.92,
                coverage_confidence=0.08,
                best_match_article_id=None,
                missing_angle=[],
                follow_up_potential=0.34,
                reason="BILD hat keinen belastbaren Treffer zum Cluster.",
                coverage_matches=[],
            )

        angle_gap = bool(best_match.missing_angle_tokens)
        is_fresh_update = best_match.freshness_delta_minutes >= 70
        entity_heat = 1.0
        if performance:
            entity_heat = max(
                [performance.entity_heat.get(entity, 1.0) for entity in cluster.entities] or [1.0]
            )
        follow_up_potential = clamp(
            0.35 * (1.0 if is_fresh_update else 0.0)
            + 0.20 * min(cluster.source_count / 5.0, 1.0)
            + 0.20 * min(cluster.document_count / 8.0, 1.0)
            + 0.25 * min(entity_heat / 1.3, 1.0)
        )

        if best_match.coverage_type == "exact" or (
            best_match.confidence >= 0.83 and best_match.freshness_delta_minutes <= 45 and not angle_gap
        ) or (
            best_match.entity_overlap >= 0.95
            and best_match.title_overlap >= 0.18
            and best_match.freshness_delta_minutes <= 30
        ):
            status = "already_covered"
            gap_score = 0.08 if not is_fresh_update else 0.28
            reason = "BILD hat das Thema bereits aktuell und konkret."
        elif angle_gap and (best_match.confidence >= 0.42 or best_match.entity_overlap >= 0.95):
            status = "angle_gap"
            gap_score = 0.76
            reason = "BILD berichtet zum Thema, aber ein klarer Winkel oder das konkrete Update fehlt."
        elif is_fresh_update and follow_up_potential >= 0.55:
            status = "follow_up"
            gap_score = 0.68
            reason = "BILD hat Vorab-Coverage, aber ein frisches Update oder Follow-up lohnt sich."
        else:
            status = "partially_covered"
            gap_score = 0.42
            reason = "BILD hat das Thema bereits, aber noch nicht komplett ausgebaut."

        return GapAssessment(
            cluster_id=cluster.cluster_id,
            coverage_status=status,
            gap_score=round(gap_score, 4),
            coverage_confidence=round(best_match.confidence, 4),
            best_match_article_id=best_match.article_id,
            missing_angle=best_match.missing_angle_tokens,
            follow_up_potential=round(follow_up_potential, 4),
            reason=reason,
            coverage_matches=matches[:5],
        )
