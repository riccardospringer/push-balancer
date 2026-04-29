"""Cluster ingestion and normalization."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from .graphd_removal import assert_graphd_absent, strip_graphd_fields
from .models import ClusterDocument, StoryCluster, parse_datetime


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return normalized[:120]


class ClusterIngestionService:
    """Normalizes incoming cluster payloads and collapses duplicates."""

    def ingest(self, payload: list[dict[str, Any]]) -> list[StoryCluster]:
        sanitized = [strip_graphd_fields(item) for item in payload]
        assert_graphd_absent(sanitized)

        by_signature: dict[str, dict[str, Any]] = {}
        duplicates = defaultdict(int)

        for item in sanitized:
            title = (item.get("title") or "").strip()
            summary = (item.get("summary") or "").strip()
            signature_source = f"{_slug(title)}::{','.join(sorted(item.get('entities', []))[:3])}"
            signature = hashlib.md5(signature_source.encode("utf-8")).hexdigest()
            if signature not in by_signature:
                by_signature[signature] = item
            else:
                duplicates[signature] += 1
                existing = by_signature[signature]
                existing["document_count"] = max(int(existing.get("document_count", 1)), int(item.get("document_count", 1)))
                existing["source_count"] = max(int(existing.get("source_count", 1)), int(item.get("source_count", 1)))
                if len(summary) > len(existing.get("summary", "")):
                    existing["summary"] = summary

        clusters: list[StoryCluster] = []
        for signature, item in by_signature.items():
            doc_payloads = item.get("documents", [])
            documents = [
                ClusterDocument(
                    document_id=str(doc.get("document_id") or doc.get("id") or f"{signature}-{idx}"),
                    source=str(doc.get("source") or "wire"),
                    title=str(doc.get("title") or item.get("title") or ""),
                    summary=str(doc.get("summary") or doc.get("description") or ""),
                    published_at=parse_datetime(doc.get("published_at") or doc.get("publishedAt")),
                    url=str(doc.get("url") or ""),
                )
                for idx, doc in enumerate(doc_payloads)
            ]
            if not documents:
                documents = [
                    ClusterDocument(
                        document_id=f"{signature}-0",
                        source=str(item.get("primary_source") or "wire"),
                        title=str(item.get("title") or ""),
                        summary=str(item.get("summary") or ""),
                        published_at=parse_datetime(item.get("first_seen_at") or item.get("firstSeenAt")),
                    )
                ]

            metadata = dict(item.get("metadata") or {})
            metadata["duplicate_cluster_count"] = duplicates.get(signature, 0)

            clusters.append(
                StoryCluster(
                    cluster_id=str(item.get("cluster_id") or item.get("clusterId") or signature),
                    title=str(item.get("title") or ""),
                    summary=str(item.get("summary") or ""),
                    entities=list(dict.fromkeys(item.get("entities") or [])),
                    topics=list(dict.fromkeys(item.get("topics") or [])),
                    countries=list(dict.fromkeys(item.get("countries") or [])),
                    source_count=max(int(item.get("source_count") or len({doc.source for doc in documents}) or 1), 1),
                    document_count=max(int(item.get("document_count") or len(documents) or 1), 1),
                    first_seen_at=parse_datetime(item.get("first_seen_at") or item.get("firstSeenAt") or documents[0].published_at),
                    last_seen_at=parse_datetime(item.get("last_seen_at") or item.get("lastSeenAt") or documents[-1].published_at),
                    documents=documents,
                    newsroom_labels=list(dict.fromkeys(item.get("newsroom_labels") or item.get("newsroomLabels") or [])),
                    metadata=metadata,
                )
            )
        return sorted(clusters, key=lambda cluster: cluster.last_seen_at, reverse=True)
