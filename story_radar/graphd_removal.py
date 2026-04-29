"""Utilities to guarantee graphD no longer participates in Story Radar."""

from __future__ import annotations

from typing import Any


GRAPH_D_FIELD_ALIASES = {
    "graphd",
    "graph_d",
    "graphd_score",
    "graphD_score",
    "graphD",
    "graphdScore",
    "graphDScore",
    "graphd_rank",
    "graphd_reason",
    "graphd_explanation",
}


def strip_graphd_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, raw_val in value.items():
            if key in GRAPH_D_FIELD_ALIASES or "graphd" in key.lower():
                continue
            cleaned[key] = strip_graphd_fields(raw_val)
        return cleaned
    if isinstance(value, list):
        return [strip_graphd_fields(item) for item in value]
    return value


def find_graphd_fields(value: Any, prefix: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, raw_val in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in GRAPH_D_FIELD_ALIASES or "graphd" in key.lower():
                matches.append(path)
            matches.extend(find_graphd_fields(raw_val, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]"
            matches.extend(find_graphd_fields(item, path))
    return matches


def assert_graphd_absent(value: Any) -> None:
    matches = find_graphd_fields(value)
    if matches:
        raise ValueError(f"graphD fields detected in Story Radar payload: {', '.join(matches)}")
