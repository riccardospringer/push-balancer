"""Shared model selection for interactive push-title generation."""

from __future__ import annotations


DEFAULT_TITLE_GENERATION_MODEL = "gpt-5.6-luna"
_STALE_TITLE_MODELS = frozenset({"gpt-4o-mini"})


def resolve_title_generation_model(configured_model: str | None) -> str:
    """Replace empty or retired title-model settings with the live default."""
    candidate = str(configured_model or "").strip()
    if not candidate or candidate.casefold() in _STALE_TITLE_MODELS:
        return DEFAULT_TITLE_GENERATION_MODEL
    return candidate
