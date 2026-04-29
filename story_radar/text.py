"""Small text and math helpers for Story Radar."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


STOPWORDS = {
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "und",
    "mit",
    "nach",
    "bei",
    "für",
    "von",
    "auf",
    "im",
    "in",
    "zu",
    "am",
    "vom",
    "des",
    "den",
    "dem",
    "oder",
    "über",
    "bericht",
    "berichts",
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9ÄÖÜäöüß]{3,}", text.lower())
        if token not in STOPWORDS
    ]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def overlap_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def recency_decay(minutes: int, half_life_minutes: float = 180.0) -> float:
    if minutes <= 0:
        return 1.0
    return clamp(math.exp(-minutes / max(half_life_minutes, 1.0)))


def most_common(values: Iterable[str], limit: int = 5) -> list[str]:
    counter = Counter(values)
    return [item for item, _ in counter.most_common(limit)]
