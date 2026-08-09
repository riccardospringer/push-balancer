"""Shared fixed-slot contract for the scheduled Power Automate transport."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo


BERLIN = ZoneInfo("Europe/Berlin")

POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS = (
    "06:00",
    "06:36",
    "07:12",
    "07:47",
    "08:23",
    "08:59",
    "12:30",
    "17:30",
    "18:49",
    "20:08",
    "21:26",
    "22:45",
)
POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS = (
    "08:00",
    "08:36",
    "09:12",
    "09:47",
    "10:23",
    "10:59",
    "12:30",
    "17:30",
    "18:49",
    "20:08",
    "21:26",
    "22:45",
)


def power_automate_slot_labels_for_date(target_date: dt.date) -> tuple[str, ...]:
    """Return the complete Berlin-local schedule for a calendar date."""
    return (
        POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS
        if target_date.weekday() >= 5
        else POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS
    )


def is_power_automate_binding_slot(slot_ts: int, label: str) -> bool:
    """Validate timestamp and label against the shared fixed-slot schedule."""
    if not isinstance(slot_ts, int) or isinstance(slot_ts, bool) or slot_ts <= 0:
        return False
    try:
        local = dt.datetime.fromtimestamp(slot_ts, BERLIN)
    except (OverflowError, OSError, ValueError):
        return False
    normalized_label = str(label or "").strip()
    return bool(
        local.second == 0
        and local.microsecond == 0
        and normalized_label == local.strftime("%H:%M")
        and normalized_label in power_automate_slot_labels_for_date(local.date())
    )
