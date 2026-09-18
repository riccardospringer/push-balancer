"""Shared fixed-slot contract for the scheduled Power Automate transport."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo


BERLIN = ZoneInfo("Europe/Berlin")

POWER_AUTOMATE_PRIMARY_DISPATCH_WINDOW_SECONDS = 5 * 60
POWER_AUTOMATE_MAX_RECOVERY_GRACE_SECONDS = 10 * 60
POWER_AUTOMATE_MAX_DISPATCH_WINDOW_SECONDS = (
    POWER_AUTOMATE_PRIMARY_DISPATCH_WINDOW_SECONDS
    + POWER_AUTOMATE_MAX_RECOVERY_GRACE_SECONDS
)

POWER_AUTOMATE_WEEKDAY_TEAMS_SLOT_LABELS = (
    "07:48",
    "08:24",
    "09:00",
    "10:15",
    "11:00",
    "12:30",
    "14:00",
    "15:45",
    "17:30",
    "18:50",
    "20:05",
    "21:25",
)
POWER_AUTOMATE_WEEKEND_TEAMS_SLOT_LABELS = (
    "08:00",
    "09:00",
    "10:15",
    "11:15",
    "12:30",
    "14:00",
    "15:30",
    "17:30",
    "18:50",
    "20:05",
    "21:25",
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


def power_automate_dispatch_window_seconds(
    recovery_grace_seconds: object,
) -> int:
    """Return the bounded total claim window; invalid extension means none."""
    if isinstance(recovery_grace_seconds, bool):
        recovery = 0
    else:
        try:
            recovery = int(recovery_grace_seconds)
        except (TypeError, ValueError, OverflowError):
            recovery = 0
    if not 0 <= recovery <= POWER_AUTOMATE_MAX_RECOVERY_GRACE_SECONDS:
        recovery = 0
    return POWER_AUTOMATE_PRIMARY_DISPATCH_WINDOW_SECONDS + recovery
