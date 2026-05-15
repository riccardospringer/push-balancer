"""Push-Schedule Router — schlanker Ersatz fuer den alten Tagesplan."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.push_schedule.service import build_push_schedule

router = APIRouter()


@router.get("/api/push-schedule")
def get_push_schedule(
    date: Optional[str] = Query(None, description="YYYY-MM-DD. Optional, default heute."),
) -> dict:
    """Tages-Pushplan aus PDF-Wochenmatrix + heutige Pushes aus DB."""
    return build_push_schedule(date)
