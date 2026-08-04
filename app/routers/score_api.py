"""Least-privilege CMS-ID adapter for the existing Render score."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.auth import require_score_key
from app.render_score_capture import (
    CapturedScore,
    RenderScoreUnavailable,
    get_captured_score,
    get_captured_scores_batch,
)

log = logging.getLogger("push-balancer")
router = APIRouter()

_NOT_FOUND_DETAIL = "No current score is available for this CMS ID."
_BATCH_MAX_SIZE = 500
_BATCH_RETRY_AFTER_SECONDS = 1
_BATCH_SOURCE_SLOTS = threading.BoundedSemaphore(2)
BatchCmsId = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{24}$")]


class EngagementScoreBreakdownResponse(BaseModel):
    """Captured numeric inputs from the legacy engagement tooltip."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["engagement"]
    relevance: float = Field(ge=0, le=30)
    urgency: float = Field(ge=0, le=25)
    curiosity: float = Field(ge=0, le=25)
    freshness: float = Field(ge=0, le=20)
    timing: float = Field(ge=0, le=15)
    titleBoost: float = Field(ge=0, le=15)
    breaking: float = Field(ge=0, le=15)
    research: float = Field(ge=0, le=12)
    pushHistory: float = Field(ge=-4, le=8)
    topicSaturation: float = Field(ge=-30, le=0)


class SportScoreBreakdownResponse(BaseModel):
    """Captured numeric inputs from the legacy sport tooltip."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sport"]
    sportRelevance: float = Field(ge=0, le=35)
    timing: float = Field(ge=0, le=30)
    drama: float = Field(ge=0, le=25)
    freshness: float = Field(ge=0, le=10)


ScoreBreakdownResponse = Annotated[
    EngagementScoreBreakdownResponse | SportScoreBreakdownResponse,
    Field(discriminator="kind"),
]

_SCORE_DETAILS_SCHEMA_BRANCHES = [
    {
        "not": {
            "anyOf": [
                {"required": ["scoreBreakdown"]},
                {"required": ["orFactor"]},
            ]
        }
    },
    {
        "required": ["scoreBreakdown", "orFactor"],
        "properties": {
            "scoreBreakdown": {"type": "null"},
            "orFactor": {"type": "null"},
        },
    },
    {
        "required": ["scoreBreakdown", "orFactor"],
        "properties": {
            "scoreBreakdown": {"not": {"type": "null"}},
            "orFactor": {"not": {"type": "null"}},
        },
    },
]


class ArticleScoreResponse(BaseModel):
    """Score contract intentionally excluding article metadata and prose."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "additionalProperties": True,
            "oneOf": _SCORE_DETAILS_SCHEMA_BRANCHES,
        },
    )

    cmsId: str = Field(description="CMS document identifier")
    score: float = Field(
        ge=0,
        le=100,
        description="Authoritative existing Push Balancer total score",
    )
    scoredAt: str = Field(
        min_length=1,
        description="Timestamp of the existing score calculation",
    )
    scoreBreakdown: ScoreBreakdownResponse | None = Field(
        default=None,
        description=(
            "Allowlisted captured numeric explanation values; existing score caps, age "
            "multipliers, and TV adjustments mean they are not guaranteed to sum to score. "
            "Null for a legacy snapshot"
        ),
    )
    orFactor: float | None = Field(
        default=None,
        ge=0.6,
        le=1.5,
        description="Captured opening-rate sorting factor, or null for a legacy snapshot",
    )

    @model_validator(mode="after")
    def details_are_complete_or_absent(self) -> ArticleScoreResponse:
        if (self.scoreBreakdown is None) != (self.orFactor is None):
            raise ValueError("scoreBreakdown and orFactor must be present together")
        return self


class BatchScoreRequest(BaseModel):
    """Bounded, body-only CMS-ID batch request."""

    model_config = ConfigDict(extra="forbid")

    cmsIds: list[BatchCmsId] = Field(min_length=1, max_length=_BATCH_MAX_SIZE)


class BatchFoundScoreResponse(BaseModel):
    """One found score; fields mirror the single lookup plus a discriminator."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": _SCORE_DETAILS_SCHEMA_BRANCHES[1:],
        },
    )

    status: Literal["found"]
    cmsId: BatchCmsId
    score: float = Field(ge=0, le=100)
    scoredAt: str = Field(min_length=1)
    scoreBreakdown: ScoreBreakdownResponse | None
    orFactor: float | None = Field(ge=0.6, le=1.5)

    @model_validator(mode="after")
    def details_are_complete_or_absent(self) -> BatchFoundScoreResponse:
        if (self.scoreBreakdown is None) != (self.orFactor is None):
            raise ValueError("scoreBreakdown and orFactor must be present together")
        return self


class BatchNotFoundScoreResponse(BaseModel):
    """Exact per-position result when no current snapshot exists."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["notFound"]
    cmsId: BatchCmsId


BatchScoreResult = Annotated[
    BatchFoundScoreResponse | BatchNotFoundScoreResponse,
    Field(discriminator="status"),
]


class BatchScoreResponse(BaseModel):
    """Ordered score results and position-based summary counts."""

    model_config = ConfigDict(extra="forbid")

    requestedCount: int = Field(ge=1, le=_BATCH_MAX_SIZE)
    uniqueCount: int = Field(ge=1, le=_BATCH_MAX_SIZE)
    foundCount: int = Field(ge=0, le=_BATCH_MAX_SIZE)
    notFoundCount: int = Field(ge=0, le=_BATCH_MAX_SIZE)
    results: list[BatchScoreResult] = Field(min_length=1, max_length=_BATCH_MAX_SIZE)

    @model_validator(mode="after")
    def counts_match_results(self) -> BatchScoreResponse:
        if self.requestedCount != len(self.results):
            raise ValueError("requestedCount must match results")
        found_count = sum(result.status == "found" for result in self.results)
        if self.foundCount != found_count:
            raise ValueError("foundCount must match results")
        if self.notFoundCount != self.requestedCount - found_count:
            raise ValueError("notFoundCount must match results")
        if not 1 <= self.uniqueCount <= self.requestedCount:
            raise ValueError("uniqueCount must be between one and requestedCount")
        return self


class ProblemResponse(BaseModel):
    """Problem-details shape returned by the shared exception handlers."""

    type: str
    title: str
    status: int
    detail: str
    instance: str


_NO_STORE_RESPONSE_HEADERS = {
    "Cache-Control": {
        "description": "Prevents storage of the CMS-ID lookup response.",
        "schema": {"type": "string", "const": "no-store"},
    },
    "Vary": {
        "description": "Separates responses by the score-only credential.",
        "schema": {"type": "string", "const": "X-Score-Key"},
    },
}

_RETRY_AFTER_RESPONSE_HEADER = {
    "Retry-After": {
        "description": "Seconds before another bounded batch attempt.",
        "schema": {"type": "string", "const": str(_BATCH_RETRY_AFTER_SECONDS)},
    }
}


def _problem_openapi_response(description: str) -> dict[str, Any]:
    return {
        "model": ProblemResponse,
        "description": description,
        "headers": _NO_STORE_RESPONSE_HEADERS,
        "content": {
            "application/problem+json": {
                "schema": {"$ref": "#/components/schemas/ProblemResponse"}
            }
        },
    }


def _captured_score_response(
    cms_id: str,
    captured: CapturedScore,
) -> ArticleScoreResponse:
    return ArticleScoreResponse(
        cmsId=cms_id,
        score=captured.score,
        scoredAt=captured.captured_at,
        scoreBreakdown=(
            _public_score_breakdown(captured.score_breakdown)
            if captured.score_breakdown is not None
            else None
        ),
        orFactor=captured.or_factor,
    )


@router.get(
    "/api/v1/scores/{cms_id}",
    response_model=ArticleScoreResponse,
    operation_id="getScoreByCmsId",
    summary="Get the existing Push Balancer score by CMS ID",
    description=(
        "Returns the latest workday score exposed by the Render score source. Fresh "
        "captured UI scores remain authoritative; the source may use its server-side "
        "candidate fallback when no current capture exists. Allowlisted numeric "
        "explanation fields are included when available."
    ),
    dependencies=[Depends(require_score_key)],
    responses={
        200: {
            "description": "Existing score projection",
            "headers": _NO_STORE_RESPONSE_HEADERS,
        },
        401: _problem_openapi_response("Invalid or missing score API key"),
        404: _problem_openapi_response("No current score is available"),
        422: _problem_openapi_response("Invalid CMS ID"),
        502: _problem_openapi_response("An upstream source is unavailable"),
        503: _problem_openapi_response("Score lookup is not configured or unavailable"),
    },
)
def get_score_by_cms_id(
    response: Response,
    cms_id: str = Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="CMS document identifier",
    ),
) -> ArticleScoreResponse:
    """Project the latest workday score that the unchanged Render UI displayed."""
    try:
        captured = get_captured_score(cms_id)
    except RenderScoreUnavailable as exc:
        raise HTTPException(status_code=502, detail="Render score source is unavailable.") from exc
    except Exception as exc:
        log.exception("Render score projection failed")
        raise HTTPException(status_code=503, detail="Score lookup is unavailable.") from exc

    if captured is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)

    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "X-Score-Key"
    return _captured_score_response(cms_id, captured)


@router.post(
    "/api/v1/scores/batch",
    response_model=BatchScoreResponse,
    operation_id="getScoresByCmsIds",
    summary="Get existing Push Balancer scores for up to 500 CMS IDs",
    description=(
        "Returns one ordered result per requested CMS-ID position while making exactly "
        "one deduplicated batch request to the Render score source. Duplicate CMS IDs "
        "remain duplicated in the public result. The adapter never fans out to single "
        "lookups and does not calculate a score itself."
    ),
    dependencies=[Depends(require_score_key)],
    responses={
        200: {
            "description": "Ordered existing-score projections and not-found results",
            "headers": _NO_STORE_RESPONSE_HEADERS,
        },
        401: _problem_openapi_response("Invalid or missing score API key"),
        422: _problem_openapi_response("Invalid batch request"),
        429: {
            **_problem_openapi_response("At most two batch lookups may run per worker"),
            "headers": _NO_STORE_RESPONSE_HEADERS | _RETRY_AFTER_RESPONSE_HEADER,
        },
        502: _problem_openapi_response("The upstream batch source is unavailable"),
        503: _problem_openapi_response("Score lookup is not configured or unavailable"),
    },
)
def get_scores_by_cms_ids(
    response: Response,
    request: BatchScoreRequest,
) -> BatchScoreResponse:
    """Project a bounded batch through one true upstream batch request."""
    if not _BATCH_SOURCE_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Too many score batch requests are already running.",
            headers={"Retry-After": str(_BATCH_RETRY_AFTER_SECONDS)},
        )

    try:
        unique_normalized_ids = list(
            dict.fromkeys(cms_id.lower() for cms_id in request.cmsIds)
        )
        try:
            source_ids, captures = get_captured_scores_batch(unique_normalized_ids)
        except RenderScoreUnavailable as exc:
            raise HTTPException(
                status_code=502,
                detail="Render score source is unavailable.",
            ) from exc
        except Exception as exc:
            log.error(
                "Render score batch projection failed (%s)",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="Score lookup is unavailable.",
            ) from exc

        if source_ids != unique_normalized_ids or len(captures) != len(source_ids):
            raise HTTPException(
                status_code=502,
                detail="Render score source is unavailable.",
            )
        capture_by_id = dict(zip(source_ids, captures, strict=True))

        results: list[BatchScoreResult] = []
        found_count = 0
        for original_cms_id in request.cmsIds:
            captured = capture_by_id[original_cms_id.lower()]
            if captured is None:
                results.append(
                    BatchNotFoundScoreResponse(
                        status="notFound",
                        cmsId=original_cms_id,
                    )
                )
                continue
            projected = _captured_score_response(original_cms_id, captured)
            results.append(
                BatchFoundScoreResponse(
                    status="found",
                    **projected.model_dump(),
                )
            )
            found_count += 1

        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "X-Score-Key"
        return BatchScoreResponse(
            requestedCount=len(request.cmsIds),
            uniqueCount=len(unique_normalized_ids),
            foundCount=found_count,
            notFoundCount=len(request.cmsIds) - found_count,
            results=results,
        )
    finally:
        _BATCH_SOURCE_SLOTS.release()


def _public_score_breakdown(score_breakdown: Any) -> dict[str, Any]:
    payload = asdict(score_breakdown)
    if payload["kind"] == "engagement":
        payload["titleBoost"] = payload.pop("title_boost")
        payload["pushHistory"] = payload.pop("push_history")
        payload["topicSaturation"] = payload.pop("topic_saturation")
    else:
        payload["sportRelevance"] = payload.pop("sport_relevance")
    return payload
