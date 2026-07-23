"""
app/routers/score_capture.py — Score-Snapshot-Endpoint.

POST /api/score-capture  — empfängt Artikel-Scores vom Kandidaten-Tab (Browser)
GET  /api/score-capture/by-cms-id/{cms_id} — gibt einen gespeicherten Score zurück
POST /api/score-capture/by-cms-id/batch — liest 1–500 gespeicherte Scores gebündelt

Der Browser sendet alle Kandidaten-Scores aus dem unveränderten Standardfilter
alle 30s wenn der Kandidaten-Tab offen ist.
Wenn ein Push rausgeht, wird der zuletzt gespeicherte Score für diese URL genutzt.
Persistenz-TTL: 8 Stunden; der CMS-ID-Endpunkt nutzt dieses Arbeitsfenster,
damit ein bereits angezeigter Original-Score nicht nach drei Minuten verschwindet.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

router = APIRouter()

_lock = threading.Lock()
# url_hash → {score, ts, url}
_score_cache: dict[str, dict] = {}
_CACHE_TTL = 8 * 3600  # 8 Stunden (ein Arbeitstag)
_CMS_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_BILD_HOSTS = frozenset({"bild.de", "www.bild.de"})
_MAX_FUTURE_SKEW_SECONDS = 30


class ScoreCaptureReadError(RuntimeError):
    """A complete score-source read could not be performed."""


class EngagementScoreBreakdown(BaseModel):
    """Numeric values displayed or applied by the candidate UI."""

    model_config = ConfigDict(extra="forbid", strict=True)

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


class SportScoreBreakdown(BaseModel):
    """Numeric values displayed or applied by the candidate UI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["sport"]
    sportRelevance: float = Field(ge=0, le=35)
    timing: float = Field(ge=0, le=30)
    drama: float = Field(ge=0, le=25)
    freshness: float = Field(ge=0, le=10)


ScoreBreakdown = Annotated[
    EngagementScoreBreakdown | SportScoreBreakdown,
    Field(discriminator="kind"),
]
_SCORE_BREAKDOWN_ADAPTER = TypeAdapter(ScoreBreakdown)


def _validated_enrichment(entry: dict) -> tuple[dict, float] | None:
    """Return a complete validated enrichment pair or ignore it as legacy data."""
    breakdown_raw = entry.get("scoreBreakdown", entry.get("score_breakdown"))
    or_factor_raw = entry.get("orFactor", entry.get("or_factor"))
    if breakdown_raw is None or or_factor_raw is None:
        return None
    if isinstance(or_factor_raw, bool) or not isinstance(or_factor_raw, (int, float)):
        return None
    or_factor = float(or_factor_raw)
    if not math.isfinite(or_factor) or not 0.6 <= or_factor <= 1.5:
        return None
    try:
        breakdown = _SCORE_BREAKDOWN_ADAPTER.validate_python(breakdown_raw)
    except ValueError:
        return None
    return breakdown.model_dump(), or_factor


def _cleanup():
    cutoff = time.time() - _CACHE_TTL
    dead = [k for k, v in _score_cache.items() if v["ts"] < cutoff]
    for k in dead:
        del _score_cache[k]


def _normalize_url(url: str) -> str:
    """Normalisiert URL für konsistentes Matching zwischen Sitemap und CMS."""
    url = url.strip().rstrip("/").lower()
    # Query-Parameter und Fragment entfernen
    url = url.split("?")[0].split("#")[0]
    # Tracking-Suffixe entfernen (bild.de spezifisch)
    for suffix in (".bild.html", ".html"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _cms_ids_in_trusted_url(url: object) -> set[str]:
    """Extract exact CMS-ID path tokens only from a trusted canonical BILD URL."""
    if not isinstance(url, str):
        return set()
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError:
        return set()
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _BILD_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        return set()
    return set(
        re.findall(
            r"(?<![0-9a-f])([0-9a-f]{24})(?![0-9a-f])",
            parsed.path.lower(),
        )
    )


def _url_matches_cms_id(url: str, cms_id: str) -> bool:
    """Matcht eine CMS-ID ausschließlich im Pfad einer vertrauenswürdigen BILD-URL."""
    return bool(
        _CMS_ID_RE.fullmatch(cms_id)
        and cms_id.lower() in _cms_ids_in_trusted_url(url)
    )


def _fresh_minimal_snapshot(
    entry: dict,
    *,
    now: int,
    max_age_seconds: int,
) -> dict[str, object] | None:
    """Validiert einen Capture-Eintrag und reduziert ihn auf den Consumer-Vertrag."""
    score_raw = entry.get("score")
    captured_at_raw = entry.get("ts", entry.get("captured_at"))
    if isinstance(score_raw, bool) or isinstance(captured_at_raw, bool):
        return None
    try:
        score = float(score_raw)
        captured_at = int(captured_at_raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(score)
        or not 0 < score <= 100
        or captured_at <= 0
        or captured_at > now + _MAX_FUTURE_SKEW_SECONDS
        or now - captured_at >= max_age_seconds
    ):
        return None
    snapshot: dict[str, object] = {"score": score, "capturedAt": captured_at}
    enrichment = _validated_enrichment(entry)
    if enrichment is not None:
        snapshot["scoreBreakdown"], snapshot["orFactor"] = enrichment
    return snapshot


def get_score_snapshot_for_url(
    url: str,
    *,
    max_age_seconds: int = _CACHE_TTL,
    allow_db_fallback: bool = True,
) -> dict[str, float | int | str] | None:
    """Liefert einen frischen Browser-Score mit Alter und technischer Quelle."""
    normalized_url = _normalize_url(url)
    max_age = max(0, int(max_age_seconds))
    if not normalized_url or max_age <= 0:
        return None

    key = hashlib.md5(normalized_url.encode()).hexdigest()
    now = int(time.time())
    # 1. Memory-Cache (schnell)
    with _lock:
        entry = _score_cache.get(key)
        if entry:
            captured_at = int(entry["ts"])
            age_seconds = max(0, now - captured_at)
            if age_seconds < max_age:
                return {
                    "score": float(entry["score"]),
                    "capturedAt": captured_at,
                    "ageSeconds": age_seconds,
                    "source": "memory",
                }
    if not allow_db_fallback:
        return None

    # 2. DB-Fallback (überlebt Restarts)
    try:
        from app.database import get_article_score_snapshot_from_db

        snapshot = get_article_score_snapshot_from_db(
            normalized_url,
            max_age_seconds=max_age,
        )
        if snapshot:
            return {
                "score": float(snapshot["score"]),
                "capturedAt": int(snapshot["captured_at"]),
                "ageSeconds": int(snapshot["age_seconds"]),
                "source": "database",
            }
    except Exception:
        pass
    return None


def get_score_for_url(url: str) -> float | None:
    snapshot = get_score_snapshot_for_url(url)
    return float(snapshot["score"]) if snapshot else None


def get_score_snapshot_for_cms_id(
    cms_id: str,
    *,
    max_age_seconds: int = _CACHE_TTL,
    allow_db_fallback: bool = True,
) -> dict[str, object] | None:
    """Liefert den neuesten UI-Capture des laufenden Arbeitstags."""
    if not _CMS_ID_RE.fullmatch(cms_id):
        return None
    normalized_cms_id = cms_id.lower()
    return get_score_snapshots_for_cms_ids(
        [normalized_cms_id],
        max_age_seconds=max_age_seconds,
        allow_db_fallback=allow_db_fallback,
    ).get(normalized_cms_id)


def get_score_snapshots_for_cms_ids(
    cms_ids: list[str],
    *,
    max_age_seconds: int = _CACHE_TTL,
    allow_db_fallback: bool = True,
) -> dict[str, dict[str, object]]:
    """Resolve requested IDs with exactly one memory and one persistent-store scan."""
    if not cms_ids or any(not _CMS_ID_RE.fullmatch(cms_id) for cms_id in cms_ids):
        return {}
    max_age = max(0, int(max_age_seconds))
    if max_age <= 0:
        return {}

    now = int(time.time())
    requested = set(cms_ids)
    newest_memory: dict[str, dict[str, object]] = {}
    with _lock:
        entries = list(_score_cache.values())
    for entry in entries:
        snapshot = _fresh_minimal_snapshot(entry, now=now, max_age_seconds=max_age)
        if snapshot is None:
            continue
        for cms_id in _cms_ids_in_trusted_url(entry.get("url")).intersection(requested):
            existing = newest_memory.get(cms_id)
            if (
                existing is None
                or int(snapshot["capturedAt"]) > int(existing["capturedAt"])
            ):
                newest_memory[cms_id] = snapshot

    if not allow_db_fallback:
        return newest_memory
    try:
        from app.database import get_article_score_snapshots_by_cms_ids_from_db

        database_rows = get_article_score_snapshots_by_cms_ids_from_db(
            cms_ids,
            max_age_seconds=max_age,
        )
    except Exception as exc:
        raise ScoreCaptureReadError("score capture storage is unavailable") from exc

    resolved = dict(newest_memory)
    for cms_id, row in database_rows.items():
        database_snapshot = _fresh_minimal_snapshot(
            row,
            now=now,
            max_age_seconds=max_age,
        )
        if database_snapshot is None:
            continue
        existing = resolved.get(cms_id)
        if (
            existing is None
            or int(database_snapshot["capturedAt"]) > int(existing["capturedAt"])
        ):
            resolved[cms_id] = database_snapshot
    return resolved


class ScoreCaptureItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    url: str = Field(min_length=1)
    score: float = Field(gt=0, le=100)
    ts: int = Field(gt=0)
    score_breakdown: ScoreBreakdown | None = Field(default=None, alias="scoreBreakdown")
    or_factor: float | None = Field(default=None, alias="orFactor", ge=0.6, le=1.5)

    @model_validator(mode="after")
    def require_complete_enrichment_pair(self) -> ScoreCaptureItem:
        if (self.score_breakdown is None) != (self.or_factor is None):
            raise ValueError("scoreBreakdown and orFactor must be provided together")
        return self


class ScoreCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scores: list[ScoreCaptureItem]


class BatchCmsScoreCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cmsIds: list[str] = Field(min_length=1, max_length=500)

    @field_validator("cmsIds")
    @classmethod
    def require_unique_lowercase_cms_ids(cls, cms_ids: list[str]) -> list[str]:
        if any(
            not _CMS_ID_RE.fullmatch(cms_id) or cms_id != cms_id.lower()
            for cms_id in cms_ids
        ):
            raise ValueError("cmsIds must contain lowercase 24-character hex IDs")
        if len(set(cms_ids)) != len(cms_ids):
            raise ValueError("cmsIds must be unique")
        return cms_ids


class CmsScoreCaptureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    score: float = Field(gt=0, le=100)
    capturedAt: int = Field(gt=0)
    scoreBreakdown: ScoreBreakdown | None = None
    orFactor: float | None = Field(default=None, ge=0.6, le=1.5)

    @model_validator(mode="after")
    def require_complete_enrichment_pair(self) -> CmsScoreCaptureResponse:
        if (self.scoreBreakdown is None) != (self.orFactor is None):
            raise ValueError("scoreBreakdown and orFactor must be provided together")
        return self


class BatchCmsScoreFoundResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cmsId: str = Field(pattern=r"^[0-9a-f]{24}$")
    status: Literal["found"]
    score: float = Field(gt=0, le=100)
    capturedAt: int = Field(gt=0)
    scoreBreakdown: ScoreBreakdown | None = None
    orFactor: float | None = Field(default=None, ge=0.6, le=1.5)

    @model_validator(mode="after")
    def require_complete_enrichment_pair(self) -> BatchCmsScoreFoundResponse:
        if (self.scoreBreakdown is None) != (self.orFactor is None):
            raise ValueError("scoreBreakdown and orFactor must be provided together")
        return self


class BatchCmsScoreNotFoundResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cmsId: str = Field(pattern=r"^[0-9a-f]{24}$")
    status: Literal["notFound"]


class BatchCmsScoreCaptureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    results: list[BatchCmsScoreFoundResponse | BatchCmsScoreNotFoundResponse]


class ScoreCaptureHealthResponse(BaseModel):
    status: str


@router.get(
    "/api/score-capture/health",
    response_model=ScoreCaptureHealthResponse,
    summary="Check score capture reachability",
    description=(
        "Returns a minimal readiness response behind the same internal CIDR gate as the "
        "CMS-ID score lookup."
    ),
)
def get_score_capture_health(response: Response) -> ScoreCaptureHealthResponse:
    response.headers["Cache-Control"] = "no-store"
    return ScoreCaptureHealthResponse(status="ok")


@router.get("/api/score-capture/debug")
def debug_score_capture() -> JSONResponse:
    """Zeigt alle gecachten Scores (für Debugging)."""
    with _lock:
        entries = sorted(_score_cache.values(), key=lambda x: x["ts"], reverse=True)
    return JSONResponse({"count": len(entries), "entries": entries[:30]})


@router.get(
    "/api/score-capture/by-cms-id/{cms_id}",
    response_model=CmsScoreCaptureResponse,
    response_model_exclude_unset=True,
    summary="Get a captured UI score by CMS ID",
    description=(
        "Returns the latest workday score and capture timestamp already produced by the "
        "unchanged candidate UI. Set includeBreakdown=1 to add its captured numeric score "
        "fields and separate OR sorting factor. Access remains protected by the service's "
        "internal CIDR gate."
    ),
)
def get_score_capture_by_cms_id(
    response: Response,
    cms_id: str = Path(pattern=r"^[0-9a-fA-F]{24}$"),
    include_breakdown: int | None = Query(
        default=None,
        alias="includeBreakdown",
        ge=1,
        le=1,
    ),
) -> CmsScoreCaptureResponse:
    try:
        snapshot = get_score_snapshot_for_cms_id(cms_id)
    except ScoreCaptureReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Score capture storage is unavailable.",
        ) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No current score is available for this CMS ID.")
    response.headers["Cache-Control"] = "no-store"
    response_data: dict[str, object] = {
        "score": float(snapshot["score"]),
        "capturedAt": int(snapshot["capturedAt"]),
    }
    if (
        include_breakdown == 1
        and snapshot.get("scoreBreakdown") is not None
        and snapshot.get("orFactor") is not None
    ):
        response_data["scoreBreakdown"] = snapshot["scoreBreakdown"]
        response_data["orFactor"] = snapshot["orFactor"]
    return CmsScoreCaptureResponse.model_validate(response_data)


@router.post(
    "/api/score-capture/by-cms-id/batch",
    response_model=BatchCmsScoreCaptureResponse,
    response_model_exclude_unset=True,
    summary="Get captured UI scores for a CMS-ID batch",
    description=(
        "Returns found or notFound in request order for 1 to 500 unique lowercase CMS "
        "IDs. The source performs one memory scan and one persistent-store scan and "
        "never recalculates or substitutes a score."
    ),
)
def post_score_capture_by_cms_id_batch(
    body: BatchCmsScoreCaptureRequest,
    request: Request,
    response: Response,
    include_breakdown: int = Query(alias="includeBreakdown", ge=1, le=1),
) -> BatchCmsScoreCaptureResponse:
    if list(request.query_params.multi_items()) != [("includeBreakdown", "1")]:
        raise HTTPException(
            status_code=422,
            detail="The exact includeBreakdown=1 query is required.",
        )
    try:
        snapshots = get_score_snapshots_for_cms_ids(body.cmsIds)
    except ScoreCaptureReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Score capture storage is unavailable.",
        ) from exc

    results: list[dict[str, object]] = []
    for cms_id in body.cmsIds:
        snapshot = snapshots.get(cms_id)
        if snapshot is None:
            results.append({"cmsId": cms_id, "status": "notFound"})
            continue
        item: dict[str, object] = {
            "cmsId": cms_id,
            "status": "found",
            "score": float(snapshot["score"]),
            "capturedAt": int(snapshot["capturedAt"]),
        }
        if (
            snapshot.get("scoreBreakdown") is not None
            and snapshot.get("orFactor") is not None
        ):
            item["scoreBreakdown"] = snapshot["scoreBreakdown"]
            item["orFactor"] = snapshot["orFactor"]
        results.append(item)

    response.headers["Cache-Control"] = "no-store"
    return BatchCmsScoreCaptureResponse.model_validate({"results": results})


@router.post("/api/score-capture")
def post_score_capture(body: ScoreCaptureRequest) -> JSONResponse:
    """Empfängt Artikel-Scores vom Kandidaten-Tab — speichert in Memory + DB."""
    from app.database import ArticleScoreWriteError, save_article_score_to_db

    now = int(time.time())
    if any(item.ts > now + _MAX_FUTURE_SKEW_SECONDS for item in body.scores):
        raise HTTPException(status_code=422, detail="Capture timestamp is too far in the future.")
    stored = 0
    with _lock:
        for item in body.scores:
            key = hashlib.md5(_normalize_url(item.url).encode()).hexdigest()
            existing = _score_cache.get(key)
            if not existing or item.ts >= existing["ts"]:
                entry: dict[str, object] = {
                    "score": round(item.score, 1),
                    "ts": item.ts,
                    "url": item.url,
                }
                if item.score_breakdown is not None and item.or_factor is not None:
                    entry["scoreBreakdown"] = item.score_breakdown.model_dump()
                    entry["orFactor"] = item.or_factor
                _score_cache[key] = entry
                stored += 1
        _cleanup()
    # Persistent in DB schreiben (überlebt Render-Restarts)
    try:
        for item in body.scores:
            score_breakdown = (
                item.score_breakdown.model_dump()
                if item.score_breakdown is not None
                else None
            )
            save_article_score_to_db(
                item.url,
                round(item.score, 1),
                captured_at=item.ts,
                score_breakdown=score_breakdown,
                or_factor=item.or_factor,
                raise_on_error=True,
            )
    except ArticleScoreWriteError as exc:
        raise HTTPException(
            status_code=503,
            detail="Score capture storage is unavailable.",
        ) from exc
    return JSONResponse({"ok": True, "stored": stored})
