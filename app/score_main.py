"""Minimal ASGI runtime for the backend-to-backend CMS score API."""

from __future__ import annotations

import ipaddress
import hashlib
import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import config
from app.render_score_capture import RenderScoreUnavailable, require_capture_source_ready
from app.routers import score_api

log = logging.getLogger("push-balancer")
_SCORE_PATH_PREFIX = "/api/v1/scores"
_FROZEN_SEED_SHA256 = "63637a93f55f009b9bb6d5a5f88e3f79addcad0468b6b134714d97996777d58e"


def _preload_local_openmp_for_development() -> None:
    """Make the bundled local macOS OpenMP shim visible to LightGBM imports."""
    if sys.platform != "darwin":
        return
    openmp_path = os.path.expanduser("~/.local/lib/libomp.dylib")
    if not os.path.isfile(openmp_path):
        return
    try:
        import ctypes

        ctypes.CDLL(openmp_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        log.warning("[ScoreRuntime] Local OpenMP preload failed")


def _frozen_seed_digest_is_valid() -> bool:
    seed_path = Path(__file__).resolve().parent / "ml" / "seed_model.pkl"
    try:
        digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    except OSError:
        return False
    return hmac.compare_digest(digest, _FROZEN_SEED_SHA256)


def _safe_request_path(path: str) -> str:
    if path == f"{_SCORE_PATH_PREFIX}/batch":
        return path
    if path == _SCORE_PATH_PREFIX or path.startswith(f"{_SCORE_PATH_PREFIX}/"):
        return f"{_SCORE_PATH_PREFIX}/{{cms_id}}"
    return path


def _ip_is_in_cidrs(client_ip: str | None, cidrs: list[str]) -> bool:
    if not client_ip:
        return False
    try:
        parsed_ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            if parsed_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            log.warning("[ScoreAccess] Invalid CIDR configuration")
    return False


def _extract_client_ip(request: Request) -> str | None:
    peer_ip = request.client.host if request.client and request.client.host else None
    if _ip_is_in_cidrs(peer_ip, config.TRUSTED_PROXY_CIDRS):
        for header_name in (
            "cf-connecting-ip",
            "true-client-ip",
            "x-real-ip",
        ):
            header_value = request.headers.get(header_name, "").strip()
            if header_value:
                return header_value
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            candidate = forwarded_for.split(",", 1)[0].strip()
            if candidate:
                return candidate
    return peer_ip


def _path_is_exempt(path: str) -> bool:
    return any(
        path == exempt_path or path.startswith(f"{exempt_path}/")
        for exempt_path in config.INTERNAL_ACCESS_EXEMPT_PATHS
    )


def _problem_response(
    request: Request,
    status_code: int,
    title: str,
    detail: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": _safe_request_path(request.url.path),
        },
        media_type="application/problem+json",
        headers=headers,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load only the frozen cold-start model; do not start mutable workers."""
    if config.ARTICLE_PREDICTION_ENRICHMENT_ENABLED:
        if not _frozen_seed_digest_is_valid():
            raise RuntimeError("Approved score seed model failed integrity validation")
        _preload_local_openmp_for_development()
        from app.ml.lightgbm_model import load_seed_model

        if not load_seed_model():
            raise RuntimeError("Approved score seed model is unavailable")
        log.info("[ScoreRuntime] Frozen seed model loaded")
    yield


app = FastAPI(
    title="Next Push Balancer Score API",
    version="3.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    title = {
        401: "Unauthorized",
        404: "Not Found",
        422: "Unprocessable Content",
        429: "Too Many Requests",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }.get(exc.status_code, "HTTP Error")
    return _problem_response(
        request,
        exc.status_code,
        title,
        str(exc.detail),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    _exc: RequestValidationError,
) -> JSONResponse:
    return _problem_response(
        request,
        422,
        "Unprocessable Content",
        "Request validation failed.",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception(
        "[ScoreRuntime] Unhandled error on %s",
        _safe_request_path(request.url.path),
        exc_info=exc,
    )
    return _problem_response(
        request,
        500,
        "Internal Server Error",
        "An unexpected server error occurred.",
    )


@app.middleware("http")
async def restrict_internal_access(request: Request, call_next) -> Response:
    if not config.INTERNAL_ACCESS_ENABLED or _path_is_exempt(request.url.path):
        return await call_next(request)

    client_ip = _extract_client_ip(request)
    if _ip_is_in_cidrs(client_ip, config.INTERNAL_ACCESS_ALLOWED_CIDRS):
        return await call_next(request)

    log.warning(
        "[ScoreAccess] Blocked request to %s",
        _safe_request_path(request.url.path),
    )
    return _problem_response(
        request,
        404,
        "Not Found",
        "The requested resource was not found.",
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == _SCORE_PATH_PREFIX or request.url.path.startswith(
        f"{_SCORE_PATH_PREFIX}/"
    ):
        response.headers["Cache-Control"] = "no-store"
        vary = {
            value.strip() for value in response.headers.get("Vary", "").split(",") if value.strip()
        }
        vary.add("X-Score-Key")
        response.headers["Vary"] = ", ".join(sorted(vary))
    return response


@app.get("/api/health", include_in_schema=False)
def get_health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/ready", include_in_schema=False)
def get_readiness() -> dict[str, str]:
    try:
        require_capture_source_ready()
    except RenderScoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="Score source is unavailable.") from exc
    return {"status": "ready"}


app.include_router(score_api.router, tags=["Score"])
