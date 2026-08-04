#!/usr/bin/env python3
"""Secret-safe live contract check for the CMS-ID score endpoint."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

_CMS_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_RESPONSE_BYTES = 64 * 1024
_DEFAULT_TIMEOUT_SECONDS = 35.0


class _RejectRedirects(HTTPRedirectHandler):
    """Do not forward the score credential to a redirected host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _score_url(base_url: str, cms_id: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("NEXT_PUSH_BALANCER_URL must not contain credentials, query, or fragment")

    is_local_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }
    if parsed.scheme != "https" and not is_local_http:
        raise ValueError("NEXT_PUSH_BALANCER_URL must use HTTPS")
    if not parsed.hostname:
        raise ValueError("NEXT_PUSH_BALANCER_URL must contain a host")
    if not _CMS_ID_PATTERN.fullmatch(cms_id):
        raise ValueError("CMS_ID has an invalid format")

    return f"{base_url.rstrip('/')}/api/v1/scores/{quote(cms_id, safe='')}"


def _validated_payload(raw: bytes, expected_cms_id: str) -> dict[str, Any]:
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds the contract size limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    if payload.get("cmsId") != expected_cms_id:
        raise ValueError("response CMS ID does not match the request")

    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("response score is not numeric")
    if not 0 <= float(score) <= 100:
        raise ValueError("response score is outside the contract range")
    if not isinstance(payload.get("scoredAt"), str) or not payload["scoredAt"].strip():
        raise ValueError("response timestamp is missing")
    return payload


def main(
    environ: Mapping[str, str] | None = None,
    opener: OpenerDirector | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Check one approved CMS ID and print only its score."""
    environ = os.environ if environ is None else environ
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    try:
        base_url = _required(environ, "NEXT_PUSH_BALANCER_URL")
        api_key = _required(environ, "SCORE_API_KEY")
        cms_id = _required(environ, "CMS_ID")
        timeout = float(environ.get("SCORE_API_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
        if not 1 <= timeout <= 120:
            raise ValueError("SCORE_API_TIMEOUT_SECONDS must be between 1 and 120")
        request_url = _score_url(base_url, cms_id)
    except (TypeError, ValueError):
        print("score API smoke configuration is invalid", file=stderr)
        return 3

    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-store",
            "X-Score-Key": api_key,
        },
        method="GET",
    )
    client = opener if opener is not None else build_opener(_RejectRedirects())

    try:
        with client.open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            print("no current score is available", file=stderr)
            return 2
        if exc.code in {500, 502, 503}:
            print(f"score API is temporarily unavailable (HTTP {exc.code})", file=stderr)
            return 4
        print(f"score API request was rejected (HTTP {exc.code})", file=stderr)
        return 3
    except (TimeoutError, URLError, OSError):
        print("score API request failed because of a network or TLS error", file=stderr)
        return 4

    try:
        payload = _validated_payload(raw, cms_id)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        print("score API response violates the v1 contract", file=stderr)
        return 3

    print(payload["score"], file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
