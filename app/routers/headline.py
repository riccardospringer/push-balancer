"""Advisory Push Headline Prompt v1.4 endpoint for the standalone UI tab.

The endpoint resolves a CMS document ID only against article metadata that is
already present in the Push Balancer (recent recommendation decisions and the
public BILD news sitemap). It never sends, schedules, scores, or persists a
push and it never forwards the CMS ID or article URL to the model.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.push_title_prompt_v14 import (
    PROMPT_VERSION,
    RUNTIME_PROFILE,
    SOURCE_PROMPT_SHA256,
    build_push_headline_escalation,
    generate_push_headline_v14,
    is_prompt_generation_enabled,
)
from app.push_titles import build_push_title_suggestions, infer_content_type
from app.routers.feed import (
    _extract_sitemap_articles,
    _fetch_url,
    _infer_article_category,
)
from app.score_api_client import resolve_cms_id
from app.config import BILD_SITEMAP

log = logging.getLogger("push-balancer")
router = APIRouter()

_CMS_ID_RE = re.compile(r"^[0-9a-f]{24}$")
_MAX_SITEMAP_ARTICLES = 250


class HeadlineGenerationRequest(BaseModel):
    """Minimal input: one validated CMS document ID."""

    model_config = ConfigDict(extra="forbid")

    articleId: str = Field(
        min_length=24,
        max_length=24,
        pattern=r"^[0-9a-fA-F]{24}$",
    )


def _normalized_article(
    *,
    cms_id: str,
    title: str,
    url: str,
    category: str,
    article_type: str | None = None,
) -> dict[str, str] | None:
    clean_title = re.sub(r"\s+", " ", str(title or "").strip())[:500]
    clean_url = str(url or "").strip()[:2048]
    if not clean_title or not clean_url:
        return None
    clean_category = str(category or "").strip().lower()
    if not clean_category:
        clean_category = _infer_article_category(clean_url, clean_title)
    content_type = article_type or infer_content_type(clean_url, clean_title)
    return {
        "articleId": cms_id,
        "title": clean_title,
        "url": clean_url,
        "category": clean_category or "news",
        "contentType": "video" if content_type == "video" else "editorial",
    }


def _article_from_recent_decisions(cms_id: str) -> dict[str, str] | None:
    """Reuse locally retained public article metadata without new data flows."""
    try:
        from app.database import teams_alert_list_recent

        rows = teams_alert_list_recent(limit=100)
    except Exception as exc:
        log.warning(
            "[Headline] Recent article metadata unavailable (%s)",
            type(exc).__name__,
        )
        return None

    for row in rows:
        row_id = str(row.get("article_id") or "").strip().lower()
        row_url = str(row.get("article_url") or "").strip()
        if row_id != cms_id and resolve_cms_id({"url": row_url}) != cms_id:
            continue
        return _normalized_article(
            cms_id=cms_id,
            title=str(row.get("article_title") or ""),
            url=row_url,
            category="",
        )
    return None


def _article_from_news_sitemap(cms_id: str) -> dict[str, str] | None:
    """Resolve the ID from the same public source as the candidate view."""
    raw = _fetch_url(BILD_SITEMAP)
    if raw is None:
        raise HTTPException(status_code=502, detail="BILD Artikeldaten sind nicht erreichbar.")
    try:
        articles = _extract_sitemap_articles(raw, max_items=_MAX_SITEMAP_ARTICLES)
    except Exception as exc:
        log.warning("[Headline] Sitemap parsing failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="BILD Artikeldaten konnten nicht gelesen werden.",
        ) from exc

    for article in articles:
        if resolve_cms_id(article) != cms_id:
            continue
        return _normalized_article(
            cms_id=cms_id,
            title=str(article.get("title") or ""),
            url=str(article.get("url") or ""),
            category=str(article.get("category") or "news"),
            article_type=str(article.get("type") or ""),
        )
    return None


def resolve_headline_article(article_id: str) -> dict[str, str]:
    """Return minimal public article metadata for one exact CMS ID."""
    cms_id = str(article_id or "").strip().lower()
    if not _CMS_ID_RE.fullmatch(cms_id):
        raise HTTPException(status_code=422, detail="Die Artikel-ID ist ungültig.")

    article = _article_from_recent_decisions(cms_id) or _article_from_news_sitemap(cms_id)
    if article is None:
        raise HTTPException(
            status_code=404,
            detail="Artikel nicht gefunden. Bitte eine aktuelle BILD Artikel-ID verwenden.",
        )
    return article


def _local_fallback(article: dict[str, str]) -> dict[str, Any]:
    """Keep the UI usable while the external prompt path is explicitly off."""
    generated = build_push_title_suggestions(
        title=article["title"],
        category=article["category"],
        url=article["url"],
    )
    raw_titles = [generated.get("title"), *(generated.get("alternativeTitles") or [])]
    for group in (generated.get("alle_kandidaten") or {}).values():
        for candidate in group or []:
            raw_titles.append(candidate.get("titel"))
    titles: list[str] = []
    for title in raw_titles:
        clean_title = str(title or "").strip()
        if clean_title and clean_title not in titles:
            titles.append(clean_title)
    variants = [
        {
            "id": identifier,
            "type": "LOKALER FALLBACK",
            "headline": title,
            "line2": "",
            "headlineLength": len(title),
            "line2Length": 0,
            "selected": index == 0,
        }
        for index, (identifier, title) in enumerate(zip(("A", "B", "C"), titles[:3]))
    ]
    generated.update(
        {
            "title": titles[0] if titles else article["title"],
            "alternativeTitles": titles[1:3],
            "variants": variants,
            "promptVersion": PROMPT_VERSION,
            "sourcePromptSha256": SOURCE_PROMPT_SHA256,
            "runtimeProfile": RUNTIME_PROFILE,
            "promptActive": False,
            "escalation": False,
            "reviewPoint": (
                "Prompt v1.4 ist serverseitig hinterlegt; der externe KI-Pfad ist in "
                "dieser Umgebung deaktiviert. Diese Vorschläge stammen aus dem lokalen Fallback."
            ),
            "article": article,
            "advisoryOnly": True,
        }
    )
    return generated


def build_headline_generation(article_id: str) -> dict[str, Any]:
    article = resolve_headline_article(article_id)
    content_type = article["contentType"]
    try:
        generated = generate_push_headline_v14(
            title=article["title"],
            category=article["category"],
            content_type=content_type,
        )
        if generated is None:
            return _local_fallback(article)
        generated["promptActive"] = True
        generated["article"] = article
        return generated
    except Exception as exc:
        # Never log the CMS ID, article metadata, prompt, provider response, or secret.
        log.warning("[Headline] Prompt v1.4 failed (%s)", type(exc).__name__)
        if is_prompt_generation_enabled():
            generated = build_push_headline_escalation(
                article["title"],
                content_type=content_type,
                reason="v1.4-Workflow nicht verfügbar; CvD-Prüfung erforderlich.",
            )
            generated["promptActive"] = True
            generated["article"] = article
            return generated
        return _local_fallback(article)


@router.post("/api/headline-generations")
def create_headline_generation(body: HeadlineGenerationRequest) -> JSONResponse:
    """Generate advisory headline variants; never send or schedule a push."""
    payload = build_headline_generation(body.articleId)
    return JSONResponse(
        content=payload,
        headers={
            "Cache-Control": "no-store",
            "X-Prompt-Version": PROMPT_VERSION,
        },
    )
