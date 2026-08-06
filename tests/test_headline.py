from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app
from app.routers import headline


client = TestClient(app, raise_server_exceptions=True)
CMS_ID = "0123456789abcdef01234567"
ARTICLE = {
    "articleId": CMS_ID,
    "title": "Bund beschließt neue Pendler-Regel",
    "url": f"https://www.bild.de/politik/test-{CMS_ID}.html",
    "category": "politik",
    "contentType": "editorial",
}


def _prompt_response():
    return {
        "title": "Bund stoppt neue Pendler-Regel",
        "line2": "Start verschiebt sich auf Montag",
        "alternativeTitles": [
            "Neue Pendler-Regel kommt später",
            "Pendler warten auf neue Regeln",
        ],
        "variants": [
            {
                "id": "A",
                "type": "FAKT",
                "headline": "Bund stoppt neue Pendler-Regel",
                "line2": "Start verschiebt sich auf Montag",
                "headlineLength": 31,
                "line2Length": 32,
                "selected": True,
            }
        ],
        "promptVersion": "1.4",
        "promptActive": True,
        "advisoryOnly": True,
    }


def test_headline_endpoint_resolves_article_id_and_uses_v14(monkeypatch):
    resolver = Mock(return_value=ARTICLE)
    generator = Mock(return_value=_prompt_response())
    monkeypatch.setattr(headline, "resolve_headline_article", resolver)
    monkeypatch.setattr(headline, "generate_push_headline_v14", generator)

    response = client.post("/api/headline-generations", json={"articleId": CMS_ID})

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-prompt-version"] == "1.4"
    data = response.json()
    assert data["article"] == ARTICLE
    assert data["title"] == "Bund stoppt neue Pendler-Regel"
    assert data["promptActive"] is True
    resolver.assert_called_once_with(CMS_ID)
    generator.assert_called_once_with(
        title=ARTICLE["title"],
        category="politik",
        content_type="editorial",
    )


def test_headline_endpoint_rejects_invalid_article_id_before_lookup(monkeypatch):
    resolver = Mock()
    monkeypatch.setattr(headline, "resolve_headline_article", resolver)

    response = client.post("/api/headline-generations", json={"articleId": "zu-kurz"})

    assert response.status_code == 422
    resolver.assert_not_called()


def test_headline_resolution_uses_matching_sitemap_article(monkeypatch):
    monkeypatch.setattr(headline, "_article_from_recent_decisions", lambda _cms_id: None)
    monkeypatch.setattr(
        headline,
        "_article_from_news_sitemap",
        lambda cms_id: {**ARTICLE, "articleId": cms_id},
    )

    assert headline.resolve_headline_article(CMS_ID) == ARTICLE


def test_headline_sitemap_lookup_searches_beyond_candidate_limit(monkeypatch):
    def sitemap_entry(cms_id: str, title: str) -> str:
        return f"""
        <url>
          <loc>https://www.bild.de/politik/test-{cms_id}</loc>
          <news:news>
            <news:publication>
              <news:name>BILD</news:name>
              <news:language>de</news:language>
            </news:publication>
            <news:publication_date>2026-08-06T10:00:00+02:00</news:publication_date>
            <news:title>{title}</news:title>
          </news:news>
        </url>
        """

    filler = [
        sitemap_entry(f"{index:024x}", f"Testartikel {index}")
        for index in range(250)
    ]
    xml = (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
        + "".join(filler)
        + sitemap_entry(CMS_ID, ARTICLE["title"])
        + "</urlset>"
    ).encode()
    monkeypatch.setattr(headline, "_fetch_url", lambda _url: xml)

    resolved = headline._article_from_news_sitemap(CMS_ID)

    assert resolved is not None
    assert resolved["articleId"] == CMS_ID
    assert resolved["title"] == ARTICLE["title"]


def test_headline_route_is_in_openapi_contract():
    operation = app.openapi()["paths"]["/api/headline-generations"]["post"]

    assert operation["operationId"] == "create_headline_generation_api_headline_generations_post"
    assert "Headline" in operation["tags"]
