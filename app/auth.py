import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app import config
from app.config import ADMIN_API_KEY

_api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)
_consumer_key_header = APIKeyHeader(name="X-Consumer-Key", auto_error=False)
_authorization_header = APIKeyHeader(name="Authorization", auto_error=False)


async def require_admin_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject admin mutations unless a configured X-Admin-Key is presented."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrative endpoints are disabled because ADMIN_API_KEY is not configured.",
        )
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )


def _extract_consumer_token(api_key: str | None, authorization: str | None) -> str:
    if api_key:
        return api_key
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
    return ""


async def require_consumer_key(
    api_key: str | None = Security(_consumer_key_header),
    authorization: str | None = Security(_authorization_header),
) -> None:
    """Reject read-only consumer API calls unless a configured key is presented."""
    if not config.CONSUMER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Consumer API is disabled because CONSUMER_API_KEY is not configured.",
        )
    token = _extract_consumer_token(api_key, authorization)
    if not token or not hmac.compare_digest(
        token.encode("utf-8"),
        config.CONSUMER_API_KEY.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing consumer API key.",
        )
