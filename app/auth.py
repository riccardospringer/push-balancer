from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import ADMIN_API_KEY

_api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


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
