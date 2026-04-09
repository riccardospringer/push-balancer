"""app/auth.py — API-Key-Authentifizierung für Admin-Endpoints.

Schützt sensible POST-Endpoints (retrain, force-promote etc.) vor
unautorisierten Zugriffen.

Verwendung als FastAPI-Dependency:
    from app.auth import require_admin_key
    @router.post("/api/ml/retrain", dependencies=[Depends(require_admin_key)])

Falls ADMIN_API_KEY nicht gesetzt ist, wird eine Warnung geloggt und
Admin-Endpoints sind lokal ohne Auth erreichbar (Entwicklungsmodus).
In Produktionsumgebungen MUSS ADMIN_API_KEY in .env gesetzt sein.
"""
import logging
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import ADMIN_API_KEY

log = logging.getLogger("push-balancer")

_api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI-Dependency: prüft X-Admin-Key Header für Admin-Endpoints.

    - Wenn ADMIN_API_KEY in config gesetzt: Header muss übereinstimmen.
    - Wenn ADMIN_API_KEY leer (Entwicklung): Warnung im Log, kein Fehler.
    """
    if not ADMIN_API_KEY:
        log.warning(
            "[Auth] ADMIN_API_KEY nicht gesetzt — Admin-Endpoint ohne Auth erreichbar. "
            "Für Produktion ADMIN_API_KEY in .env setzen!"
        )
        return
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Ungültiger oder fehlender X-Admin-Key Header",
        )
