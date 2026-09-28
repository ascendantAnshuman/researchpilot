"""Health-check endpoint used by Docker, Render/Railway and the frontend."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status

from app import __version__
from app.config import Settings, get_settings
from app.db import check_database
from app.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and configuration check",
    responses={503: {"model": HealthResponse, "description": "The database is unreachable."}},
)
def health(response: Response, settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report whether the API and database are up, plus non-secret configuration.

    Returns HTTP 503 when the database is unreachable so hosting platforms can
    restart the service.
    """
    database_ok = check_database()
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        database="ok" if database_ok else "unavailable",
        llm_model=settings.llm_model,
        llm_configured=settings.llm_configured,
        critic_enabled=settings.enable_critic,
        server_time=datetime.now(UTC),
    )
