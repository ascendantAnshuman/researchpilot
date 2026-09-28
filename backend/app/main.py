"""FastAPI application factory and ASGI entry point.

Run the development server from the ``backend/`` folder with::

    uvicorn app.main:app --reload --reload-dir app

The module-level ``app`` object is what uvicorn and the Docker image import.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.db import create_tables, dispose_engine, init_engine
from app.logging_config import setup_logging
from app.routes import auth, chat, health, reports, research
from app.services.run_manager import RunManager, Runner, default_runner
from app.services.session_store import fail_interrupted_sessions

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage, the database and the run manager; shut them down cleanly."""
    settings: Settings = app.state.settings
    settings.ensure_data_dirs()
    init_engine(settings.sqlalchemy_database_url)
    create_tables()
    fail_interrupted_sessions()
    app.state.run_manager = RunManager(runner=app.state.runner, max_concurrent_runs=settings.max_concurrent_runs)
    logger.info(
        "Starting %s v%s | env=%s | model=%s | data_dir=%s",
        settings.app_name,
        __version__,
        settings.environment,
        settings.llm_model,
        settings.data_dir,
    )
    if settings.jwt_secret_is_ephemeral:
        logger.warning(
            "JWT_SECRET_KEY is not set: using a random per-process secret. "
            "Logins will not survive a restart - set JWT_SECRET_KEY in .env."
        )
    if not settings.llm_configured:
        logger.warning(
            "ANTHROPIC_API_KEY is not set: the research agents cannot call Claude until it is added to .env."
        )
    yield
    await app.state.run_manager.shutdown()
    dispose_engine()
    logger.info("%s shut down cleanly", settings.app_name)


def create_app(settings: Settings | None = None, *, runner: Runner | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional settings to build the app with (tests pass their own).
            Defaults to the cached global settings.
        runner: The function that executes a research run (tests pass a fake
            pipeline). Defaults to the real multi-agent pipeline.

    Returns:
        A fully configured FastAPI instance.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description="Multi-agent research copilot: plan, search, read, synthesize and verify.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.runner = runner or default_runner
    # Every route that depends on get_settings sees the same instance the app was built with.
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    for router in (health.router, auth.router, research.router, reports.router, chat.router):
        app.include_router(router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Point people who open the bare API URL at the docs and health check."""
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
