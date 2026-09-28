"""Shared FastAPI dependencies (overridable in tests via ``app.dependency_overrides``)."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.llm import LLM, LLMClient
from app.services.run_manager import RunManager
from app.services.vector_store import VectorStore, get_vector_store

LLM_NOT_CONFIGURED = "Claude is not configured on the server: set ANTHROPIC_API_KEY in the .env file."


def get_run_manager(request: Request) -> RunManager:
    """The application's background run manager (created at start-up)."""
    manager: RunManager = request.app.state.run_manager
    return manager


def get_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    """The shared vector store."""
    return get_vector_store(settings)


def require_llm_configured(settings: Settings = Depends(get_settings)) -> None:
    """Reject requests that need Claude when no API key is configured (HTTP 503)."""
    if not settings.llm_configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LLM_NOT_CONFIGURED)


def get_chat_llm(
    request: Request,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_llm_configured),
) -> LLM:
    """A Claude client for chat, created once and reused across requests."""
    llm: LLM | None = getattr(request.app.state, "chat_llm", None)
    if llm is None:
        llm = LLMClient(settings)
        request.app.state.chat_llm = llm
    return llm
