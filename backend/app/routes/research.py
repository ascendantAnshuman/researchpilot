"""Research routes: start a run, stream its live progress (SSE), check its status.

Streaming protocol (``GET /api/research/{id}/events``, ``text/event-stream``)::

    event: running    data: {"message": "Research started", ...}
    event: progress   data: {"stage": "search", "status": "progress", "message": "...", "data": {...}}
    ...
    event: done       data: {"session_id": "...", "status": "completed", "title": "...", ...}
      or
    event: error      data: {"session_id": "...", "message": "why it failed"}

Past events are replayed first, so reconnecting never loses progress. The
browser reads this with ``fetch`` (not ``EventSource``) so it can send the
``Authorization: Bearer`` header.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sse_starlette import EventSourceResponse

from app.auth import get_current_user
from app.config import Settings, get_settings
from app.db import get_db
from app.deps import get_run_manager, require_llm_configured
from app.models import SessionStatus, User
from app.schemas import ResearchCreate, ResearchStarted, SessionSummary
from app.services import session_store
from app.services.run_manager import TERMINAL_EVENTS, RunManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

SSE_PING_SECONDS = 15


def _sse(item: dict[str, Any], event_id: int) -> dict[str, str]:
    """Format a run-manager message for sse-starlette."""
    return {"event": item["event"], "data": json.dumps(item["data"]), "id": str(event_id)}


def _final_event(summary: SessionSummary) -> dict[str, Any]:
    """The terminal message for a run that is no longer in memory (e.g. after a restart)."""
    now = datetime.now(UTC).isoformat()
    if summary.status == SessionStatus.COMPLETED:
        return {"event": "done", "data": {"session_id": summary.id, "status": "completed", "title": summary.title,
                                          "timestamp": now}}
    reason = summary.error_message or "This run is no longer active. Please start it again."
    return {"event": "error", "data": {"session_id": summary.id, "message": reason, "timestamp": now}}


@router.post(
    "",
    response_model=ResearchStarted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_llm_configured)],
    summary="Start a research run in the background",
)
async def start_research(
    payload: ResearchCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    manager: RunManager = Depends(get_run_manager),
) -> ResearchStarted:
    """Create a session and start the pipeline; progress is streamed from ``events_url``."""
    if session_store.count_active_sessions(db, user.id) >= settings.max_active_runs_per_user:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You already have {settings.max_active_runs_per_user} research runs in progress. "
            "Please wait for one to finish.",
        )
    overrides: dict[str, Any] = {}
    if payload.enable_critic is not None:
        overrides["enable_critic"] = payload.enable_critic
    if payload.top_n is not None:
        overrides["top_n_papers"] = payload.top_n
    run_settings = settings.model_copy(update=overrides) if overrides else settings

    session = session_store.create_session(
        db, user.id, payload.question, session_store.settings_snapshot(run_settings)
    )
    manager.start(session.id, payload.question, run_settings)
    logger.info("User %s started research session %s", user.id, session.id)
    return ResearchStarted(
        session=session_store.session_summary(session),
        events_url=f"{settings.api_prefix}/research/{session.id}/events",
    )


@router.get("/{session_id}", response_model=SessionSummary, summary="Status of a research run")
def get_status(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionSummary:
    """Current status and stage of one of the user's runs."""
    session = session_store.get_user_session(db, user.id, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research session not found")
    return session_store.session_summary(session)


@router.get(
    "/{session_id}/events",
    summary="Live progress of a research run (Server-Sent Events)",
    response_class=EventSourceResponse,
)
async def stream_events(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    manager: RunManager = Depends(get_run_manager),
) -> EventSourceResponse:
    """Replay the run's events so far, then stream new ones until it finishes."""
    session = session_store.get_user_session(db, user.id, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research session not found")
    summary = session_store.session_summary(session)
    subscription = manager.subscribe(session_id)

    async def events() -> AsyncIterator[dict[str, str]]:
        if subscription is None:
            yield _sse(_final_event(summary), 0)
            return
        history, queue = subscription
        try:
            event_id = 0
            for item in history:
                yield _sse(item, event_id)
                event_id += 1
                if item["event"] in TERMINAL_EVENTS:
                    return
            while True:
                item = await queue.get()
                yield _sse(item, event_id)
                event_id += 1
                if item["event"] in TERMINAL_EVENTS:
                    return
        finally:
            manager.unsubscribe(session_id, queue)

    return EventSourceResponse(events(), ping=SSE_PING_SECONDS)
