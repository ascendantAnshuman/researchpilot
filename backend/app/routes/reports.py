"""Report routes: history list, full report, delete, and Markdown/PDF export."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db import get_db
from app.deps import get_run_manager
from app.models import ResearchSession, User
from app.schemas import ResearchReport, SessionDetail, SessionSummary
from app.services import session_store
from app.services.export import export_filename, report_to_markdown, report_to_pdf
from app.services.run_manager import RunManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


def _owned_session(db: Session, user: User, session_id: str) -> ResearchSession:
    """Load one of the user's sessions or answer 404."""
    session = session_store.get_user_session(db, user.id, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research session not found")
    return session


def _completed_report(session: ResearchSession) -> ResearchReport:
    """The session's report, or 409 if the run has not produced one."""
    report = session_store.load_report(session)
    if report is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"This research run is {session.status}")
    return report


@router.get("", response_model=list[SessionSummary], summary="The user's research history, newest first")
def list_reports(
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SessionSummary]:
    """Sessions for the history sidebar."""
    return [session_store.session_summary(session) for session in session_store.list_user_sessions(db, user.id, limit)]


@router.get("/{session_id}", response_model=SessionDetail, summary="A research session with its full report")
def get_report(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionDetail:
    """Everything the report view needs: report, papers, verification and metrics."""
    return session_store.session_detail(_owned_session(db, user, session_id))


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a research session")
def delete_report(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    manager: RunManager = Depends(get_run_manager),
) -> Response:
    """Delete a session with its report, papers and chat history."""
    session = _owned_session(db, user, session_id)
    if manager.is_active(session_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This research run is still in progress")
    db.delete(session)
    db.commit()
    logger.info("User %s deleted session %s", user.id, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/export", summary="Download the report as Markdown or PDF")
async def export_report(
    session_id: str,
    export_format: Literal["markdown", "pdf"] = Query(default="markdown", alias="format", description="markdown or pdf"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Return the report as a file download."""
    report = _completed_report(_owned_session(db, user, session_id))
    if export_format == "pdf":
        content = await asyncio.to_thread(report_to_pdf, report)
        media_type, filename = "application/pdf", export_filename(report, "pdf")
    else:
        content = report_to_markdown(report)
        media_type, filename = "text/markdown; charset=utf-8", export_filename(report, "md")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
