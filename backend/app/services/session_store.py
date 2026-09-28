"""Database access for research sessions, reports, papers and chat history.

Route handlers pass in their request-scoped ``Session``; the background run
manager uses the functions that open their own transaction via
:func:`app.db.session_scope`. Every lookup is scoped to the owning user, so one
user can never see another user's sessions (they get a 404, not a 403).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import session_scope
from app.models import ChatMessage, Report, ResearchSession, SessionPaper, SessionStatus
from app.schemas import (
    ChatCitation,
    ChatMessageRead,
    ResearchReport,
    RunMetrics,
    SessionDetail,
    SessionSummary,
)

logger = logging.getLogger(__name__)

ACTIVE_STATUSES: tuple[str, ...] = (SessionStatus.PENDING, SessionStatus.RUNNING)
INTERRUPTED_MESSAGE = "The server restarted while this run was in progress. Please start it again."


def settings_snapshot(settings: Settings) -> dict[str, Any]:
    """The settings that shape a run, stored with it for reproducibility."""
    return {
        "llm_model": settings.llm_model,
        "llm_fallback_models": settings.llm_fallback_models,
        "llm_effort": settings.llm_effort,
        "enable_critic": settings.enable_critic,
        "max_revisions": settings.max_revisions,
        "top_n_papers": settings.top_n_papers,
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
    }


# ---------------------------------------------------------------------------
# Request-scoped helpers (used by route handlers)
# ---------------------------------------------------------------------------


def create_session(db: Session, user_id: int, question: str, snapshot: dict[str, Any]) -> ResearchSession:
    """Insert a new pending session."""
    session = ResearchSession(user_id=user_id, question=question, settings_snapshot=snapshot)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def count_active_sessions(db: Session, user_id: int) -> int:
    """How many of the user's runs are pending or running."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(ResearchSession)
            .where(ResearchSession.user_id == user_id, ResearchSession.status.in_(ACTIVE_STATUSES))
        )
        or 0
    )


def get_user_session(db: Session, user_id: int, session_id: str) -> ResearchSession | None:
    """A session if it exists and belongs to ``user_id``."""
    return db.scalar(
        select(ResearchSession).where(ResearchSession.id == session_id, ResearchSession.user_id == user_id)
    )


def list_user_sessions(db: Session, user_id: int, limit: int = 50) -> list[ResearchSession]:
    """The user's sessions, newest first."""
    return list(
        db.scalars(
            select(ResearchSession)
            .where(ResearchSession.user_id == user_id)
            .order_by(ResearchSession.created_at.desc())
            .limit(limit)
        )
    )


def session_summary(session: ResearchSession) -> SessionSummary:
    """API view of a session (without the report)."""
    return SessionSummary(
        id=session.id,
        question=session.question,
        title=session.title,
        status=session.status,  # type: ignore[arg-type]
        current_stage=session.current_stage,
        error_message=session.error_message,
        created_at=session.created_at,
        completed_at=session.completed_at,
        metrics=RunMetrics.model_validate(session.metrics) if session.metrics else None,
    )


def load_report(session: ResearchSession) -> ResearchReport | None:
    """The stored report of a completed session."""
    if session.report is None or not session.report.content:
        return None
    return ResearchReport.model_validate(session.report.content)


def session_detail(session: ResearchSession) -> SessionDetail:
    """API view of a session including its report."""
    return SessionDetail(**session_summary(session).model_dump(), report=load_report(session))


def chat_message_read(message: ChatMessage) -> ChatMessageRead:
    """API view of a stored chat message."""
    return ChatMessageRead(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        citations=[ChatCitation.model_validate(citation) for citation in message.citations or []],
        created_at=message.created_at,
    )


def add_chat_message(
    db: Session, session_id: str, role: str, content: str, citations: list[ChatCitation] | None = None
) -> ChatMessage:
    """Store one chat turn."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        citations=[citation.model_dump() for citation in citations or []],
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


# ---------------------------------------------------------------------------
# Background helpers (open their own transaction; used by the run manager)
# ---------------------------------------------------------------------------


def mark_running(session_id: str) -> None:
    """Flag a session as running."""
    with session_scope() as db:
        db.execute(
            update(ResearchSession)
            .where(ResearchSession.id == session_id)
            .values(status=SessionStatus.RUNNING, current_stage="planner")
        )


def update_stage(session_id: str, stage: str) -> None:
    """Record which pipeline stage is currently running."""
    with session_scope() as db:
        db.execute(update(ResearchSession).where(ResearchSession.id == session_id).values(current_stage=stage))


def mark_failed(session_id: str, message: str) -> None:
    """Flag a session as failed with a user-readable reason."""
    with session_scope() as db:
        db.execute(
            update(ResearchSession)
            .where(ResearchSession.id == session_id)
            .values(
                status=SessionStatus.FAILED,
                error_message=message[:2000],
                current_stage=None,
                completed_at=datetime.now(UTC),
            )
        )


def save_report(session_id: str, report: ResearchReport) -> None:
    """Persist a finished run: report, ranked papers, metrics and status."""
    with session_scope() as db:
        session = db.get(ResearchSession, session_id)
        if session is None:
            logger.warning("Session %s disappeared before its report could be saved", session_id)
            return
        session.title = report.title[:300]
        session.status = SessionStatus.COMPLETED
        session.current_stage = None
        session.error_message = None
        session.completed_at = datetime.now(UTC)
        session.metrics = report.metrics.model_dump(mode="json")
        session.report = Report(
            title=report.title[:300],
            executive_summary=report.executive_summary,
            markdown=report.markdown,
            content=report.model_dump(mode="json"),
            verification=[claim.model_dump(mode="json") for claim in report.verification],
        )
        session.papers = [
            SessionPaper(
                paper_id=digest.paper.paper_id,
                rank=rank,
                title=digest.paper.title,
                authors=digest.paper.authors,
                year=digest.paper.year,
                venue=digest.paper.venue,
                abstract=digest.paper.abstract,
                url=digest.paper.url,
                pdf_url=digest.paper.pdf_url,
                doi=digest.paper.doi,
                citation_count=digest.paper.citation_count,
                source=",".join(digest.paper.sources),
                score=digest.paper.score,
                full_text=digest.full_text,
                summary=digest.summary.model_dump() if digest.summary else {},
            )
            for rank, digest in enumerate(report.papers, start=1)
        ]


def fail_interrupted_sessions() -> int:
    """At start-up, mark runs left pending/running by a previous process as failed."""
    with session_scope() as db:
        result = db.execute(
            update(ResearchSession)
            .where(ResearchSession.status.in_(ACTIVE_STATUSES))
            .values(status=SessionStatus.FAILED, error_message=INTERRUPTED_MESSAGE, current_stage=None)
        )
        count = int(getattr(result, "rowcount", 0) or 0)
    if count:
        logger.warning("Marked %d interrupted research run(s) as failed", count)
    return count
