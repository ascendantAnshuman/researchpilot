"""SQLAlchemy ORM models.

Tables:
    users            registered accounts
    research_sessions one research question and its run status/metrics
    reports          the final literature review (Markdown + structured JSON + verification)
    session_papers   the ranked papers a session used, with per-paper structured summaries
    chat_messages    follow-up questions and answers over a session's papers

Deleting a user deletes their sessions; deleting a session deletes its report,
papers and chat history (both through the ORM and through ``ON DELETE CASCADE``).
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Dialect,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def new_session_id() -> str:
    """A random 32-character hex id, safe to expose in URLs."""
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator[datetime]):
    """Store datetimes as naive UTC and always load them as timezone-aware UTC.

    SQLite has no timezone support and returns naive datetimes. Re-attaching UTC
    means the API always serialises times with an explicit ``Z``/``+00:00`` offset,
    so the browser shows the correct local time.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Convert an outgoing value to naive UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Attach UTC to a value loaded from the database."""
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class SessionStatus(enum.StrEnum):
    """Lifecycle of a research session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    """A registered account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(120), default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    research_sessions: Mapped[list[ResearchSession]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ResearchSession.created_at.desc()",
    )

    def __repr__(self) -> str:
        """Readable representation that never includes the password hash."""
        return f"User(id={self.id!r}, email={self.email!r})"


class ResearchSession(Base):
    """One research question asked by a user, plus the run's status and metrics."""

    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_session_id)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(String(300), default=None)
    status: Mapped[str] = mapped_column(String(20), default=SessionStatus.PENDING, index=True)
    current_stage: Mapped[str | None] = mapped_column(String(40), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    # Model, critic on/off, top-N, ... captured at run time so results are reproducible.
    settings_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Per-stage latency, token counts and cost, filled in when the run finishes.
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    user: Mapped[User] = relationship(back_populates="research_sessions")
    report: Mapped[Report | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )
    papers: Mapped[list[SessionPaper]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SessionPaper.rank",
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.id",
    )

    def __repr__(self) -> str:
        """Readable representation for logs and debugging."""
        return f"ResearchSession(id={self.id!r}, status={self.status!r})"


class Report(Base):
    """The final literature review produced for a session."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    executive_summary: Mapped[str] = mapped_column(Text, default="")
    markdown: Mapped[str] = mapped_column(Text, default="")
    # Structured report: themed sections, comparison table, references.
    content: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # One entry per atomic claim: text, label (supported/partial/unsupported), evidence.
    verification: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="report")


class SessionPaper(Base):
    """A paper selected for a session, with its ranking and structured summary."""

    __tablename__ = "session_papers"
    __table_args__ = (UniqueConstraint("session_id", "paper_id", name="uq_session_paper"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    paper_id: Mapped[str] = mapped_column(String(128))
    rank: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, default=None)
    venue: Mapped[str | None] = mapped_column(String(300), default=None)
    abstract: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str | None] = mapped_column(String(1000), default=None)
    pdf_url: Mapped[str | None] = mapped_column(String(1000), default=None)
    doi: Mapped[str | None] = mapped_column(String(200), default=None)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(40), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    # True when the full PDF was parsed, False when only the abstract was available.
    full_text: Mapped[bool] = mapped_column(Boolean, default=False)
    # Reader output: problem, method, dataset, metrics, key results, limitations.
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    session: Mapped[ResearchSession] = relationship(back_populates="papers")


class ChatMessage(Base):
    """One turn of the follow-up chat over a session's papers."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="chat_messages")
