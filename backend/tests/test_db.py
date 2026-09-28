"""Tests for the database layer: tables, constraints, cascades and UTC timestamps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import check_database, get_engine, get_session_factory, session_scope
from app.models import ChatMessage, Report, ResearchSession, SessionPaper, SessionStatus, User


def make_user(db: Session, email: str = "ada@university.edu") -> User:
    """Insert and return a user with a placeholder password hash."""
    user = User(email=email, hashed_password="not-a-real-hash")
    db.add(user)
    db.commit()
    return user


def test_all_tables_are_created(db_session: Session) -> None:
    """Start-up creates every table the app needs."""
    tables = set(inspect(get_engine()).get_table_names())

    assert {"users", "research_sessions", "reports", "session_papers", "chat_messages"} <= tables


def test_sqlite_pragmas_are_applied(db_session: Session) -> None:
    """Foreign keys are enforced and WAL journaling is on."""
    assert db_session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    assert db_session.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"


def test_user_email_is_unique(db_session: Session) -> None:
    """Two accounts cannot share an email."""
    make_user(db_session)
    db_session.add(User(email="ada@university.edu", hashed_password="x"))

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_foreign_keys_are_enforced(db_session: Session) -> None:
    """A research session must belong to an existing user."""
    db_session.add(ResearchSession(user_id=9999, question="orphan?"))

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_research_session_defaults(db_session: Session) -> None:
    """New sessions get a URL-safe hex id, pending status and empty JSON fields."""
    user = make_user(db_session)
    research = ResearchSession(user_id=user.id, question="What is RAG?")
    db_session.add(research)
    db_session.commit()

    assert len(research.id) == 32
    int(research.id, 16)  # raises if the id is not hexadecimal
    assert research.status == SessionStatus.PENDING
    assert research.settings_snapshot == {}
    assert research.metrics == {}
    assert research.completed_at is None


def test_timestamps_come_back_timezone_aware(db_session: Session) -> None:
    """Datetimes loaded from SQLite are UTC-aware, not naive."""
    user = make_user(db_session)
    user_id = user.id

    with get_session_factory()() as fresh_session:
        loaded = fresh_session.get(User, user_id)
        assert loaded is not None
        assert loaded.created_at.tzinfo is not None
        assert loaded.created_at.utcoffset() == timedelta(0)


def test_non_utc_datetimes_are_stored_as_utc(db_session: Session) -> None:
    """A time written in another timezone is read back as the same instant in UTC."""
    user = make_user(db_session)
    plus_five_thirty = timezone(timedelta(hours=5, minutes=30))
    local_time = datetime(2026, 1, 1, 17, 30, tzinfo=plus_five_thirty)  # == 12:00 UTC
    research = ResearchSession(user_id=user.id, question="q", completed_at=local_time)
    db_session.add(research)
    db_session.commit()
    research_id = research.id

    with get_session_factory()() as fresh_session:
        loaded = fresh_session.get(ResearchSession, research_id)
        assert loaded is not None
        assert loaded.completed_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_deleting_a_user_cascades_to_everything(db_session: Session) -> None:
    """Removing a user removes their sessions, reports, papers and chat history."""
    user = make_user(db_session)
    research = ResearchSession(user_id=user.id, question="What reduces hallucination in RAG?")
    research.report = Report(title="Review", markdown="# Review", content={"sections": []})
    research.papers = [SessionPaper(paper_id="arxiv:2401.00001", title="Paper A", rank=1)]
    research.chat_messages = [ChatMessage(role="user", content="Which dataset is most common?")]
    db_session.add(research)
    db_session.commit()

    db_session.delete(user)
    db_session.commit()

    for model in (ResearchSession, Report, SessionPaper, ChatMessage):
        assert db_session.scalar(select(func.count()).select_from(model)) == 0


def test_database_level_cascade_without_orm(db_session: Session) -> None:
    """ON DELETE CASCADE also works for raw SQL deletes (e.g. from a DB browser)."""
    user = make_user(db_session)
    research = ResearchSession(user_id=user.id, question="q")
    research.papers = [SessionPaper(paper_id="s2:abc", title="Paper B")]
    db_session.add(research)
    db_session.commit()

    db_session.execute(text("DELETE FROM research_sessions"))
    db_session.commit()

    assert db_session.scalar(select(func.count()).select_from(SessionPaper)) == 0


def test_same_paper_cannot_be_added_twice_to_a_session(db_session: Session) -> None:
    """(session_id, paper_id) is unique so de-duplication bugs surface immediately."""
    user = make_user(db_session)
    research = ResearchSession(user_id=user.id, question="q")
    research.papers = [
        SessionPaper(paper_id="arxiv:1", title="A"),
        SessionPaper(paper_id="arxiv:1", title="A again"),
    ]
    db_session.add(research)

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_user_sessions_are_listed_newest_first(db_session: Session) -> None:
    """The history sidebar relies on newest-first ordering."""
    user = make_user(db_session)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for day, question in enumerate(["first", "second", "third"]):
        db_session.add(ResearchSession(user_id=user.id, question=question, created_at=base + timedelta(days=day)))
    db_session.commit()
    db_session.expire_all()

    assert [s.question for s in user.research_sessions] == ["third", "second", "first"]


def test_session_scope_commits_on_success(db_session: Session) -> None:
    """session_scope persists changes when the block finishes normally."""
    with session_scope() as scoped:
        scoped.add(User(email="grace@university.edu", hashed_password="x"))

    assert db_session.scalar(select(func.count()).select_from(User)) == 1


def test_session_scope_rolls_back_on_error(db_session: Session) -> None:
    """session_scope discards changes when the block raises."""
    with pytest.raises(RuntimeError), session_scope() as scoped:
        scoped.add(User(email="linus@university.edu", hashed_password="x"))
        scoped.flush()
        raise RuntimeError("boom")

    assert db_session.scalar(select(func.count()).select_from(User)) == 0


def test_check_database_reports_ok(db_session: Session) -> None:
    """The health helper can reach the test database."""
    assert check_database() is True
