"""Tests for persisting research sessions, reports and chat messages."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.graph import run_research
from app.config import Settings
from app.models import ResearchSession, SessionPaper, SessionStatus, User
from app.schemas import ChatCitation, ResearchReport
from app.services import session_store
from app.services.vector_store import VectorStore
from tests.builders import make_context

QUESTION = "What are recent methods for reducing hallucination in RAG systems?"


@pytest.fixture()
async def report(test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient) -> ResearchReport:
    """A complete report from the offline pipeline."""
    return await run_research(QUESTION, context=make_context(test_settings, vector_store, http_client))


@pytest.fixture()
def user(db_session: Session) -> User:
    """A stored user."""
    account = User(email="ada@university.edu", hashed_password="x")
    db_session.add(account)
    db_session.commit()
    return account


def new_session(db: Session, user: User, test_settings: Settings) -> ResearchSession:
    """A pending session with a settings snapshot."""
    return session_store.create_session(db, user.id, QUESTION, session_store.settings_snapshot(test_settings))


def test_created_session_is_pending_with_snapshot(db_session: Session, user: User, test_settings: Settings) -> None:
    """New sessions start pending and remember the settings they will run with."""
    session = new_session(db_session, user, test_settings)

    assert session.status == SessionStatus.PENDING
    assert session.settings_snapshot["llm_model"] == test_settings.llm_model
    assert session.settings_snapshot["enable_critic"] is True
    assert session_store.count_active_sessions(db_session, user.id) == 1


def test_save_report_round_trip(
    db_session: Session, user: User, test_settings: Settings, report: ResearchReport
) -> None:
    """Everything saved comes back identical, and papers get their own rows."""
    session = new_session(db_session, user, test_settings)
    session_store.mark_running(session.id)

    session_store.save_report(session.id, report)
    db_session.expire_all()

    stored = session_store.get_user_session(db_session, user.id, session.id)
    assert stored is not None
    assert stored.status == SessionStatus.COMPLETED
    assert stored.completed_at is not None
    assert session_store.load_report(stored) == report
    papers = list(db_session.scalars(select(SessionPaper).where(SessionPaper.session_id == session.id)))
    assert [paper.rank for paper in papers] == [1, 2, 3]
    assert papers[0].summary["datasets"] == ["HaluEval"]
    detail = session_store.session_detail(stored)
    assert detail.metrics is not None and detail.metrics.faithfulness == report.metrics.faithfulness
    assert session_store.count_active_sessions(db_session, user.id) == 0


def test_stage_and_failure_updates(db_session: Session, user: User, test_settings: Settings) -> None:
    """The current stage is tracked and failures keep their reason."""
    session = new_session(db_session, user, test_settings)

    session_store.mark_running(session.id)
    session_store.update_stage(session.id, "reader")
    db_session.expire_all()
    assert (session.status, session.current_stage) == (SessionStatus.RUNNING, "reader")

    session_store.mark_failed(session.id, "Claude is temporarily unavailable")
    db_session.expire_all()
    assert session.status == SessionStatus.FAILED
    assert session.error_message == "Claude is temporarily unavailable"
    assert session.current_stage is None


def test_interrupted_runs_are_failed_at_startup(db_session: Session, user: User, test_settings: Settings) -> None:
    """Runs left pending/running by a previous server process are marked failed."""
    pending = new_session(db_session, user, test_settings)
    running = new_session(db_session, user, test_settings)
    session_store.mark_running(running.id)

    assert session_store.fail_interrupted_sessions() == 2
    db_session.expire_all()

    assert pending.status == running.status == SessionStatus.FAILED
    assert pending.error_message == session_store.INTERRUPTED_MESSAGE
    assert session_store.fail_interrupted_sessions() == 0


def test_sessions_are_scoped_to_their_owner(db_session: Session, user: User, test_settings: Settings) -> None:
    """Lookups by another user find nothing."""
    session = new_session(db_session, user, test_settings)
    other = User(email="grace@university.edu", hashed_password="x")
    db_session.add(other)
    db_session.commit()

    assert session_store.get_user_session(db_session, other.id, session.id) is None
    assert session_store.list_user_sessions(db_session, other.id) == []
    assert [s.id for s in session_store.list_user_sessions(db_session, user.id)] == [session.id]


def test_chat_messages_round_trip(db_session: Session, user: User, test_settings: Settings) -> None:
    """Chat turns keep their citations."""
    session = new_session(db_session, user, test_settings)
    citation = ChatCitation(number=1, paper_id="arxiv:1", title="T", chunk_id="arxiv:1::c000", section="Abstract",
                            page=0, snippet="Grounding helps.")

    stored = session_store.add_chat_message(db_session, session.id, "assistant", "Grounding helps [1].", [citation])

    read = session_store.chat_message_read(stored)
    assert read.role == "assistant"
    assert read.citations == [citation]
