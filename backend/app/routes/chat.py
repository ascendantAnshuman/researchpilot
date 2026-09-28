"""Follow-up chat routes: ask questions about a finished report, read the history."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import Settings, get_settings
from app.db import get_db
from app.deps import get_chat_llm, get_store
from app.llm import LLM, LLMConfigurationError, LLMError, LLMUnavailableError
from app.models import ResearchSession, User
from app.schemas import ChatExchange, ChatMessageRead, ChatRequest
from app.services import session_store
from app.services.chat_service import answer_follow_up
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _owned_session(db: Session, user: User, session_id: str) -> ResearchSession:
    """Load one of the user's sessions or answer 404."""
    session = session_store.get_user_session(db, user.id, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research session not found")
    return session


@router.get("/{session_id}", response_model=list[ChatMessageRead], summary="Chat history of a session")
def chat_history(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatMessageRead]:
    """All follow-up questions and answers, oldest first."""
    session = _owned_session(db, user, session_id)
    return [session_store.chat_message_read(message) for message in session.chat_messages]


@router.post("/{session_id}", response_model=ChatExchange, summary="Ask a follow-up question")
async def ask(
    session_id: str,
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    llm: LLM = Depends(get_chat_llm),
    store: VectorStore = Depends(get_store),
) -> ChatExchange:
    """Answer a question using RAG over this session's papers; both turns are stored."""
    session = _owned_session(db, user, session_id)
    report = session_store.load_report(session)
    if report is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chat is available once the report is ready")

    history = [(message.role, message.content) for message in session.chat_messages]
    paper_ids = [paper.paper_id for paper in session.papers]
    try:
        answer = await answer_follow_up(payload.message, report, paper_ids, history, llm, store, settings)
    except LLMConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except LLMUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Claude is busy right now, please retry shortly"
        ) from error
    except LLMError as error:
        logger.warning("Chat answer failed for session %s: %s", session_id, error)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    question = session_store.add_chat_message(db, session_id, "user", payload.message)
    reply = session_store.add_chat_message(db, session_id, "assistant", answer.text, answer.citations)
    return ChatExchange(
        question=session_store.chat_message_read(question), answer=session_store.chat_message_read(reply)
    )
