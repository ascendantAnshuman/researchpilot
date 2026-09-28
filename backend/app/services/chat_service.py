"""Follow-up chat: answers questions about a finished report with RAG over its papers.

For each question:
    1. Retrieve the most relevant chunks, limited to the papers of this session.
    2. Send Claude the report's title and summary, the recent conversation, and
       the passages labelled [E1], [E2], ...; the answer must cite them.
    3. Convert [E#] to the report's reference numbers [n] (papers the report
       did not cite get the next free numbers) and return the cited passages.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.agents.critic import best_sentence
from app.agents.finalize import CitationNumberer
from app.config import Settings
from app.llm import LLM, lighter_effort
from app.schemas import ChatCitation, ResearchReport, RetrievedChunk
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are ResearchPilot's research assistant. You answer follow-up questions about a literature review.

Rules:
- Use ONLY the evidence passages provided in the latest message. Do not use outside knowledge.
- End every factual sentence with its evidence citations in square brackets, e.g. "... [E2]" or "... [E1, E4]".
- If the passages do not answer the question, say so plainly and suggest a more specific follow-up search.
- Be concise: at most about 200 words unless the user asks for more detail. Plain prose, no headings."""


@dataclass
class ChatAnswer:
    """Claude's answer with citations converted to reference numbers."""

    text: str
    citations: list[ChatCitation]


def format_chat_context(report: ResearchReport, evidence: dict[str, RetrievedChunk], question: str) -> str:
    """The context block sent with each follow-up question."""
    passages = "\n\n".join(
        f"[{label}] {chunk.title} ({'abstract' if chunk.source == 'abstract' else f'{chunk.section}, p. {chunk.page}'})"
        f"\n{chunk.text}"
        for label, chunk in evidence.items()
    )
    return (
        f"REPORT: {report.title}\nSUMMARY: {report.executive_summary}\n\n"
        f"EVIDENCE PASSAGES\n{passages or '(no relevant passages found)'}\n\n"
        f"QUESTION: {question}"
    )


async def answer_follow_up(
    question: str,
    report: ResearchReport,
    paper_ids: Sequence[str],
    history: Sequence[tuple[str, str]],
    llm: LLM,
    store: VectorStore,
    settings: Settings,
) -> ChatAnswer:
    """Answer a follow-up question using the session's stored papers.

    Args:
        question: The user's question.
        report: The session's report (for context and reference numbers).
        paper_ids: Papers of the session (retrieval is limited to them).
        history: Earlier ``(role, content)`` turns, oldest first.
        llm: The Claude client.
        store: The vector store holding the papers' chunks.
        settings: Configuration.
    """
    chunks = await asyncio.to_thread(store.query, question, paper_ids=list(paper_ids), top_k=settings.retrieval_top_k)
    evidence = {f"E{index}": chunk for index, chunk in enumerate(chunks, start=1)}

    recent = list(history[-2 * settings.chat_history_turns :]) if settings.chat_history_turns else []
    while recent and recent[0][0] != "user":  # the conversation sent to Claude must start with the user
        recent.pop(0)
    messages: list[dict[str, object]] = [{"role": role, "content": content} for role, content in recent]
    messages.append({"role": "user", "content": format_chat_context(report, evidence, question)})

    raw_answer = await llm.text(
        stage="chat", system=CHAT_SYSTEM_PROMPT, messages=messages, effort=lighter_effort(settings.llm_effort)
    )

    numberer = CitationNumberer(evidence, initial={ref.paper_id: ref.number for ref in report.references})
    used = numberer.labels_used(raw_answer)
    text = numberer.convert(raw_answer)
    citations: list[ChatCitation] = []
    seen_chunks: set[str] = set()
    for label in used:
        chunk = evidence[label]
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)
        citations.append(
            ChatCitation(
                number=numberer.number_for(chunk.paper_id),
                paper_id=chunk.paper_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                page=chunk.page,
                snippet=best_sentence(chunk.text, raw_answer),
            )
        )
    citations.sort(key=lambda citation: citation.number)
    return ChatAnswer(text=text or "I could not produce an answer from the stored papers.", citations=citations)
