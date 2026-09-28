"""Tests for the Synthesizer agent: evidence gathering, prompting, sanitising and revision (fake LLM)."""

from __future__ import annotations

import httpx
import pytest

from app.agents.state import DraftComparisonRow, DraftReport, DraftSection
from app.agents.synthesizer import (
    build_messages,
    gather_evidence,
    label_evidence,
    label_papers,
    sanitize_draft,
    synthesizer_node,
)
from app.config import Settings
from app.llm import LLMOutputError
from app.schemas import PaperDigest, VerifiedClaim
from app.services.pdf_parser import TextBlock, blocks_from_abstract, chunk_blocks
from app.services.vector_store import VectorStore
from tests.builders import FALSE_SENTENCE_START, PLAN, RAG_PAPERS, SUMMARY, default_llm, fake_runtime, make_context
from tests.fakes import FakeLLM

QUESTION = "What reduces hallucination in retrieval augmented generation?"


@pytest.fixture()
def digests(vector_store: VectorStore) -> list[PaperDigest]:
    """The three RAG papers stored as abstract chunks, with summaries."""
    for paper in RAG_PAPERS:
        vector_store.add_chunks(
            chunk_blocks(blocks_from_abstract(paper.abstract), paper_id=paper.paper_id, title=paper.title, source="abstract")
        )
    return [PaperDigest(paper=paper, summary=SUMMARY, chunk_count=1) for paper in RAG_PAPERS]


def test_evidence_covers_every_paper_in_ranking_order(
    digests: list[PaperDigest], vector_store: VectorStore, test_settings: Settings
) -> None:
    """Each paper contributes at least one passage; passages are grouped by paper rank."""
    evidence = gather_evidence(QUESTION, PLAN, digests, vector_store, test_settings)

    assert [chunk.paper_id for chunk in evidence] == [paper.paper_id for paper in RAG_PAPERS]


def test_evidence_respects_the_budget_but_keeps_coverage(vector_store: VectorStore, test_settings: Settings) -> None:
    """With many chunks, the budget is enforced and every paper is still represented."""
    long_digests = []
    for paper in RAG_PAPERS:
        text = " ".join(f"Sentence {i} about hallucination in retrieval augmented generation." for i in range(400))
        chunks = chunk_blocks([TextBlock(1, "Body", text)], paper_id=paper.paper_id, title=paper.title)
        vector_store.add_chunks(chunks)
        long_digests.append(PaperDigest(paper=paper, chunk_count=len(chunks), full_text=True))
    settings = test_settings.model_copy(update={"max_evidence_chunks": 5})

    evidence = gather_evidence(QUESTION, PLAN, long_digests, vector_store, settings)

    assert len(evidence) == 5
    assert {chunk.paper_id for chunk in evidence} == {paper.paper_id for paper in RAG_PAPERS}


def test_labels_are_sequential(digests: list[PaperDigest], vector_store: VectorStore, test_settings: Settings) -> None:
    """Evidence is labelled E1.., usable papers P1.."""
    evidence = gather_evidence(QUESTION, PLAN, digests, vector_store, test_settings)

    assert list(label_evidence(evidence)) == ["E1", "E2", "E3"]
    assert list(label_papers(digests)) == ["P1", "P2", "P3"]


def test_sanitize_draft_removes_bad_citations_and_rows() -> None:
    """Citations to unknown evidence and table rows for unknown papers are dropped."""
    draft = DraftReport(
        title=" Title ",
        executive_summary="Grounding helps [E1, E9]. Unknown source [E9].",
        sections=[
            DraftSection(heading="Methods", theme="methods", paragraphs=["Critics help [E2; E2].", "Only bad [E99]"]),
            DraftSection(heading="Empty", theme="other", paragraphs=["[E42]"]),
        ],
        comparison_table=[
            DraftComparisonRow(paper_label="[p1]", method="m", dataset="d", key_result="r"),
            DraftComparisonRow(paper_label="P1", method="dup", dataset="d", key_result="r"),
            DraftComparisonRow(paper_label="P9", method="m", dataset="d", key_result="r"),
        ],
    )

    cleaned, warnings = sanitize_draft(draft, {"E1", "E2"}, {"P1", "P2"})

    assert cleaned.title == "Title"
    assert cleaned.executive_summary == "Grounding helps [E1]. Unknown source."
    assert cleaned.sections[0].paragraphs == ["Critics help [E2].", "Only bad"]
    assert [section.heading for section in cleaned.sections] == ["Methods"]
    assert [row.paper_label for row in cleaned.comparison_table] == ["P1"]
    assert warnings == ["Removed citations to unknown evidence: E42, E9, E99"]


def test_messages_mark_the_context_block_as_cacheable() -> None:
    """Papers+evidence form a cacheable prefix; the revision adds feedback and the old draft after it."""
    previous = DraftReport(title="Old", executive_summary="x [E1].", sections=[], comparison_table=[])
    flagged = [
        VerifiedClaim(claim_id="C3", text="Bad claim", section="Methods", sentence="Bad claim [E1].",
                      label="unsupported", rationale="Not in evidence.")
    ]

    first = build_messages(QUESTION, PLAN, "PAPERS...", "EVIDENCE...")
    revision = build_messages(QUESTION, PLAN, "PAPERS...", "EVIDENCE...", previous, flagged)

    first_blocks, revision_blocks = first[0]["content"], revision[0]["content"]
    assert first_blocks[0] == revision_blocks[0]  # identical prefix -> prompt-cache hit on revision
    assert first_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "Write the literature review." in first_blocks[1]["text"]
    assert "[UNSUPPORTED] Bad claim" in revision_blocks[1]["text"]
    assert '"title":"Old"' in revision_blocks[1]["text"].replace(" ", "")


async def test_first_draft(
    digests: list[PaperDigest], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """The node gathers evidence and returns a draft citing it."""
    context = make_context(test_settings, vector_store, http_client)
    runtime, _ = fake_runtime(context)

    update = await synthesizer_node({"question": QUESTION, "plan": PLAN, "digests": digests}, runtime)

    draft = update["draft"]
    assert "revisions" not in update
    assert len(update["evidence"]) == 3
    assert any(FALSE_SENTENCE_START in p for s in draft.sections for p in s.paragraphs)
    assert [row.paper_label for row in draft.comparison_table] == ["P1", "P2"]
    assert context.llm.calls_for("synthesizer")[0]["effort"] == test_settings.llm_effort


async def test_revision_uses_same_evidence_and_feedback(
    digests: list[PaperDigest], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """A revision reuses the evidence, sends the flagged claims and increments the counter."""
    context = make_context(test_settings, vector_store, http_client)
    runtime, _ = fake_runtime(context)
    first = await synthesizer_node({"question": QUESTION, "plan": PLAN, "digests": digests}, runtime)
    flagged = VerifiedClaim(claim_id="C4", text=f"{FALSE_SENTENCE_START} eliminate all errors.", section="Methods",
                            sentence="...", label="unsupported", rationale="Not stated.")

    update = await synthesizer_node(
        {"question": QUESTION, "plan": PLAN, "digests": digests, "draft": first["draft"],
         "evidence": first["evidence"], "verification": [flagged], "revisions": 0},
        runtime,
    )

    assert update["revisions"] == 1
    assert update["evidence"] == first["evidence"]
    assert not any(FALSE_SENTENCE_START in p for s in update["draft"].sections for p in s.paragraphs)
    revision_prompt = context.llm.calls_for("synthesizer")[1]["messages"][0]["content"][1]["text"]
    assert "FLAGGED CLAIMS" in revision_prompt


async def test_failed_revision_keeps_previous_draft(
    digests: list[PaperDigest], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """If the revision call fails, the old draft stays and the run continues."""
    llm = default_llm()
    context = make_context(test_settings, vector_store, http_client, llm=llm)
    runtime, events = fake_runtime(context)
    first = await synthesizer_node({"question": QUESTION, "plan": PLAN, "digests": digests}, runtime)
    llm.handlers[DraftReport] = lambda call: LLMOutputError("truncated")
    flagged = VerifiedClaim(claim_id="C1", text="x", section="s", sentence="x", label="unsupported", rationale="r")

    update = await synthesizer_node(
        {"question": QUESTION, "plan": PLAN, "digests": digests, "draft": first["draft"],
         "evidence": first["evidence"], "verification": [flagged], "revisions": 0},
        runtime,
    )

    assert update["revisions"] == 1
    assert "draft" not in update
    assert any(event["status"] == "warning" for event in events)


async def test_first_draft_failure_propagates(
    digests: list[PaperDigest], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """Without a first draft there is no report, so the error is raised."""
    llm = FakeLLM({DraftReport: lambda call: LLMOutputError("truncated")})
    runtime, _ = fake_runtime(make_context(test_settings, vector_store, http_client, llm=llm))

    with pytest.raises(LLMOutputError):
        await synthesizer_node({"question": QUESTION, "plan": PLAN, "digests": digests}, runtime)
