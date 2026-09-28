"""Tests for the Critic agent: claim splitting, evidence pooling, verdicts and routing (fake LLM)."""

from __future__ import annotations

import httpx
import pytest

from app.agents.critic import (
    MAX_POOL_PER_BATCH,
    best_sentence,
    critic_node,
    render_draft,
    route_after_critic,
    verify_claims,
    verify_draft,
)
from app.agents.state import ClaimList, DraftReport, DraftSection, ExtractedClaim, JudgementList
from app.config import Settings
from app.llm import LLMUnavailableError
from app.schemas import RetrievedChunk, VerifiedClaim
from app.services.pdf_parser import blocks_from_abstract, chunk_blocks
from app.services.vector_store import VectorStore
from tests.builders import FALSE_SENTENCE_START, RAG_PAPERS, default_llm, fake_runtime, make_context
from tests.fakes import FakeLLM, prompt_text


@pytest.fixture()
def evidence(vector_store: VectorStore) -> list[RetrievedChunk]:
    """Abstract chunks of the RAG papers, stored and returned as evidence E1..E3."""
    chunks = []
    for paper in RAG_PAPERS:
        paper_chunks = chunk_blocks(
            blocks_from_abstract(paper.abstract), paper_id=paper.paper_id, title=paper.title, source="abstract"
        )
        vector_store.add_chunks(paper_chunks)
        chunks += [RetrievedChunk(**chunk.model_dump(), score=0.9) for chunk in paper_chunks]
    return chunks


DRAFT = DraftReport(
    title="Review",
    executive_summary="Verification lowers hallucination in RAG systems [E1, E2].",
    sections=[
        DraftSection(
            heading="Methods",
            theme="methods",
            paragraphs=[
                "Retrieval grounding reduces hallucination [E1]. A critic verifies each generated claim [E2]. "
                f"{FALSE_SENTENCE_START} eliminate all errors [E1]."
            ],
        )
    ],
    comparison_table=[],
)


def claim(text: str, labels: list[str] | None = None) -> ExtractedClaim:
    """An extracted claim."""
    return ExtractedClaim(claim=text, sentence=f"{text} [E1].", section="Methods", evidence_labels=labels or [])


def test_render_draft_includes_every_section() -> None:
    """The claim splitter sees the summary and every section with citations intact."""
    rendered = render_draft(DRAFT)

    assert rendered.startswith("# Review")
    assert "## Executive summary" in rendered and "## Methods" in rendered
    assert "[E1, E2]" in rendered


def test_best_sentence_picks_most_overlapping_sentence() -> None:
    """Without a quote from Claude, the closest sentence of the passage is shown."""
    text = "Unrelated opening sentence. The hallucination rate drops from 21% to 9%. Closing remark."

    assert best_sentence(text, "hallucination rate drops") == "The hallucination rate drops from 21% to 9%."


async def test_verify_draft_labels_claims_and_attaches_evidence(
    evidence: list[RetrievedChunk], vector_store: VectorStore, test_settings: Settings
) -> None:
    """Every cited sentence becomes a claim; the false one is unsupported, the rest supported with quotes."""
    llm = default_llm()
    paper_ids = [paper.paper_id for paper in RAG_PAPERS]

    verified = await verify_draft(DRAFT, evidence, vector_store, paper_ids, llm, test_settings)

    assert [item.claim_id for item in verified] == ["C1", "C2", "C3", "C4"]
    labels = {item.text: item.label for item in verified}
    assert labels[f"{FALSE_SENTENCE_START} eliminate all errors."] == "unsupported"
    assert sum(label == "supported" for label in labels.values()) == 3
    supported = next(item for item in verified if item.label == "supported")
    assert supported.evidence[0].chunk_id == evidence[0].chunk_id
    assert supported.evidence[0].text == "the hallucination rate drops from 21% to 9%"
    assert llm.calls_for("critic")[0]["effort"] == "medium"  # claim splitting is a light task
    assert llm.calls_for("critic")[1]["effort"] == test_settings.llm_effort


async def test_missing_verdict_counts_as_unsupported(
    evidence: list[RetrievedChunk], vector_store: VectorStore, test_settings: Settings
) -> None:
    """If Claude skips a claim, it is conservatively treated as unsupported."""
    llm = FakeLLM({JudgementList: lambda call: JudgementList(judgements=[])})

    verified = await verify_claims([claim("Grounding helps", ["E1"])], evidence, vector_store,
                                   [RAG_PAPERS[0].paper_id], llm, "high")

    assert verified[0].label == "unsupported"
    assert "no verdict" in verified[0].rationale
    assert verified[0].evidence  # the closest passage is still shown


async def test_claims_are_verified_in_batches_with_bounded_evidence_pools(
    evidence: list[RetrievedChunk], vector_store: VectorStore, test_settings: Settings
) -> None:
    """Twenty claims -> three batches of at most eight, each with at most 12 passages."""
    llm = default_llm()
    claims = [claim(f"Claim number {i} about hallucination", ["E1", "E2", "E3"]) for i in range(20)]

    verified = await verify_claims(claims, evidence, vector_store, [p.paper_id for p in RAG_PAPERS], llm, "high")

    judge_calls = llm.calls_for("critic")
    assert len(judge_calls) == 3
    for call in judge_calls:
        passages = prompt_text(call).split("CLAIMS TO VERIFY")[0].count("\n[")
        assert passages <= MAX_POOL_PER_BATCH
    assert len(verified) == 20


async def test_uncited_claims_get_retrieved_evidence(
    evidence: list[RetrievedChunk], vector_store: VectorStore, test_settings: Settings
) -> None:
    """A claim citing nothing is still checked against passages found by vector search."""
    llm = default_llm()

    await verify_claims([claim("benchmark for hallucination detection", [])], evidence[:1], vector_store,
                        [p.paper_id for p in RAG_PAPERS], llm, "high")

    prompt = prompt_text(llm.calls_for("critic")[0])
    assert "candidate evidence: X1" in prompt


@pytest.mark.parametrize(
    ("labels", "revisions", "max_revisions", "expected"),
    [
        (["supported", "unsupported"], 0, 1, "synthesizer"),
        (["supported", "unsupported"], 1, 1, "finalize"),
        (["supported", "partial"], 0, 1, "finalize"),
        (["unsupported"], 0, 0, "finalize"),
        ([], 0, 1, "finalize"),
    ],
)
def test_route_after_critic(labels: list[str], revisions: int, max_revisions: int, expected: str) -> None:
    """Only unsupported claims trigger a revision, and only while revisions remain."""
    verification = [
        VerifiedClaim(claim_id=f"C{i}", text="t", section="s", sentence="s", label=label, rationale="r")  # type: ignore[arg-type]
        for i, label in enumerate(labels)
    ]
    state = {"verification": verification, "revisions": revisions, "max_revisions": max_revisions}

    assert route_after_critic(state) == expected  # type: ignore[arg-type]


async def test_critic_node_reports_counts(
    evidence: list[RetrievedChunk], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """The node returns the verification and a summary event."""
    runtime, events = fake_runtime(make_context(test_settings, vector_store, http_client))

    update = await critic_node({"draft": DRAFT, "evidence": evidence}, runtime)

    assert len(update["verification"]) == 4
    assert any("3 supported, 0 partial, 1 unsupported" in event["message"] for event in events)


async def test_critic_node_survives_llm_outage(
    evidence: list[RetrievedChunk], test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """If Claude is unavailable, the report continues unverified with a warning."""
    llm = FakeLLM({ClaimList: lambda call: LLMUnavailableError("overloaded")})
    runtime, events = fake_runtime(make_context(test_settings, vector_store, http_client, llm=llm))

    update = await critic_node({"draft": DRAFT, "evidence": evidence}, runtime)

    assert update["verification"] == []
    assert "Critic failed" in update["warnings"][0]
    assert any(event["status"] == "warning" for event in events)
