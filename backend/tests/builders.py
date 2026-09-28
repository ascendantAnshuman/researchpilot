"""Test data builders shared by the agent and pipeline tests.

The fake-Claude handlers here read the prompts they receive (evidence labels,
claim ids, draft sentences) and answer consistently, so the whole pipeline can
run end-to-end offline. One deliberately false sentence ("Quantum widgets ...")
lets tests check that the Critic catches it and the revision removes it.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import httpx

from app.agents.state import (
    ClaimJudgement,
    ClaimList,
    DraftComparisonRow,
    DraftReport,
    DraftSection,
    ExtractedClaim,
    JudgementList,
    PipelineContext,
    ResearchPlan,
    SubQuery,
)
from app.config import Settings
from app.schemas import Paper, PaperSummary
from app.services.vector_store import VectorStore
from tests.fakes import FakeLLM, FakeSearchClient, prompt_text

FALSE_SENTENCE_START = "Quantum widgets"


def make_paper(
    number: int,
    title: str,
    abstract: str,
    *,
    year: int = 2024,
    citations: int = 0,
    pdf_url: str | None = None,
    source: str = "arxiv",
    doi: str | None = None,
) -> Paper:
    """A paper as a search client would return it."""
    arxiv_id = f"2401.{number:05d}" if source == "arxiv" else None
    return Paper(
        paper_id=f"arxiv:{arxiv_id}" if arxiv_id else f"s2:paper{number}",
        sources=[source],  # type: ignore[list-item]
        title=title,
        authors=["Ada Lovelace", "Alan Turing"],
        year=year,
        venue="arXiv" if source == "arxiv" else "ACL",
        abstract=abstract,
        url=f"https://example.org/{number}",
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        citation_count=citations,
    )


RAG_PAPERS: list[Paper] = [
    make_paper(
        1,
        "Reducing Hallucination in Retrieval-Augmented Generation",
        "We reduce hallucination in retrieval augmented generation by grounding answers in retrieved passages. "
        "On HaluEval the hallucination rate drops from 21% to 9%.",
        citations=120,
    ),
    make_paper(
        2,
        "A Critic Model for Verifying Generated Claims",
        "We train a critic that verifies each generated claim against retrieved evidence and flags unsupported "
        "statements in retrieval augmented generation outputs.",
        citations=40,
        year=2025,
    ),
    make_paper(
        3,
        "Benchmarking Hallucination Detection",
        "We introduce a benchmark for hallucination detection in retrieval augmented generation with human labels "
        "and evaluate several detectors on it.",
        citations=15,
        year=2023,
    ),
]

PLAN = ResearchPlan(
    title="Hallucination in RAG",
    sub_queries=[
        SubQuery(
            query="methods for reducing hallucination in retrieval augmented generation",
            keywords=['"retrieval augmented generation"', "hallucination"],
            rationale="Methods.",
        ),
        SubQuery(
            query="hallucination detection benchmarks for RAG",
            keywords=["hallucination", "benchmark"],
            rationale="Datasets.",
        ),
        SubQuery(
            query="limitations of hallucination mitigation in RAG",
            keywords=["hallucination", "limitations"],
            rationale="Open problems.",
        ),
    ],
)

SUMMARY = PaperSummary(
    problem="Language models hallucinate unsupported facts.",
    method="Ground answers in retrieved passages and verify them.",
    datasets=["HaluEval"],
    metrics=["hallucination rate"],
    key_results=["Hallucination rate drops from 21% to 9% on HaluEval."],
    limitations=["Evaluated only in English."],
)

_EVIDENCE_LABEL_RE = re.compile(r"^\[(E\d+)\]", re.MULTILINE)
_PAPER_LABEL_RE = re.compile(r"^\[(P\d+)\]", re.MULTILINE)
_CITED_SENTENCE_RE = re.compile(r"[^.\n]*?\[E\d+(?:, E\d+)*\]\.")
_CLAIM_LINE_RE = re.compile(r"^\[(C\d+)\] (.+)$", re.MULTILINE)


def draft_handler(call: dict[str, Any]) -> DraftReport:
    """Writes a small draft citing the evidence it was given; the first draft contains one false sentence."""
    text = prompt_text(call)
    evidence = _EVIDENCE_LABEL_RE.findall(text)
    papers = _PAPER_LABEL_RE.findall(text)
    first, second = evidence[0], evidence[1] if len(evidence) > 1 else evidence[0]
    methods = [
        f"Retrieval grounding reduces hallucination [{first}].",
        f"A critic verifies each generated claim [{second}].",
    ]
    if "FLAGGED CLAIMS" not in text:
        methods.append(f"{FALSE_SENTENCE_START} eliminate all errors [{first}].")
    return DraftReport(
        title="Review: hallucination in RAG",
        executive_summary=f"Verification lowers hallucination in RAG systems [{first}, {second}].",
        sections=[
            DraftSection(heading="Methods", theme="methods", paragraphs=[" ".join(methods)]),
            DraftSection(heading="Open problems", theme="open_problems", paragraphs=[f"Evaluation remains hard [{second}]."]),
        ],
        comparison_table=[
            DraftComparisonRow(paper_label=label, method="Grounded generation", dataset="HaluEval", key_result="21% to 9%")
            for label in papers[:2]
        ],
    )


def claims_handler(call: dict[str, Any]) -> ClaimList:
    """Turns every cited sentence of the draft into one claim."""
    draft_text = prompt_text(call).split("DRAFT", 1)[-1]
    claims = []
    for sentence in _CITED_SENTENCE_RE.findall(draft_text):
        sentence = sentence.strip()
        claims.append(
            ExtractedClaim(
                claim=sentence.split(" [")[0] + ".",
                sentence=sentence,
                section="Methods",
                evidence_labels=re.findall(r"E\d+", sentence),
            )
        )
    return ClaimList(claims=claims)


def judgement_handler(call: dict[str, Any]) -> JudgementList:
    """Supports every claim except the deliberately false one."""
    judgements = []
    for claim_id, claim in _CLAIM_LINE_RE.findall(prompt_text(call)):
        false = claim.startswith(FALSE_SENTENCE_START)
        judgements.append(
            ClaimJudgement(
                claim_id=claim_id,
                label="unsupported" if false else "supported",
                evidence_labels=[] if false else ["E1"],
                quote="" if false else "the hallucination rate drops from 21% to 9%",
                rationale="Not stated in any passage." if false else "Stated in E1.",
            )
        )
    return JudgementList(judgements=judgements)


def default_llm() -> FakeLLM:
    """A fake Claude answering every pipeline schema consistently."""
    return FakeLLM(
        {
            ResearchPlan: lambda call: PLAN,
            PaperSummary: lambda call: SUMMARY,
            DraftReport: draft_handler,
            ClaimList: claims_handler,
            JudgementList: judgement_handler,
        }
    )


def make_context(
    settings: Settings,
    store: VectorStore,
    http: httpx.AsyncClient,
    *,
    llm: FakeLLM | None = None,
    arxiv: FakeSearchClient | None = None,
    semantic_scholar: FakeSearchClient | None = None,
) -> PipelineContext:
    """A pipeline context wired with fakes."""
    return PipelineContext(
        settings=settings,
        llm=llm or default_llm(),
        vector_store=store,
        arxiv=arxiv or FakeSearchClient(RAG_PAPERS),
        semantic_scholar=semantic_scholar or FakeSearchClient([]),
        http=http,
    )


def fake_runtime(context: PipelineContext) -> tuple[Any, list[dict[str, Any]]]:
    """An object shaped like LangGraph's Runtime, collecting streamed events."""
    events: list[dict[str, Any]] = []
    return SimpleNamespace(context=context, stream_writer=events.append), events
