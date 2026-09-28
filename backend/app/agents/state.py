"""Shared building blocks for the LangGraph pipeline.

* :class:`ResearchState` - the dictionary passed from node to node. Each node
  returns only the keys it changes; LangGraph merges them in.
* :class:`PipelineContext` - run-scoped dependencies (settings, LLM, vector store,
  search clients). LangGraph hands it to every node as ``runtime.context``, which
  is what lets tests swap in fakes.
* :func:`emit` and :func:`stage_node` - progress events streamed to the CLI now
  and to the browser (over SSE) in Phase 5.
* Pydantic models describing Claude's structured outputs for the Planner,
  Synthesizer and Critic.
"""

from __future__ import annotations

import functools
import logging
import operator
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol, TypedDict

import httpx
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.config import Settings
from app.llm import LLM
from app.schemas import (
    Paper,
    PaperDigest,
    ProgressEvent,
    ProgressStatus,
    ResearchReport,
    RetrievedChunk,
    SectionTheme,
    VerifiedClaim,
)
from app.services.vector_store import Embedder, VectorStore

logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("planner", "search", "reader", "synthesizer", "critic", "finalize")


# ---------------------------------------------------------------------------
# Claude output schemas
# ---------------------------------------------------------------------------


class SubQuery(BaseModel):
    """One focused search the Planner wants to run."""

    query: str = Field(description="A natural-language search query for Semantic Scholar.")
    keywords: list[str] = Field(description="2-5 arXiv keywords; wrap multi-word technical terms in double quotes.")
    rationale: str = Field(description="Which part of the question this sub-query covers.")


class ResearchPlan(BaseModel):
    """The Planner's output."""

    title: str = Field(description="A short working title for the literature review.")
    sub_queries: list[SubQuery] = Field(description="3 to 5 complementary sub-queries.")


class DraftSection(BaseModel):
    """A themed section written by the Synthesizer, citing evidence as [E#]."""

    heading: str
    theme: SectionTheme
    paragraphs: list[str]


class DraftComparisonRow(BaseModel):
    """A comparison-table row referring to a paper by its [P#] label."""

    paper_label: str = Field(description="The paper's label, e.g. 'P3'.")
    method: str
    dataset: str
    key_result: str


class DraftReport(BaseModel):
    """The Synthesizer's output before citations are renumbered and claims verified."""

    title: str
    executive_summary: str
    sections: list[DraftSection]
    comparison_table: list[DraftComparisonRow]


class ExtractedClaim(BaseModel):
    """One atomic, checkable statement taken from the draft."""

    claim: str = Field(description="The claim as a short standalone sentence.")
    sentence: str = Field(description="The draft sentence it comes from, copied exactly.")
    section: str = Field(description="Heading of the section, or 'Executive summary'.")
    evidence_labels: list[str] = Field(description="The [E#] labels cited in that sentence, e.g. ['E3'].")


class ClaimList(BaseModel):
    """The Critic's claim-splitting output."""

    claims: list[ExtractedClaim]


class ClaimJudgement(BaseModel):
    """The Critic's verdict on one claim."""

    claim_id: str = Field(description="The claim's id, e.g. 'C4'.")
    label: Literal["supported", "partial", "unsupported"]
    evidence_labels: list[str] = Field(description="Evidence labels that support the claim, best first.")
    quote: str = Field(description="The exact supporting sentence from the best passage, or '' if none.")
    rationale: str = Field(description="One sentence explaining the verdict.")


class JudgementList(BaseModel):
    """The Critic's verification output for one batch of claims."""

    judgements: list[ClaimJudgement]


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


class PaperSearchClient(Protocol):
    """Anything that can search for papers (ArxivClient, SemanticScholarClient, fakes)."""

    async def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        """Return papers matching ``query``."""
        ...


class PaperLookupClient(PaperSearchClient, Protocol):
    """A search client that can also look papers up by id (Semantic Scholar)."""

    async def get_papers(self, ids: Sequence[str]) -> list[Paper | None]:
        """Return papers aligned with ``ids`` (None where unknown)."""
        ...


@dataclass
class PipelineContext:
    """Dependencies shared by all nodes of one research run."""

    settings: Settings
    llm: LLM
    vector_store: VectorStore
    arxiv: PaperSearchClient
    semantic_scholar: PaperLookupClient
    http: httpx.AsyncClient

    @property
    def embedder(self) -> Embedder:
        """The embedding model (shared with the vector store)."""
        return self.vector_store.embedder


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


def add_seconds(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Reducer: sum stage durations (the Synthesizer and Critic can run twice)."""
    merged = dict(left)
    for stage, seconds in right.items():
        merged[stage] = merged.get(stage, 0.0) + seconds
    return merged


class ResearchState(TypedDict, total=False):
    """Everything the pipeline knows about one research question."""

    question: str
    critic_enabled: bool
    max_revisions: int
    plan: ResearchPlan
    papers: list[Paper]
    digests: list[PaperDigest]
    evidence: list[RetrievedChunk]
    draft: DraftReport
    verification: list[VerifiedClaim]
    revisions: int
    report: ResearchReport
    warnings: Annotated[list[str], operator.add]
    stage_seconds: Annotated[dict[str, float], add_seconds]


PipelineRuntime = Runtime[PipelineContext]
NodeFunction = Callable[[ResearchState, PipelineRuntime], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------


def emit(
    runtime: PipelineRuntime,
    stage: str,
    message: str,
    status: ProgressStatus = "progress",
    **data: Any,
) -> None:
    """Log a progress message and stream it to whoever is listening (CLI, SSE)."""
    event = ProgressEvent(stage=stage, status=status, message=message, data=data, timestamp=datetime.now(UTC))
    log = logger.warning if status in ("warning", "error") else logger.info
    log("[%s] %s", stage, message)
    runtime.stream_writer(event.model_dump(mode="json"))


def stage_node(stage: str) -> Callable[[NodeFunction], NodeFunction]:
    """Decorator for graph nodes: emits started/completed events and records the stage's duration."""

    def decorator(function: NodeFunction) -> NodeFunction:
        @functools.wraps(function)
        async def wrapper(state: ResearchState, runtime: PipelineRuntime) -> dict[str, Any]:
            emit(runtime, stage, f"{stage.capitalize()} started", "started")
            started = time.perf_counter()
            update = dict(await function(state, runtime))
            seconds = time.perf_counter() - started
            update["stage_seconds"] = {stage: seconds}
            emit(runtime, stage, f"{stage.capitalize()} finished in {seconds:.1f}s", "completed", seconds=round(seconds, 2))
            return update

        return wrapper

    return decorator


def usable_digests(digests: Sequence[PaperDigest]) -> list[PaperDigest]:
    """Papers for which at least some text (full text or abstract) was stored."""
    return [digest for digest in digests if digest.chunk_count > 0]
