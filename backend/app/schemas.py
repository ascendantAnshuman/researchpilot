"""Pydantic v2 request/response schemas shared across the API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

BCRYPT_MAX_PASSWORD_BYTES = 72


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Payload returned by ``GET /api/health``."""

    status: Literal["ok", "degraded"] = "ok"
    app: str = Field(description="Application name.")
    version: str = Field(description="Backend version string.")
    environment: str = Field(description="development, test or production.")
    database: Literal["ok", "unavailable"] = Field(description="Result of a SELECT 1 against the database.")
    llm_model: str = Field(description="Claude model the agents will call.")
    llm_configured: bool = Field(description="Whether an Anthropic API key is set.")
    critic_enabled: bool = Field(description="Whether the Critic agent runs in the pipeline.")
    server_time: datetime = Field(description="Current server time (UTC).")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Registration payload."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128, description="8+ characters with a letter and a digit.")
    full_name: str | None = Field(default=None, max_length=120)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        """Store emails lower-cased so logins are case-insensitive."""
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def _check_password_strength(cls, value: str) -> str:
        """Require a letter and a digit, and stay within bcrypt's 72-byte limit."""
        if len(value.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
            raise ValueError(f"Password must be at most {BCRYPT_MAX_PASSWORD_BYTES} bytes")
        if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
            raise ValueError("Password must contain at least one letter and one digit")
        return value

    @field_validator("full_name")
    @classmethod
    def _clean_full_name(cls, value: str | None) -> str | None:
        """Trim the name and treat an empty name as missing."""
        if value is None:
            return None
        value = value.strip()
        return value or None


class LoginRequest(BaseModel):
    """JSON login payload used by the frontend."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class UserRead(BaseModel):
    """Public view of a user (never includes the password hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str | None
    created_at: datetime


class TokenResponse(BaseModel):
    """Returned by register/login: a bearer token plus the user it belongs to."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")
    user: UserRead


# ---------------------------------------------------------------------------
# Research domain: papers and text chunks
# ---------------------------------------------------------------------------

PaperSource = Literal["arxiv", "semantic_scholar"]
ChunkSource = Literal["pdf", "abstract"]


class Paper(BaseModel):
    """A paper normalised from arXiv or Semantic Scholar.

    ``paper_id`` is canonical so the same paper found by both sources gets the
    same id: ``arxiv:<id>`` when an arXiv id is known, else ``doi:<doi>``, else
    ``s2:<semantic scholar id>``.
    """

    paper_id: str
    sources: list[PaperSource] = Field(default_factory=list)
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    published: str | None = Field(default=None, description="Publication date, YYYY-MM-DD.")
    venue: str | None = None
    abstract: str = ""
    url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    citation_count: int = 0
    source_rank: int = Field(default=0, description="Position in the source's result list (0 = top hit).")
    relevance: float = Field(default=0.0, description="Cosine similarity of title+abstract to the question.")
    score: float = Field(default=0.0, description="Final ranking score (relevance + citations + recency).")


class Chunk(BaseModel):
    """A ~800-token passage of a paper, the unit that is embedded, retrieved and cited."""

    chunk_id: str = Field(description="Stable id '<paper_id>::c<index>' used as the citation anchor.")
    paper_id: str
    title: str
    text: str
    section: str = "Body"
    page: int = Field(default=0, description="First page (1-based); 0 when the text is the abstract.")
    page_end: int = 0
    chunk_index: int = 0
    token_count: int = 0
    source: ChunkSource = "pdf"


class RetrievedChunk(Chunk):
    """A chunk returned by a similarity search, with its cosine similarity to the query."""

    score: float = Field(description="Cosine similarity in [-1, 1]; higher is more relevant.")


# ---------------------------------------------------------------------------
# Research domain: per-paper extraction and the final report
# ---------------------------------------------------------------------------

SupportLabel = Literal["supported", "partial", "unsupported"]
SectionTheme = Literal["methods", "datasets", "results", "open_problems", "other"]


class PaperSummary(BaseModel):
    """Structured facts the Reader extracts from one paper (also Claude's output schema)."""

    problem: str = Field(description="The problem the paper addresses, in one or two sentences.")
    method: str = Field(description="The proposed method or approach, in one or two sentences.")
    datasets: list[str] = Field(description="Datasets or benchmarks used; empty if none are reported.")
    metrics: list[str] = Field(description="Evaluation metrics used; empty if none are reported.")
    key_results: list[str] = Field(description="Main quantitative or qualitative findings, as short statements.")
    limitations: list[str] = Field(description="Limitations stated or evident; empty if none are reported.")


class PaperDigest(BaseModel):
    """A paper after the Reader: how much text was available and what was extracted."""

    paper: Paper
    summary: PaperSummary | None = None
    full_text: bool = False
    chunk_count: int = 0
    note: str | None = Field(default=None, description="Why only the abstract was used, if applicable.")


class Reference(BaseModel):
    """A numbered bibliography entry; ``[n]`` in the report text points here."""

    number: int
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None


class ReportSection(BaseModel):
    """One themed section of the literature review."""

    heading: str
    theme: SectionTheme = "other"
    paragraphs: list[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    """One row of the paper comparison table."""

    reference: int
    title: str
    year: int | None = None
    method: str
    dataset: str
    key_result: str


class EvidenceSnippet(BaseModel):
    """The source passage shown next to a verified claim."""

    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    text: str


class VerifiedClaim(BaseModel):
    """An atomic claim from the report and the Critic's verdict on it."""

    claim_id: str
    text: str
    section: str
    sentence: str = Field(description="The report sentence the claim was taken from.")
    label: SupportLabel
    rationale: str
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    action: Literal["kept", "removed", "flagged"] = "kept"


class StageMetrics(BaseModel):
    """Latency and LLM usage of one pipeline stage."""

    seconds: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RunMetrics(BaseModel):
    """Everything the evaluation needs about one research run."""

    llm_model: str
    critic_enabled: bool
    revisions: int = 0
    total_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    stages: dict[str, StageMetrics] = Field(default_factory=dict)
    papers_found: int = 0
    papers_full_text: int = 0
    claims_total: int = 0
    claims_supported: int = 0
    claims_partial: int = 0
    claims_unsupported: int = 0
    faithfulness: float | None = Field(default=None, description="Supported claims / all claims (None without the Critic).")


class ReportSentence(BaseModel):
    """One sentence of the report and the exact evidence chunks it cites (used by the evaluation)."""

    section: str
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final, cited and verified literature review."""

    question: str
    title: str
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    comparison_table: list[ComparisonRow] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    papers: list[PaperDigest] = Field(default_factory=list)
    verification: list[VerifiedClaim] = Field(default_factory=list)
    sentences: list[ReportSentence] = Field(default_factory=list)
    markdown: str = ""
    metrics: RunMetrics
    warnings: list[str] = Field(default_factory=list)


ProgressStatus = Literal["started", "progress", "completed", "warning", "error"]


class ProgressEvent(BaseModel):
    """A live update from the pipeline (streamed to the browser over SSE)."""

    stage: str
    status: ProgressStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


# ---------------------------------------------------------------------------
# Research sessions API
# ---------------------------------------------------------------------------

SessionStatusValue = Literal["pending", "running", "completed", "failed"]


class ResearchCreate(BaseModel):
    """Request body for starting a research run."""

    question: str = Field(min_length=10, max_length=1000, description="The research question.")
    enable_critic: bool | None = Field(default=None, description="Override ENABLE_CRITIC for this run.")
    top_n: int | None = Field(default=None, ge=1, le=30, description="Override TOP_N_PAPERS for this run.")

    @field_validator("question")
    @classmethod
    def _normalise_question(cls, value: str) -> str:
        """Collapse whitespace and re-check the minimum length."""
        value = " ".join(value.split())
        if len(value) < 10:
            raise ValueError("Please ask a complete research question (at least 10 characters)")
        return value


class SessionSummary(BaseModel):
    """A research session as shown in the history sidebar."""

    id: str
    question: str
    title: str | None = None
    status: SessionStatusValue
    current_stage: str | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    metrics: RunMetrics | None = None


class SessionDetail(SessionSummary):
    """A research session with its full report (once completed)."""

    report: ResearchReport | None = None


class ResearchStarted(BaseModel):
    """Response to ``POST /api/research``."""

    session: SessionSummary
    events_url: str = Field(description="Server-Sent Events stream with live progress.")


# ---------------------------------------------------------------------------
# Follow-up chat API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """A follow-up question about a finished report."""

    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        """Reject whitespace-only messages."""
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty")
        return value


class ChatCitation(BaseModel):
    """A passage cited in a chat answer; ``number`` matches the report's reference list."""

    number: int
    paper_id: str
    title: str
    chunk_id: str
    section: str
    page: int
    snippet: str


class ChatMessageRead(BaseModel):
    """One stored chat message."""

    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: datetime


class ChatExchange(BaseModel):
    """The stored question and Claude's answer."""

    question: ChatMessageRead
    answer: ChatMessageRead
