"""Synthesizer agent: writes the literature review from retrieved evidence only.

Evidence gathering (no LLM):
    The question and every sub-query are run against the vector store, limited to
    this session's papers. The best chunk of every paper is always included (so
    each paper can be cited), then the highest-scoring remaining chunks fill the
    budget (``MAX_EVIDENCE_CHUNKS``). Passages are labelled [E1], [E2], ...

Writing (one Claude call, structured output):
    Claude gets the paper list ([P#] labels with the Reader's summaries) and the
    evidence passages, and must end every factual sentence with [E#] citations.
    Those labels map to exact chunk ids, which is what makes the Critic's
    verification and the citation-accuracy metric possible.

Revision (at most ``MAX_REVISIONS`` times):
    When the Critic flags unsupported or partially supported claims, the same
    evidence block is sent again (so Claude's prompt cache is reused), along with
    the previous draft and the flagged claims, and Claude rewrites the draft.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

from app.agents.state import (
    DraftComparisonRow,
    DraftReport,
    DraftSection,
    PipelineRuntime,
    ResearchPlan,
    ResearchState,
    emit,
    stage_node,
    usable_digests,
)
from app.config import Effort, Settings
from app.llm import LLM, LLMError
from app.schemas import PaperDigest, RetrievedChunk, VerifiedClaim
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

CITATION_GROUP_RE = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]")
LABEL_RE = re.compile(r"E\d+")

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer in ResearchPilot. You write a literature review that answers a research question using ONLY the evidence passages provided.

Grounding rules (most important):
- End every factual sentence with one or more evidence citations in square brackets, e.g. "... [E3]" or "... [E3, E7]". Cite the passages that directly support that sentence.
- Use only facts stated in the evidence passages. Never add facts, numbers, method names or papers from memory.
- If the evidence does not cover something, say so briefly (e.g. "The retrieved papers do not report ...") instead of guessing.
- Copy numbers exactly as they appear in the evidence.
- Cite only [E#] labels in the prose. [P#] labels are used only in the comparison table.

Structure:
- title: a concise title for the review.
- executive_summary: 4-6 sentences that answer the question directly, with citations.
- sections: 4-6 themed sections with at least one section for each theme "methods", "datasets", "results" and "open_problems" (use "other" for background or anything else). Each section has 1-3 paragraphs of 3-6 sentences.
- comparison_table: one row for each of the most relevant papers (up to 8), identified by its [P#] label (e.g. "P2"). Take method, dataset and key_result from the paper summaries; write "Not reported" when unknown.

Style: precise, neutral and concise, written for a graduate student."""

REVISION_INSTRUCTIONS = """A fact-checker verified your previous draft against the evidence and flagged the claims listed below.
Revise the draft:
- UNSUPPORTED claims: delete them, or rewrite them to state only what the evidence says, with correct [E#] citations.
- PARTIAL claims: make the wording match the evidence precisely (numbers, scope, conditions).
- Keep all other content, structure and citations unless they must change.
Return the complete revised review in the same format."""


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def gather_evidence(
    question: str,
    plan: ResearchPlan,
    digests: Sequence[PaperDigest],
    store: VectorStore,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Retrieve the passages the Synthesizer may cite, ordered by paper then position."""
    papers = [digest.paper.paper_id for digest in usable_digests(digests)]
    if not papers:
        return []

    best: dict[str, RetrievedChunk] = {}

    def keep(chunk: RetrievedChunk) -> None:
        if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
            best[chunk.chunk_id] = chunk

    for query in [question, *(sub_query.query for sub_query in plan.sub_queries)]:
        for chunk in store.query(query, paper_ids=papers, top_k=settings.retrieval_top_k):
            keep(chunk)
    for paper_id in papers:
        if not any(chunk.paper_id == paper_id for chunk in best.values()):
            for chunk in store.query(question, paper_ids=[paper_id], top_k=1):
                keep(chunk)

    ranked = sorted(best.values(), key=lambda chunk: chunk.score, reverse=True)
    selected: dict[str, RetrievedChunk] = {}
    for paper_id in papers:  # coverage first: the best passage of every paper
        top = next((chunk for chunk in ranked if chunk.paper_id == paper_id), None)
        if top is not None and len(selected) < settings.max_evidence_chunks:
            selected[top.chunk_id] = top
    for chunk in ranked:  # then fill the budget by score
        if len(selected) >= settings.max_evidence_chunks:
            break
        selected.setdefault(chunk.chunk_id, chunk)

    paper_order = {paper_id: index for index, paper_id in enumerate(papers)}
    return sorted(selected.values(), key=lambda chunk: (paper_order[chunk.paper_id], chunk.chunk_index))


def label_evidence(evidence: Sequence[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    """Map ``E1``, ``E2``, ... to evidence chunks in order."""
    return {f"E{index}": chunk for index, chunk in enumerate(evidence, start=1)}


def label_papers(digests: Sequence[PaperDigest]) -> dict[str, PaperDigest]:
    """Map ``P1``, ``P2``, ... to usable papers in ranking order."""
    return {f"P{index}": digest for index, digest in enumerate(usable_digests(digests), start=1)}


def format_paper_list(papers: dict[str, PaperDigest]) -> str:
    """The [P#] paper list with the Reader's structured summaries."""
    lines: list[str] = []
    for label, digest in papers.items():
        paper = digest.paper
        authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
        lines.append(f"[{label}] {paper.title} ({paper.year or 'n.d.'}; {paper.venue or 'venue unknown'}; {authors or 'authors unknown'})")
        lines.append(f"    text used: {'full text' if digest.full_text else 'abstract only'}")
        if digest.summary is not None:
            summary = digest.summary
            lines.append(f"    problem: {summary.problem}")
            lines.append(f"    method: {summary.method}")
            lines.append(f"    datasets: {'; '.join(summary.datasets) or 'Not reported'}")
            lines.append(f"    metrics: {'; '.join(summary.metrics) or 'Not reported'}")
            lines.append(f"    key results: {'; '.join(summary.key_results) or 'Not reported'}")
            lines.append(f"    limitations: {'; '.join(summary.limitations) or 'Not reported'}")
    return "\n".join(lines)


def format_evidence(labelled: dict[str, RetrievedChunk], paper_labels: dict[str, str]) -> str:
    """The [E#] evidence passages, each with its paper, section and page."""
    blocks = []
    for label, chunk in labelled.items():
        location = "abstract" if chunk.source == "abstract" else f"{chunk.section}, p. {chunk.page}"
        paper_label = paper_labels.get(chunk.paper_id, "?")
        blocks.append(f"[{label}] from [{paper_label}] {chunk.title} ({location})\n{chunk.text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _feedback_text(flagged: Sequence[VerifiedClaim]) -> str:
    """Flagged claims formatted for the revision prompt."""
    return "\n".join(f"- [{claim.label.upper()}] {claim.text}\n    reason: {claim.rationale}" for claim in flagged)


def build_messages(
    question: str,
    plan: ResearchPlan,
    paper_block: str,
    evidence_block: str,
    previous: DraftReport | None = None,
    flagged: Sequence[VerifiedClaim] = (),
) -> list[dict[str, object]]:
    """User message: a cacheable context block (papers + evidence) followed by the task."""
    context_block = f"PAPERS\n{paper_block}\n\nEVIDENCE PASSAGES\n{evidence_block}"
    sub_topics = "\n".join(f"- {sub_query.query}" for sub_query in plan.sub_queries)
    task = f"Research question: {question}\n\nSub-topics to cover:\n{sub_topics}\n\n"
    if previous is None:
        task += "Write the literature review."
    else:
        task += (
            f"{REVISION_INSTRUCTIONS}\n\nFLAGGED CLAIMS\n{_feedback_text(flagged)}\n\n"
            f"PREVIOUS DRAFT (JSON)\n{previous.model_dump_json(indent=1)}"
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": task},
            ],
        }
    ]


def _clean_citations(text: str, valid: set[str], unknown: set[str]) -> str:
    """Drop citation labels that do not exist; remove citation brackets left empty."""

    def replace(match: re.Match[str]) -> str:
        labels = LABEL_RE.findall(match.group(1))
        kept = [label for label in dict.fromkeys(labels) if label in valid]
        unknown.update(label for label in labels if label not in valid)
        return f"[{', '.join(kept)}]" if kept else ""

    return re.sub(r"\s+([.,;:])", r"\1", CITATION_GROUP_RE.sub(replace, text)).strip()


def sanitize_draft(
    draft: DraftReport, evidence_labels: set[str], paper_labels: set[str]
) -> tuple[DraftReport, list[str]]:
    """Remove citations to non-existent evidence and table rows for unknown papers."""
    unknown: set[str] = set()
    sections = [
        DraftSection(
            heading=section.heading.strip(),
            theme=section.theme,
            paragraphs=[
                cleaned
                for paragraph in section.paragraphs
                if (cleaned := _clean_citations(paragraph, evidence_labels, unknown))
            ],
        )
        for section in draft.sections
    ]
    rows: list[DraftComparisonRow] = []
    seen_papers: set[str] = set()
    for row in draft.comparison_table:
        label = row.paper_label.strip().strip("[]").upper()
        if label in paper_labels and label not in seen_papers:
            seen_papers.add(label)
            rows.append(row.model_copy(update={"paper_label": label}))
    cleaned = DraftReport(
        title=draft.title.strip(),
        executive_summary=_clean_citations(draft.executive_summary, evidence_labels, unknown),
        sections=[section for section in sections if section.paragraphs],
        comparison_table=rows,
    )
    warnings = [f"Removed citations to unknown evidence: {', '.join(sorted(unknown))}"] if unknown else []
    return cleaned, warnings


async def write_draft(
    question: str,
    plan: ResearchPlan,
    digests: Sequence[PaperDigest],
    evidence: Sequence[RetrievedChunk],
    llm: LLM,
    effort: Effort,
    previous: DraftReport | None = None,
    flagged: Sequence[VerifiedClaim] = (),
) -> tuple[DraftReport, list[str]]:
    """Ask Claude for a (revised) draft and sanitise its citations."""
    papers = label_papers(digests)
    labelled = label_evidence(evidence)
    paper_label_by_id = {digest.paper.paper_id: label for label, digest in papers.items()}
    draft = await llm.structured(
        stage="synthesizer",
        system=SYNTHESIZER_SYSTEM_PROMPT,
        messages=build_messages(
            question,
            plan,
            format_paper_list(papers),
            format_evidence(labelled, paper_label_by_id),
            previous,
            flagged,
        ),
        output_model=DraftReport,
        effort=effort,
    )
    return sanitize_draft(draft, set(labelled), set(papers))


@stage_node("synthesizer")
async def synthesizer_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: write the first draft, or revise it after the Critic."""
    context = runtime.context
    flagged = [claim for claim in state.get("verification") or [] if claim.label != "supported"]
    revising = bool(flagged) and "draft" in state

    if revising:
        evidence = state["evidence"]
        emit(runtime, "synthesizer", f"Revising the draft to address {len(flagged)} flagged claims")
    else:
        evidence = await asyncio.to_thread(
            gather_evidence, state["question"], state["plan"], state["digests"], context.vector_store, context.settings
        )
        emit(runtime, "synthesizer", f"Writing the review from {len(evidence)} evidence passages", evidence=len(evidence))

    try:
        draft, warnings = await write_draft(
            state["question"],
            state["plan"],
            state["digests"],
            evidence,
            context.llm,
            context.settings.llm_effort,
            previous=state["draft"] if revising else None,
            flagged=flagged if revising else (),
        )
    except LLMError as exc:
        if not revising:
            raise
        emit(runtime, "synthesizer", f"Revision failed, keeping the previous draft ({exc})", "warning")
        return {"revisions": state.get("revisions", 0) + 1, "warnings": [f"Revision failed: {exc}"]}

    update: dict[str, object] = {"draft": draft, "evidence": list(evidence), "warnings": warnings}
    if revising:
        update["revisions"] = state.get("revisions", 0) + 1
    emit(runtime, "synthesizer", f"Draft ready: {len(draft.sections)} sections, {len(draft.comparison_table)} table rows")
    return update
