"""Finalize step: turns the verified draft into the report users see.

* Claims still **unsupported** after the revision are removed from the text
  (``UNSUPPORTED_CLAIM_POLICY=remove``) or marked ``[unverified]`` (``flag``).
  Either way they stay listed in the Verification panel.
* Evidence citations ``[E3, E7]`` become numbered paper references ``[1, 4]``,
  numbered in order of first appearance; the chunk-level link is kept in the
  verification data.
* The comparison table's ``[P#]`` labels are resolved to the same numbers.
* The report is rendered to Markdown and run metrics (latency, tokens, cost,
  faithfulness) are attached.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.agents.critic import count_labels
from app.agents.state import (
    STAGES,
    DraftReport,
    DraftSection,
    PipelineRuntime,
    ResearchState,
    emit,
    stage_node,
    usable_digests,
)
from app.agents.synthesizer import CITATION_GROUP_RE, LABEL_RE, label_evidence, label_papers
from app.config import Settings
from app.llm import UsageTracker
from app.services.pdf_parser import split_sentences
from app.schemas import (
    ComparisonRow,
    PaperDigest,
    Reference,
    ReportSection,
    ReportSentence,
    ResearchReport,
    RetrievedChunk,
    RunMetrics,
    StageMetrics,
    VerifiedClaim,
)

logger = logging.getLogger(__name__)

UNVERIFIED_MARKER = "[unverified]"
EXECUTIVE_SUMMARY = "Executive summary"
_SPACES_RE = re.compile(r"\s{2,}")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,;:])")


# ---------------------------------------------------------------------------
# Unsupported-claim policy
# ---------------------------------------------------------------------------


def _remove_sentence(text: str, sentence: str) -> str:
    """Delete ``sentence`` from ``text`` and tidy the spacing."""
    return _SPACES_RE.sub(" ", text.replace(sentence, " ")).strip()


def apply_claim_policy(
    draft: DraftReport, verification: Sequence[VerifiedClaim], policy: str
) -> tuple[DraftReport, list[VerifiedClaim]]:
    """Remove or flag the sentences behind unsupported claims.

    Returns:
        The edited draft and the claims with their ``action`` set
        (``kept``, ``removed`` or ``flagged``).
    """
    summary = draft.executive_summary
    paragraphs = [list(section.paragraphs) for section in draft.sections]
    handled: dict[str, str] = {}
    updated: list[VerifiedClaim] = []

    for claim in verification:
        if claim.label != "unsupported":
            updated.append(claim)
            continue
        sentence = claim.sentence.strip()
        if sentence in handled:
            updated.append(claim.model_copy(update={"action": handled[sentence]}))
            continue
        found = bool(sentence) and (sentence in summary or any(sentence in p for group in paragraphs for p in group))
        if not found:
            action = "flagged"
        elif policy == "remove":
            summary = _remove_sentence(summary, sentence)
            paragraphs = [[_remove_sentence(p, sentence) for p in group] for group in paragraphs]
            action = "removed"
        else:
            marked = f"{sentence} {UNVERIFIED_MARKER}"
            summary = summary.replace(sentence, marked)
            paragraphs = [[p.replace(sentence, marked) for p in group] for group in paragraphs]
            action = "flagged"
        handled[sentence] = action
        updated.append(claim.model_copy(update={"action": action}))

    sections = [
        DraftSection(heading=section.heading, theme=section.theme, paragraphs=[p for p in group if p])
        for section, group in zip(draft.sections, paragraphs, strict=True)
    ]
    edited = draft.model_copy(
        update={"executive_summary": summary, "sections": [section for section in sections if section.paragraphs]}
    )
    return edited, updated


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


class CitationNumberer:
    """Assigns reference numbers to papers in order of first citation."""

    def __init__(self, evidence: dict[str, RetrievedChunk], initial: dict[str, int] | None = None) -> None:
        """Create a numberer for the labelled evidence ``E1..En``.

        Args:
            evidence: Evidence label -> chunk.
            initial: Numbers already assigned (the chat reuses the report's reference numbers).
        """
        self._evidence = evidence
        self.numbers: dict[str, int] = dict(initial or {})

    def number_for(self, paper_id: str) -> int:
        """The reference number of a paper, assigning the next free one if needed."""
        if paper_id not in self.numbers:
            self.numbers[paper_id] = max(self.numbers.values(), default=0) + 1
        return self.numbers[paper_id]

    def labels_used(self, text: str) -> list[str]:
        """Valid evidence labels cited in ``text``, in order of first appearance."""
        labels = [label for group in CITATION_GROUP_RE.findall(text) for label in LABEL_RE.findall(group)]
        return [label for label in dict.fromkeys(labels) if label in self._evidence]

    def convert(self, text: str) -> str:
        """Replace ``[E3, E7]`` with ``[1, 2]`` (one number per distinct paper)."""

        def replace(match: re.Match[str]) -> str:
            numbers: list[int] = []
            for label in LABEL_RE.findall(match.group(1)):
                chunk = self._evidence.get(label)
                if chunk is not None:
                    number = self.number_for(chunk.paper_id)
                    if number not in numbers:
                        numbers.append(number)
            return f"[{', '.join(str(n) for n in sorted(numbers))}]" if numbers else ""

        converted = _SPACES_RE.sub(" ", CITATION_GROUP_RE.sub(replace, text))
        return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", converted).strip()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table_cell(value: str) -> str:
    """Make a value safe inside a Markdown table cell."""
    return " ".join(value.replace("|", "\\|").split())


def render_markdown(report: ResearchReport) -> str:
    """Render the report as Markdown (used for display and export)."""
    lines = [f"# {report.title}", "", f"> **Research question:** {report.question}", "", "## Executive summary", ""]
    lines += [report.executive_summary, ""]
    for section in report.sections:
        lines += [f"## {section.heading}", ""]
        for paragraph in section.paragraphs:
            lines += [paragraph, ""]

    if report.comparison_table:
        lines += ["## Comparison of key papers", ""]
        lines += ["| Ref | Paper | Year | Method | Dataset | Key result |", "|---|---|---|---|---|---|"]
        for row in report.comparison_table:
            cells = [
                f"[{row.reference}]",
                _table_cell(row.title),
                str(row.year or "n.d."),
                _table_cell(row.method),
                _table_cell(row.dataset),
                _table_cell(row.key_result),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if report.verification:
        counts = count_labels(report.verification)
        removed = sum(claim.action == "removed" for claim in report.verification)
        lines += ["## Verification", ""]
        lines.append(
            f"{len(report.verification)} claims checked against the source passages: "
            f"{counts['supported']} supported, {counts['partial']} partially supported, "
            f"{counts['unsupported']} unsupported"
            + (f" ({removed} removed from the text)." if removed else ".")
        )
        lines.append("")

    if report.references:
        lines += ["## References", ""]
        for reference in report.references:
            authors = ", ".join(reference.authors[:3]) + (" et al." if len(reference.authors) > 3 else "")
            venue = f" *{reference.venue}*." if reference.venue else ""
            link = f" {reference.url}" if reference.url else ""
            lines.append(f"[{reference.number}] {authors or 'Unknown authors'} ({reference.year or 'n.d.'}). {reference.title}.{venue}{link}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_metrics(
    settings: Settings,
    tracker: UsageTracker,
    stage_seconds: dict[str, float],
    digests: Sequence[PaperDigest],
    verification: Sequence[VerifiedClaim],
    revisions: int,
    critic_ran: bool,
) -> RunMetrics:
    """Collect latency, token, cost and faithfulness numbers for the run."""
    stages: dict[str, StageMetrics] = {}
    for stage in STAGES:
        usage = tracker.stages.get(stage)
        stages[stage] = StageMetrics(
            seconds=round(stage_seconds.get(stage, 0.0), 3),
            llm_calls=usage.calls if usage else 0,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cost_usd=round(usage.cost_usd, 6) if usage else 0.0,
        )
    total = tracker.total()
    counts = count_labels(verification)
    return RunMetrics(
        llm_model=settings.llm_model,
        critic_enabled=settings.enable_critic,
        revisions=revisions,
        total_seconds=round(sum(stage_seconds.values()), 3),
        total_input_tokens=total.input_tokens + total.cache_read_tokens + total.cache_write_tokens,
        total_output_tokens=total.output_tokens,
        total_cost_usd=round(total.cost_usd, 6),
        stages=stages,
        papers_found=len(digests),
        papers_full_text=sum(digest.full_text for digest in digests),
        claims_total=len(verification),
        claims_supported=counts["supported"],
        claims_partial=counts["partial"],
        claims_unsupported=counts["unsupported"],
        faithfulness=round(counts["supported"] / len(verification), 4) if critic_ran and verification else None,
    )


def report_sentences(
    draft: DraftReport, numberer: CitationNumberer, evidence: dict[str, RetrievedChunk]
) -> list[ReportSentence]:
    """Every sentence of the final text with the chunk ids it cites (for citation-accuracy evaluation)."""
    blocks = [(EXECUTIVE_SUMMARY, draft.executive_summary)]
    blocks += [(section.heading, paragraph) for section in draft.sections for paragraph in section.paragraphs]
    sentences: list[ReportSentence] = []
    for section, text in blocks:
        for sentence in split_sentences(text):
            chunk_ids = [evidence[label].chunk_id for label in numberer.labels_used(sentence)]
            sentences.append(
                ReportSentence(section=section, text=numberer.convert(sentence), chunk_ids=list(dict.fromkeys(chunk_ids)))
            )
    return sentences


def build_report(state: ResearchState, settings: Settings, tracker: UsageTracker) -> ResearchReport:
    """Assemble the final :class:`ResearchReport` from the pipeline state."""
    question = state["question"]
    digests = state.get("digests", [])
    warnings = list(state.get("warnings", []))
    verification = list(state.get("verification") or [])
    revisions = state.get("revisions", 0)
    critic_ran = settings.enable_critic and bool(verification)
    draft = state.get("draft")

    if draft is None:
        reason = warnings[-1] if warnings else "No usable papers were found."
        report = ResearchReport(
            question=question,
            title="No literature review could be produced",
            executive_summary=f"{reason} Try rephrasing the question with more specific technical terms.",
            papers=digests,
            metrics=build_metrics(settings, tracker, state.get("stage_seconds", {}), digests, [], revisions, False),
            warnings=warnings,
        )
        report.markdown = render_markdown(report)
        return report

    draft, verification = apply_claim_policy(draft, verification, settings.unsupported_claim_policy)
    evidence = label_evidence(state.get("evidence", []))
    numberer = CitationNumberer(evidence)
    executive_summary = numberer.convert(draft.executive_summary)
    sections = [
        ReportSection(
            heading=section.heading,
            theme=section.theme,
            paragraphs=[converted for paragraph in section.paragraphs if (converted := numberer.convert(paragraph))],
        )
        for section in draft.sections
    ]
    sentences = report_sentences(draft, numberer, evidence)

    papers_by_label = label_papers(digests)
    comparison: list[ComparisonRow] = []
    for row in draft.comparison_table:
        digest = papers_by_label.get(row.paper_label)
        if digest is None:
            continue
        comparison.append(
            ComparisonRow(
                reference=numberer.number_for(digest.paper.paper_id),
                title=digest.paper.title,
                year=digest.paper.year,
                method=row.method,
                dataset=row.dataset,
                key_result=row.key_result,
            )
        )

    verification = [
        claim.model_copy(update={"sentence": numberer.convert(claim.sentence)}) for claim in verification
    ]
    papers_by_id = {digest.paper.paper_id: digest.paper for digest in usable_digests(digests)}
    references = [
        Reference(
            number=number,
            paper_id=paper_id,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            url=paper.url or paper.pdf_url,
        )
        for paper_id, number in sorted(numberer.numbers.items(), key=lambda item: item[1])
        if (paper := papers_by_id.get(paper_id)) is not None
    ]

    report = ResearchReport(
        question=question,
        title=draft.title or state["plan"].title,
        executive_summary=executive_summary,
        sections=[section for section in sections if section.paragraphs],
        comparison_table=sorted(comparison, key=lambda row: row.reference),
        references=references,
        papers=digests,
        verification=verification,
        sentences=sentences,
        metrics=build_metrics(
            settings, tracker, state.get("stage_seconds", {}), digests, verification, revisions, critic_ran
        ),
        warnings=warnings,
    )
    report.markdown = render_markdown(report)
    return report


@stage_node("finalize")
async def finalize_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: assemble the final report."""
    context = runtime.context
    report = build_report(state, context.settings, context.llm.tracker)
    removed = sum(claim.action == "removed" for claim in report.verification)
    emit(
        runtime,
        "finalize",
        f"Report ready: {len(report.sections)} sections, {len(report.references)} references"
        + (f", {removed} unsupported claims removed" if removed else ""),
        cost_usd=report.metrics.total_cost_usd,
    )
    return {"report": report}
