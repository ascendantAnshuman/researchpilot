"""Tests for report assembly: claim policy, citation numbering, Markdown and metrics."""

from __future__ import annotations

from app.agents.finalize import (
    UNVERIFIED_MARKER,
    CitationNumberer,
    apply_claim_policy,
    build_report,
    render_markdown,
)
from app.agents.state import DraftComparisonRow, DraftReport, DraftSection
from app.config import Settings
from app.llm import UsageTracker
from app.schemas import PaperDigest, RetrievedChunk, VerifiedClaim
from tests.builders import PLAN, RAG_PAPERS, SUMMARY

A, B, C = (paper.paper_id for paper in RAG_PAPERS)


def chunk(paper_id: str, index: int) -> RetrievedChunk:
    """An evidence chunk."""
    return RetrievedChunk(chunk_id=f"{paper_id}::c{index:03d}", paper_id=paper_id, title="T", text="text", score=0.9)


EVIDENCE = {"E1": chunk(A, 0), "E2": chunk(B, 0), "E3": chunk(A, 1)}

DRAFT = DraftReport(
    title="Review",
    executive_summary="Grounding helps [E2]. Critics help more [E1, E3].",
    sections=[
        DraftSection(heading="Methods", theme="methods", paragraphs=["Good claim [E1]. Bad claim [E2]."]),
        DraftSection(heading="Doomed", theme="other", paragraphs=["Only bad [E3]."]),
    ],
    comparison_table=[
        DraftComparisonRow(paper_label="P2", method="Critic", dataset="HaluEval", key_result="Better | faster"),
        DraftComparisonRow(paper_label="P1", method="Grounding", dataset="HaluEval", key_result="21% to 9%"),
    ],
)


def verdict(claim_id: str, sentence: str, label: str) -> VerifiedClaim:
    """A verified claim for ``sentence``."""
    return VerifiedClaim(claim_id=claim_id, text=sentence, section="s", sentence=sentence, label=label, rationale="r")  # type: ignore[arg-type]


VERIFICATION = [
    verdict("C1", "Good claim [E1].", "supported"),
    verdict("C2", "Bad claim [E2].", "unsupported"),
    verdict("C3", "Only bad [E3].", "unsupported"),
    verdict("C4", "Not in the text at all.", "unsupported"),
    verdict("C5", "Critics help more [E1, E3].", "partial"),
]


def test_citations_are_numbered_by_first_appearance_per_paper() -> None:
    """E-labels map to one number per paper, in order of first citation."""
    numberer = CitationNumberer(EVIDENCE)

    assert numberer.convert("First [E2]. Then [E1, E3]. Again [E3, E2].") == "First [1]. Then [2]. Again [1, 2]."
    assert numberer.numbers == {B: 1, A: 2}


def test_unknown_labels_are_dropped() -> None:
    """A citation to missing evidence disappears rather than printing a broken reference."""
    assert CitationNumberer(EVIDENCE).convert("Claim [E9].") == "Claim."


def test_remove_policy_deletes_unsupported_sentences_and_empty_sections() -> None:
    """Unsupported sentences are removed; a section left empty disappears."""
    edited, claims = apply_claim_policy(DRAFT, VERIFICATION, "remove")

    assert edited.sections[0].paragraphs == ["Good claim [E1]."]
    assert [section.heading for section in edited.sections] == ["Methods"]
    assert [claim.action for claim in claims] == ["kept", "removed", "removed", "flagged", "kept"]


def test_flag_policy_marks_sentences() -> None:
    """With the flag policy the text stays but is marked [unverified]."""
    edited, claims = apply_claim_policy(DRAFT, VERIFICATION, "flag")

    assert edited.sections[0].paragraphs == [f"Good claim [E1]. Bad claim [E2]. {UNVERIFIED_MARKER}"]
    assert claims[1].action == "flagged"


def test_two_claims_from_one_sentence_are_handled_once() -> None:
    """A sentence with two unsupported claims is removed once; both claims say 'removed'."""
    claims = [verdict("C1", "Bad claim [E2].", "unsupported"), verdict("C2", "Bad claim [E2].", "unsupported")]

    edited, updated = apply_claim_policy(DRAFT, claims, "remove")

    assert [claim.action for claim in updated] == ["removed", "removed"]
    assert edited.sections[0].paragraphs == ["Good claim [E1]."]


def make_state(verification: list[VerifiedClaim]) -> dict[str, object]:
    """A pipeline state ready for finalisation."""
    return {
        "question": "What reduces hallucination in RAG?",
        "plan": PLAN,
        "digests": [PaperDigest(paper=paper, summary=SUMMARY, chunk_count=1, full_text=i == 0)
                    for i, paper in enumerate(RAG_PAPERS)],
        "evidence": list(EVIDENCE.values()),
        "draft": DRAFT,
        "verification": verification,
        "revisions": 1,
        "warnings": ["Semantic Scholar search failed (HTTP 429); continuing without it"],
        "stage_seconds": {"planner": 1.5, "critic": 4.0},
    }


def test_build_report_assembles_everything(test_settings: Settings) -> None:
    """References, table, verification, Markdown and metrics are all consistent."""
    tracker = UsageTracker()
    tracker.record("critic", input_tokens=1000, output_tokens=100, cost_usd=0.02, seconds=2.0)

    report = build_report(make_state(VERIFICATION), test_settings, tracker)  # type: ignore[arg-type]

    assert report.executive_summary == "Grounding helps [1]. Critics help more [2]."
    assert [ref.paper_id for ref in report.references] == [B, A]
    assert [(row.reference, row.method) for row in report.comparison_table] == [(1, "Critic"), (2, "Grounding")]
    assert report.verification[0].sentence == "Good claim [2]."
    assert "Bad claim" not in report.markdown
    assert "## Comparison of key papers" in report.markdown
    assert "Better \\| faster" in report.markdown
    assert "[1] Ada Lovelace, Alan Turing (2025). A Critic Model for Verifying Generated Claims." in report.markdown
    metrics = report.metrics
    assert (metrics.claims_total, metrics.claims_supported, metrics.claims_unsupported) == (5, 1, 3)
    assert metrics.faithfulness == 0.2
    assert metrics.revisions == 1
    assert metrics.papers_full_text == 1
    assert metrics.stages["critic"].llm_calls == 1
    assert metrics.stages["critic"].seconds == 4.0
    assert metrics.total_cost_usd == 0.02
    assert report.warnings


def test_report_sentences_keep_chunk_level_citations(test_settings: Settings) -> None:
    """Each final sentence records the exact chunks it cites, even though the text shows paper numbers."""
    report = build_report(make_state(VERIFICATION), test_settings, UsageTracker())  # type: ignore[arg-type]

    by_text = {sentence.text: sentence for sentence in report.sentences}
    assert by_text["Critics help more [2]."].chunk_ids == [f"{A}::c000", f"{A}::c001"]
    assert by_text["Grounding helps [1]."].chunk_ids == [f"{B}::c000"]
    assert by_text["Good claim [2]."].section == "Methods"
    assert all("Bad claim" not in sentence.text for sentence in report.sentences)


def test_faithfulness_is_none_without_the_critic(test_settings: Settings) -> None:
    """The ablation run has no verification, so faithfulness is not reported."""
    settings = test_settings.model_copy(update={"enable_critic": False})

    report = build_report(make_state([]), settings, UsageTracker())  # type: ignore[arg-type]

    assert report.metrics.faithfulness is None
    assert report.verification == []
    assert "## Verification" not in report.markdown


def test_report_without_a_draft_explains_why(test_settings: Settings) -> None:
    """When no papers were found the report says so instead of failing."""
    state = {"question": "Obscure question?", "warnings": ["No papers were found for this question."], "stage_seconds": {}}

    report = build_report(state, test_settings, UsageTracker())  # type: ignore[arg-type]

    assert report.title == "No literature review could be produced"
    assert "No papers were found" in report.executive_summary
    assert report.markdown.startswith("# No literature review could be produced")


def test_render_markdown_structure(test_settings: Settings) -> None:
    """Headings appear in reading order."""
    report = build_report(make_state(VERIFICATION), test_settings, UsageTracker())  # type: ignore[arg-type]

    markdown = render_markdown(report)
    order = ["# Review", "## Executive summary", "## Methods", "## Comparison of key papers", "## Verification",
             "## References"]
    positions = [markdown.index(heading) for heading in order]
    assert positions == sorted(positions)
