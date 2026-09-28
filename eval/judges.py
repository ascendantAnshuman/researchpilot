"""LLM-as-judge metrics used by the evaluation.

* **Citation accuracy** - for every (sentence, cited chunk) pair in the final
  report, a judge decides whether the chunk supports the sentence.
  Strict accuracy counts only full support; lenient accuracy also counts partial support.
* **Faithfulness** - the final report is split into atomic claims and each is
  verified against the source passages, reusing the Critic's own extraction and
  verification code, applied identically to both arms of the ablation.
* **Answer relevance** - a judge scores how well the review answers the question (1-5).

The judge model is configurable (``--judge-model``) so it can differ from the
pipeline model, which reduces the risk of a model grading its own style.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

import eval  # noqa: F401  (puts backend/ on sys.path)
from app.agents.critic import count_labels, verify_draft
from app.agents.state import DraftReport, DraftSection
from app.config import Settings
from app.llm import LLM, lighter_effort
from app.schemas import Chunk, ResearchReport, RetrievedChunk
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

CITATION_BATCH_SIZE = 8
MAX_REPORT_CHARS_FOR_RELEVANCE = 24000
_CITATION_MARKER_RE = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*|unverified)\]")

CITATION_JUDGE_PROMPT = """You are an impartial evaluator of citations in scientific writing.
Each numbered pair contains a SENTENCE from a literature review and a PASSAGE from the paper it cites.
Decide whether the passage supports the sentence:
- "supports": everything the sentence states is backed by the passage (paraphrase is fine; numbers must match).
- "partial": the passage backs only part of the sentence, or backs it with a different scope, number or condition.
- "does_not_support": the passage does not back the sentence, or contradicts it.
Judge every pair independently and only from the passage text. Return exactly one verdict per pair id."""

RELEVANCE_JUDGE_PROMPT = """You are an impartial evaluator of literature reviews. Rate how well the review answers the research question:
5 = answers the question directly and completely, with specific methods, evidence and open problems;
4 = answers the question well, with minor gaps;
3 = partially answers it: relevant but generic, or misses an important aspect of the question;
2 = mostly off-target: answers a different or much narrower question;
1 = irrelevant, or no review was produced.
Judge only relevance and coverage of the question - not writing style, and not factual accuracy (checked separately)."""


class CitationVerdict(BaseModel):
    """The judge's verdict on one sentence/passage pair."""

    pair_id: str = Field(description="The pair id, e.g. 'P3'.")
    label: Literal["supports", "partial", "does_not_support"]
    rationale: str = Field(description="One short sentence.")


class CitationVerdictList(BaseModel):
    """Verdicts for one batch of pairs."""

    verdicts: list[CitationVerdict]


class RelevanceVerdict(BaseModel):
    """The judge's relevance score."""

    score: int = Field(description="An integer from 1 to 5.")
    rationale: str = Field(description="Two sentences at most.")


@dataclass
class CitationScore:
    """Outcome of the citation-accuracy judge for one report."""

    pairs: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0

    @property
    def accuracy(self) -> float | None:
        """Share of citations whose passage fully supports the sentence."""
        return self.supported / self.pairs if self.pairs else None

    @property
    def lenient_accuracy(self) -> float | None:
        """Share of citations whose passage supports the sentence at least partially."""
        return (self.supported + self.partial) / self.pairs if self.pairs else None


@dataclass
class FaithfulnessScore:
    """Outcome of the claim-level verification for one report."""

    claims: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0

    @property
    def faithfulness(self) -> float | None:
        """Share of atomic claims fully supported by the sources."""
        return self.supported / self.claims if self.claims else None


def strip_citations(text: str) -> str:
    """Remove ``[1, 2]`` / ``[unverified]`` markers from report text."""
    return _CITATION_MARKER_RE.sub("", text).strip()


async def _load_chunks(store: VectorStore, chunk_ids: list[str]) -> dict[str, Chunk]:
    """Fetch chunks by id (unknown ids are silently absent)."""
    chunks = await asyncio.to_thread(store.get_chunks, chunk_ids)
    return {chunk.chunk_id: chunk for chunk in chunks}


async def judge_citations(report: ResearchReport, store: VectorStore, llm: LLM) -> CitationScore:
    """Judge whether each cited chunk supports the sentence citing it."""
    wanted = list(dict.fromkeys(chunk_id for sentence in report.sentences for chunk_id in sentence.chunk_ids))
    chunks = await _load_chunks(store, wanted)
    pairs: list[tuple[str, str, Chunk]] = []
    for sentence in report.sentences:
        for chunk_id in sentence.chunk_ids:
            if chunk_id in chunks:
                pairs.append((f"P{len(pairs) + 1}", strip_citations(sentence.text), chunks[chunk_id]))
            else:
                logger.warning("Cited chunk %s is missing from the vector store; skipping it", chunk_id)
    if not pairs:
        return CitationScore()

    async def judge(batch: list[tuple[str, str, Chunk]]) -> dict[str, str]:
        prompt = "\n\n".join(
            f"[{pair_id}]\nSENTENCE: {sentence}\nPASSAGE ({chunk.title}; {chunk.section}): {chunk.text}"
            for pair_id, sentence, chunk in batch
        )
        result = await llm.structured(
            stage="judge_citations",
            system=CITATION_JUDGE_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_model=CitationVerdictList,
            effort="medium",
        )
        return {verdict.pair_id.strip().strip("[]").upper(): verdict.label for verdict in result.verdicts}

    batches = [pairs[start : start + CITATION_BATCH_SIZE] for start in range(0, len(pairs), CITATION_BATCH_SIZE)]
    labels: dict[str, str] = {}
    for result in await asyncio.gather(*(judge(batch) for batch in batches)):
        labels.update(result)

    counts: defaultdict[str, int] = defaultdict(int)
    for pair_id, _, _ in pairs:
        counts[labels.get(pair_id, "does_not_support")] += 1  # a missing verdict counts against the citation
    return CitationScore(len(pairs), counts["supports"], counts["partial"], counts["does_not_support"])


def draft_from_report(report: ResearchReport, labels: dict[str, str]) -> DraftReport:
    """Rebuild an [E#]-cited draft from the final report so the Critic's verifier can check it."""

    def relabel(text: str, chunk_ids: list[str]) -> str:
        plain = strip_citations(text)
        cited = [labels[chunk_id] for chunk_id in chunk_ids if chunk_id in labels]
        if not cited:
            return plain
        marker = f" [{', '.join(cited)}]"
        return f"{plain[:-1]}{marker}{plain[-1]}" if plain[-1:] in ".!?" else f"{plain}{marker}"

    summary: list[str] = []
    sections: dict[str, list[str]] = {}
    for sentence in report.sentences:
        text = relabel(sentence.text, sentence.chunk_ids)
        if sentence.section == "Executive summary":
            summary.append(text)
        else:
            sections.setdefault(sentence.section, []).append(text)
    return DraftReport(
        title=report.title,
        executive_summary=" ".join(summary),
        sections=[
            DraftSection(heading=heading, theme="other", paragraphs=[" ".join(texts)])
            for heading, texts in sections.items()
        ],
        comparison_table=[],
    )


async def judge_faithfulness(
    report: ResearchReport, store: VectorStore, llm: LLM, settings: Settings
) -> FaithfulnessScore:
    """Split the final report into claims and verify each against the sources (Critic procedure)."""
    if not report.sentences:
        return FaithfulnessScore()
    wanted = list(dict.fromkeys(chunk_id for sentence in report.sentences for chunk_id in sentence.chunk_ids))
    chunks = await _load_chunks(store, wanted)
    evidence = [RetrievedChunk(**chunks[chunk_id].model_dump(), score=1.0) for chunk_id in wanted if chunk_id in chunks]
    labels = {chunk.chunk_id: f"E{index}" for index, chunk in enumerate(evidence, start=1)}
    paper_ids = [digest.paper.paper_id for digest in report.papers if digest.chunk_count > 0]

    verified = await verify_draft(draft_from_report(report, labels), evidence, store, paper_ids, llm, settings)
    counts = count_labels(verified)
    return FaithfulnessScore(len(verified), counts["supported"], counts["partial"], counts["unsupported"])


def report_text_for_relevance(report: ResearchReport) -> str:
    """The review text (without citations) as shown to the relevance judge."""
    parts = [f"TITLE: {report.title}", f"EXECUTIVE SUMMARY: {strip_citations(report.executive_summary)}"]
    for section in report.sections:
        parts.append(f"## {section.heading}")
        parts.extend(strip_citations(paragraph) for paragraph in section.paragraphs)
    for row in report.comparison_table:
        parts.append(f"- {row.title}: method {row.method}; dataset {row.dataset}; result {row.key_result}")
    return "\n".join(parts)[:MAX_REPORT_CHARS_FOR_RELEVANCE]


async def judge_relevance(report: ResearchReport, llm: LLM, effort: str = "medium") -> RelevanceVerdict:
    """Score (1-5) how well the report answers its research question."""
    verdict = await llm.structured(
        stage="judge_relevance",
        system=RELEVANCE_JUDGE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"RESEARCH QUESTION: {report.question}\n\nREVIEW\n{report_text_for_relevance(report)}",
            }
        ],
        output_model=RelevanceVerdict,
        effort=lighter_effort(effort),  # type: ignore[arg-type]
    )
    return verdict.model_copy(update={"score": min(5, max(1, verdict.score))})
