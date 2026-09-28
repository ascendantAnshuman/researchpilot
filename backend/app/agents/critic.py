"""Critic agent: fact-checks every claim in the draft against the source passages.

1. **Claim splitting** (one Claude call): the draft is broken into atomic claims,
   each keeping the exact sentence it came from and the [E#] labels it cited.
2. **Evidence per claim** (no LLM): the passages the sentence cited, plus the
   closest passages found by a fresh vector search for the claim - so a claim
   citing the wrong passage can still be recognised as supported elsewhere.
3. **Verification** (Claude calls in batches of 8 claims): each claim is labelled
   *supported*, *partial* or *unsupported*, with a supporting quote. Claims in a
   batch share one pool of passages, which keeps the prompt small.

The graph then sends the draft back to the Synthesizer once if any claim is
unsupported (:func:`route_after_critic`). The same :func:`verify_draft` function
is reused by the evaluation module to score runs made without the Critic.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.agents.state import (
    ClaimJudgement,
    ClaimList,
    DraftReport,
    ExtractedClaim,
    JudgementList,
    PipelineRuntime,
    ResearchState,
    emit,
    stage_node,
)
from app.config import Effort, Settings
from app.llm import LLM, LLMOutputError, LLMRefusalError, LLMUnavailableError, lighter_effort
from app.schemas import Chunk, EvidenceSnippet, RetrievedChunk, VerifiedClaim
from app.services.pdf_parser import split_sentences
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
MAX_CITED_PER_CLAIM = 3
RETRIEVED_PER_CLAIM = 2
MAX_POOL_PER_BATCH = 12
MAX_SNIPPET_CHARS = 400
_WORD_RE = re.compile(r"[a-z0-9]+")

CLAIM_SYSTEM_PROMPT = """You are the Critic in ResearchPilot. Split a literature-review draft into atomic factual claims so that each one can be checked against the source papers.

Rules:
- An atomic claim states exactly one checkable fact (what a method does, which dataset is used, what result was reported, what limitation exists). Write it as a short standalone sentence.
- Split sentences containing several facts into several claims.
- Skip sentences without factual content (transitions, or statements that the evidence does not cover a topic).
- "sentence": copy the complete draft sentence the claim comes from EXACTLY, including its [E#] citations.
- "evidence_labels": the [E#] labels cited in that sentence (e.g. ["E3", "E7"]); [] if it cites none.
- "section": the section heading, or "Executive summary".
- Return at most {max_claims} claims, prioritising the most specific and important ones."""

VERIFY_SYSTEM_PROMPT = """You are the Critic in ResearchPilot. For each claim, decide whether the evidence passages support it.

Labels:
- "supported": the passages state the claim (paraphrase is fine; numbers and conditions must match).
- "partial": the passages support part of it, but something is overstated or over-generalised, or a number or condition differs.
- "unsupported": no passage states it, or the passages contradict it.

Judge only against the passages shown - never against your own knowledge. The candidate labels listed with each claim are hints; any passage may be used.
For each claim return: claim_id; label; evidence_labels (the supporting passages, best first; [] if unsupported); quote (the single most relevant sentence copied exactly from the best passage, or "" if none); rationale (one sentence)."""


@dataclass
class _ClaimWork:
    """A claim being verified, with its candidate evidence labels."""

    claim_id: str
    claim: ExtractedClaim
    candidates: list[str] = field(default_factory=list)


def render_draft(draft: DraftReport) -> str:
    """Plain-text draft (with [E#] citations) as shown to the claim splitter."""
    parts = [f"# {draft.title}", "## Executive summary", draft.executive_summary]
    for section in draft.sections:
        parts.append(f"## {section.heading}")
        parts.extend(section.paragraphs)
    return "\n\n".join(parts)


def best_sentence(text: str, claim: str) -> str:
    """The sentence of ``text`` sharing the most words with ``claim`` (a fallback evidence quote)."""
    claim_words = set(_WORD_RE.findall(claim.lower()))
    sentences = split_sentences(text) or [text]
    best = max(sentences, key=lambda sentence: len(claim_words & set(_WORD_RE.findall(sentence.lower()))))
    return best[:MAX_SNIPPET_CHARS]


def snippet_for(chunk: Chunk, text: str) -> EvidenceSnippet:
    """Package a passage (or a quote from it) for the Verification panel."""
    return EvidenceSnippet(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        title=chunk.title,
        section=chunk.section,
        page=chunk.page,
        text=text[:MAX_SNIPPET_CHARS],
    )


async def extract_claims(draft: DraftReport, llm: LLM, settings: Settings) -> list[ExtractedClaim]:
    """Split the draft into at most ``MAX_CLAIMS`` atomic claims."""
    result = await llm.structured(
        stage="critic",
        system=CLAIM_SYSTEM_PROMPT.format(max_claims=settings.max_claims),
        messages=[{"role": "user", "content": f"DRAFT\n\n{render_draft(draft)}"}],
        output_model=ClaimList,
        effort=lighter_effort(settings.llm_effort),
    )
    claims = [claim for claim in result.claims if claim.claim.strip()]
    return claims[: settings.max_claims]


def _batch_prompt(batch: Sequence[_ClaimWork], pool: dict[str, Chunk]) -> str:
    """Evidence pool followed by the claims to judge."""
    passages = "\n\n".join(
        f"[{label}] {chunk.title} ({chunk.section}, p. {chunk.page or 'n/a'})\n{chunk.text}" for label, chunk in pool.items()
    )
    claims = "\n".join(
        f"[{work.claim_id}] {work.claim.claim}\n    candidate evidence: {', '.join(work.candidates) or 'none'}"
        for work in batch
    )
    return f"EVIDENCE PASSAGES\n{passages}\n\nCLAIMS TO VERIFY\n{claims}"


def _to_verified(work: _ClaimWork, judgement: ClaimJudgement | None, labelled: dict[str, Chunk]) -> VerifiedClaim:
    """Combine a claim with its judgement (a missing judgement counts as unsupported)."""
    if judgement is None:
        closest = [labelled[label] for label in work.candidates[:1] if label in labelled]
        return VerifiedClaim(
            claim_id=work.claim_id,
            text=work.claim.claim,
            section=work.claim.section,
            sentence=work.claim.sentence,
            label="unsupported",
            rationale="The critic returned no verdict for this claim.",
            evidence=[snippet_for(chunk, best_sentence(chunk.text, work.claim.claim)) for chunk in closest],
        )

    supporting = [label for label in judgement.evidence_labels if label in labelled]
    shown = supporting or [label for label in work.candidates if label in labelled][:1]
    snippets = []
    for position, label in enumerate(shown[:2]):
        chunk = labelled[label]
        quote = judgement.quote.strip() if position == 0 and judgement.quote.strip() else ""
        snippets.append(snippet_for(chunk, quote or best_sentence(chunk.text, work.claim.claim)))
    return VerifiedClaim(
        claim_id=work.claim_id,
        text=work.claim.claim,
        section=work.claim.section,
        sentence=work.claim.sentence,
        label=judgement.label,
        rationale=judgement.rationale.strip(),
        evidence=snippets,
    )


async def verify_claims(
    claims: Sequence[ExtractedClaim],
    evidence: Sequence[RetrievedChunk],
    store: VectorStore,
    paper_ids: Sequence[str],
    llm: LLM,
    effort: Effort,
) -> list[VerifiedClaim]:
    """Judge every claim against its cited and retrieved passages."""
    labelled: dict[str, Chunk] = {f"E{index}": chunk for index, chunk in enumerate(evidence, start=1)}
    label_by_chunk = {chunk.chunk_id: label for label, chunk in labelled.items()}

    retrieved = await asyncio.to_thread(
        lambda: [store.query(claim.claim, paper_ids=paper_ids, top_k=RETRIEVED_PER_CLAIM) for claim in claims]
    )
    work_items: list[_ClaimWork] = []
    for index, (claim, found) in enumerate(zip(claims, retrieved, strict=True), start=1):
        cited = [label for label in dict.fromkeys(claim.evidence_labels) if label in labelled][:MAX_CITED_PER_CLAIM]
        extra = []
        for chunk in found:
            if chunk.chunk_id not in label_by_chunk:
                label = f"X{len(label_by_chunk) - len(evidence) + 1}"
                label_by_chunk[chunk.chunk_id] = label
                labelled[label] = chunk
            extra.append(label_by_chunk[chunk.chunk_id])
        work_items.append(_ClaimWork(f"C{index}", claim, list(dict.fromkeys([*cited, *extra]))))

    async def judge(batch: list[_ClaimWork]) -> dict[str, ClaimJudgement]:
        pool_labels = list(dict.fromkeys(label for work in batch for label in work.candidates))[:MAX_POOL_PER_BATCH]
        pool = {label: labelled[label] for label in pool_labels}
        result = await llm.structured(
            stage="critic",
            system=VERIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _batch_prompt(batch, pool)}],
            output_model=JudgementList,
            effort=effort,
        )
        return {judgement.claim_id.strip().strip("[]").upper(): judgement for judgement in result.judgements}

    batches = [work_items[start : start + BATCH_SIZE] for start in range(0, len(work_items), BATCH_SIZE)]
    verdicts: dict[str, ClaimJudgement] = {}
    for batch_result in await asyncio.gather(*(judge(batch) for batch in batches)):
        verdicts.update(batch_result)
    return [_to_verified(work, verdicts.get(work.claim_id), labelled) for work in work_items]


async def verify_draft(
    draft: DraftReport,
    evidence: Sequence[RetrievedChunk],
    store: VectorStore,
    paper_ids: Sequence[str],
    llm: LLM,
    settings: Settings,
) -> list[VerifiedClaim]:
    """Split a draft into claims and verify them (used by the Critic node and the evaluation)."""
    claims = await extract_claims(draft, llm, settings)
    if not claims:
        return []
    return await verify_claims(claims, evidence, store, paper_ids, llm, settings.llm_effort)


def count_labels(claims: Sequence[VerifiedClaim]) -> dict[str, int]:
    """How many claims carry each label."""
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for claim in claims:
        counts[claim.label] += 1
    return counts


def route_after_critic(state: ResearchState) -> str:
    """Send the draft back for one revision if any claim is unsupported, otherwise finish."""
    unsupported = any(claim.label == "unsupported" for claim in state.get("verification") or [])
    if unsupported and state.get("revisions", 0) < state.get("max_revisions", 1):
        return "synthesizer"
    return "finalize"


@stage_node("critic")
async def critic_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: verify the current draft."""
    context = runtime.context
    paper_ids = list(dict.fromkeys(chunk.paper_id for chunk in state["evidence"]))
    try:
        verified = await verify_draft(
            state["draft"], state["evidence"], context.vector_store, paper_ids, context.llm, context.settings
        )
    except (LLMOutputError, LLMRefusalError, LLMUnavailableError) as exc:
        emit(runtime, "critic", f"Verification failed; the report will be marked unverified ({exc})", "warning")
        return {"verification": [], "warnings": [f"Critic failed: {exc}"]}

    counts = count_labels(verified)
    emit(
        runtime,
        "critic",
        f"Checked {len(verified)} claims: {counts['supported']} supported, "
        f"{counts['partial']} partial, {counts['unsupported']} unsupported",
        **counts,
    )
    return {"verification": verified}
