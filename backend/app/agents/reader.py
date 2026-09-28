"""Reader agent: turns each selected paper into stored chunks and a structured summary.

For every paper (several in parallel, ``READER_CONCURRENCY``):
    1. Reuse its full text if an earlier run already stored it.
    2. Otherwise download the PDF, extract text and chunk it (~800/100 tokens).
    3. If there is no PDF, or it fails (paywall, corrupt, too large, scanned),
       fall back to the abstract so the paper still contributes.
    4. Embed and store the chunks in ChromaDB (replacing old abstract-only chunks).
    5. Ask Claude to extract problem, method, datasets, metrics, key results and
       limitations as validated JSON (:class:`~app.schemas.PaperSummary`).

A failure on one paper never stops the run: it is recorded as a note/warning.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import httpx

from app.agents.state import PipelineContext, PipelineRuntime, ResearchState, emit, stage_node, usable_digests
from app.llm import LLMOutputError, LLMRefusalError, LLMUnavailableError, lighter_effort
from app.schemas import Chunk, Paper, PaperDigest, PaperSummary
from app.services.http_utils import ExternalServiceError
from app.services.pdf_parser import (
    PdfError,
    blocks_from_abstract,
    chunk_blocks,
    download_pdf,
    extract_text_blocks,
)

logger = logging.getLogger(__name__)

MAX_EXTRACTION_TOKENS = 6000
PRIORITY_SECTION_WORDS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "method",
    "approach",
    "model",
    "experiment",
    "evaluation",
    "result",
    "discussion",
    "limitation",
    "conclusion",
)

READER_SYSTEM_PROMPT = """You are the Reader in ResearchPilot. You extract structured facts from one academic paper.

Use ONLY the excerpts provided. Do not add knowledge from outside the text.
- problem, method: one or two plain sentences each.
- datasets, metrics: names exactly as written in the paper; an empty list if none are mentioned.
- key_results: short factual statements, keeping the paper's numbers exactly (e.g. "Reduces hallucination rate from 21% to 9% on HaluEval").
- limitations: stated or clearly evident limitations; an empty list if none are mentioned.
If only the abstract is available, extract what it states and leave the rest empty rather than guessing."""


def select_extraction_chunks(chunks: Sequence[Chunk], max_tokens: int = MAX_EXTRACTION_TOKENS) -> list[Chunk]:
    """Pick the most informative chunks (abstract, intro, method, results...) within a token budget.

    The first chunk is always included; the selection keeps reading order.
    """
    if not chunks:
        return []

    def priority(chunk: Chunk) -> int:
        section = chunk.section.lower()
        return 0 if any(word in section for word in PRIORITY_SECTION_WORDS) else 1

    ordered = [chunks[0], *sorted(chunks[1:], key=lambda chunk: (priority(chunk), chunk.chunk_index))]
    selected: list[Chunk] = []
    budget = max_tokens
    for chunk in ordered:
        if chunk.token_count <= budget or not selected:
            selected.append(chunk)
            budget -= chunk.token_count
    return sorted(selected, key=lambda chunk: chunk.chunk_index)


def format_paper_excerpts(paper: Paper, chunks: Sequence[Chunk]) -> str:
    """Paper metadata plus the selected excerpts, as sent to Claude."""
    header = [
        f"Title: {paper.title}",
        f"Authors: {', '.join(paper.authors[:8]) or 'unknown'}",
        f"Year: {paper.year or 'unknown'}",
        f"Venue: {paper.venue or 'unknown'}",
        f"Text available: {'full text' if any(chunk.source == 'pdf' for chunk in chunks) else 'abstract only'}",
        "",
    ]
    body = [f"--- {chunk.section} (page {chunk.page or 'n/a'}) ---\n{chunk.text}" for chunk in chunks]
    return "\n".join(header) + "\n\n".join(body)


async def load_paper_chunks(paper: Paper, context: PipelineContext) -> tuple[list[Chunk], bool, str | None]:
    """Get a paper's chunks: cached full text, fresh PDF text, or the abstract fallback.

    Returns:
        ``(chunks, full_text, note)`` where ``note`` explains any fallback.
    """
    store = context.vector_store
    settings = context.settings

    if store.has_paper(paper.paper_id, full_text=True):
        return await asyncio.to_thread(store.get_paper_chunks, paper.paper_id), True, None

    note: str | None = None
    if paper.pdf_url:
        try:
            path = await download_pdf(paper.pdf_url, paper.paper_id, client=context.http, settings=settings)
            blocks = await asyncio.to_thread(extract_text_blocks, path, settings.max_pdf_pages)
            if blocks:
                chunks = chunk_blocks(
                    blocks,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    chunk_size=settings.chunk_size_tokens,
                    overlap=settings.chunk_overlap_tokens,
                    source="pdf",
                )
                await asyncio.to_thread(store.delete_paper, paper.paper_id)
                await asyncio.to_thread(store.add_chunks, chunks)
                return chunks, True, None
            note = "PDF contained no extractable text (possibly scanned); used the abstract"
        except (PdfError, ExternalServiceError, httpx.HTTPError, OSError) as exc:
            note = f"PDF unavailable ({type(exc).__name__}: {str(exc)[:120]}); used the abstract"
            logger.info("Falling back to abstract for %s: %s", paper.paper_id, exc)
    else:
        note = "No open-access PDF; used the abstract"

    if store.has_paper(paper.paper_id):
        return await asyncio.to_thread(store.get_paper_chunks, paper.paper_id), False, note

    chunks = chunk_blocks(
        blocks_from_abstract(paper.abstract),
        paper_id=paper.paper_id,
        title=paper.title,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
        source="abstract",
    )
    if not chunks:
        return [], False, "No PDF and no abstract available; paper skipped"
    await asyncio.to_thread(store.add_chunks, chunks)
    return chunks, False, note


async def extract_summary(paper: Paper, chunks: Sequence[Chunk], context: PipelineContext) -> PaperSummary:
    """Ask Claude for the paper's structured summary."""
    excerpts = format_paper_excerpts(paper, select_extraction_chunks(chunks))
    return await context.llm.structured(
        stage="reader",
        system=READER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": excerpts}],
        output_model=PaperSummary,
        effort=lighter_effort(context.settings.llm_effort),
    )


async def read_paper(paper: Paper, context: PipelineContext) -> PaperDigest:
    """Load, store and summarise one paper; never raises for per-paper problems."""
    chunks, full_text, note = await load_paper_chunks(paper, context)
    if not chunks:
        return PaperDigest(paper=paper, full_text=False, chunk_count=0, note=note)
    try:
        summary: PaperSummary | None = await extract_summary(paper, chunks, context)
    except (LLMOutputError, LLMRefusalError, LLMUnavailableError) as exc:
        summary = None
        note = f"{note + '; ' if note else ''}summary extraction failed ({exc})"
    return PaperDigest(paper=paper, summary=summary, full_text=full_text, chunk_count=len(chunks), note=note)


@stage_node("reader")
async def reader_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: read every selected paper with bounded concurrency."""
    context = runtime.context
    papers = state.get("papers", [])
    semaphore = asyncio.Semaphore(context.settings.reader_concurrency)
    finished = 0

    async def read_one(paper: Paper) -> PaperDigest:
        nonlocal finished
        async with semaphore:
            digest = await read_paper(paper, context)
        finished += 1
        kind = "full text" if digest.full_text else "abstract only" if digest.chunk_count else "skipped"
        emit(
            runtime,
            "reader",
            f"Read {finished}/{len(papers)}: {paper.title[:80]} ({kind}, {digest.chunk_count} chunks)",
            paper_id=paper.paper_id,
            full_text=digest.full_text,
            chunks=digest.chunk_count,
        )
        return digest

    digests = list(await asyncio.gather(*(read_one(paper) for paper in papers)))
    warnings = [f"{digest.paper.title[:80]}: {digest.note}" for digest in digests if digest.note]
    usable = usable_digests(digests)
    emit(
        runtime,
        "reader",
        f"{sum(digest.full_text for digest in digests)} papers with full text, "
        f"{len(usable) - sum(digest.full_text for digest in digests)} abstract only",
    )
    if not usable:
        warnings.append("None of the selected papers had readable text.")
    return {"digests": digests, "warnings": warnings}
