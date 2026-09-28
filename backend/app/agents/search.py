"""Search agent: finds candidate papers on arXiv and Semantic Scholar and keeps the best N.

Steps:
    1. Run every sub-query against both sources. Each source is searched
       sequentially (they are rate limited anyway), both sources in parallel.
       If a source keeps failing (e.g. Semantic Scholar HTTP 429), it is skipped
       for the remaining sub-queries and the run continues with the other source.
    2. De-duplicate: the same paper is merged across sources by arXiv id, DOI or
       normalised title, keeping the best fields of each copy.
    3. Enrich arXiv-only papers with citation counts via one Semantic Scholar
       batch lookup (skipped if Semantic Scholar is unavailable).
    4. Rank by   w_rel * relevance + w_cit * citations + w_rec * recency
       where relevance is the embedding similarity between the question and the
       title+abstract, citations are log-scaled, and recency decays over 10 years.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from app.agents.planner import keyword_query
from app.agents.state import (
    PaperLookupClient,
    PaperSearchClient,
    PipelineRuntime,
    ResearchPlan,
    ResearchState,
    emit,
    stage_node,
)
from app.config import Settings
from app.schemas import Paper
from app.services.http_utils import ExternalServiceError
from app.services.vector_store import Embedder

logger = logging.getLogger(__name__)

RECENCY_WINDOW_YEARS = 10
UNKNOWN_YEAR_RECENCY = 0.3
MIN_RELEVANCE = 0.2
_TITLE_NORMALISER = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def normalise_title(title: str) -> str:
    """Lower-case, punctuation-free title used to spot the same paper across sources."""
    return _TITLE_NORMALISER.sub(" ", title.lower()).strip()


def paper_keys(paper: Paper) -> list[str]:
    """Every identifier under which this paper could appear in another source."""
    keys = [paper.paper_id]
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        keys.append(f"doi:{paper.doi.lower()}")
    title = normalise_title(paper.title)
    if len(title) > 15:
        keys.append(f"title:{title}")
    return keys


def combine_papers(first: Paper, second: Paper) -> Paper:
    """Merge two records of the same paper, keeping the most informative value of each field."""
    arxiv_id = first.arxiv_id or second.arxiv_id
    venue_candidates = [venue for venue in (first.venue, second.venue) if venue]
    published_venue = next((venue for venue in venue_candidates if not venue.startswith("arXiv")), None)
    return first.model_copy(
        update={
            "paper_id": f"arxiv:{arxiv_id}" if arxiv_id else first.paper_id,
            "sources": list(dict.fromkeys([*first.sources, *second.sources])),
            "authors": first.authors if len(first.authors) >= len(second.authors) else second.authors,
            "year": first.year or second.year,
            "published": first.published or second.published,
            "venue": published_venue or (venue_candidates[0] if venue_candidates else None),
            "abstract": first.abstract if len(first.abstract) >= len(second.abstract) else second.abstract,
            "url": first.url or second.url,
            "pdf_url": first.pdf_url or second.pdf_url,
            "doi": first.doi or second.doi,
            "arxiv_id": arxiv_id,
            "s2_id": first.s2_id or second.s2_id,
            "citation_count": max(first.citation_count, second.citation_count),
            "source_rank": min(first.source_rank, second.source_rank),
        }
    )


def merge_papers(papers: Sequence[Paper]) -> list[Paper]:
    """De-duplicate papers from all sources and sub-queries, preserving first-seen order."""
    merged: list[Paper] = []
    index_by_key: dict[str, int] = {}
    for paper in papers:
        existing = next((index_by_key[key] for key in paper_keys(paper) if key in index_by_key), None)
        if existing is None:
            merged.append(paper)
            existing = len(merged) - 1
        else:
            merged[existing] = combine_papers(merged[existing], paper)
        for key in paper_keys(merged[existing]):
            index_by_key[key] = existing
    return merged


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def recency_score(year: int | None, current_year: int) -> float:
    """1.0 for this year's papers, falling linearly to 0 after ten years."""
    if year is None:
        return UNKNOWN_YEAR_RECENCY
    return max(0.0, 1.0 - max(0, current_year - year) / RECENCY_WINDOW_YEARS)


def rank_papers(
    question: str,
    papers: Sequence[Paper],
    embedder: Embedder,
    settings: Settings,
    current_year: int | None = None,
) -> list[Paper]:
    """Score and sort papers (best first) by relevance, citations and recency.

    Papers whose raw relevance is below ``MIN_RELEVANCE`` are dropped as off-topic,
    unless that would leave fewer than ``TOP_N_PAPERS`` papers.
    """
    if not papers:
        return []
    current_year = current_year or datetime.now(UTC).year

    vectors = np.asarray(embedder.embed_documents([f"{paper.title}. {paper.abstract}" for paper in papers]))
    query = np.asarray(embedder.embed_query(question))
    similarities = vectors @ query
    low, high = float(similarities.min()), float(similarities.max())
    max_citations = max(paper.citation_count for paper in papers)
    weight_sum = settings.rank_weight_relevance + settings.rank_weight_citations + settings.rank_weight_recency

    scored: list[Paper] = []
    for paper, similarity in zip(papers, similarities, strict=True):
        relevance = (float(similarity) - low) / (high - low) if high > low else 1.0
        citations = math.log1p(paper.citation_count) / math.log1p(max_citations) if max_citations > 0 else 0.0
        score = (
            settings.rank_weight_relevance * relevance
            + settings.rank_weight_citations * citations
            + settings.rank_weight_recency * recency_score(paper.year, current_year)
        ) / weight_sum
        scored.append(paper.model_copy(update={"relevance": round(float(similarity), 4), "score": round(score, 4)}))

    scored.sort(key=lambda paper: paper.score, reverse=True)
    on_topic = [paper for paper in scored if paper.relevance >= MIN_RELEVANCE]
    return on_topic if len(on_topic) >= settings.top_n_papers else scored


# ---------------------------------------------------------------------------
# Running the searches
# ---------------------------------------------------------------------------


async def _search_source(
    name: str,
    client: PaperSearchClient,
    queries: Sequence[str],
    max_results: int,
) -> tuple[list[Paper], list[str]]:
    """Run ``queries`` one by one against a source; stop early if the source is down."""
    papers: list[Paper] = []
    warnings: list[str] = []
    for query in queries:
        try:
            papers.extend(await client.search(query, max_results=max_results))
        except ExternalServiceError as exc:
            reason = f"HTTP {exc.status_code}" if exc.status_code else str(exc)
            warnings.append(f"{name} search failed ({reason}); continuing without it")
            logger.warning("%s unavailable, skipping its remaining queries: %s", name, exc)
            break
    return papers, warnings


async def find_papers(
    question: str,
    plan: ResearchPlan,
    arxiv: PaperSearchClient,
    semantic_scholar: PaperLookupClient,
    settings: Settings,
) -> tuple[list[Paper], list[str], bool]:
    """Search both sources for every sub-query and merge the results.

    Returns:
        ``(merged papers, warnings, semantic_scholar_ok)``.
    """
    per_query = settings.search_results_per_query
    arxiv_queries = [keyword_query(sub_query) for sub_query in plan.sub_queries]
    s2_queries = [sub_query.query for sub_query in plan.sub_queries]
    (arxiv_papers, arxiv_warnings), (s2_papers, s2_warnings) = await asyncio.gather(
        _search_source("arXiv", arxiv, arxiv_queries, per_query),
        _search_source("Semantic Scholar", semantic_scholar, s2_queries, per_query),
    )
    return merge_papers([*arxiv_papers, *s2_papers]), [*arxiv_warnings, *s2_warnings], not s2_warnings


async def enrich_citation_counts(papers: list[Paper], semantic_scholar: PaperLookupClient) -> tuple[list[Paper], list[str]]:
    """Fill in citation counts for arXiv-only papers with one Semantic Scholar batch call."""
    targets = [paper for paper in papers if paper.arxiv_id and "semantic_scholar" not in paper.sources]
    if not targets:
        return papers, []
    try:
        found = await semantic_scholar.get_papers([f"ARXIV:{paper.arxiv_id}" for paper in targets])
    except ExternalServiceError as exc:
        return papers, [f"Could not fetch citation counts from Semantic Scholar ({exc})"]

    enriched = {
        target.paper_id: combine_papers(target, match) for target, match in zip(targets, found, strict=True) if match
    }
    return [enriched.get(paper.paper_id, paper) for paper in papers], []


@stage_node("search")
async def search_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: search, de-duplicate, enrich and rank papers."""
    context = runtime.context
    settings = context.settings
    papers, warnings, s2_ok = await find_papers(
        state["question"], state["plan"], context.arxiv, context.semantic_scholar, settings
    )
    for warning in warnings:
        emit(runtime, "search", warning, "warning")
    emit(runtime, "search", f"Found {len(papers)} unique papers across all sub-queries", candidates=len(papers))

    if not papers:
        emit(runtime, "search", "No papers found for this question", "warning")
        return {"papers": [], "warnings": [*warnings, "No papers were found for this question."]}

    if s2_ok:
        papers, enrich_warnings = await enrich_citation_counts(papers, context.semantic_scholar)
        warnings.extend(enrich_warnings)

    ranked = await asyncio.to_thread(rank_papers, state["question"], papers, context.embedder, settings)
    top = ranked[: settings.top_n_papers]
    emit(
        runtime,
        "search",
        f"Selected the top {len(top)} papers",
        papers=[{"paper_id": paper.paper_id, "title": paper.title, "score": paper.score} for paper in top],
    )
    return {"papers": top, "warnings": warnings}
