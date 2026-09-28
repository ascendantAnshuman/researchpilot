"""Tests for the Search agent: merging, ranking and graceful degradation (fake clients)."""

from __future__ import annotations

import httpx

from app.agents.search import (
    combine_papers,
    enrich_citation_counts,
    find_papers,
    merge_papers,
    rank_papers,
    recency_score,
    search_node,
)
from app.config import Settings
from app.schemas import Paper
from app.services.vector_store import VectorStore
from tests.builders import PLAN, RAG_PAPERS, fake_runtime, make_context, make_paper
from tests.fakes import FakeSearchClient, HashingEmbedder

QUESTION = "reducing hallucination in retrieval augmented generation"
PROTEIN = make_paper(9, "Protein Folding with Deep Networks", "We predict protein structures.", citations=5000, year=2020)


def s2_copy(paper: Paper, citations: int, venue: str = "ACL", doi: str | None = "10.1/abc") -> Paper:
    """The same paper as Semantic Scholar would return it."""
    return paper.model_copy(
        update={"sources": ["semantic_scholar"], "citation_count": citations, "venue": venue, "doi": doi, "s2_id": "s2x"}
    )


def test_same_arxiv_paper_from_both_sources_is_merged() -> None:
    """Fields are combined: both sources, highest citations, published venue, DOI."""
    arxiv_version = RAG_PAPERS[0].model_copy(update={"citation_count": 0})

    (merged,) = merge_papers([arxiv_version, s2_copy(RAG_PAPERS[0], 250)])

    assert merged.paper_id == RAG_PAPERS[0].paper_id
    assert merged.sources == ["arxiv", "semantic_scholar"]
    assert merged.citation_count == 250
    assert merged.venue == "ACL"
    assert merged.doi == "10.1/abc"
    assert merged.s2_id == "s2x"


def test_papers_are_matched_by_doi() -> None:
    """A journal record without an arXiv id merges with its arXiv preprint via the DOI."""
    preprint = RAG_PAPERS[0].model_copy(update={"doi": "10.9/xyz"})
    journal = make_paper(50, "Different Title Entirely For Journal", "", source="semantic_scholar", doi="10.9/xyz")

    (merged,) = merge_papers([journal, preprint])

    assert merged.paper_id == preprint.paper_id


def test_papers_are_matched_by_normalised_title() -> None:
    """Punctuation and case differences in titles do not create duplicates."""
    other = make_paper(77, "reducing hallucination in retrieval augmented generation!", "", source="semantic_scholar")

    assert len(merge_papers([RAG_PAPERS[0], other])) == 1


def test_distinct_papers_stay_separate() -> None:
    """Different papers are all kept, in first-seen order."""
    assert [p.paper_id for p in merge_papers(RAG_PAPERS)] == [p.paper_id for p in RAG_PAPERS]


def test_combine_prefers_longer_abstract_and_existing_pdf() -> None:
    """The more complete value of each field wins."""
    short = RAG_PAPERS[1].model_copy(update={"abstract": "short", "pdf_url": None})
    full = RAG_PAPERS[1].model_copy(update={"pdf_url": "https://arxiv.org/pdf/2401.00002"})

    combined = combine_papers(short, full)

    assert combined.abstract == RAG_PAPERS[1].abstract
    assert combined.pdf_url == "https://arxiv.org/pdf/2401.00002"


def test_recency_score() -> None:
    """New papers score 1, ten-year-old papers 0, unknown years 0.3."""
    assert recency_score(2026, 2026) == 1.0
    assert recency_score(2021, 2026) == 0.5
    assert recency_score(2010, 2026) == 0.0
    assert recency_score(None, 2026) == 0.3


def test_ranking_by_relevance_only(test_settings: Settings) -> None:
    """With only the relevance weight, the on-topic paper beats the highly cited off-topic one."""
    settings = test_settings.model_copy(update={"rank_weight_citations": 0.0, "rank_weight_recency": 0.0})

    ranked = rank_papers(QUESTION, [PROTEIN, RAG_PAPERS[0]], HashingEmbedder(), settings, current_year=2026)

    assert ranked[0].paper_id == RAG_PAPERS[0].paper_id
    assert ranked[0].score == 1.0
    assert ranked[0].relevance > ranked[1].relevance


def test_ranking_by_citations_only(test_settings: Settings) -> None:
    """With only the citation weight, the most cited paper wins."""
    settings = test_settings.model_copy(update={"rank_weight_relevance": 0.0, "rank_weight_recency": 0.0})

    ranked = rank_papers(QUESTION, RAG_PAPERS, HashingEmbedder(), settings, current_year=2026)

    assert [p.citation_count for p in ranked] == [120, 40, 15]


def test_off_topic_papers_are_dropped_when_enough_remain(test_settings: Settings) -> None:
    """Papers with near-zero similarity are removed if at least TOP_N on-topic papers remain."""
    settings = test_settings.model_copy(update={"top_n_papers": 1})

    ranked = rank_papers(QUESTION, [PROTEIN, *RAG_PAPERS], HashingEmbedder(), settings, current_year=2026)

    assert PROTEIN.paper_id not in {p.paper_id for p in ranked}


async def test_search_continues_when_semantic_scholar_is_down(test_settings: Settings) -> None:
    """A 429 from Semantic Scholar produces one warning; arXiv results are still used."""
    arxiv = FakeSearchClient(RAG_PAPERS)
    s2 = FakeSearchClient(error_status=429)

    papers, warnings, s2_ok = await find_papers(QUESTION, PLAN, arxiv, s2, test_settings)

    assert len(papers) == 3
    assert s2_ok is False
    assert warnings == ["Semantic Scholar search failed (HTTP 429); continuing without it"]
    assert len(s2.queries) == 1  # stopped after the first failure
    assert arxiv.queries == ['"retrieval augmented generation" hallucination', "hallucination benchmark",
                             "hallucination limitations"]


async def test_citation_counts_are_enriched_for_arxiv_only_papers() -> None:
    """One batch lookup fills in citation counts."""
    s2 = FakeSearchClient(lookup={f"ARXIV:{RAG_PAPERS[2].arxiv_id}": s2_copy(RAG_PAPERS[2], 999)})

    papers, warnings = await enrich_citation_counts(list(RAG_PAPERS), s2)

    assert s2.lookups == [[f"ARXIV:{paper.arxiv_id}" for paper in RAG_PAPERS]]
    assert papers[2].citation_count == 999
    assert papers[0].citation_count == RAG_PAPERS[0].citation_count
    assert warnings == []


async def test_enrichment_failure_is_only_a_warning() -> None:
    """If the lookup fails the papers are returned unchanged."""
    papers, warnings = await enrich_citation_counts(list(RAG_PAPERS), FakeSearchClient(error_status=503))

    assert papers == RAG_PAPERS
    assert "citation counts" in warnings[0]


async def test_search_node_keeps_top_n(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """The node returns the TOP_N best papers, sorted by score."""
    settings = test_settings.model_copy(update={"top_n_papers": 2})
    runtime, events = fake_runtime(make_context(settings, vector_store, http_client))

    update = await search_node({"question": QUESTION, "plan": PLAN}, runtime)

    papers = update["papers"]
    assert len(papers) == 2
    assert papers[0].score >= papers[1].score
    assert any("Selected the top 2 papers" in event["message"] for event in events)


async def test_search_node_with_no_results(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """No papers means an empty list and a warning, not an exception."""
    context = make_context(test_settings, vector_store, http_client, arxiv=FakeSearchClient([]))
    runtime, events = fake_runtime(context)

    update = await search_node({"question": QUESTION, "plan": PLAN}, runtime)

    assert update["papers"] == []
    assert "No papers were found" in update["warnings"][-1]
    assert any(event["status"] == "warning" for event in events)
