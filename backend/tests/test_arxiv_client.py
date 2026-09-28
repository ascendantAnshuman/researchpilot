"""Tests for the arXiv client: query building, Atom parsing and retry behaviour (HTTP mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.services.arxiv_client import (
    ARXIV_API_URL,
    ArxivClient,
    build_search_query,
    extract_search_terms,
    normalise_arxiv_id,
    parse_arxiv_feed,
)
from app.services.http_utils import ExternalServiceError

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query</title>
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <updated>2024-02-01T10:00:00Z</updated>
    <published>2024-01-03T18:30:00Z</published>
    <title>Reducing Hallucination in
      Retrieval-Augmented Generation</title>
    <summary>  We propose a verification step that
      checks every generated claim.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:doi>10.1234/RAG.2024</arxiv:doi>
    <arxiv:journal_ref>ACL 2024</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <published>1999-01-05T00:00:00Z</published>
    <title>An Old-Style Identifier Paper</title>
    <summary>Classic abstract.</summary>
    <author><name>Grace Hopper</name></author>
    <arxiv:primary_category term="hep-th" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>ArXiv Query</title></feed>"""

ERROR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
    <summary>incorrect id format</summary>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def test_search_terms_drop_stopwords_and_keep_hyphenated_phrases() -> None:
    """Filler words vanish and 'retrieval-augmented' becomes a phrase."""
    terms = extract_search_terms("What are recent methods for reducing hallucination in retrieval-augmented generation?")

    assert terms == ["reducing", "hallucination", '"retrieval augmented"', "generation"]


def test_search_terms_keep_quoted_phrases() -> None:
    """Quoted phrases from the planner stay together."""
    assert extract_search_terms('"large language models" evaluation benchmarks') == [
        '"large language models"',
        "evaluation",
        "benchmarks",
    ]


def test_search_terms_are_capped_and_deduplicated() -> None:
    """At most six distinct terms are used."""
    terms = extract_search_terms("alpha beta gamma delta epsilon zeta eta theta alpha")

    assert terms == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def test_build_search_query() -> None:
    """Terms are prefixed with all: and joined by the operator."""
    assert build_search_query(["rag", '"dense retrieval"'], "AND") == 'all:rag AND all:"dense retrieval"'
    assert build_search_query(["rag", "llm"], "OR") == "all:rag OR all:llm"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2401.01234v2", "2401.01234"),
        ("https://arxiv.org/pdf/2401.01234v1", "2401.01234"),
        ("arXiv:1706.03762", "1706.03762"),
        ("http://arxiv.org/abs/hep-th/9901001v1", "hep-th/9901001"),
        ("2312.10997", "2312.10997"),
        ("not an id", None),
    ],
)
def test_normalise_arxiv_id(raw: str, expected: str | None) -> None:
    """Versions and URL prefixes are stripped from arXiv ids."""
    assert normalise_arxiv_id(raw) == expected


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_feed_normalises_fields() -> None:
    """Every useful Atom field ends up on the Paper."""
    first, second = parse_arxiv_feed(SAMPLE_FEED)

    assert first.paper_id == "arxiv:2401.01234"
    assert first.arxiv_id == "2401.01234"
    assert first.title == "Reducing Hallucination in Retrieval-Augmented Generation"
    assert first.abstract == "We propose a verification step that checks every generated claim."
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.year == 2024
    assert first.published == "2024-01-03"
    assert first.doi == "10.1234/rag.2024"
    assert first.venue == "ACL 2024"
    assert first.url == "https://arxiv.org/abs/2401.01234v2"
    assert first.pdf_url == "https://arxiv.org/pdf/2401.01234v2"
    assert first.sources == ["arxiv"]
    assert first.source_rank == 0

    assert second.paper_id == "arxiv:hep-th/9901001"
    assert second.pdf_url == "https://arxiv.org/pdf/hep-th/9901001"
    assert second.venue == "arXiv (hep-th)"
    assert second.doi is None
    assert second.source_rank == 1


def test_parse_error_feed_returns_no_papers() -> None:
    """arXiv reports bad queries as a fake 'Error' entry, which is skipped."""
    assert parse_arxiv_feed(ERROR_FEED) == []


def test_parse_invalid_xml_raises() -> None:
    """A non-XML body (e.g. an HTML error page) is a service error."""
    with pytest.raises(ExternalServiceError, match="invalid Atom XML"):
        parse_arxiv_feed("<html><body>Service Unavailable")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


async def test_search_sends_relevance_sorted_and_query(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """The client sends a relevance-sorted AND query and parses the feed."""
    route = respx_mock.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=SAMPLE_FEED))

    async with ArxivClient(test_settings, rate_limiter=None) as client:
        papers = await client.search("hallucination in retrieval-augmented generation", max_results=5)

    params = route.calls.last.request.url.params
    assert params["search_query"] == 'all:hallucination AND all:"retrieval augmented" AND all:generation'
    assert params["max_results"] == "5"
    assert params["sortBy"] == "relevance"
    assert [paper.paper_id for paper in papers] == ["arxiv:2401.01234", "arxiv:hep-th/9901001"]


async def test_search_uses_configured_default_limit(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """Without max_results the SEARCH_RESULTS_PER_QUERY setting is used."""
    route = respx_mock.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=SAMPLE_FEED))

    async with ArxivClient(test_settings, rate_limiter=None) as client:
        await client.search("dense retrieval")

    assert route.calls.last.request.url.params["max_results"] == str(test_settings.search_results_per_query)


async def test_search_falls_back_to_or_when_and_finds_nothing(
    respx_mock: respx.MockRouter, test_settings: Settings
) -> None:
    """An empty AND result triggers one broader OR search."""
    route = respx_mock.get(ARXIV_API_URL).mock(
        side_effect=[httpx.Response(200, text=EMPTY_FEED), httpx.Response(200, text=SAMPLE_FEED)]
    )

    async with ArxivClient(test_settings, rate_limiter=None) as client:
        papers = await client.search("hallucination verification critic")

    assert route.call_count == 2
    assert " AND " in route.calls[0].request.url.params["search_query"]
    assert " OR " in route.calls[1].request.url.params["search_query"]
    assert len(papers) == 2


async def test_search_with_only_stopwords_makes_no_request(
    respx_mock: respx.MockRouter, test_settings: Settings
) -> None:
    """Nothing searchable means no API call at all."""
    async with ArxivClient(test_settings, rate_limiter=None) as client:
        assert await client.search("what are the recent methods") == []

    assert respx_mock.calls.call_count == 0


async def test_search_retries_when_arxiv_is_overloaded(
    respx_mock: respx.MockRouter, test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """A 503 from arXiv is retried and the search still succeeds."""
    respx_mock.get(ARXIV_API_URL).mock(side_effect=[httpx.Response(503), httpx.Response(200, text=SAMPLE_FEED)])

    async with ArxivClient(test_settings, rate_limiter=None) as client:
        papers = await client.search("retrieval augmented generation")

    assert len(papers) == 2
    assert len(recorded_sleeps) == 1
