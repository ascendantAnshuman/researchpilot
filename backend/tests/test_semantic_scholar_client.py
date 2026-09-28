"""Tests for the Semantic Scholar client (HTTP mocked)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.config import Settings
from app.services.http_utils import ExternalServiceError
from app.services.semantic_scholar_client import S2_API_URL, SemanticScholarClient, canonical_paper_id

SEARCH_URL = f"{S2_API_URL}/paper/search"
BATCH_URL = f"{S2_API_URL}/paper/batch"

ARXIV_ITEM: dict[str, Any] = {
    "paperId": "abc123",
    "externalIds": {"ArXiv": "2401.01234", "DOI": "10.1234/RAG.2024"},
    "title": "Reducing Hallucination in RAG",
    "abstract": "We  verify\nclaims.",
    "year": 2024,
    "venue": "ACL",
    "publicationDate": "2024-01-03",
    "citationCount": 42,
    "authors": [{"authorId": "1", "name": "Ada Lovelace"}],
    "openAccessPdf": None,
    "url": "https://www.semanticscholar.org/paper/abc123",
}
DOI_ITEM: dict[str, Any] = {
    "paperId": "def456",
    "externalIds": {"DOI": "10.5555/Journal.99"},
    "title": "Journal Only Paper",
    "abstract": None,
    "year": 2023,
    "venue": "",
    "publicationDate": None,
    "citationCount": 7,
    "authors": [],
    "openAccessPdf": {"url": "https://example.org/paper.pdf", "status": "GREEN"},
    "url": "https://www.semanticscholar.org/paper/def456",
}
BARE_ITEM: dict[str, Any] = {
    "paperId": "ghi789",
    "externalIds": {},
    "title": "No External Ids",
    "abstract": "Short.",
    "year": None,
    "citationCount": None,
    "authors": [{"name": None}],
    "openAccessPdf": {"url": "", "status": None},
}
BROKEN_ITEM: dict[str, Any] = {"paperId": "zzz", "title": None}

SEARCH_RESPONSE = {"total": 4, "offset": 0, "data": [ARXIV_ITEM, DOI_ITEM, BARE_ITEM, BROKEN_ITEM]}


def test_canonical_paper_id_prefers_arxiv_then_doi() -> None:
    """The same paper found via arXiv and S2 must get the same id."""
    assert canonical_paper_id("2401.01234", "10.1/x", "s2id") == "arxiv:2401.01234"
    assert canonical_paper_id(None, "10.1/X", "s2id") == "doi:10.1/x"
    assert canonical_paper_id(None, None, "s2id") == "s2:s2id"


async def test_search_parses_and_normalises_results(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """Results become Papers with canonical ids, PDF links and safe defaults."""
    respx_mock.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=SEARCH_RESPONSE))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        papers = await client.search("hallucination in RAG")

    assert [paper.paper_id for paper in papers] == ["arxiv:2401.01234", "doi:10.5555/journal.99", "s2:ghi789"]
    arxiv_paper, doi_paper, bare_paper = papers

    assert arxiv_paper.citation_count == 42
    assert arxiv_paper.pdf_url == "https://arxiv.org/pdf/2401.01234"
    assert arxiv_paper.abstract == "We verify claims."
    assert arxiv_paper.s2_id == "abc123"
    assert arxiv_paper.sources == ["semantic_scholar"]

    assert doi_paper.pdf_url == "https://example.org/paper.pdf"
    assert doi_paper.abstract == ""
    assert doi_paper.venue is None

    assert bare_paper.pdf_url is None
    assert bare_paper.citation_count == 0
    assert bare_paper.authors == []
    assert [paper.source_rank for paper in papers] == [0, 1, 2]


async def test_search_request_parameters_without_api_key(
    respx_mock: respx.MockRouter, test_settings: Settings
) -> None:
    """Query, limit and fields are sent; no key header when none is configured."""
    route = respx_mock.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"total": 0, "data": []}))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        await client.search('"dense retrieval"   benchmarks', max_results=7)

    request = route.calls.last.request
    assert request.url.params["query"] == "dense retrieval benchmarks"
    assert request.url.params["limit"] == "7"
    assert "citationCount" in request.url.params["fields"]
    assert "x-api-key" not in request.headers


async def test_search_sends_api_key_when_configured(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """SEMANTIC_SCHOLAR_API_KEY is sent as the x-api-key header."""
    route = respx_mock.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    keyed = test_settings.model_copy(update={"semantic_scholar_api_key": SecretStr("s2-test-key")})

    async with SemanticScholarClient(keyed, rate_limiter=None) as client:
        await client.search("rag")

    assert route.calls.last.request.headers["x-api-key"] == "s2-test-key"


async def test_search_with_no_results(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """A response without a data list means no papers, not an error."""
    respx_mock.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"total": 0, "offset": 0}))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        assert await client.search("an extremely obscure query") == []


async def test_blank_query_makes_no_request(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """Empty queries short-circuit."""
    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        assert await client.search('  ""  ') == []

    assert respx_mock.calls.call_count == 0


async def test_rate_limit_is_retried(
    respx_mock: respx.MockRouter, test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """The common unauthenticated 429 is retried transparently."""
    respx_mock.get(SEARCH_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"data": [ARXIV_ITEM]})]
    )

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        papers = await client.search("rag")

    assert len(papers) == 1
    assert len(recorded_sleeps) == 1


async def test_invalid_json_is_a_service_error(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """A 200 with a non-JSON body is reported clearly."""
    respx_mock.get(SEARCH_URL).mock(return_value=httpx.Response(200, text="<html>oops</html>"))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        with pytest.raises(ExternalServiceError, match="not valid JSON"):
            await client.search("rag")


async def test_batch_lookup_is_aligned_with_requested_ids(
    respx_mock: respx.MockRouter, test_settings: Settings
) -> None:
    """get_papers returns one entry per id, None for unknown ids."""
    route = respx_mock.post(BATCH_URL).mock(return_value=httpx.Response(200, json=[ARXIV_ITEM, None]))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        found, missing = await client.get_papers(["ARXIV:2401.01234", "ARXIV:9999.99999"])

    assert json.loads(route.calls.last.request.content) == {"ids": ["ARXIV:2401.01234", "ARXIV:9999.99999"]}
    assert found is not None and found.citation_count == 42
    assert missing is None


async def test_batch_lookup_rejects_mismatched_response(
    respx_mock: respx.MockRouter, test_settings: Settings
) -> None:
    """A batch response of the wrong length is an error, not silently misaligned data."""
    respx_mock.post(BATCH_URL).mock(return_value=httpx.Response(200, json=[ARXIV_ITEM]))

    async with SemanticScholarClient(test_settings, rate_limiter=None) as client:
        with pytest.raises(ExternalServiceError, match="did not match"):
            await client.get_papers(["ARXIV:1", "ARXIV:2"])
