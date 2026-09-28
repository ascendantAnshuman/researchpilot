"""Async client for the Semantic Scholar Graph API.

Semantic Scholar adds what arXiv lacks: citation counts, venues, DOIs and
open-access PDF links for papers published outside arXiv.

Without an API key, requests share a small public quota and HTTP 429 responses
are common; they are retried with exponential backoff. A free key
(``SEMANTIC_SCHOLAR_API_KEY``) makes searches much more reliable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.schemas import Paper
from app.services.arxiv_client import clean_whitespace, normalise_arxiv_id
from app.services.http_utils import ExternalServiceError, RateLimiter, create_http_client, request_with_retry

logger = logging.getLogger(__name__)

SERVICE_NAME = "Semantic Scholar"
S2_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,publicationDate,citationCount,authors,openAccessPdf,url"
)
S2_MAX_SEARCH_LIMIT = 100
S2_MAX_BATCH_SIZE = 500
S2_RATE_LIMITER = RateLimiter(1.0)


def canonical_paper_id(arxiv_id: str | None, doi: str | None, s2_id: str) -> str:
    """Pick the id shared across sources: arXiv id, then DOI, then Semantic Scholar id."""
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi.lower()}"
    return f"s2:{s2_id}"


def parse_s2_paper(item: dict[str, Any] | None, rank: int = 0) -> Paper | None:
    """Convert one Semantic Scholar paper object into a :class:`Paper` (None if unusable)."""
    if not item or not item.get("paperId") or not item.get("title"):
        return None

    external: dict[str, Any] = item.get("externalIds") or {}
    raw_arxiv = external.get("ArXiv")
    arxiv_id = normalise_arxiv_id(str(raw_arxiv)) if raw_arxiv else None
    doi = str(external.get("DOI") or "").strip().lower() or None

    open_access: dict[str, Any] = item.get("openAccessPdf") or {}
    pdf_url = str(open_access.get("url") or "").strip() or None
    if pdf_url is None and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    year = item.get("year")
    return Paper(
        paper_id=canonical_paper_id(arxiv_id, doi, str(item["paperId"])),
        sources=["semantic_scholar"],
        title=clean_whitespace(str(item["title"])),
        authors=[str(author["name"]) for author in item.get("authors") or [] if author.get("name")],
        year=int(year) if isinstance(year, int) else None,
        published=item.get("publicationDate") or None,
        venue=item.get("venue") or None,
        abstract=clean_whitespace(str(item.get("abstract") or "")),
        url=item.get("url") or None,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_id=str(item["paperId"]),
        citation_count=int(item.get("citationCount") or 0),
        source_rank=rank,
    )


def _json(response: httpx.Response) -> Any:
    """Decode a JSON body or raise :class:`ExternalServiceError`."""
    try:
        return response.json()
    except ValueError as exc:
        raise ExternalServiceError(SERVICE_NAME, "response was not valid JSON") from exc


class SemanticScholarClient:
    """Search Semantic Scholar and look papers up in bulk.

    Use as an async context manager so the underlying HTTP client is closed::

        async with SemanticScholarClient() as s2:
            papers = await s2.search("hallucination in retrieval augmented generation")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        rate_limiter: RateLimiter | None = S2_RATE_LIMITER,
        api_url: str = S2_API_URL,
    ) -> None:
        """Create a client; pass ``http_client`` to share a connection pool (it is then not closed here)."""
        self._settings = settings or get_settings()
        self._client = http_client or create_http_client(self._settings)
        self._owns_client = http_client is None
        self._rate_limiter = rate_limiter
        self._api_url = api_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        """Authentication header when an API key is configured."""
        key = self._settings.semantic_scholar_api_key
        return {"x-api-key": key.get_secret_value()} if key else {}

    async def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        """Relevance search over titles and abstracts.

        Returns:
            Papers in Semantic Scholar's relevance order; empty if nothing matched.
        """
        query = " ".join(query.replace('"', " ").split())
        if not query:
            return []
        limit = min(max_results or self._settings.search_results_per_query, S2_MAX_SEARCH_LIMIT)

        response = await request_with_retry(
            self._client,
            "GET",
            f"{self._api_url}/paper/search",
            params={"query": query, "limit": limit, "fields": S2_FIELDS},
            headers=self._headers(),
            service=SERVICE_NAME,
            max_retries=self._settings.http_max_retries,
            rate_limiter=self._rate_limiter,
        )
        payload = _json(response)
        items = payload.get("data") if isinstance(payload, dict) else None
        papers = [paper for rank, item in enumerate(items or []) if (paper := parse_s2_paper(item, rank))]
        logger.info("Semantic Scholar: %d papers for %r", len(papers), query)
        return papers

    async def get_papers(self, ids: Sequence[str]) -> list[Paper | None]:
        """Look papers up by id, e.g. ``ARXIV:2401.01234``, ``DOI:10.1000/xyz`` or an S2 id.

        Returns:
            A list aligned with ``ids``; entries are None for ids Semantic Scholar does not know.
        """
        results: list[Paper | None] = []
        for start in range(0, len(ids), S2_MAX_BATCH_SIZE):
            batch = list(ids[start : start + S2_MAX_BATCH_SIZE])
            response = await request_with_retry(
                self._client,
                "POST",
                f"{self._api_url}/paper/batch",
                params={"fields": S2_FIELDS},
                json={"ids": batch},
                headers=self._headers(),
                service=SERVICE_NAME,
                max_retries=self._settings.http_max_retries,
                rate_limiter=self._rate_limiter,
            )
            items = _json(response)
            if not isinstance(items, list) or len(items) != len(batch):
                raise ExternalServiceError(SERVICE_NAME, "batch response did not match the requested ids")
            results.extend(parse_s2_paper(item) for item in items)
        return results

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> SemanticScholarClient:
        """Enter the async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client on exit."""
        await self.aclose()
