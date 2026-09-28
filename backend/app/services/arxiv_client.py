"""Async client for the arXiv search API (Atom feed served by export.arxiv.org).

arXiv asks API users to leave at least 3 seconds between requests, so every call
goes through a process-wide :class:`~app.services.http_utils.RateLimiter`.

Free-text queries are turned into arXiv's field syntax, e.g.
``reducing hallucination in retrieval-augmented generation`` becomes
``all:reducing AND all:hallucination AND all:"retrieval augmented" AND all:generation``.
If an AND query finds nothing, the client retries once with OR.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from types import TracebackType
from typing import Literal

import httpx

from app.config import Settings, get_settings
from app.schemas import Paper
from app.services.http_utils import ExternalServiceError, RateLimiter, create_http_client, request_with_retry

logger = logging.getLogger(__name__)

SERVICE_NAME = "arXiv"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_SECONDS = 3.0
ARXIV_RATE_LIMITER = RateLimiter(ARXIV_MIN_INTERVAL_SECONDS)
MAX_QUERY_TERMS = 6

NAMESPACES: dict[str, str] = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after all also an and any are as at be been being between both but by can could
    do does done during each for from has have how if in into is it its latest may method methods
    approach approaches technique techniques more most new novel of on or other our over recent
    such than that the their them then there these they this those through to toward towards under
    use used using via was we were what when where which while who why will with within without
    """.split()
)

_ARXIV_ID_RE = re.compile(
    r"(?P<id>\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE
)
_PHRASE_RE = re.compile(r'"([^"]+)"')
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace (arXiv titles and abstracts contain hard line breaks)."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalise_arxiv_id(raw: str) -> str | None:
    """Extract a version-less arXiv id from a URL or string.

    ``http://arxiv.org/abs/2401.01234v2`` -> ``2401.01234``;
    ``hep-th/9901001v1`` -> ``hep-th/9901001``. Returns None if no id is found.
    """
    tail = raw.strip().rsplit("/abs/", 1)[-1].rsplit("/pdf/", 1)[-1]
    tail = tail.removeprefix("arXiv:").removeprefix("arxiv:")
    match = _ARXIV_ID_RE.search(tail)
    return match.group("id") if match else None


def extract_search_terms(query: str) -> list[str]:
    """Turn a free-text query into at most six arXiv search terms.

    Quoted phrases are kept together, hyphenated words become phrases, and
    stop-words such as "what", "recent" or "methods" are dropped.
    """
    terms: list[str] = []

    def add(term: str) -> None:
        if term and term not in terms:
            terms.append(term)

    for phrase in _PHRASE_RE.findall(query):
        words = " ".join(_WORD_RE.findall(phrase.replace("-", " "))).lower()
        if words:
            add(f'"{words}"' if " " in words else words)

    for word in _WORD_RE.findall(_PHRASE_RE.sub(" ", query)):
        token = word.lower().strip("-")
        if len(token) < 2 or token in STOPWORDS:
            continue
        add(f'"{token.replace("-", " ")}"' if "-" in token else token)

    return terms[:MAX_QUERY_TERMS]


def build_search_query(terms: list[str], operator: Literal["AND", "OR"] = "AND") -> str:
    """Join search terms into an arXiv ``search_query`` string over all fields."""
    return f" {operator} ".join(f"all:{term}" for term in terms)


def _text(element: ET.Element, path: str) -> str:
    """Whitespace-cleaned text of a child element, or an empty string."""
    found = element.find(path, NAMESPACES)
    return clean_whitespace(found.text) if found is not None and found.text else ""


def _https(url: str) -> str:
    """Upgrade arXiv links to HTTPS."""
    return url.replace("http://", "https://", 1) if url.startswith("http://") else url


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into :class:`Paper` objects (in result order).

    Error entries (arXiv reports bad queries as a fake entry) are skipped.

    Raises:
        ExternalServiceError: If the response is not valid XML.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ExternalServiceError(SERVICE_NAME, f"invalid Atom XML: {exc}") from exc

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NAMESPACES):
        raw_id = _text(entry, "atom:id")
        arxiv_id = None if "/api/errors" in raw_id else normalise_arxiv_id(raw_id)
        title = _text(entry, "atom:title")
        if not arxiv_id or not title:
            logger.warning("Skipping arXiv entry %r: %s", raw_id, _text(entry, "atom:summary")[:150])
            continue

        abs_url: str | None = None
        pdf_url: str | None = None
        for link in entry.findall("atom:link", NAMESPACES):
            href = link.get("href")
            if not href:
                continue
            if link.get("title") == "pdf":
                pdf_url = href
            elif link.get("rel") == "alternate":
                abs_url = href

        published = _text(entry, "atom:published")
        doi = _text(entry, "arxiv:doi")
        journal_ref = _text(entry, "arxiv:journal_ref")
        primary = entry.find("arxiv:primary_category", NAMESPACES)
        category = primary.get("term") if primary is not None else None

        papers.append(
            Paper(
                paper_id=f"arxiv:{arxiv_id}",
                sources=["arxiv"],
                title=title,
                authors=[
                    name
                    for author in entry.findall("atom:author", NAMESPACES)
                    if (name := _text(author, "atom:name"))
                ],
                year=int(published[:4]) if published[:4].isdigit() else None,
                published=published[:10] or None,
                venue=journal_ref or (f"arXiv ({category})" if category else "arXiv"),
                abstract=_text(entry, "atom:summary"),
                url=_https(abs_url or f"https://arxiv.org/abs/{arxiv_id}"),
                pdf_url=_https(pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"),
                doi=doi.lower() or None,
                arxiv_id=arxiv_id,
                source_rank=len(papers),
            )
        )
    return papers


class ArxivClient:
    """Search arXiv and return normalised :class:`Paper` objects.

    Use as an async context manager so the underlying HTTP client is closed::

        async with ArxivClient() as arxiv:
            papers = await arxiv.search("hallucination retrieval augmented generation")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        rate_limiter: RateLimiter | None = ARXIV_RATE_LIMITER,
        api_url: str = ARXIV_API_URL,
    ) -> None:
        """Create a client; pass ``http_client`` to share a connection pool (it is then not closed here)."""
        self._settings = settings or get_settings()
        self._client = http_client or create_http_client(self._settings)
        self._owns_client = http_client is None
        self._rate_limiter = rate_limiter
        self._api_url = api_url

    async def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        """Search arXiv by relevance.

        Args:
            query: Free text; quoted phrases are respected.
            max_results: Number of results (defaults to ``SEARCH_RESULTS_PER_QUERY``).

        Returns:
            Papers in arXiv's relevance order; empty if nothing matched.
        """
        limit = max_results or self._settings.search_results_per_query
        terms = extract_search_terms(query)
        if not terms:
            logger.info("arXiv: no searchable terms in %r", query)
            return []

        papers = await self._fetch(build_search_query(terms, "AND"), limit)
        if not papers and len(terms) > 1:
            logger.info("arXiv: AND query found nothing for %r, retrying with OR", query)
            papers = await self._fetch(build_search_query(terms, "OR"), limit)
        logger.info("arXiv: %d papers for %r", len(papers), query)
        return papers

    async def _fetch(self, search_query: str, limit: int) -> list[Paper]:
        """Run one API request and parse the feed."""
        response = await request_with_retry(
            self._client,
            "GET",
            self._api_url,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            service=SERVICE_NAME,
            max_retries=self._settings.http_max_retries,
            rate_limiter=self._rate_limiter,
        )
        return parse_arxiv_feed(response.text)

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ArxivClient:
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
