"""PDF download, text extraction and chunking.

Pipeline for one paper::

    download_pdf()        -> cached file in DATA_DIR/pdfs (size-limited, checked to really be a PDF)
    extract_text_blocks() -> [(page, section, text)], stopping at References, skipping Acknowledgements
    chunk_blocks()        -> ~800-token chunks with ~100-token overlap, split on sentence boundaries

Chunks never cross a section boundary, so every chunk (and every citation that
points at it) has one well-defined section and page range.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from app.config import Settings, get_settings
from app.schemas import Chunk, ChunkSource
from app.services.http_utils import raise_for_status, retry_async

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

PDF_MAGIC = b"%PDF-"
MAGIC_SEARCH_BYTES = 1024
DOWNLOAD_PIECE_BYTES = 64 * 1024
DEFAULT_SECTION = "Front matter"
ABSTRACT_SECTION = "Abstract"
MIN_BLOCK_CHARS = 20
MAX_HEADING_WORDS = 10

KNOWN_HEADINGS: frozenset[str] = frozenset(
    {
        "abstract", "introduction", "background", "related work", "related works", "preliminaries",
        "method", "methods", "methodology", "approach", "proposed method", "our approach", "model",
        "experiments", "experimental setup", "experimental results", "evaluation", "results",
        "results and discussion", "analysis", "discussion", "ablation study", "ablation studies",
        "conclusion", "conclusions", "conclusion and future work", "limitations", "future work",
        "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
        "references", "bibliography", "appendix",
    }
)
STOP_SECTIONS: frozenset[str] = frozenset({"references", "bibliography"})
SKIP_SECTIONS: frozenset[str] = frozenset(
    {"acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}
)

_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,2}\.?|[IVX]{1,5}\.|[A-H]\.)\s+(?P<text>[A-Z][^.!?]{1,80})$"
)
_INLINE_ABSTRACT_RE = re.compile(r"^abstract\s*[—–:.\-]\s*(?P<rest>\S.*)$", re.IGNORECASE)
_HYPHENATED_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bSec\.)"
    r"(?<!\bTab\.)(?<!\bvs\.)(?<!\bNo\.)(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])"
)
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class PdfError(RuntimeError):
    """Base class for PDF problems; the Reader falls back to the abstract on these."""


class PdfDownloadError(PdfError):
    """The file could not be downloaded, was too large, or was not a PDF."""


class PdfParseError(PdfError):
    """The file is corrupt, encrypted or otherwise unreadable."""


@dataclass(frozen=True)
class TextBlock:
    """Cleaned text from one section on one page (page 0 means "from the abstract")."""

    page: int
    section: str
    text: str


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def safe_filename(paper_id: str) -> str:
    """Turn a paper id such as ``arxiv:2401.01234`` into a safe file name stem."""
    return _UNSAFE_FILENAME_RE.sub("_", paper_id).strip("._") or "paper"


def _looks_like_pdf(path: Path) -> bool:
    """True if ``path`` exists, is non-empty and starts with the PDF signature."""
    try:
        with path.open("rb") as handle:
            return PDF_MAGIC in handle.read(MAGIC_SEARCH_BYTES)
    except OSError:
        return False


async def download_pdf(
    url: str,
    paper_id: str,
    *,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> Path:
    """Download a paper PDF into the cache (or reuse a cached copy) and return its path.

    The download is streamed, aborted once it exceeds ``MAX_PDF_MB``, retried on
    transient network errors, and rejected if the content is not a PDF (for
    example an HTML paywall page).

    Raises:
        PdfDownloadError: Too large or not a PDF.
        ExternalServiceError: HTTP error or retries exhausted.
    """
    settings = settings or get_settings()
    settings.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.pdf_cache_dir / f"{safe_filename(paper_id)}.pdf"
    if _looks_like_pdf(destination):
        logger.debug("PDF cache hit for %s", paper_id)
        return destination

    max_bytes = int(settings.max_pdf_mb * 1024 * 1024)
    partial = destination.with_suffix(".part")

    async def _once() -> Path:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response, "PDF download")
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > max_bytes:
                raise PdfDownloadError(
                    f"{paper_id}: PDF is {declared / 1e6:.1f} MB, over the {settings.max_pdf_mb} MB limit"
                )
            received = 0
            head = b""
            with partial.open("wb") as handle:
                async for piece in response.aiter_bytes(DOWNLOAD_PIECE_BYTES):
                    received += len(piece)
                    if received > max_bytes:
                        raise PdfDownloadError(f"{paper_id}: PDF exceeds the {settings.max_pdf_mb} MB limit")
                    if len(head) < MAGIC_SEARCH_BYTES:
                        head += piece[: MAGIC_SEARCH_BYTES - len(head)]
                    handle.write(piece)
            if PDF_MAGIC not in head:
                content_type = response.headers.get("Content-Type", "unknown")
                raise PdfDownloadError(f"{paper_id}: response is not a PDF (Content-Type: {content_type})")
        partial.replace(destination)
        return destination

    try:
        path = await retry_async(_once, service="PDF download", max_retries=settings.http_max_retries)
    finally:
        partial.unlink(missing_ok=True)
    logger.info("Downloaded PDF for %s (%.1f MB)", paper_id, path.stat().st_size / 1e6)
    return path


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def clean_text(raw: str) -> str:
    """Normalise extracted PDF text.

    Expands ligatures (NFKC), re-joins words hyphenated across line breaks,
    removes soft hyphens and collapses whitespace.
    """
    text = unicodedata.normalize("NFKC", raw).replace("­", "")
    text = _HYPHENATED_BREAK_RE.sub(r"\1\2", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def heading_key(heading: str) -> str:
    """Heading text without numbering, lower-cased (``"7 References"`` -> ``"references"``)."""
    match = _NUMBERED_HEADING_RE.match(heading)
    text = match.group("text") if match else heading
    return text.strip().rstrip(":").lower()


def detect_heading(line: str) -> tuple[str | None, str]:
    """Decide whether a line of PDF text is a section heading.

    Returns:
        ``(heading, remainder)``. ``heading`` is None for ordinary text. ``remainder``
        is body text that shares the heading's line, as in IEEE's ``Abstract—We study ...``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None, ""

    inline_abstract = _INLINE_ABSTRACT_RE.match(stripped)
    if inline_abstract:
        return ABSTRACT_SECTION, inline_abstract.group("rest")

    candidate = stripped.rstrip(":").strip()
    if candidate.lower() in KNOWN_HEADINGS:
        return (candidate.title() if candidate.isupper() else candidate), ""

    match = _NUMBERED_HEADING_RE.match(candidate)
    if match:
        text = match.group("text").strip()
        if any(char.isdigit() for char in text) or len(text.split()) > MAX_HEADING_WORDS:
            return None, ""
        if text.isupper():
            text = text.title()
        return f"{match.group('number')} {text}", ""
    return None, ""


def extract_text_blocks(source: Path | bytes, max_pages: int) -> list[TextBlock]:
    """Extract cleaned text blocks with page numbers and section names.

    Reading stops at the References/Bibliography heading, and Acknowledgements
    are skipped, since neither helps answer research questions. A PDF with no
    extractable text (for example a scan) returns an empty list.

    Args:
        source: Path to a PDF file or the raw PDF bytes.
        max_pages: Only the first ``max_pages`` pages are read.

    Raises:
        PdfParseError: If the PDF is corrupt or password-protected.
    """
    try:
        document = pymupdf.open(source) if isinstance(source, Path) else pymupdf.open(stream=source, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated error types for bad files.
        raise PdfParseError(f"cannot open PDF: {exc}") from exc

    blocks: list[TextBlock] = []
    section = DEFAULT_SECTION
    skipping = False

    def flush(buffer: list[str], page_number: int) -> None:
        text = clean_text("\n".join(buffer))
        buffer.clear()
        if len(text) >= MIN_BLOCK_CHARS:
            blocks.append(TextBlock(page=page_number, section=section, text=text))

    with document:
        if document.needs_pass:
            raise PdfParseError("PDF is password-protected")
        if document.page_count == 0:
            raise PdfParseError("PDF has no pages")
        for page_number in range(1, min(document.page_count, max_pages) + 1):
            try:
                raw = document[page_number - 1].get_text("text", sort=True)
            except Exception as exc:  # A single broken page should not lose the whole paper.
                logger.warning("Skipping unreadable page %d: %s", page_number, exc)
                continue

            buffer: list[str] = []
            for line in raw.splitlines():
                heading, remainder = detect_heading(line)
                if heading is None:
                    if not skipping:
                        buffer.append(line)
                    continue
                flush(buffer, page_number)
                key = heading_key(heading)
                if key in STOP_SECTIONS:
                    return blocks
                skipping = key in SKIP_SECTIONS
                section = heading
                if remainder and not skipping:
                    buffer.append(remainder)
            flush(buffer, page_number)
    return blocks


def blocks_from_abstract(abstract: str) -> list[TextBlock]:
    """Fallback when no PDF is available: treat the abstract as the only text block."""
    text = clean_text(abstract)
    return [TextBlock(page=0, section=ABSTRACT_SECTION, text=text)] if text else []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Approximate LLM token count using the common rule of ~4 tokens per 3 English words."""
    return math.ceil(len(text.split()) * 4 / 3)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking on "e.g.", "et al.", "Fig." and similar."""
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


@dataclass(frozen=True)
class _Unit:
    """A sentence (or piece of an over-long sentence) waiting to be packed into a chunk."""

    text: str
    page: int
    tokens: int


def _section_units(blocks: Sequence[TextBlock], chunk_size: int, count_tokens: TokenCounter) -> list[_Unit]:
    """Sentences of one section, with over-long sentences split into word windows."""
    units: list[_Unit] = []
    for block in blocks:
        for sentence in split_sentences(block.text):
            tokens = count_tokens(sentence)
            if tokens <= chunk_size:
                units.append(_Unit(sentence, block.page, tokens))
                continue
            words = sentence.split()
            words_per_piece = max(1, len(words) * chunk_size // max(tokens, 1))
            for start in range(0, len(words), words_per_piece):
                piece = " ".join(words[start : start + words_per_piece])
                units.append(_Unit(piece, block.page, count_tokens(piece)))
    return units


def _overlap_tail(window: Sequence[_Unit], overlap: int, count_tokens: TokenCounter) -> list[_Unit]:
    """Trailing units of a finished chunk, about ``overlap`` tokens, to start the next chunk."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(window):
        if total + unit.tokens > overlap:
            break
        tail.insert(0, unit)
        total += unit.tokens
    if not tail and window and overlap > 0:
        # The last sentence alone is longer than the overlap: carry over its final words instead.
        last = window[-1]
        piece = " ".join(last.text.split()[-max(1, overlap * 3 // 4) :])
        tail = [_Unit(piece, last.page, count_tokens(piece))]
    return tail


def chunk_blocks(
    blocks: Sequence[TextBlock],
    *,
    paper_id: str,
    title: str,
    chunk_size: int = 800,
    overlap: int = 100,
    source: ChunkSource = "pdf",
    count_tokens: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    """Pack text blocks into overlapping, sentence-aligned chunks.

    Args:
        blocks: Output of :func:`extract_text_blocks` or :func:`blocks_from_abstract`.
        paper_id: Canonical paper id; chunk ids are ``<paper_id>::c000``, ``::c001``, ...
        title: Paper title, stored on every chunk for display and prompting.
        chunk_size: Maximum tokens per chunk (default 800).
        overlap: Tokens repeated at the start of the next chunk in the same section (default 100).
        source: ``"pdf"`` for full text, ``"abstract"`` for the abstract fallback.
        count_tokens: Token counter; defaults to :func:`estimate_tokens`.

    Returns:
        Chunks in reading order.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []

    def emit(window: Sequence[_Unit], section: str) -> None:
        text = " ".join(unit.text for unit in window)
        pages = [unit.page for unit in window]
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}::c{len(chunks):03d}",
                paper_id=paper_id,
                title=title,
                text=text,
                section=section,
                page=min(pages),
                page_end=max(pages),
                chunk_index=len(chunks),
                token_count=count_tokens(text),
                source=source,
            )
        )

    for section, group in itertools.groupby(blocks, key=lambda block: block.section):
        window: list[_Unit] = []
        window_tokens = 0
        for unit in _section_units(list(group), chunk_size, count_tokens):
            if window and window_tokens + unit.tokens > chunk_size:
                emit(window, section)
                window = _overlap_tail(window, overlap, count_tokens)
                window_tokens = sum(item.tokens for item in window)
                if window_tokens + unit.tokens > chunk_size:
                    window, window_tokens = [], 0
            window.append(unit)
            window_tokens += unit.tokens
        if window:
            emit(window, section)

    logger.debug("Chunked %s into %d chunks", paper_id, len(chunks))
    return chunks
