"""Tests for PDF download (HTTP mocked), text extraction and heading detection."""

from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest
import respx

from app.config import Settings
from app.services.http_utils import ExternalServiceError
from app.services.pdf_parser import (
    PdfDownloadError,
    PdfParseError,
    blocks_from_abstract,
    clean_text,
    detect_heading,
    download_pdf,
    extract_text_blocks,
    heading_key,
    safe_filename,
)

PDF_URL = "https://arxiv.org/pdf/2401.01234"

SAMPLE_PAGES: list[list[str]] = [
    [
        "Reducing Hallucination in RAG",
        "Ada Lovelace",
        "Abstract",
        "We study hallucination in retrieval-augmented generation systems.",
        "1 Introduction",
        "Large language models often produce hallu-",
        "cinated content that is not supported by evidence.",
        "We propose a critic that verifies each claim.",
    ],
    [
        "2 Method",
        "Our method retrieves passages and checks claims one by one.",
        "The critic labels each claim as supported or unsupported.",
        "Acknowledgments",
        "We thank the reviewers for their helpful comments.",
        "References",
        "[1] Lewis et al. Retrieval-augmented generation. 2020.",
    ],
    [
        "Appendix",
        "Extra experimental details that should never be read.",
    ],
]


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a small text PDF in memory, one line per entry."""
    document = pymupdf.open()
    for lines in pages:
        page = document.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    data = document.tobytes()
    document.close()
    return data


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def test_extracts_sections_pages_and_stops_at_references() -> None:
    """Sections are detected, hyphenation fixed, acknowledgements skipped, references dropped."""
    blocks = extract_text_blocks(make_pdf(SAMPLE_PAGES), max_pages=40)

    assert [block.section for block in blocks] == ["Front matter", "Abstract", "1 Introduction", "2 Method"]
    by_section = {block.section: block for block in blocks}
    assert by_section["1 Introduction"].page == 1
    assert by_section["2 Method"].page == 2
    assert "hallucinated content" in by_section["1 Introduction"].text
    assert "hallu-" not in by_section["1 Introduction"].text

    all_text = " ".join(block.text for block in blocks)
    assert "reviewers" not in all_text
    assert "Lewis" not in all_text
    assert "never be read" not in all_text


def test_max_pages_limits_reading() -> None:
    """Only the first max_pages pages are processed."""
    blocks = extract_text_blocks(make_pdf(SAMPLE_PAGES), max_pages=1)

    assert {block.page for block in blocks} == {1}
    assert "2 Method" not in {block.section for block in blocks}


def test_extract_from_file_path(tmp_path: Path) -> None:
    """A path on disk works the same as raw bytes."""
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(SAMPLE_PAGES))

    assert len(extract_text_blocks(path, max_pages=40)) == 4


def test_corrupt_pdf_raises_parse_error() -> None:
    """Garbage bytes raise PdfParseError, which the Reader turns into an abstract fallback."""
    with pytest.raises(PdfParseError):
        extract_text_blocks(b"%PDF-1.7 this is not really a pdf", max_pages=5)


def test_pdf_without_text_returns_no_blocks() -> None:
    """A scanned (image-only) PDF yields nothing rather than crashing."""
    assert extract_text_blocks(make_pdf([[]]), max_pages=5) == []


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("3.2 Retrieval Module", "3.2 Retrieval Module"),
        ("II. RELATED WORK", "II. Related Work"),
        ("ABSTRACT", "Abstract"),
        ("Conclusions:", "Conclusions"),
        ("A. Additional Results", "A. Additional Results"),
        ("1 BERT-base 110M", None),
        ("In this paper we propose a method.", None),
        ("2 We show that the proposed approach clearly improves results on every benchmark", None),
        ("", None),
    ],
)
def test_detect_heading(line: str, expected: str | None) -> None:
    """Numbered, Roman-numeral and well-known headings are recognised; body text is not."""
    assert detect_heading(line)[0] == expected


def test_detect_inline_ieee_abstract() -> None:
    """'Abstract—text' yields the Abstract heading plus the text after the dash."""
    assert detect_heading("Abstract—We study RAG systems.") == ("Abstract", "We study RAG systems.")


def test_heading_key_strips_numbering() -> None:
    """Numbering is ignored when deciding whether a heading means References etc."""
    assert heading_key("7 References") == "references"
    assert heading_key("II. Related Work") == "related work"


def test_clean_text_fixes_common_pdf_artifacts() -> None:
    """Ligatures, line-break hyphenation, soft hyphens and spacing are normalised."""
    raw = "eﬃcient ﬁne-tuning of hallu-\ncination­ detectors   with\n\nnew  lines"

    assert clean_text(raw) == "efficient fine-tuning of hallucination detectors with new lines"


def test_blocks_from_abstract() -> None:
    """The abstract fallback is a single page-0 block, or nothing if empty."""
    assert blocks_from_abstract("   ") == []
    (block,) = blocks_from_abstract("We  study\nRAG.")
    assert (block.page, block.section, block.text) == (0, "Abstract", "We study RAG.")


def test_safe_filename() -> None:
    """Paper ids become safe file names on every OS."""
    assert safe_filename("arxiv:hep-th/9901001") == "arxiv_hep-th_9901001"
    assert safe_filename("doi:10.1145/3442188.3445922") == "doi_10.1145_3442188.3445922"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


async def test_download_saves_pdf_and_reuses_cache(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """The first call downloads, the second is served from DATA_DIR/pdfs."""
    pdf_bytes = make_pdf(SAMPLE_PAGES)
    route = respx_mock.get(PDF_URL).mock(
        return_value=httpx.Response(200, content=pdf_bytes, headers={"Content-Type": "application/pdf"})
    )

    async with httpx.AsyncClient() as client:
        first = await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=test_settings)
        second = await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=test_settings)

    assert first == second == test_settings.pdf_cache_dir / "arxiv_2401.01234.pdf"
    assert first.read_bytes() == pdf_bytes
    assert route.call_count == 1


async def test_html_page_is_rejected(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """A paywall/HTML page served instead of a PDF is rejected and nothing is cached."""
    respx_mock.get(PDF_URL).mock(
        return_value=httpx.Response(200, text="<html>Please log in</html>", headers={"Content-Type": "text/html"})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(PdfDownloadError, match="not a PDF"):
            await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=test_settings)

    assert list(test_settings.pdf_cache_dir.iterdir()) == []


async def test_oversized_pdf_is_rejected(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """Files over MAX_PDF_MB are refused so a laptop never chokes on a huge download."""
    respx_mock.get(PDF_URL).mock(return_value=httpx.Response(200, content=b"%PDF-1.7\n" + b"0" * 5000))
    tiny_limit = test_settings.model_copy(update={"max_pdf_mb": 0.001})

    async with httpx.AsyncClient() as client:
        with pytest.raises(PdfDownloadError, match="limit"):
            await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=tiny_limit)

    assert list(test_settings.pdf_cache_dir.iterdir()) == []


async def test_missing_pdf_raises_service_error(respx_mock: respx.MockRouter, test_settings: Settings) -> None:
    """A 404 is not retried and surfaces as ExternalServiceError."""
    respx_mock.get(PDF_URL).mock(return_value=httpx.Response(404, text="Not Found"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ExternalServiceError) as error:
            await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=test_settings)

    assert error.value.status_code == 404


async def test_transient_download_failure_is_retried(
    respx_mock: respx.MockRouter, test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """A 503 during download is retried and the file still arrives."""
    pdf_bytes = make_pdf(SAMPLE_PAGES)
    respx_mock.get(PDF_URL).mock(side_effect=[httpx.Response(503), httpx.Response(200, content=pdf_bytes)])

    async with httpx.AsyncClient() as client:
        path = await download_pdf(PDF_URL, "arxiv:2401.01234", client=client, settings=test_settings)

    assert path.read_bytes() == pdf_bytes
    assert len(recorded_sleeps) == 1
