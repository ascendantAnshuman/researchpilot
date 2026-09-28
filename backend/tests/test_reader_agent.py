"""Tests for the Reader agent: PDF path, abstract fallback, caching and extraction (fake LLM, mocked HTTP)."""

from __future__ import annotations

import httpx
import respx

from app.agents.reader import reader_node, read_paper, select_extraction_chunks
from app.config import Settings
from app.llm import LLMOutputError
from app.schemas import Chunk, PaperSummary
from app.services.vector_store import VectorStore
from tests.builders import RAG_PAPERS, SUMMARY, default_llm, fake_runtime, make_context, make_paper
from tests.fakes import FakeLLM
from tests.test_pdf_parser import SAMPLE_PAGES, make_pdf

PDF_URL = "https://arxiv.org/pdf/2401.00001"
PDF_PAPER = RAG_PAPERS[0].model_copy(update={"pdf_url": PDF_URL})


async def test_full_text_is_downloaded_chunked_stored_and_summarised(
    respx_mock: respx.MockRouter, test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """The happy path: PDF -> chunks in ChromaDB -> structured summary."""
    respx_mock.get(PDF_URL).mock(return_value=httpx.Response(200, content=make_pdf(SAMPLE_PAGES)))
    context = make_context(test_settings, vector_store, http_client)

    digest = await read_paper(PDF_PAPER, context)

    assert digest.full_text is True
    assert digest.chunk_count >= 1
    assert digest.summary == SUMMARY
    assert digest.note is None
    assert vector_store.has_paper(PDF_PAPER.paper_id, full_text=True)
    excerpt_prompt = context.llm.calls_for("reader")[0]["messages"][0]["content"]
    assert "Text available: full text" in excerpt_prompt
    assert "hallucinated content" in excerpt_prompt


async def test_failed_pdf_falls_back_to_abstract(
    respx_mock: respx.MockRouter, test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """A 404 PDF still yields a usable (abstract-only) paper."""
    respx_mock.get(PDF_URL).mock(return_value=httpx.Response(404))

    digest = await read_paper(PDF_PAPER, make_context(test_settings, vector_store, http_client))

    assert digest.full_text is False
    assert digest.chunk_count == 1
    assert "PDF unavailable" in (digest.note or "")
    assert vector_store.get_paper_chunks(PDF_PAPER.paper_id)[0].source == "abstract"


async def test_paper_without_pdf_uses_abstract(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """No open-access PDF: abstract only, with an explanatory note."""
    digest = await read_paper(RAG_PAPERS[1], make_context(test_settings, vector_store, http_client))

    assert (digest.full_text, digest.chunk_count) == (False, 1)
    assert digest.note == "No open-access PDF; used the abstract"


async def test_cached_full_text_is_reused_without_downloading(
    respx_mock: respx.MockRouter, test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """A paper stored by an earlier run is not downloaded again."""
    vector_store.add_chunks(
        [Chunk(chunk_id=f"{PDF_PAPER.paper_id}::c000", paper_id=PDF_PAPER.paper_id, title="T", text="Stored text.")]
    )

    digest = await read_paper(PDF_PAPER, make_context(test_settings, vector_store, http_client))

    assert digest.full_text is True
    assert respx_mock.calls.call_count == 0


async def test_abstract_chunks_are_replaced_when_full_text_arrives(
    respx_mock: respx.MockRouter, test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """If a later run finds the PDF, the old abstract-only chunks are replaced."""
    context = make_context(test_settings, vector_store, http_client)
    await read_paper(RAG_PAPERS[0], context)
    assert vector_store.paper_chunk_count(RAG_PAPERS[0].paper_id, "abstract") == 1

    respx_mock.get(PDF_URL).mock(return_value=httpx.Response(200, content=make_pdf(SAMPLE_PAGES)))
    await read_paper(PDF_PAPER, context)

    assert vector_store.paper_chunk_count(PDF_PAPER.paper_id, "abstract") == 0
    assert vector_store.has_paper(PDF_PAPER.paper_id, full_text=True)


async def test_paper_without_any_text_is_skipped(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """No PDF and no abstract: zero chunks and no LLM call."""
    context = make_context(test_settings, vector_store, http_client)

    digest = await read_paper(make_paper(42, "Title Only Paper", ""), context)

    assert digest.chunk_count == 0
    assert digest.summary is None
    assert context.llm.calls == []


async def test_summary_failure_is_recorded_not_raised(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """If extraction fails the paper is still usable, with a note."""
    llm = FakeLLM({PaperSummary: lambda call: LLMOutputError("schema mismatch")})

    digest = await read_paper(RAG_PAPERS[1], make_context(test_settings, vector_store, http_client, llm=llm))

    assert digest.summary is None
    assert digest.chunk_count == 1
    assert "summary extraction failed" in (digest.note or "")


def test_select_extraction_chunks_prioritises_key_sections_within_budget() -> None:
    """The first chunk plus method/results sections are chosen; the result stays in reading order."""
    sections = ["Front matter", "Background", "3 Method", "Related Work", "5 Results", "Appendix"]
    chunks = [
        Chunk(chunk_id=f"p::c{i:03d}", paper_id="p", title="T", text="x", section=section, chunk_index=i, token_count=800)
        for i, section in enumerate(sections)
    ]

    selected = select_extraction_chunks(chunks, max_tokens=2400)

    assert [chunk.section for chunk in selected] == ["Front matter", "3 Method", "5 Results"]


async def test_reader_node_reads_every_paper(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """All papers are read, progress is reported per paper, notes become warnings."""
    runtime, events = fake_runtime(make_context(test_settings, vector_store, http_client, llm=default_llm()))

    update = await reader_node({"papers": RAG_PAPERS}, runtime)

    assert [digest.paper.paper_id for digest in update["digests"]] == [paper.paper_id for paper in RAG_PAPERS]
    assert sum("Read " in event["message"] for event in events) == 3
    assert len(update["warnings"]) == 3
