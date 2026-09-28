"""Live integration tests: real arXiv, Semantic Scholar, PDF download and embedding model.

Skipped by default. Run them with (PowerShell)::

    $env:RP_RUN_INTEGRATION = "1"; pytest -m integration -v

They need internet access; the first run downloads the ~90 MB MiniLM model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from app.config import Settings
from app.schemas import Chunk
from app.services.arxiv_client import ArxivClient
from app.services.http_utils import ExternalServiceError, create_http_client
from app.services.pdf_parser import chunk_blocks, download_pdf, extract_text_blocks
from app.services.semantic_scholar_client import SemanticScholarClient
from app.services.vector_store import SentenceTransformerEmbedder, VectorStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RP_RUN_INTEGRATION") != "1", reason="set RP_RUN_INTEGRATION=1 to run live tests"),
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


async def test_live_arxiv_search(test_settings: Settings) -> None:
    """arXiv returns relevant papers with PDF links."""
    async with ArxivClient(test_settings) as client:
        papers = await client.search("hallucination retrieval augmented generation", max_results=5)

    assert papers
    assert all(paper.paper_id.startswith("arxiv:") and paper.pdf_url for paper in papers)
    assert any("hallucinat" in (paper.title + paper.abstract).lower() for paper in papers)


async def test_live_semantic_scholar_search(test_settings: Settings) -> None:
    """Semantic Scholar returns papers with citation counts (skips if the public quota is exhausted)."""
    try:
        async with SemanticScholarClient(test_settings) as client:
            papers = await client.search("retrieval augmented generation hallucination", max_results=5)
    except ExternalServiceError as error:
        if error.status_code == 429:
            pytest.skip("Semantic Scholar public quota exhausted - set SEMANTIC_SCHOLAR_API_KEY")
        raise

    assert papers
    assert any(paper.citation_count > 0 for paper in papers)


async def test_live_pdf_download_parse_and_chunk(test_settings: Settings) -> None:
    """A real arXiv PDF ('Attention Is All You Need') is downloaded, parsed and chunked."""
    async with create_http_client(test_settings) as client:
        path = await download_pdf(
            "https://arxiv.org/pdf/1706.03762", "arxiv:1706.03762", client=client, settings=test_settings
        )

    blocks = extract_text_blocks(path, max_pages=test_settings.max_pdf_pages)
    sections = [block.section for block in blocks]
    assert any("Introduction" in section for section in sections)
    assert not any("References" in section for section in sections)

    chunks = chunk_blocks(blocks, paper_id="arxiv:1706.03762", title="Attention Is All You Need")
    assert len(chunks) >= 5
    assert max(chunk.token_count for chunk in chunks) <= test_settings.chunk_size_tokens


def test_real_embedder_windows_long_chunks_and_ranks_by_meaning(tmp_path: Path) -> None:
    """MiniLM splits an 800-token chunk into windows and retrieval matches meaning, not just words."""
    embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)
    long_text = " ".join(["Transformers use self-attention to model long-range dependencies."] * 90)
    assert len(embedder.split_into_windows(long_text)) > 1

    query, related, unrelated = embedder.embed_documents(
        [
            "How can we stop language models from making things up?",
            "We reduce hallucinations in large language models by grounding answers in retrieved documents.",
            "The recipe needs two cups of flour and a pinch of salt.",
        ]
    )
    assert float(np.dot(query, related)) > float(np.dot(query, unrelated)) + 0.2

    store = VectorStore(tmp_path / "chroma", embedder)
    store.add_chunks(
        [
            Chunk(
                chunk_id="a::c000",
                paper_id="a",
                title="A",
                text="Grounding LLM answers in retrieved evidence reduces hallucination.",
            ),
            Chunk(
                chunk_id="b::c000",
                paper_id="b",
                title="B",
                text="Convolutional networks classify images of cats and dogs.",
            ),
        ]
    )
    assert store.query("preventing fabricated facts in chatbots", top_k=1)[0].paper_id == "a"
    SharedSystemClient.clear_system_cache()
