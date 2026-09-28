"""Tests for the ChromaDB vector store (offline hashing embedder) and windowed pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from chromadb.api.shared_system_client import SharedSystemClient

from app.schemas import Chunk, ChunkSource
from app.services.vector_store import SentenceTransformerEmbedder, VectorStore, collection_name_for
from tests.fakes import FakeSentenceModel, HashingEmbedder


def make_chunk(paper_id: str, index: int, text: str, source: ChunkSource = "pdf") -> Chunk:
    """Build a chunk with realistic metadata."""
    return Chunk(
        chunk_id=f"{paper_id}::c{index:03d}",
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        text=text,
        section="2 Method",
        page=index + 1,
        page_end=index + 2,
        chunk_index=index,
        token_count=len(text.split()),
        source=source,
    )


CORPUS = [
    make_chunk("arxiv:1", 0, "hallucination detection in retrieval augmented generation with a critic"),
    make_chunk("arxiv:1", 1, "image segmentation with convolutional networks on medical scans"),
    make_chunk("arxiv:2", 0, "reducing hallucination by verifying generated claims against retrieved evidence"),
    make_chunk("arxiv:3", 0, "protein folding prediction with deep learning"),
]


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


def test_add_and_query_ranks_relevant_chunks_first(vector_store: VectorStore) -> None:
    """The most similar chunk comes first and scores are sorted descending."""
    vector_store.add_chunks(CORPUS)

    results = vector_store.query("hallucination in retrieval augmented generation", top_k=3)

    assert results[0].chunk_id == "arxiv:1::c000"
    assert [result.score for result in results] == sorted((result.score for result in results), reverse=True)
    assert all(result.score <= 1.0 for result in results)


def test_query_can_be_restricted_to_session_papers(vector_store: VectorStore) -> None:
    """Only the listed papers are searched."""
    vector_store.add_chunks(CORPUS)

    results = vector_store.query("hallucination", paper_ids=["arxiv:2", "arxiv:3"], top_k=10)

    assert {result.paper_id for result in results} == {"arxiv:2", "arxiv:3"}
    assert results[0].paper_id == "arxiv:2"


def test_query_single_paper_filter(vector_store: VectorStore) -> None:
    """A one-element paper list uses a simple equality filter."""
    vector_store.add_chunks(CORPUS)

    results = vector_store.query("networks", paper_ids=["arxiv:1"], top_k=10)

    assert {result.paper_id for result in results} == {"arxiv:1"}


def test_query_with_empty_paper_list_returns_nothing(vector_store: VectorStore) -> None:
    """An empty session has nothing to search."""
    vector_store.add_chunks(CORPUS)

    assert vector_store.query("hallucination", paper_ids=[]) == []


def test_upsert_is_idempotent(vector_store: VectorStore) -> None:
    """Adding the same chunks twice does not duplicate them."""
    vector_store.add_chunks(CORPUS)
    vector_store.add_chunks(CORPUS)

    assert vector_store.count() == len(CORPUS)


def test_chunks_round_trip_with_all_metadata(vector_store: VectorStore) -> None:
    """Everything stored comes back unchanged, in the requested order, skipping unknown ids."""
    vector_store.add_chunks(CORPUS)

    fetched = vector_store.get_chunks(["arxiv:2::c000", "missing::c999", "arxiv:1::c001"])

    assert fetched == [CORPUS[2], CORPUS[1]]


def test_paper_chunks_come_back_in_reading_order(vector_store: VectorStore) -> None:
    """get_paper_chunks sorts by chunk index regardless of insertion order."""
    chunks = [make_chunk("arxiv:9", i, f"text number {i}") for i in range(5)]
    vector_store.add_chunks(list(reversed(chunks)))

    assert [chunk.chunk_index for chunk in vector_store.get_paper_chunks("arxiv:9")] == [0, 1, 2, 3, 4]


def test_abstract_only_paper_can_be_upgraded_to_full_text(vector_store: VectorStore) -> None:
    """has_paper(full_text=True) distinguishes PDF text from the abstract fallback."""
    vector_store.add_chunks([make_chunk("s2:abc", 0, "abstract text only", source="abstract")])
    assert vector_store.has_paper("s2:abc")
    assert not vector_store.has_paper("s2:abc", full_text=True)

    vector_store.delete_paper("s2:abc")
    assert not vector_store.has_paper("s2:abc")

    vector_store.add_chunks([make_chunk("s2:abc", i, f"full text part {i}") for i in range(3)])
    assert vector_store.has_paper("s2:abc", full_text=True)
    assert vector_store.paper_chunk_count("s2:abc") == 3


def test_data_persists_across_restarts(tmp_path: Path) -> None:
    """A new store on the same folder sees previously stored chunks."""
    VectorStore(tmp_path / "chroma", HashingEmbedder()).add_chunks(CORPUS)
    SharedSystemClient.clear_system_cache()

    reopened = VectorStore(tmp_path / "chroma", HashingEmbedder())

    assert reopened.count() == len(CORPUS)
    SharedSystemClient.clear_system_cache()


def test_collection_name_is_derived_from_model() -> None:
    """Each embedding model gets its own collection."""
    assert collection_name_for("sentence-transformers/all-MiniLM-L6-v2") == "chunks-all-minilm-l6-v2"
    assert collection_name_for("BAAI/bge-small-en-v1.5") == "chunks-bge-small-en-v1-5"


# ---------------------------------------------------------------------------
# Windowed mean pooling
# ---------------------------------------------------------------------------


def test_short_text_is_a_single_window() -> None:
    """Text within the model limit is embedded as-is."""
    embedder = SentenceTransformerEmbedder("fake", model=FakeSentenceModel(max_seq_length=6))

    assert embedder.split_into_windows("one two three") == ["one two three"]


def test_long_text_is_split_into_model_sized_windows() -> None:
    """With a 4-token limit, 10 tokens become windows of 4, 4 and 2."""
    embedder = SentenceTransformerEmbedder("fake", model=FakeSentenceModel(max_seq_length=6))

    assert embedder.split_into_windows("a b c d e f g h i j") == ["a b c d", "e f g h", "i j"]


def test_long_text_embedding_is_normalised_mean_of_windows() -> None:
    """Every window contributes: the chunk vector is the re-normalised average."""
    model = FakeSentenceModel(max_seq_length=6)
    embedder = SentenceTransformerEmbedder("fake", model=model)

    short_vector, long_vector = embedder.embed_documents(["short text", "a b c d e f g h i j"])

    assert model.encoded_batches == [["short text", "a b c d", "e f g h", "i j"]]
    window_vectors = np.asarray(HashingEmbedder(dimensions=16).embed_documents(["a b c d", "e f g h", "i j"]))
    expected = window_vectors.mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(long_vector, expected, rtol=1e-5)
    assert abs(float(np.linalg.norm(short_vector)) - 1.0) < 1e-5


def test_embed_empty_list() -> None:
    """No texts, no work."""
    model = FakeSentenceModel()
    embedder = SentenceTransformerEmbedder("fake", model=model)

    assert embedder.embed_documents([]) == []
    assert model.encoded_batches == []
