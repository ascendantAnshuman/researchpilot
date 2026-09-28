"""Tests for sentence splitting and the 800/100-token chunker."""

from __future__ import annotations

import pytest

from app.services.pdf_parser import (
    TextBlock,
    blocks_from_abstract,
    chunk_blocks,
    estimate_tokens,
    split_sentences,
)


def paragraph(sentence_count: int, topic: str = "retrieval") -> str:
    """Distinct 13-word sentences (18 estimated tokens each)."""
    return " ".join(
        f"Sentence {i} discusses {topic} augmented generation and hallucination control in large language models."
        for i in range(sentence_count)
    )


def test_estimate_tokens() -> None:
    """Roughly 4 tokens per 3 words, rounded up."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") == 4
    assert estimate_tokens(" ".join(["word"] * 600)) == 800


def test_split_sentences_respects_abbreviations() -> None:
    """'et al.', 'Fig.' and 'e.g.' do not end sentences."""
    text = "We follow Lewis et al. (2020) closely. See Fig. 3 for details. This works, e.g. for RAG. Final sentence!"

    assert split_sentences(text) == [
        "We follow Lewis et al. (2020) closely.",
        "See Fig. 3 for details.",
        "This works, e.g. for RAG.",
        "Final sentence!",
    ]


def test_chunks_respect_size_and_overlap() -> None:
    """No chunk exceeds the budget and each chunk starts with the end of the previous one."""
    blocks = [TextBlock(page=1, section="1 Introduction", text=paragraph(60))]

    chunks = chunk_blocks(blocks, paper_id="arxiv:1", title="T", chunk_size=100, overlap=20)

    assert len(chunks) > 5
    assert all(chunk.token_count <= 100 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = split_sentences(current.text)[0]
        assert first_sentence in previous.text


def test_default_sizes_follow_project_spec() -> None:
    """With the defaults, chunks are close to (but never over) 800 tokens."""
    blocks = [TextBlock(page=1, section="Body", text=paragraph(200))]

    chunks = chunk_blocks(blocks, paper_id="arxiv:1", title="T")

    assert all(chunk.token_count <= 800 for chunk in chunks)
    assert max(chunk.token_count for chunk in chunks[:-1]) >= 700


def test_chunks_never_cross_sections() -> None:
    """A new section always starts a new chunk, with no overlap from the previous section."""
    blocks = [
        TextBlock(page=1, section="1 Introduction", text=paragraph(3, "intro")),
        TextBlock(page=2, section="2 Method", text=paragraph(3, "method")),
    ]

    intro, method = chunk_blocks(blocks, paper_id="arxiv:1", title="T")

    assert (intro.section, method.section) == ("1 Introduction", "2 Method")
    assert "intro" not in method.text


def test_chunk_ids_and_indices_are_sequential_across_sections() -> None:
    """Ids are '<paper_id>::c000', '::c001', ... in reading order."""
    blocks = [
        TextBlock(page=1, section="A", text=paragraph(12)),
        TextBlock(page=2, section="B", text=paragraph(12)),
    ]

    chunks = chunk_blocks(blocks, paper_id="arxiv:2401.01234", title="T", chunk_size=100, overlap=20)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_id for chunk in chunks] == [f"arxiv:2401.01234::c{i:03d}" for i in range(len(chunks))]
    assert all(chunk.paper_id == "arxiv:2401.01234" and chunk.title == "T" for chunk in chunks)


def test_page_range_is_tracked() -> None:
    """A chunk built from two pages records both its first and last page."""
    blocks = [
        TextBlock(page=3, section="Results", text=paragraph(2)),
        TextBlock(page=4, section="Results", text=paragraph(2)),
    ]

    (chunk,) = chunk_blocks(blocks, paper_id="p", title="T")

    assert (chunk.page, chunk.page_end) == (3, 4)


def test_overlong_sentence_is_split() -> None:
    """A 'sentence' longer than a chunk (e.g. a table dump) is broken into word windows."""
    blocks = [TextBlock(page=1, section="Table", text=" ".join(f"cell{i}" for i in range(3000)))]

    chunks = chunk_blocks(blocks, paper_id="p", title="T", chunk_size=200, overlap=30)

    assert len(chunks) >= 15
    assert all(chunk.token_count <= 200 for chunk in chunks)
    assert chunks[0].text.startswith("cell0 cell1")
    assert chunks[-1].text.endswith("cell2999")


def test_word_level_overlap_when_sentences_are_long() -> None:
    """If the last sentence is longer than the overlap, its final words are carried over."""
    long_sentence = " ".join(["token"] * 44) + " end."
    blocks = [TextBlock(page=1, section="S", text=" ".join(f"Part{i} " + long_sentence for i in range(6)))]

    chunks = chunk_blocks(blocks, paper_id="p", title="T", chunk_size=150, overlap=20)

    assert len(chunks) >= 2
    carried = 20 * 3 // 4
    assert chunks[1].text.split()[:carried] == chunks[0].text.split()[-carried:]


def test_empty_input_gives_no_chunks() -> None:
    """No text, no chunks."""
    assert chunk_blocks([], paper_id="p", title="T") == []


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """An overlap as large as the chunk would never make progress."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_blocks([TextBlock(1, "S", "Some text here.")], paper_id="p", title="T", chunk_size=50, overlap=50)


def test_abstract_fallback_chunks_are_labelled() -> None:
    """Chunks built from the abstract say so, so the UI can mark 'abstract only'."""
    blocks = blocks_from_abstract("We study hallucination in RAG. We propose a critic.")

    (chunk,) = chunk_blocks(blocks, paper_id="s2:abc", title="T", source="abstract")

    assert chunk.source == "abstract"
    assert (chunk.page, chunk.section) == (0, "Abstract")
