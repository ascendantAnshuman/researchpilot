"""Rebuild the vector store from the saved research sessions and the PDF cache.

Use it when the ChromaDB index (``data/chroma``) was lost or corrupted. Stop the web app first, move the
old ``data/chroma`` folder aside, then run from ``backend/``::

    .\\.venv\\Scripts\\python -m app.services.reindex

Every paper of every saved session is chunked again from its cached PDF (or its abstract when no PDF was
read), with the same chunk ids as before, so existing reports, their citations and follow-up chat work
again. Nothing is downloaded and no language model is called.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import create_tables, session_scope
from app.logging_config import setup_logging
from app.models import SessionPaper
from app.services.pdf_parser import PdfError, blocks_from_abstract, chunk_blocks, extract_text_blocks, safe_filename
from app.services.vector_store import VectorStore, VectorStoreInUseError, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ReindexResult:
    """What a rebuild did."""

    papers: int = 0
    from_pdf: int = 0
    from_abstract: int = 0
    already_indexed: int = 0
    skipped: int = 0
    chunks: int = 0


def reindex(settings: Settings, store: VectorStore | None = None) -> ReindexResult:
    """Re-chunk and re-embed every paper of every saved session that is missing from the store."""
    store = store or get_vector_store(settings)
    with session_scope() as db:
        rows = db.execute(
            select(SessionPaper.paper_id, SessionPaper.title, SessionPaper.abstract, SessionPaper.full_text)
        ).all()
    # A paper can appear in several sessions; one copy each, preferring a session that read its full text.
    papers: dict[str, tuple[str, str, bool]] = {}
    for paper_id, title, abstract, full_text in rows:
        if paper_id not in papers or (full_text and not papers[paper_id][2]):
            papers[paper_id] = (title, abstract or "", bool(full_text))

    result = ReindexResult(papers=len(papers))
    for paper_id, (title, abstract, full_text) in papers.items():
        if store.has_paper(paper_id, full_text=full_text):
            result.already_indexed += 1
            continue
        blocks, source = [], "abstract"
        pdf_path = settings.pdf_cache_dir / f"{safe_filename(paper_id)}.pdf"
        if full_text and pdf_path.exists():
            try:
                blocks, source = extract_text_blocks(pdf_path, settings.max_pdf_pages), "pdf"
            except PdfError as exc:
                logger.warning("Could not re-read the cached PDF of %s (%s); using its abstract", paper_id, exc)
        if not blocks:
            blocks, source = blocks_from_abstract(abstract), "abstract"
        chunks = chunk_blocks(
            blocks,
            paper_id=paper_id,
            title=title,
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
            source=source,  # type: ignore[arg-type]
        )
        if not chunks:
            result.skipped += 1
            continue
        store.delete_paper(paper_id)
        result.chunks += store.add_chunks(chunks)
        if source == "pdf":
            result.from_pdf += 1
        else:
            result.from_abstract += 1
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    argparse.ArgumentParser(prog="python -m app.services.reindex", description=__doc__.split("\n\n")[0]).parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_data_dirs()
    create_tables()
    try:
        result = reindex(settings)
    except VectorStoreInUseError as error:
        logger.error("%s", error)
        return 2
    logger.info(
        "Rebuilt the vector store: %d paper(s) - %d from cached PDFs, %d from abstracts, %d already indexed, "
        "%d without text; %d chunks stored",
        result.papers,
        result.from_pdf,
        result.from_abstract,
        result.already_indexed,
        result.skipped,
        result.chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
