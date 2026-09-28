"""The vector store is locked against a second process, and can be rebuilt from saved sessions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.models import ResearchSession, SessionPaper, User
from app.services.pdf_parser import safe_filename
from app.services.reindex import reindex
from app.services.vector_store import VectorStore, release_store_directory
from tests.test_pdf_parser import make_pdf

BACKEND_DIR = Path(__file__).resolve().parents[1]
TRY_LOCK = """
import sys
from pathlib import Path
from app.services.vector_store import VectorStoreInUseError, lock_store_directory
try:
    lock_store_directory(Path(sys.argv[1]))
except VectorStoreInUseError as error:
    print(error)
    sys.exit(3)
print("locked")
"""


def try_lock_in_another_process(directory: Path) -> subprocess.CompletedProcess[str]:
    """Run a separate Python process that tries to lock ``directory``."""
    return subprocess.run(
        [sys.executable, "-c", TRY_LOCK, str(directory)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_second_process_cannot_open_a_store_in_use(vector_store: VectorStore, tmp_path: Path) -> None:
    """Two processes writing one ChromaDB folder corrupt it, so the second gets a clear error instead."""
    blocked = try_lock_in_another_process(tmp_path / "chroma")
    assert blocked.returncode == 3, blocked.stderr
    assert "Another ResearchPilot process" in blocked.stdout

    release_store_directory(tmp_path / "chroma")
    freed = try_lock_in_another_process(tmp_path / "chroma")
    assert freed.returncode == 0, freed.stderr
    assert "locked" in freed.stdout


def test_reindex_rebuilds_saved_sessions_from_the_pdf_cache(
    test_settings: Settings, db_session: Session, vector_store: VectorStore
) -> None:
    """Papers come back with the same chunk ids: from the cached PDF when it was read, else the abstract."""
    pdf_paper, abstract_paper = "arxiv:2401.00001", "s2:abc123"
    text = "Retrieval augmented generation grounds answers in retrieved passages to reduce hallucination."
    test_settings.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    (test_settings.pdf_cache_dir / f"{safe_filename(pdf_paper)}.pdf").write_bytes(
        make_pdf([["1 Introduction", *[text] * 12]])
    )
    user = User(email="ada@university.edu", hashed_password="x")
    db_session.add(user)
    db_session.flush()
    research = ResearchSession(user_id=user.id, question="What reduces hallucination in RAG?")
    research.papers = [
        SessionPaper(paper_id=pdf_paper, title="RAG paper", rank=1, full_text=True, abstract="Short abstract."),
        SessionPaper(paper_id=abstract_paper, title="Abstract only", rank=2, full_text=False,
                     abstract="An abstract about verifying generated claims against sources."),
    ]
    db_session.add(research)
    db_session.commit()

    result = reindex(test_settings, store=vector_store)

    assert (result.papers, result.from_pdf, result.from_abstract) == (2, 1, 1)
    assert vector_store.has_paper(pdf_paper, full_text=True)
    assert vector_store.get_paper_chunks(pdf_paper)[0].chunk_id == f"{pdf_paper}::c000"
    assert "hallucination" in vector_store.get_paper_chunks(pdf_paper)[0].text
    assert vector_store.has_paper(abstract_paper)
    assert reindex(test_settings, store=vector_store).already_indexed == 2
