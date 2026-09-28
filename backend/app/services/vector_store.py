"""Persistent vector store for paper chunks (ChromaDB + local sentence-transformers).

Embeddings:
    ``all-MiniLM-L6-v2`` reads at most 256 word-pieces, but chunks are ~800
    tokens. :class:`SentenceTransformerEmbedder` therefore splits each chunk into
    windows that fit the model, embeds every window, and averages them (mean
    pooling, then re-normalising). Every part of the chunk counts for retrieval,
    not just its first third.

Storage:
    One global collection per embedding model. Chunk ids are
    ``<paper_id>::c<index>``, so a paper that shows up in several research
    sessions is embedded once and reused. Queries are restricted to the papers
    of the current session with a ``paper_id`` metadata filter.

ChromaDB is synchronous; async callers should use ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings
from app.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 256
MAX_TITLE_METADATA_CHARS = 500
LOCK_FILENAME = ".researchpilot.lock"

# Store directories this process has locked, with their open lock files (kept open for the process lifetime).
_HELD_LOCKS: dict[str, Any] = {}
_HELD_LOCKS_GUARD = threading.Lock()


class VectorStoreInUseError(RuntimeError):
    """Another process already has this vector store open."""


def lock_store_directory(persist_dir: Path) -> None:
    """Take an exclusive lock on a store directory for the rest of this process's life.

    ChromaDB's persistent client is not safe for two processes at once: concurrent writers can corrupt the
    index so badly that merely opening it later crashes Python. Several stores in one process are fine, and
    the operating system releases the lock when the process exits, even after a crash.

    Raises:
        VectorStoreInUseError: If another process holds the lock.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    key = str(persist_dir.resolve())
    with _HELD_LOCKS_GUARD:
        if key in _HELD_LOCKS:
            return
        handle = open(persist_dir / LOCK_FILENAME, "a+b")  # noqa: SIM115 - must stay open to keep the lock
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise VectorStoreInUseError(
                f"Another ResearchPilot process (the web app, the CLI or an evaluation) is using the vector store in "
                f"{persist_dir}. ChromaDB cannot be shared safely between processes: stop the other process first."
            ) from exc
        _HELD_LOCKS[key] = handle


def release_store_directory(persist_dir: Path) -> None:
    """Give up this process's lock on a store directory (used by tests; normally it lasts until exit)."""
    with _HELD_LOCKS_GUARD:
        handle = _HELD_LOCKS.pop(str(persist_dir.resolve()), None)
    if handle is None:
        return
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    handle.close()


class Embedder(Protocol):
    """Anything that can turn texts into unit-length vectors."""

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model (used to name the collection)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class SentenceTransformerEmbedder:
    """Local CPU embeddings with windowed mean pooling for texts longer than the model limit."""

    def __init__(self, model_name: str, batch_size: int = 32, model: Any | None = None) -> None:
        """Configure the embedder; the model is loaded lazily on first use.

        Args:
            model_name: Hugging Face model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.
            batch_size: Windows embedded per forward pass.
            model: Pre-built model object (tests inject a fake here).
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = model
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """The configured model id."""
        return self._model_name

    def _get_model(self) -> Any:
        """Load the sentence-transformers model once (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s (first run downloads ~90 MB)", self._model_name)
                    self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def split_into_windows(self, text: str) -> list[str]:
        """Split ``text`` into consecutive pieces that each fit the model's input limit."""
        model = self._get_model()
        limit = max(1, int(model.max_seq_length) - 2)  # leave room for [CLS] and [SEP]
        encoding = model.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets: list[tuple[int, int]] = list(encoding["offset_mapping"])
        if len(offsets) <= limit:
            return [text]
        windows = []
        for start in range(0, len(offsets), limit):
            piece = offsets[start : start + limit]
            windows.append(text[piece[0][0] : piece[-1][1]])
        return windows

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text as the normalised mean of its window embeddings."""
        if not texts:
            return []
        windows: list[str] = []
        owners: list[int] = []
        for index, text in enumerate(texts):
            for window in self.split_into_windows(text or " "):
                windows.append(window)
                owners.append(index)

        vectors = np.asarray(
            self._get_model().encode(
                windows,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        np.add.at(pooled, np.asarray(owners), vectors)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled /= np.where(norms == 0, 1.0, norms)
        return pooled.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self.embed_documents([text])[0]


def collection_name_for(model_name: str) -> str:
    """Chroma collection name derived from the model, e.g. ``chunks-all-minilm-l6-v2``.

    Changing ``EMBEDDING_MODEL`` therefore starts a fresh collection instead of
    mixing vectors from different models.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower().rsplit("/", 1)[-1]).strip("-")
    return f"chunks-{slug or 'default'}"[:60].rstrip("-")


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Flatten a chunk's fields into Chroma metadata (scalars only, no None)."""
    return {
        "paper_id": chunk.paper_id,
        "title": chunk.title[:MAX_TITLE_METADATA_CHARS],
        "section": chunk.section,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "source": chunk.source,
    }


def _chunk_from_record(chunk_id: str, document: str | None, metadata: Any) -> Chunk:
    """Rebuild a :class:`Chunk` from what Chroma returns."""
    meta: dict[str, Any] = dict(metadata or {})
    return Chunk(
        chunk_id=chunk_id,
        paper_id=str(meta.get("paper_id", "")),
        title=str(meta.get("title", "")),
        text=document or "",
        section=str(meta.get("section", "Body")),
        page=int(meta.get("page", 0)),
        page_end=int(meta.get("page_end", 0)),
        chunk_index=int(meta.get("chunk_index", 0)),
        token_count=int(meta.get("token_count", 0)),
        source="abstract" if meta.get("source") == "abstract" else "pdf",
    )


def _paper_filter(paper_ids: Sequence[str]) -> dict[str, Any]:
    """Chroma ``where`` clause matching any of ``paper_ids``."""
    ids = list(dict.fromkeys(paper_ids))
    return {"paper_id": ids[0]} if len(ids) == 1 else {"paper_id": {"$in": ids}}


class VectorStore:
    """Store and search paper chunks."""

    def __init__(self, persist_dir: Path, embedder: Embedder, collection_name: str | None = None) -> None:
        """Open (or create) the persistent collection in ``persist_dir``.

        Raises:
            VectorStoreInUseError: If another process has the same directory open.
        """
        lock_store_directory(persist_dir)
        self._embedder = embedder
        # Writes can arrive from several Reader threads at once; serialise them.
        self._write_lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Vectors are always supplied by us, so Chroma's built-in embedder is never used.
        self._collection = self._client.get_or_create_collection(
            name=collection_name or collection_name_for(embedder.model_name),
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )
        logger.info("Vector store ready: %s (%d chunks)", self._collection.name, self._collection.count())

    @property
    def collection_name(self) -> str:
        """Name of the underlying Chroma collection."""
        return self._collection.name

    def count(self) -> int:
        """Total number of stored chunks."""
        return int(self._collection.count())

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks (re-adding the same chunk id replaces it). Returns the number stored."""
        if not chunks:
            return 0
        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = list(chunks[start : start + UPSERT_BATCH_SIZE])
            embeddings = self._embedder.embed_documents([chunk.text for chunk in batch])
            with self._write_lock:
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=[chunk.text for chunk in batch],
                    metadatas=[_chunk_metadata(chunk) for chunk in batch],  # type: ignore[misc]
                )
        logger.info("Stored %d chunks for %d paper(s)", len(chunks), len({chunk.paper_id for chunk in chunks}))
        return len(chunks)

    def paper_chunk_count(self, paper_id: str, source: str | None = None) -> int:
        """How many chunks are stored for a paper, optionally only from ``"pdf"`` or ``"abstract"``."""
        where: dict[str, Any] = {"paper_id": paper_id}
        if source is not None:
            where = {"$and": [{"paper_id": paper_id}, {"source": source}]}
        return len(self._collection.get(where=where, include=[])["ids"])

    def has_paper(self, paper_id: str, *, full_text: bool = False) -> bool:
        """Whether a paper is stored; with ``full_text=True``, only if its PDF text is stored."""
        return self.paper_chunk_count(paper_id, "pdf" if full_text else None) > 0

    def delete_paper(self, paper_id: str) -> None:
        """Remove every chunk of a paper (used before replacing abstract-only chunks with full text)."""
        with self._write_lock:
            self._collection.delete(where={"paper_id": paper_id})

    @property
    def embedder(self) -> Embedder:
        """The embedder used for this collection (also used to rank papers by relevance)."""
        return self._embedder

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id, in the requested order (unknown ids are skipped)."""
        if not chunk_ids:
            return []
        result = self._collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        found = {
            chunk_id: _chunk_from_record(chunk_id, documents[i], metadatas[i])
            for i, chunk_id in enumerate(result["ids"])
        }
        return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]

    def get_paper_chunks(self, paper_id: str) -> list[Chunk]:
        """All chunks of one paper in reading order."""
        result = self._collection.get(where={"paper_id": paper_id}, include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks = [
            _chunk_from_record(chunk_id, documents[i], metadatas[i]) for i, chunk_id in enumerate(result["ids"])
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def query(
        self,
        text: str,
        *,
        paper_ids: Sequence[str] | None = None,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``text``.

        Args:
            text: The query (a sub-question, a claim to verify, a chat question...).
            paper_ids: Restrict the search to these papers; None searches everything.
            top_k: Maximum number of results.

        Returns:
            Chunks sorted by descending cosine similarity.
        """
        if paper_ids is not None and not paper_ids:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(text)],  # type: ignore[arg-type]
            n_results=max(1, top_k),
            where=_paper_filter(paper_ids) if paper_ids is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0] if result["ids"] else []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                **_chunk_from_record(chunk_id, documents[i], metadatas[i]).model_dump(),
                score=round(1.0 - float(distances[i]), 6),
            )
            for i, chunk_id in enumerate(ids)
        ]


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """One embedder (and loaded model) per model name per process."""
    return SentenceTransformerEmbedder(model_name)


@lru_cache(maxsize=2)
def _cached_store(persist_dir: str, model_name: str) -> VectorStore:
    """One store per (directory, model) per process."""
    return VectorStore(Path(persist_dir), get_embedder(model_name))


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """The shared vector store for the configured ``DATA_DIR`` and ``EMBEDDING_MODEL``."""
    settings = settings or get_settings()
    return _cached_store(str(settings.chroma_dir), settings.embedding_model)
