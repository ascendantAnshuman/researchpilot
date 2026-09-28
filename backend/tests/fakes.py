"""Lightweight fakes so unit tests never download models or call real services."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.llm import UsageTracker
from app.schemas import Paper
from app.services.http_utils import ExternalServiceError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    model_name = "test/hashing-embedder"

    def __init__(self, dimensions: int = 256) -> None:
        """Create an embedder producing ``dimensions``-long unit vectors."""
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        return self._vector(text)


class FakeTokenizer:
    """Mimics a Hugging Face fast tokenizer: whitespace tokens with character offsets."""

    def __call__(self, text: str, **_: Any) -> dict[str, list[Any]]:
        """Tokenise ``text`` into whitespace-separated tokens."""
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


class FakeSentenceModel:
    """Mimics ``SentenceTransformer``: tiny input limit and recorded ``encode`` calls."""

    def __init__(self, max_seq_length: int = 6) -> None:
        """Create a model that accepts ``max_seq_length - 2`` tokens per window."""
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()
        self.encoded_batches: list[list[str]] = []
        self._embedder = HashingEmbedder(dimensions=16)

    def encode(self, sentences: list[str], **_: Any) -> np.ndarray:
        """Return unit vectors for each input window."""
        self.encoded_batches.append(list(sentences))
        return np.asarray(self._embedder.embed_documents(sentences), dtype=np.float32)


# ---------------------------------------------------------------------------
# LLM fakes
# ---------------------------------------------------------------------------

Handler = Callable[[dict[str, Any]], Any]


def prompt_text(call: dict[str, Any]) -> str:
    """All text a fake LLM call received (system prompt + every message block)."""
    parts = [call["system"]]
    for message in call["messages"]:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(block["text"] for block in content if block.get("type") == "text")
    return "\n".join(parts)


class FakeLLM:
    """Stands in for LLMClient: answers each output schema with a handler and records calls.

    A handler receives the call (``stage``, ``system``, ``messages``, ``output_model``,
    ``effort``) and returns the model instance - or an exception instance to raise.
    """

    def __init__(
        self,
        handlers: dict[type[BaseModel], Handler],
        model: str = "fake-model",
        text_handler: Handler | None = None,
    ) -> None:
        """Create a fake answering ``output_model`` types with ``handlers`` and text calls with ``text_handler``."""
        self.handlers = handlers
        self.text_handler = text_handler
        self.model = model
        self.tracker = UsageTracker()
        self.calls: list[dict[str, Any]] = []

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        """Calls made by one pipeline stage."""
        return [call for call in self.calls if call["stage"] == stage]

    async def structured(self, *, stage: str, system: str, messages: list[Any], output_model: type[BaseModel],
                         effort: str | None = None, max_tokens: int | None = None) -> Any:
        """Return (or raise) whatever the handler for ``output_model`` produces."""
        call = {"stage": stage, "system": system, "messages": messages, "output_model": output_model, "effort": effort}
        self.calls.append(call)
        self.tracker.record(stage, input_tokens=1000, output_tokens=200, cost_usd=0.01, seconds=0.0)
        result = self.handlers[output_model](call)
        if isinstance(result, BaseException):
            raise result
        return result

    async def text(self, *, stage: str, system: str, messages: list[Any], effort: str | None = None,
                   max_tokens: int | None = None) -> str:
        """Return (or raise) whatever ``text_handler`` produces."""
        if self.text_handler is None:
            raise NotImplementedError("FakeLLM was created without a text_handler")
        call = {"stage": stage, "system": system, "messages": messages, "output_model": None, "effort": effort}
        self.calls.append(call)
        self.tracker.record(stage, input_tokens=500, output_tokens=100, cost_usd=0.005, seconds=0.0)
        result = self.text_handler(call)
        if isinstance(result, BaseException):
            raise result
        return str(result)


class FakeSearchClient:
    """Stands in for ArxivClient / SemanticScholarClient."""

    def __init__(self, papers: Sequence[Paper] = (), error_status: int | None = None,
                 lookup: dict[str, Paper] | None = None) -> None:
        """Return ``papers`` for every query, or raise HTTP ``error_status``."""
        self.papers = list(papers)
        self.error_status = error_status
        self.lookup = lookup or {}
        self.queries: list[str] = []
        self.lookups: list[list[str]] = []

    async def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        """Record the query and return the canned papers (or fail)."""
        self.queries.append(query)
        if self.error_status is not None:
            raise ExternalServiceError("Fake", "unavailable", self.error_status)
        return [paper.model_copy() for paper in self.papers]

    async def get_papers(self, ids: Sequence[str]) -> list[Paper | None]:
        """Batch lookup against ``lookup``."""
        self.lookups.append(list(ids))
        if self.error_status is not None:
            raise ExternalServiceError("Fake", "unavailable", self.error_status)
        return [self.lookup.get(paper_id) for paper_id in ids]


# ---------------------------------------------------------------------------
# Anthropic SDK fake (for testing app.llm itself)
# ---------------------------------------------------------------------------


def fake_message(
    *,
    parsed: Any = None,
    text: str = "",
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_write: int = 0,
    cache_read: int = 0,
    category: str | None = None,
) -> SimpleNamespace:
    """An object shaped like the SDK's final (parsed) message."""
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        parsed_output=parsed,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
    )


class _FakeStream:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    async def __aenter__(self) -> _FakeStream:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self._outcome


class _FakeMessages:
    def __init__(self, owner: FakeAnthropic, endpoint: str) -> None:
        self._owner = owner
        self._endpoint = endpoint

    def stream(self, **kwargs: Any) -> _FakeStream:
        self._owner.requests.append({"endpoint": self._endpoint, **kwargs})
        return _FakeStream(self._owner.outcomes.pop(0))


def fake_gemini_response(
    text: str = "",
    *,
    finish: str = "STOP",
    prompt_tokens: int = 1000,
    output_tokens: int = 200,
    thoughts: int = 50,
    cached: int = 0,
    block_reason: str | None = None,
) -> SimpleNamespace:
    """An object shaped like google-genai's ``GenerateContentResponse``."""
    from google.genai import types

    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=types.FinishReason(finish))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts,
            cached_content_token_count=cached,
        ),
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
    )


class FakeGemini:
    """Mimics ``genai.Client``: records generate_content() calls and replays queued outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        """Queue responses (or exceptions) to return from successive calls."""
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, *, model: str, contents: Any, config: Any) -> Any:
        self.requests.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeAnthropic:
    """Mimics ``AsyncAnthropic``: records stream() kwargs and replays queued outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        """Queue messages (or exceptions) to return from successive calls."""
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.messages = _FakeMessages(self, "messages")
        self.beta = SimpleNamespace(messages=_FakeMessages(self, "beta"))
