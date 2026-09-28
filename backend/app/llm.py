"""The single gateway to the language model. Every agent calls the LLM through this module.

Two providers sit behind one interface, chosen by ``LLM_MODEL``:

* **Anthropic Claude** (``claude-*``) through the official ``anthropic`` SDK:
  streamed requests, ``effort`` where the model supports it, and server-side
  refusal fallback on Claude Opus 5 / Fable 5.1.
* **Google Gemini** (``gemini-*``) through the official ``google-genai`` SDK:
  JSON-schema output, thinking level mapped from ``LLM_EFFORT``, and automatic
  retries on the free tier's HTTP 429 responses. Free-tier usage costs $0.

What every call gets, whichever provider serves it:

* **Structured output** - ``structured()`` returns a validated Pydantic object.
* **Resilience** - SDK retries for 429/5xx/network errors (``LLM_MAX_RETRIES``),
  a cap on parallel requests (``LLM_MAX_CONCURRENCY``) and an optional
  requests-per-minute limit (``LLM_REQUESTS_PER_MINUTE``, useful on free tiers).
* **Metering** - tokens, latency and USD cost per pipeline stage in a
  :class:`UsageTracker`, which the evaluation reports.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol, TypeVar

import anthropic
import httpx
import pydantic
from pydantic import BaseModel

from app.config import GOOGLE_MODEL_PREFIXES, Effort, Settings, get_settings
from app.services import http_utils
from app.services.http_utils import RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Message = dict[str, Any]

# USD per million tokens (input, output) - Anthropic first-party API list prices.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
FREE_TIER_PRICES: tuple[float, float] = (0.0, 0.0)
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

EFFORT_ORDER: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
# Claude models accepting every effort level, including xhigh and max.
_FULL_EFFORT_MODELS = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)
# Claude models accepting low/medium/high/max but not xhigh.
_NO_XHIGH_MODELS = ("claude-opus-4-6", "claude-sonnet-4-6")
# Claude models accepting only low/medium/high.
_BASIC_EFFORT_MODELS = ("claude-opus-4-5",)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS: frozenset[str] = frozenset({"claude-opus-5", "claude-fable-5-1"})

# 429 is deliberately absent: quota errors are handled below (skip the model, or switch to another one).
GEMINI_RETRY_STATUS_CODES = [408, 500, 502, 503, 504]
GEMINI_REFUSAL_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"})
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
_QUOTA_ID_RE = re.compile(r"quotaId['\"]?\s*:\s*['\"]([A-Za-z0-9_-]+)")
MAX_FAILOVER_ROUNDS = 3  # rounds allowed while models are overloaded (503) or unreachable
MAX_RATE_LIMITED_ROUNDS = 8  # per-minute limits clear within a minute, so they may be waited out longer
MAX_RATE_LIMIT_WAIT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Quota and pacing state shared by every client in the process
# ---------------------------------------------------------------------------


class QuotaRegistry:
    """Remembers which models have used up their daily quota, so later calls skip them straight away."""

    def __init__(self) -> None:
        """Start with no exhausted models."""
        self._exhausted: dict[str, date] = {}
        self._missing: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def quota_day() -> date:
        """The current quota day. Free-tier quotas reset at midnight Pacific time (UTC-8 is used, never early)."""
        return (datetime.now(UTC) - timedelta(hours=8)).date()

    def mark_exhausted(self, model: str) -> None:
        """Record that ``model`` cannot be used again until the next quota day."""
        with self._lock:
            self._exhausted[model] = self.quota_day()

    def mark_missing(self, model: str) -> None:
        """Record that ``model`` does not exist for this key (HTTP 404), so it is never tried again."""
        with self._lock:
            self._missing.add(model)

    def is_missing(self, model: str) -> bool:
        """Whether ``model`` was reported as not found."""
        return model in self._missing

    def is_exhausted(self, model: str) -> bool:
        """Whether ``model`` cannot be used today (daily quota used up, or the model does not exist)."""
        return model in self._missing or self._exhausted.get(model) == self.quota_day()

    def reset(self) -> None:
        """Forget everything (used by tests)."""
        with self._lock:
            self._exhausted.clear()
            self._missing.clear()


QUOTAS = QuotaRegistry()
_LIMITERS: dict[tuple[str, int], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def limiter_for(model: str, requests_per_minute: int) -> RateLimiter | None:
    """The process-wide pacing limiter for one model (None when unlimited)."""
    if requests_per_minute <= 0:
        return None
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault((model, requests_per_minute), RateLimiter(60.0 / requests_per_minute))


def reset_llm_state() -> None:
    """Clear quota memory and pacing limiters (used by tests)."""
    QUOTAS.reset()
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


def retry_delay_seconds(error_text: str) -> float | None:
    """The ``retryDelay`` a Gemini 429 error asks for, in seconds."""
    match = _RETRY_DELAY_RE.search(error_text)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every failure of an LLM call."""


class LLMConfigurationError(LLMError):
    """Missing/invalid API key or unknown model: retrying will not help."""


class LLMModelNotFoundError(LLMConfigurationError):
    """The model name does not exist, or is no longer offered to this API key (HTTP 404)."""


class LLMUnavailableError(LLMError):
    """Rate limits, overload or network problems persisted after all retries."""


class LLMRateLimitedError(LLMUnavailableError):
    """A per-minute limit was hit; ``retry_after`` says when the model accepts requests again."""

    def __init__(self, message: str, retry_after: float = 15.0) -> None:
        """Remember how long to wait before this model can be used again."""
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaExhaustedError(LLMUnavailableError):
    """The model's daily quota is used up (free tiers reset around midnight Pacific time)."""


class LLMRefusalError(LLMError):
    """The model (and any fallback model) declined the request."""


class LLMOutputError(LLMError):
    """The response was cut off or did not match the requested schema."""


# ---------------------------------------------------------------------------
# Model capabilities and pricing
# ---------------------------------------------------------------------------


def _matches(model: str, families: Sequence[str]) -> bool:
    """True if ``model`` equals or extends one of the model ``families``."""
    return any(model == family or model.startswith(f"{family}-") for family in families)


def resolve_effort(model: str, requested: Effort) -> Effort | None:
    """The Claude effort level to send for ``model``, or None if the model has no effort setting."""
    if _matches(model, _FULL_EFFORT_MODELS):
        return requested
    if _matches(model, _NO_XHIGH_MODELS):
        return "high" if requested == "xhigh" else requested
    if _matches(model, _BASIC_EFFORT_MODELS):
        return "high" if requested in ("xhigh", "max") else requested
    return None


def gemini_thinking_level(model: str, effort: Effort) -> str | None:
    """Gemini 3.x thinking level for an effort setting (older Gemini models use their default)."""
    if not model.startswith("gemini-3"):
        return None
    return {"low": "LOW", "medium": "MEDIUM"}.get(effort, "HIGH")


def lighter_effort(effort: Effort, cap: Effort = "medium") -> Effort:
    """The lower of ``effort`` and ``cap`` - used for simple extraction-style calls."""
    return EFFORT_ORDER[min(EFFORT_ORDER.index(effort), EFFORT_ORDER.index(cap))]


def model_prices(model: str, settings: Settings) -> tuple[float, float] | None:
    """USD per million (input, output) tokens: ``.env`` overrides, else free for Gemini, else the Claude table."""
    if settings.llm_input_cost_per_mtok is not None and settings.llm_output_cost_per_mtok is not None:
        return settings.llm_input_cost_per_mtok, settings.llm_output_cost_per_mtok
    if model.startswith(GOOGLE_MODEL_PREFIXES):
        return FREE_TIER_PRICES
    for family in sorted(MODEL_PRICES, key=len, reverse=True):
        if model == family or model.startswith(f"{family}-"):
            return MODEL_PRICES[family]
    return None


def compute_cost(
    prices: tuple[float, float] | None,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """USD cost of one call; cache writes cost 1.25x and cache reads 0.1x the input price."""
    if prices is None:
        return 0.0
    input_price, output_price = prices
    billed_input = (
        input_tokens + cache_write_tokens * CACHE_WRITE_MULTIPLIER + cache_read_tokens * CACHE_READ_MULTIPLIER
    )
    return (billed_input * input_price + output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


@dataclass
class StageUsage:
    """Accumulated LLM usage for one pipeline stage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def add(self, other: StageUsage) -> None:
        """Add another usage record into this one."""
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd
        self.seconds += other.seconds


class UsageTracker:
    """Collects token usage and cost per stage for one research run."""

    def __init__(self) -> None:
        """Start with no recorded usage."""
        self.stages: dict[str, StageUsage] = {}

    def record(
        self,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        seconds: float,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """Add one LLM call to ``stage``."""
        self.stages.setdefault(stage, StageUsage()).add(
            StageUsage(1, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, cost_usd, seconds)
        )

    def total(self) -> StageUsage:
        """Usage summed over all stages."""
        total = StageUsage()
        for usage in self.stages.values():
            total.add(usage)
        return total

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Per-stage usage as plain dictionaries (for JSON/logging)."""
        return {stage: asdict(usage) for stage, usage in self.stages.items()}


# ---------------------------------------------------------------------------
# Provider back-ends
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    """A provider-neutral model response."""

    text: str
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    stop: Literal["end", "max_tokens", "refusal"] = "end"
    refusal_detail: str | None = None


class _Backend(Protocol):
    """One provider's way of executing a request."""

    client: Any

    async def complete(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort,
        max_tokens: int,
        output_model: type[BaseModel] | None,
    ) -> Completion:
        """Run one request and return a :class:`Completion` (or raise an ``LLMError``)."""
        ...


class _AnthropicBackend:
    """Claude via the ``anthropic`` SDK (streamed, with refusal fallback where available)."""

    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        self.model = settings.llm_model
        if client is not None:
            self.client = client
            return
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value().strip():
            raise LLMConfigurationError("No Anthropic API key found. Set ANTHROPIC_API_KEY in the .env file.")
        try:
            self.client = anthropic.AsyncAnthropic(
                api_key=key.get_secret_value() if key else None,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        except anthropic.AnthropicError as exc:
            raise LLMConfigurationError("No Anthropic credentials found. Set ANTHROPIC_API_KEY in the .env file.") from exc

    @property
    def uses_refusal_fallback(self) -> bool:
        return self.settings.llm_refusal_fallback and self.model in FALLBACK_MODELS

    def request_kwargs(
        self,
        system: str,
        messages: list[Message],
        effort: Effort,
        max_tokens: int,
        output_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        """SDK arguments, leaving out features the model does not support."""
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        resolved_effort = resolve_effort(self.model, effort)
        if resolved_effort is not None:
            kwargs["output_config"] = {"effort": resolved_effort}
        if output_model is not None:
            kwargs["output_format"] = output_model
        return kwargs

    async def complete(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort,
        max_tokens: int,
        output_model: type[BaseModel] | None,
    ) -> Completion:
        kwargs = self.request_kwargs(system, messages, effort, max_tokens, output_model)
        try:
            if self.uses_refusal_fallback:
                manager = self.client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                manager = self.client.messages.stream(**kwargs)
            async with manager as stream:
                message = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMConfigurationError("ANTHROPIC_API_KEY was rejected - check the key in .env") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMConfigurationError(f"The API key cannot use model {self.model!r}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMModelNotFoundError(
                f"Model {self.model!r} was not found - check LLM_MODEL and LLM_FALLBACK_MODEL"
            ) from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"{stage}: request rejected by the API: {exc.message}") from exc
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
            raise LLMUnavailableError(f"{stage}: Claude is unavailable after retries ({exc})") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"{stage}: API error {exc.status_code}: {exc.message}") from exc
        except pydantic.ValidationError as exc:
            raise LLMOutputError(f"{stage}: the model's JSON did not match the expected schema: {exc}") from exc

        usage = message.usage
        stop: Literal["end", "max_tokens", "refusal"] = "end"
        detail = None
        if message.stop_reason == "refusal":
            stop = "refusal"
            details = getattr(message, "stop_details", None)
            detail = getattr(details, "category", None) if details is not None else None
        elif message.stop_reason == "max_tokens":
            stop = "max_tokens"
        return Completion(
            text="".join(block.text for block in message.content if getattr(block, "type", None) == "text").strip(),
            parsed=getattr(message, "parsed_output", None),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            stop=stop,
            refusal_detail=detail,
        )


def _gemini_contents(messages: list[Message]) -> list[Any]:
    """Convert Anthropic-style messages into Gemini ``Content`` objects (roles user/model)."""
    from google.genai import types

    contents = []
    for message in messages:
        content = message["content"]
        texts = [content] if isinstance(content, str) else [
            block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
        ]
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=text) for text in texts if text]))
    return contents


class _GeminiBackend:
    """Gemini via the ``google-genai`` SDK (JSON-schema output, retries on 429/5xx)."""

    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        self.model = settings.llm_model
        if client is not None:
            self.client = client
            return
        from google import genai
        from google.genai import types

        key = settings.gemini_api_key
        if key is None or not key.get_secret_value().strip():
            raise LLMConfigurationError("No Gemini API key found. Set GEMINI_API_KEY in the .env file.")
        self.client = genai.Client(
            api_key=key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(
                    attempts=settings.llm_max_retries + 1,
                    initial_delay=2.0,
                    max_delay=60.0,
                    exp_base=2.0,
                    jitter=1.0,
                    http_status_codes=GEMINI_RETRY_STATUS_CODES,
                ),
            ),
        )

    def request_config(
        self, system: str, effort: Effort, max_tokens: int, output_model: type[BaseModel] | None
    ) -> Any:
        """The ``GenerateContentConfig`` for a request."""
        from google.genai import types

        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            # No tools are passed, so the SDK's automatic function calling is switched off explicitly.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        level = gemini_thinking_level(self.model, effort)
        if level is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        if output_model is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = output_model
        return types.GenerateContentConfig(**config)

    async def complete(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort,
        max_tokens: int,
        output_model: type[BaseModel] | None,
    ) -> Completion:
        from google.genai import errors

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=_gemini_contents(messages),
                config=self.request_config(system, effort, max_tokens, output_model),
            )
        except errors.ClientError as exc:
            code = getattr(exc, "code", None)
            text = str(exc)
            if code == 429:
                if "PerDay" in text:
                    raise LLMQuotaExhaustedError(
                        f"{stage}: {self.model} has used up its free daily quota (resets at midnight Pacific time)"
                    ) from exc
                quota = _QUOTA_ID_RE.search(text)
                raise LLMRateLimitedError(
                    f"{stage}: {self.model} hit its free-tier per-minute limit"
                    + (f" ({quota.group(1)})" if quota else ""),
                    retry_after=retry_delay_seconds(text) or 15.0,
                ) from exc
            if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
                raise LLMConfigurationError("GEMINI_API_KEY was rejected - check the key in .env") from exc
            if code == 404:
                raise LLMModelNotFoundError(
                    f"Gemini model {self.model!r} was not found - check LLM_MODEL and LLM_FALLBACK_MODEL"
                ) from exc
            raise LLMError(f"{stage}: request rejected by Gemini: {text}") from exc
        except errors.ServerError as exc:
            raise LLMUnavailableError(f"{stage}: Gemini is unavailable after retries ({exc})") from exc
        except errors.APIError as exc:
            raise LLMError(f"{stage}: Gemini API error: {exc}") from exc
        except (httpx.TransportError, OSError) as exc:  # network failures after the SDK's own retries
            raise LLMUnavailableError(f"{stage}: cannot reach Gemini ({exc})") from exc

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0)

        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback is not None else None
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish_name = str(getattr(finish, "value", finish) or "")
        stop: Literal["end", "max_tokens", "refusal"] = "end"
        if block_reason is not None or finish_name in GEMINI_REFUSAL_REASONS:
            stop = "refusal"
        elif finish_name == "MAX_TOKENS":
            stop = "max_tokens"

        text = ""
        if stop != "refusal":
            try:
                text = (response.text or "").strip()
            except (ValueError, AttributeError):
                text = ""
        parsed: BaseModel | None = None
        if output_model is not None and stop == "end" and text:
            try:
                parsed = output_model.model_validate_json(text)
            except pydantic.ValidationError as exc:
                raise LLMOutputError(f"{stage}: Gemini's JSON did not match the expected schema: {exc}") from exc
        return Completion(
            text=text,
            parsed=parsed,
            input_tokens=max(0, input_tokens - cache_read),
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            stop=stop,
            refusal_detail=str(getattr(block_reason, "value", block_reason) or finish_name or "") or None,
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLM(Protocol):
    """What agents need from an LLM client (lets tests substitute a fake)."""

    model: str
    tracker: UsageTracker

    async def structured(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        output_model: type[T],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return the model's answer parsed into ``output_model``."""
        ...

    async def text(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's plain-text answer."""
        ...


class LLMClient:
    """The client used by all agents (see the module docstring for what it adds)."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        """Create a client for ``LLM_MODEL``.

        Args:
            settings: Configuration (defaults to the global settings).
            client: A pre-built provider SDK client (tests pass a fake).
            tracker: Usage tracker to record into (a new one by default).

        Raises:
            LLMConfigurationError: If no credentials for the provider can be found.
        """
        self.settings = settings or get_settings()
        self.model = self.settings.llm_model
        self.provider = self.settings.llm_provider
        self.tracker = tracker or UsageTracker()
        self._prices = model_prices(self.model, self.settings)
        if self._prices is None:
            logger.warning("No price known for model %s: cost will be reported as $0", self.model)
        backend_class = _GeminiBackend if self.provider == "gemini" else _AnthropicBackend
        self._backend: _Backend = backend_class(self.settings, client)
        # The main model first, then LLM_FALLBACK_MODEL entries of the same provider, tried in order when a
        # model is overloaded, rate limited or out of daily quota.
        self._backends: list[_Backend] = [self._backend]
        for fallback in self.settings.llm_fallback_models:
            if fallback in {backend.model for backend in self._backends}:
                continue
            fallback_settings = self.settings.model_copy(update={"llm_model": fallback})
            if fallback_settings.llm_provider != self.provider:
                logger.warning("Fallback model %s uses a different provider than %s; ignoring it", fallback, self.model)
                continue
            self._backends.append(backend_class(fallback_settings, self._backend.client))
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def models(self) -> list[str]:
        """The main model followed by its fallbacks, in the order they are tried."""
        return [backend.model for backend in self._backends]

    @property
    def uses_refusal_fallback(self) -> bool:
        """Whether requests opt into Claude's server-side refusal fallbacks."""
        return isinstance(self._backend, _AnthropicBackend) and self._backend.uses_refusal_fallback

    async def structured(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        output_model: type[T],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Ask for JSON matching ``output_model`` and return the validated object.

        Raises:
            LLMOutputError: If the answer was cut off or could not be parsed.
            LLMRefusalError, LLMUnavailableError, LLMConfigurationError: See :meth:`_call`.
        """
        completion = await self._call(
            stage=stage, system=system, messages=messages, effort=effort, max_tokens=max_tokens, output_model=output_model
        )
        if not isinstance(completion.parsed, output_model):
            raise LLMOutputError(f"{stage}: the model's answer could not be parsed as {output_model.__name__}")
        return completion.parsed

    async def text(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Ask for a plain-text answer (thinking is not included)."""
        completion = await self._call(stage=stage, system=system, messages=messages, effort=effort, max_tokens=max_tokens)
        return completion.text

    async def _call(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort | None,
        max_tokens: int | None,
        output_model: type[BaseModel] | None = None,
    ) -> Completion:
        """Run one request with concurrency/rate limits, record usage, and check how it ended.

        Raises:
            LLMConfigurationError: Bad key, missing permission or unknown model.
            LLMUnavailableError: Rate limit, overload or network failure after retries.
            LLMRefusalError: The model declined.
            LLMOutputError: Output truncated at ``max_tokens`` or failed validation.
            LLMError: Any other rejected request.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)
        limit = max_tokens or self.settings.llm_max_tokens
        request: dict[str, Any] = {
            "stage": stage,
            "system": system,
            "messages": messages,
            "effort": effort or self.settings.llm_effort,
            "max_tokens": limit,
            "output_model": output_model,
        }
        async with self._semaphore:
            started = time.perf_counter()
            completion = await self._complete_with_failover(stage, request)
            elapsed = time.perf_counter() - started

        cost = compute_cost(
            self._prices,
            completion.input_tokens,
            completion.output_tokens,
            completion.cache_write_tokens,
            completion.cache_read_tokens,
        )
        self.tracker.record(
            stage,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_write_tokens=completion.cache_write_tokens,
            cache_read_tokens=completion.cache_read_tokens,
            cost_usd=cost,
            seconds=elapsed,
        )
        logger.debug(
            "%s: %d in / %d out tokens (cache r/w %d/%d), $%.4f, %.1fs",
            stage,
            completion.input_tokens,
            completion.output_tokens,
            completion.cache_read_tokens,
            completion.cache_write_tokens,
            cost,
            elapsed,
        )

        if completion.stop == "refusal":
            raise LLMRefusalError(
                f"{stage}: the model declined the request (reason: {completion.refusal_detail or 'unspecified'})"
            )
        if completion.stop == "max_tokens":
            raise LLMOutputError(f"{stage}: the answer was cut off at LLM_MAX_TOKENS={limit}; raise it in .env")
        return completion

    async def _complete_with_failover(self, stage: str, request: dict[str, Any]) -> Completion:
        """Try the models in order, skipping exhausted ones, until one answers.

        * Daily quota used up -> remember it (process-wide) and move to the next model.
        * Model not found (HTTP 404) while other models remain -> skip it for good, with a warning.
        * Per-minute limit or overload -> move to the next model; if every model is
          busy, wait for the shortest ``retryDelay`` and try another round.

        Raises:
            LLMModelNotFoundError: If the only model, or every model, does not exist for this key.
            LLMUnavailableError: If no model answered within ``MAX_FAILOVER_ROUNDS`` rounds of
                overload (``MAX_RATE_LIMITED_ROUNDS`` of per-minute limits), or every model's
                daily quota is used up.
        """
        last_error: LLMUnavailableError | None = None
        overloaded_rounds = 0
        for round_number in range(1, MAX_RATE_LIMITED_ROUNDS + 1):
            if all(QUOTAS.is_missing(backend.model) for backend in self._backends):
                raise LLMModelNotFoundError(
                    f"None of the configured models exist for this API key ({', '.join(self.models)}) - "
                    "check LLM_MODEL and LLM_FALLBACK_MODEL"
                )
            available = [backend for backend in self._backends if not QUOTAS.is_exhausted(backend.model)]
            if not available:
                raise LLMUnavailableError(
                    f"{stage}: every configured model ({', '.join(self.models)}) has used up its free daily quota. "
                    "Quotas reset at midnight Pacific time; add more models to LLM_FALLBACK_MODEL or continue tomorrow."
                )
            waits: list[float] = []
            overloaded = False
            for index, backend in enumerate(available):
                next_step = "trying the next model" if index + 1 < len(available) else "waiting"
                limiter = limiter_for(backend.model, self.settings.llm_requests_per_minute)
                if limiter is not None:
                    await limiter.wait()
                try:
                    return await backend.complete(**request)
                except LLMModelNotFoundError:
                    if len(self._backends) == 1:
                        raise
                    QUOTAS.mark_missing(backend.model)
                    logger.warning("%s: model %s is not available to this API key (HTTP 404); skipping it - "
                                   "remove it from LLM_MODEL / LLM_FALLBACK_MODEL", stage, backend.model)
                except LLMQuotaExhaustedError as exc:
                    QUOTAS.mark_exhausted(backend.model)
                    logger.warning("%s: %s used up its daily quota; switching model", stage, backend.model)
                    last_error = exc
                except LLMRateLimitedError as exc:
                    logger.info("%s: %s is rate limited (retry in %.0fs); %s", stage, backend.model,
                                exc.retry_after, next_step)
                    waits.append(exc.retry_after)
                    last_error = exc
                except LLMUnavailableError as exc:
                    logger.warning("%s: %s unavailable (%s); %s", stage, backend.model, exc, next_step)
                    overloaded = True
                    waits.append(5.0 * (overloaded_rounds + 1))
                    last_error = exc
            if not waits:
                continue  # Every remaining model just ran out of daily quota: the next round reports it.
            overloaded_rounds += overloaded
            if overloaded_rounds >= MAX_FAILOVER_ROUNDS or round_number == MAX_RATE_LIMITED_ROUNDS:
                break
            await http_utils._sleep(min(max(min(waits), 1.0), MAX_RATE_LIMIT_WAIT_SECONDS))
        raise LLMUnavailableError(f"{stage}: no model could answer after {round_number} rounds ({last_error})")
