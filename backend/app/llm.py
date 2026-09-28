"""The single gateway to Claude. Every agent calls the LLM through this module.

What it adds on top of the official ``anthropic`` SDK:

* **Model swapping** - the model comes from ``LLM_MODEL``; features a model does
  not support (e.g. the ``effort`` setting on Haiku 4.5) are left out automatically.
* **Structured output** - ``structured()`` returns a validated Pydantic object
  (the API constrains Claude's output to the model's JSON schema).
* **Streaming** - every request is streamed, so long answers never hit HTTP timeouts.
* **Resilience** - the SDK retries 429/5xx/network errors with exponential backoff
  (``LLM_MAX_RETRIES``); a semaphore caps concurrent requests (``LLM_MAX_CONCURRENCY``);
  on Claude Opus 5 / Fable 5.1 a declined request is re-run server-side on
  Anthropic's recommended fallback model (``LLM_REFUSAL_FALLBACK``).
* **Metering** - token usage, latency and USD cost are recorded per pipeline
  stage in a :class:`UsageTracker`, which the evaluation reports.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel

from app.config import Effort, Settings, get_settings

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
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

EFFORT_ORDER: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
# Models accepting every effort level, including xhigh and max.
_FULL_EFFORT_MODELS = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)
# Models accepting low/medium/high/max but not xhigh.
_NO_XHIGH_MODELS = ("claude-opus-4-6", "claude-sonnet-4-6")
# Models accepting only low/medium/high.
_BASIC_EFFORT_MODELS = ("claude-opus-4-5",)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS: frozenset[str] = frozenset({"claude-opus-5", "claude-fable-5-1"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every failure of an LLM call."""


class LLMConfigurationError(LLMError):
    """Missing/invalid API key or unknown model: retrying will not help."""


class LLMUnavailableError(LLMError):
    """Rate limits, overload or network problems persisted after all retries."""


class LLMRefusalError(LLMError):
    """Claude (and any fallback model) declined the request."""


class LLMOutputError(LLMError):
    """The response was cut off or did not match the requested schema."""


# ---------------------------------------------------------------------------
# Model capabilities and pricing
# ---------------------------------------------------------------------------


def _matches(model: str, families: Sequence[str]) -> bool:
    """True if ``model`` equals or extends one of the model ``families``."""
    return any(model == family or model.startswith(f"{family}-") for family in families)


def resolve_effort(model: str, requested: Effort) -> Effort | None:
    """The effort level to send for ``model``, or None if the model has no effort setting."""
    if _matches(model, _FULL_EFFORT_MODELS):
        return requested
    if _matches(model, _NO_XHIGH_MODELS):
        return "high" if requested == "xhigh" else requested
    if _matches(model, _BASIC_EFFORT_MODELS):
        return "high" if requested in ("xhigh", "max") else requested
    return None


def lighter_effort(effort: Effort, cap: Effort = "medium") -> Effort:
    """The lower of ``effort`` and ``cap`` - used for simple extraction-style calls."""
    return EFFORT_ORDER[min(EFFORT_ORDER.index(effort), EFFORT_ORDER.index(cap))]


def model_prices(model: str, settings: Settings) -> tuple[float, float] | None:
    """USD per million (input, output) tokens: ``.env`` overrides first, then the built-in table."""
    if settings.llm_input_cost_per_mtok is not None and settings.llm_output_cost_per_mtok is not None:
        return settings.llm_input_cost_per_mtok, settings.llm_output_cost_per_mtok
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
        """Return Claude's answer parsed into ``output_model``."""
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
        """Return Claude's plain-text answer."""
        ...


class LLMClient:
    """Claude client used by all agents (see the module docstring for what it adds)."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        """Create a client for ``LLM_MODEL``.

        Args:
            settings: Configuration (defaults to the global settings).
            client: A pre-built ``anthropic.AsyncAnthropic`` (tests pass a fake).
            tracker: Usage tracker to record into (a new one by default).

        Raises:
            LLMConfigurationError: If no Anthropic credentials can be found.
        """
        self.settings = settings or get_settings()
        self.model = self.settings.llm_model
        self.tracker = tracker or UsageTracker()
        self._prices = model_prices(self.model, self.settings)
        if self._prices is None:
            logger.warning("No price known for model %s: cost will be reported as $0", self.model)
        self._semaphore: asyncio.Semaphore | None = None
        if client is not None:
            self._client = client
            return
        key = self.settings.anthropic_api_key
        try:
            self._client = anthropic.AsyncAnthropic(
                api_key=key.get_secret_value() if key else None,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
        except anthropic.AnthropicError as exc:
            raise LLMConfigurationError(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY in the .env file."
            ) from exc

    @property
    def uses_refusal_fallback(self) -> bool:
        """Whether requests opt into server-side refusal fallbacks."""
        return self.settings.llm_refusal_fallback and self.model in FALLBACK_MODELS

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
        """Ask Claude for JSON matching ``output_model`` and return the validated object.

        Raises:
            LLMOutputError: If the answer was cut off or could not be parsed.
            LLMRefusalError, LLMUnavailableError, LLMConfigurationError: See :meth:`_call`.
        """
        message = await self._call(
            stage=stage,
            system=system,
            messages=messages,
            effort=effort,
            max_tokens=max_tokens,
            output_model=output_model,
        )
        parsed = getattr(message, "parsed_output", None)
        if not isinstance(parsed, output_model):
            raise LLMOutputError(f"{stage}: Claude's answer could not be parsed as {output_model.__name__}")
        return parsed

    async def text(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Ask Claude for a plain-text answer (thinking blocks are ignored)."""
        message = await self._call(stage=stage, system=system, messages=messages, effort=effort, max_tokens=max_tokens)
        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text").strip()

    def _request_kwargs(
        self,
        system: str,
        messages: list[Message],
        effort: Effort | None,
        max_tokens: int | None,
        output_model: type[BaseModel] | None,
    ) -> dict[str, Any]:
        """Build SDK arguments, leaving out features the model does not support."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "system": system,
            "messages": messages,
        }
        resolved_effort = resolve_effort(self.model, effort or self.settings.llm_effort)
        if resolved_effort is not None:
            kwargs["output_config"] = {"effort": resolved_effort}
        if output_model is not None:
            kwargs["output_format"] = output_model
        return kwargs

    async def _call(
        self,
        *,
        stage: str,
        system: str,
        messages: list[Message],
        effort: Effort | None,
        max_tokens: int | None,
        output_model: type[BaseModel] | None = None,
    ) -> Any:
        """Stream one request, record usage, and translate SDK errors.

        Raises:
            LLMConfigurationError: Bad key, missing permission or unknown model.
            LLMUnavailableError: Rate limit, overload or network failure after retries.
            LLMRefusalError: Claude declined (``stop_reason == "refusal"``).
            LLMOutputError: Output truncated at ``max_tokens`` or failed validation.
            LLMError: Any other rejected request.
        """
        kwargs = self._request_kwargs(system, messages, effort, max_tokens, output_model)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)

        async with self._semaphore:
            started = time.perf_counter()
            try:
                if self.uses_refusal_fallback:
                    stream_manager = self._client.beta.messages.stream(
                        **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                    )
                else:
                    stream_manager = self._client.messages.stream(**kwargs)
                async with stream_manager as stream:
                    message = await stream.get_final_message()
            except anthropic.AuthenticationError as exc:
                raise LLMConfigurationError("ANTHROPIC_API_KEY was rejected - check the key in .env") from exc
            except anthropic.PermissionDeniedError as exc:
                raise LLMConfigurationError(f"The API key cannot use model {self.model!r}") from exc
            except anthropic.NotFoundError as exc:
                raise LLMConfigurationError(f"Model {self.model!r} was not found - check LLM_MODEL") from exc
            except anthropic.BadRequestError as exc:
                raise LLMError(f"{stage}: request rejected by the API: {exc.message}") from exc
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
                raise LLMUnavailableError(f"{stage}: Claude is unavailable after retries ({exc})") from exc
            except anthropic.APIStatusError as exc:
                raise LLMError(f"{stage}: API error {exc.status_code}: {exc.message}") from exc
            except pydantic.ValidationError as exc:
                raise LLMOutputError(f"{stage}: Claude's JSON did not match the expected schema: {exc}") from exc
            elapsed = time.perf_counter() - started

        self._record_usage(stage, message, elapsed)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMRefusalError(f"{stage}: Claude declined the request (category: {category or 'unspecified'})")
        if message.stop_reason == "max_tokens":
            raise LLMOutputError(
                f"{stage}: the answer was cut off at LLM_MAX_TOKENS={kwargs['max_tokens']}; raise it in .env"
            )
        return message

    def _record_usage(self, stage: str, message: Any, seconds: float) -> None:
        """Record tokens and cost of one response."""
        usage = message.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cost = compute_cost(self._prices, input_tokens, output_tokens, cache_write, cache_read)
        self.tracker.record(
            stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            cost_usd=cost,
            seconds=seconds,
        )
        logger.debug(
            "%s: %d in / %d out tokens (cache r/w %d/%d), $%.4f, %.1fs",
            stage,
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
            cost,
            seconds,
        )
