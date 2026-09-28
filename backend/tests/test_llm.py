"""Tests for the Claude gateway (the Anthropic SDK is replaced by a fake; no network)."""

from __future__ import annotations

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, SecretStr

from app.config import Settings
from app.llm import (
    FALLBACK_BETA,
    LLMClient,
    LLMConfigurationError,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
    UsageTracker,
    compute_cost,
    lighter_effort,
    model_prices,
    resolve_effort,
)
from tests.fakes import FakeAnthropic, fake_message


class Answer(BaseModel):
    """A tiny structured-output schema."""

    value: str


def settings_for(test_settings: Settings, model: str, **extra: object) -> Settings:
    """Copy of the test settings using ``model``."""
    return test_settings.model_copy(update={"llm_model": model, **extra})


def status_error(error_class: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    """Build an SDK status error the way the SDK itself would."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_class("error", response=httpx2.Response(status, request=request), body=None)


# ---------------------------------------------------------------------------
# Capabilities and pricing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "requested", "expected"),
    [
        ("claude-opus-5", "xhigh", "xhigh"),
        ("claude-sonnet-5", "max", "max"),
        ("claude-fable-5-1", "low", "low"),
        ("claude-sonnet-4-6", "xhigh", "high"),
        ("claude-opus-4-5", "max", "high"),
        ("claude-haiku-4-5", "high", None),
        ("some-other-model", "high", None),
    ],
)
def test_resolve_effort_respects_model_support(model: str, requested: str, expected: str | None) -> None:
    """Effort is sent only where supported, capped where a level does not exist."""
    assert resolve_effort(model, requested) == expected  # type: ignore[arg-type]


def test_lighter_effort_caps_at_medium() -> None:
    """Simple extraction calls never use more than medium effort."""
    assert lighter_effort("max") == "medium"
    assert lighter_effort("low") == "low"


def test_model_prices_table_and_overrides(test_settings: Settings) -> None:
    """Built-in prices are used unless both .env overrides are set."""
    assert model_prices("claude-opus-5", test_settings) == (5.0, 25.0)
    assert model_prices("claude-opus-5-5", test_settings) == (4.0, 20.0)
    assert model_prices("claude-sonnet-5", test_settings) == (2.0, 10.0)
    assert model_prices("unknown-model", test_settings) is None
    custom = test_settings.model_copy(update={"llm_input_cost_per_mtok": 1.5, "llm_output_cost_per_mtok": 7.5})
    assert model_prices("claude-opus-5", custom) == (1.5, 7.5)


def test_compute_cost_includes_cache_pricing() -> None:
    """Cache writes cost 1.25x input, cache reads 0.1x input."""
    assert compute_cost((5.0, 25.0), 1000, 500) == pytest.approx(0.0175)
    assert compute_cost((5.0, 25.0), 0, 0, cache_write_tokens=1000) == pytest.approx(0.00625)
    assert compute_cost((5.0, 25.0), 0, 0, cache_read_tokens=1000) == pytest.approx(0.0005)
    assert compute_cost(None, 10**6, 10**6) == 0.0


def test_usage_tracker_totals_per_stage() -> None:
    """Usage accumulates per stage and in total."""
    tracker = UsageTracker()
    tracker.record("planner", input_tokens=100, output_tokens=10, cost_usd=0.01, seconds=1.0)
    tracker.record("critic", input_tokens=300, output_tokens=30, cost_usd=0.03, seconds=2.0)
    tracker.record("critic", input_tokens=200, output_tokens=20, cost_usd=0.02, seconds=1.0)

    assert tracker.stages["critic"].calls == 2
    assert tracker.stages["critic"].input_tokens == 500
    total = tracker.total()
    assert (total.calls, total.input_tokens, total.output_tokens) == (3, 600, 60)
    assert total.cost_usd == pytest.approx(0.06)
    assert tracker.as_dict()["planner"]["seconds"] == 1.0


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


async def test_opus_5_uses_refusal_fallback_and_effort(test_settings: Settings) -> None:
    """Claude Opus 5 requests go to the beta endpoint with fallbacks='default'."""
    fake = FakeAnthropic(fake_message(parsed=Answer(value="ok")))
    client = LLMClient(settings_for(test_settings, "claude-opus-5", llm_effort="high"), client=fake)

    answer = await client.structured(
        stage="planner", system="sys", messages=[{"role": "user", "content": "hi"}], output_model=Answer, effort="low"
    )

    request = fake.requests[0]
    assert answer == Answer(value="ok")
    assert request["endpoint"] == "beta"
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "low"}
    assert request["output_format"] is Answer
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == test_settings.llm_max_tokens
    assert "temperature" not in request


async def test_fallback_can_be_switched_off(test_settings: Settings) -> None:
    """LLM_REFUSAL_FALLBACK=false uses the regular endpoint."""
    fake = FakeAnthropic(fake_message(parsed=Answer(value="ok")))
    client = LLMClient(settings_for(test_settings, "claude-opus-5", llm_refusal_fallback=False), client=fake)

    await client.structured(stage="s", system="sys", messages=[], output_model=Answer)

    assert fake.requests[0]["endpoint"] == "messages"
    assert "fallbacks" not in fake.requests[0]


async def test_haiku_gets_no_effort_and_no_fallback(test_settings: Settings) -> None:
    """Features Haiku 4.5 does not support are left out."""
    fake = FakeAnthropic(fake_message(text="hello"))
    client = LLMClient(settings_for(test_settings, "claude-haiku-4-5"), client=fake)

    assert await client.text(stage="chat", system="sys", messages=[]) == "hello"
    request = fake.requests[0]
    assert request["endpoint"] == "messages"
    assert "output_config" not in request
    assert "output_format" not in request


async def test_default_effort_comes_from_settings(test_settings: Settings) -> None:
    """Without an explicit effort, LLM_EFFORT is used."""
    fake = FakeAnthropic(fake_message(text="x"))
    client = LLMClient(settings_for(test_settings, "claude-sonnet-5", llm_effort="medium"), client=fake)

    await client.text(stage="s", system="sys", messages=[])

    assert fake.requests[0]["output_config"] == {"effort": "medium"}


async def test_usage_and_cost_are_recorded_per_stage(test_settings: Settings) -> None:
    """Every call's tokens and cost land in the tracker under its stage."""
    fake = FakeAnthropic(
        fake_message(parsed=Answer(value="a"), input_tokens=2000, output_tokens=1000),
        fake_message(parsed=Answer(value="b"), input_tokens=1000, output_tokens=0, cache_read=10000),
    )
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=fake)

    await client.structured(stage="reader", system="s", messages=[], output_model=Answer)
    await client.structured(stage="reader", system="s", messages=[], output_model=Answer)

    usage = client.tracker.stages["reader"]
    assert usage.calls == 2
    assert usage.input_tokens == 3000
    assert usage.cache_read_tokens == 10000
    assert usage.cost_usd == pytest.approx((2000 * 5 + 1000 * 25) / 1e6 + (1000 * 5 + 10000 * 0.5) / 1e6)


async def test_text_ignores_thinking_blocks(test_settings: Settings) -> None:
    """Only text blocks make up the answer."""
    fake = FakeAnthropic(fake_message(text="  final answer  "))
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=fake)

    assert await client.text(stage="s", system="s", messages=[]) == "final answer"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


async def test_refusal_raises_but_usage_is_still_recorded(test_settings: Settings) -> None:
    """A refusal is an error for the caller, and its tokens are still billed and tracked."""
    fake = FakeAnthropic(fake_message(stop_reason="refusal", category="cyber"))
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=fake)

    with pytest.raises(LLMRefusalError, match="cyber"):
        await client.structured(stage="critic", system="s", messages=[], output_model=Answer)
    assert client.tracker.stages["critic"].calls == 1


async def test_truncated_answer_raises_output_error(test_settings: Settings) -> None:
    """Hitting max_tokens is reported with a hint to raise LLM_MAX_TOKENS."""
    fake = FakeAnthropic(fake_message(stop_reason="max_tokens"))
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=fake)

    with pytest.raises(LLMOutputError, match="LLM_MAX_TOKENS"):
        await client.structured(stage="synthesizer", system="s", messages=[], output_model=Answer)


async def test_missing_parsed_output_raises_output_error(test_settings: Settings) -> None:
    """A response without a parsed object is an output error, not a crash."""
    fake = FakeAnthropic(fake_message(parsed=None))
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=fake)

    with pytest.raises(LLMOutputError, match="Answer"):
        await client.structured(stage="planner", system="s", messages=[], output_model=Answer)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(anthropic.AuthenticationError, 401), LLMConfigurationError),
        (status_error(anthropic.NotFoundError, 404), LLMConfigurationError),
        (status_error(anthropic.RateLimitError, 429), LLMUnavailableError),
        (status_error(anthropic.InternalServerError, 529), LLMUnavailableError),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
            LLMUnavailableError,
        ),
    ],
)
async def test_sdk_errors_are_translated(
    test_settings: Settings, error: Exception, expected: type[Exception], recorded_sleeps: list[float]
) -> None:
    """SDK exceptions become the pipeline's own error types (after the SDK's own retries)."""
    client = LLMClient(settings_for(test_settings, "claude-opus-5"), client=FakeAnthropic(error))

    with pytest.raises(expected):
        await client.structured(stage="planner", system="s", messages=[], output_model=Answer)


def test_missing_anthropic_key_is_a_configuration_error(test_settings: Settings) -> None:
    """A Claude model without ANTHROPIC_API_KEY fails fast with a clear message (not a raw SDK TypeError)."""
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        LLMClient(settings_for(test_settings, "claude-opus-5"))


def test_real_client_is_built_from_settings(test_settings: Settings) -> None:
    """Without an injected client, an AsyncAnthropic with the configured retries is created."""
    keyed = test_settings.model_copy(update={"anthropic_api_key": SecretStr("sk-ant-test"), "llm_max_retries": 4})

    client = LLMClient(keyed)

    assert client.provider == "anthropic"
    assert isinstance(client._backend.client, anthropic.AsyncAnthropic)
    assert client._backend.client.max_retries == 4
