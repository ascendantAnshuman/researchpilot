"""Tests for the Gemini back-end of the LLM gateway (the google-genai SDK is replaced by a fake; no network)."""

from __future__ import annotations

import httpx
import pytest
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, SecretStr

from app.config import Settings
from app.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
    gemini_thinking_level,
    model_prices,
)
from tests.fakes import FakeGemini, fake_gemini_response


class Answer(BaseModel):
    """A tiny structured-output schema."""

    value: str


def gemini_settings(test_settings: Settings, model: str = "gemini-3.8-flash", **extra: object) -> Settings:
    """Test settings pointing at a Gemini model."""
    return test_settings.model_copy(update={"llm_model": model, "gemini_api_key": SecretStr("gm-test"), **extra})


def client_error(code: int, message: str, status: str) -> errors.ClientError:
    """A 4xx error as the SDK raises it."""
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def test_provider_is_chosen_from_the_model_name(test_settings: Settings) -> None:
    """gemini-* models use Gemini (and GEMINI_API_KEY); everything else uses Anthropic."""
    gemini = gemini_settings(test_settings)
    assert (gemini.llm_provider, gemini.llm_key_name, gemini.llm_configured) == ("gemini", "GEMINI_API_KEY", True)
    assert test_settings.llm_provider == "anthropic"
    assert test_settings.model_copy(update={"llm_model": "gemini-3.8-flash"}).llm_configured is False


def test_gemini_free_tier_costs_nothing(test_settings: Settings) -> None:
    """Gemini usage is priced at $0 unless paid-tier prices are configured."""
    assert model_prices("gemini-3.8-flash", test_settings) == (0.0, 0.0)
    paid = test_settings.model_copy(update={"llm_input_cost_per_mtok": 0.3, "llm_output_cost_per_mtok": 2.5})
    assert model_prices("gemini-3.8-flash", paid) == (0.3, 2.5)


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gemini-3.8-flash", "low", "LOW"),
        ("gemini-3.8-flash", "medium", "MEDIUM"),
        ("gemini-3.5-flash-lite", "max", "HIGH"),
        ("gemini-2.5-flash-lite", "high", None),
    ],
)
def test_thinking_level_mapping(model: str, effort: str, expected: str | None) -> None:
    """Gemini 3.x gets a thinking level from LLM_EFFORT; older models keep their default."""
    assert gemini_thinking_level(model, effort) == expected  # type: ignore[arg-type]


async def test_structured_output_request_and_usage(test_settings: Settings) -> None:
    """The request carries the system prompt, JSON schema and roles; usage and cost are recorded."""
    fake = FakeGemini(fake_gemini_response('{"value": "ok"}', prompt_tokens=1200, output_tokens=300, thoughts=100))
    client = LLMClient(gemini_settings(test_settings), client=fake)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "context", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "task"},
        ]},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "follow-up"},
    ]

    answer = await client.structured(stage="planner", system="be precise", messages=messages, output_model=Answer,
                                     effort="low")

    assert answer == Answer(value="ok")
    request = fake.requests[0]
    config = request["config"]
    assert request["model"] == "gemini-3.8-flash"
    assert config.system_instruction == "be precise"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer
    assert config.max_output_tokens == test_settings.llm_max_tokens
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert [content.role for content in request["contents"]] == ["user", "model", "user"]
    assert [part.text for part in request["contents"][0].parts] == ["context", "task"]
    usage = client.tracker.stages["planner"]
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (1200, 400, 0.0)
    assert client.uses_refusal_fallback is False


async def test_text_answer(test_settings: Settings) -> None:
    """Plain-text calls return the text and send no JSON schema."""
    fake = FakeGemini(fake_gemini_response("  Grounding helps [E1].  "))
    client = LLMClient(gemini_settings(test_settings, model="gemini-2.5-flash-lite"), client=fake)

    assert await client.text(stage="chat", system="s", messages=[{"role": "user", "content": "q"}]) == "Grounding helps [E1]."
    config = fake.requests[0]["config"]
    assert config.response_schema is None
    assert config.thinking_config is None


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (fake_gemini_response('{"value": "x"}', finish="MAX_TOKENS"), LLMOutputError),
        (fake_gemini_response("", finish="SAFETY"), LLMRefusalError),
        (fake_gemini_response("", block_reason="PROHIBITED_CONTENT"), LLMRefusalError),
        (fake_gemini_response('{"wrong": 1}'), LLMOutputError),
        (fake_gemini_response("not json at all"), LLMOutputError),
    ],
)
async def test_bad_endings_raise_clear_errors(test_settings: Settings, response: object, error: type[Exception]) -> None:
    """Truncation, safety blocks and schema mismatches become the pipeline's error types."""
    client = LLMClient(gemini_settings(test_settings), client=FakeGemini(response))

    with pytest.raises(error):
        await client.structured(stage="critic", system="s", messages=[{"role": "user", "content": "q"}],
                                output_model=Answer)


@pytest.mark.parametrize(
    ("exception", "expected", "message"),
    [
        (client_error(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED"), LLMUnavailableError, "free-tier"),
        (client_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"),
         LLMConfigurationError, "GEMINI_API_KEY"),
        (client_error(403, "Permission denied", "PERMISSION_DENIED"), LLMConfigurationError, "GEMINI_API_KEY"),
        (client_error(404, "models/gemini-9 is not found", "NOT_FOUND"), LLMConfigurationError, "LLM_MODEL"),
        (client_error(400, "Invalid argument", "INVALID_ARGUMENT"), LLMError, "rejected"),
        (errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}),
         LLMUnavailableError, "unavailable"),
        (httpx.ConnectError("no route to host"), LLMUnavailableError, "cannot reach"),
    ],
)
async def test_sdk_errors_are_translated(
    test_settings: Settings, exception: Exception, expected: type[Exception], message: str
) -> None:
    """Every SDK failure maps to a readable pipeline error (after the SDK's own retries)."""
    client = LLMClient(gemini_settings(test_settings), client=FakeGemini(exception))

    with pytest.raises(expected, match=message):
        await client.text(stage="planner", system="s", messages=[{"role": "user", "content": "q"}])


async def test_requests_per_minute_limit_spaces_out_calls(test_settings: Settings, recorded_sleeps: list[float]) -> None:
    """LLM_REQUESTS_PER_MINUTE=30 leaves ~2 s between calls (keeps free tiers from rejecting requests)."""
    fake = FakeGemini(fake_gemini_response("a"), fake_gemini_response("b"))
    client = LLMClient(gemini_settings(test_settings, llm_requests_per_minute=30), client=fake)

    await client.text(stage="s", system="s", messages=[{"role": "user", "content": "1"}])
    await client.text(stage="s", system="s", messages=[{"role": "user", "content": "2"}])

    assert recorded_sleeps == [pytest.approx(2.0, abs=0.2)]


def test_real_gemini_client_is_built_with_retries(test_settings: Settings) -> None:
    """Without an injected client, a genai.Client is created (no network call happens)."""
    client = LLMClient(gemini_settings(test_settings, llm_max_retries=4))

    assert client.provider == "gemini"
    assert isinstance(client._backend.client, genai.Client)


def test_missing_gemini_key_is_a_configuration_error(test_settings: Settings) -> None:
    """A Gemini model without GEMINI_API_KEY fails fast with a clear message."""
    settings = test_settings.model_copy(update={"llm_model": "gemini-3.8-flash", "gemini_api_key": None})

    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        LLMClient(settings)
