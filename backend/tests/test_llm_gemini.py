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
    QUOTAS,
    LLMUnavailableError,
    gemini_thinking_level,
    model_prices,
    retry_delay_seconds,
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


def quota_error(per: str, retry_delay: str = "7s") -> errors.ClientError:
    """A free-tier 429 exactly as Gemini reports it (quota id + retry delay in the details)."""
    return errors.ClientError(429, {"error": {
        "code": 429,
        "message": "You exceeded your current quota.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": f"GenerateRequestsPer{per}PerProjectPerModel-FreeTier",
                "quotaValue": "20" if per == "Day" else "5",
            }]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }})


def test_provider_is_chosen_from_the_model_name(test_settings: Settings) -> None:
    """gemini-* models use Gemini (and GEMINI_API_KEY); everything else uses Anthropic."""
    gemini = gemini_settings(test_settings)
    assert (gemini.llm_provider, gemini.llm_key_name, gemini.llm_configured) == ("gemini", "GEMINI_API_KEY", True)
    assert test_settings.llm_provider == "anthropic"
    assert test_settings.model_copy(update={"llm_model": "gemini-3.8-flash"}).llm_configured is False


def test_gemma_models_are_served_by_the_gemini_api(test_settings: Settings) -> None:
    """Google's open Gemma models use the same API, key and free pricing as Gemini."""
    gemma = gemini_settings(test_settings, model="gemma-4-31b-it")

    assert (gemma.llm_provider, gemma.llm_key_name) == ("gemini", "GEMINI_API_KEY")
    assert model_prices("gemma-4-31b-it", gemma) == (0.0, 0.0)
    assert gemini_thinking_level("gemma-4-31b-it", "high") is None
    assert LLMClient(gemma, client=FakeGemini()).provider == "gemini"


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
        (client_error(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED"), LLMUnavailableError, "per-minute"),
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
    test_settings: Settings, exception: Exception, expected: type[Exception], message: str, recorded_sleeps: list[float]
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


async def test_overloaded_model_falls_back_to_the_second_model(test_settings: Settings) -> None:
    """With LLM_FALLBACK_MODEL set, a 503 on the main model is retried once on the fallback model."""
    overloaded = errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
    fake = FakeGemini(overloaded, fake_gemini_response('{"value": "from fallback"}'))
    client = LLMClient(gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash"), client=fake)

    answer = await client.structured(stage="reader", system="s", messages=[{"role": "user", "content": "q"}],
                                     output_model=Answer)

    assert answer.value == "from fallback"
    assert [request["model"] for request in fake.requests] == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert client.tracker.stages["reader"].calls == 1


async def test_quota_errors_also_use_the_fallback_but_config_errors_do_not(test_settings: Settings) -> None:
    """Quota (429) moves to the fallback model; a rejected key does not (it would fail there too)."""
    quota = client_error(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED")
    bad_key = client_error(400, "API key not valid", "INVALID_ARGUMENT")
    settings = gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash")

    recovered = LLMClient(settings, client=FakeGemini(quota, fake_gemini_response("ok")))
    assert await recovered.text(stage="s", system="s", messages=[{"role": "user", "content": "q"}]) == "ok"

    with pytest.raises(LLMConfigurationError):
        await LLMClient(settings, client=FakeGemini(bad_key)).text(
            stage="s", system="s", messages=[{"role": "user", "content": "q"}]
        )


def test_fallback_from_another_provider_is_ignored(test_settings: Settings) -> None:
    """A Claude fallback for a Gemini model cannot share the client, so it is ignored."""
    client = LLMClient(gemini_settings(test_settings, llm_fallback_model="claude-sonnet-5"), client=FakeGemini())

    assert client.models == ["gemini-3.8-flash"]


def test_fallback_models_accept_a_comma_separated_list(test_settings: Settings) -> None:
    """LLM_FALLBACK_MODEL may list several models; duplicates and the main model are dropped."""
    settings = gemini_settings(
        test_settings, llm_fallback_model="gemini-3.6-flash, gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.6-flash"
    )

    assert settings.llm_fallback_models == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert LLMClient(settings, client=FakeGemini()).models == [
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
    ]


def test_retry_delay_is_read_from_the_error() -> None:
    """Gemini's RetryInfo delay is parsed from the error text."""
    assert retry_delay_seconds(str(quota_error("Minute", "35s"))) == 35.0
    assert retry_delay_seconds("no delay here") is None


async def test_rate_limit_error_names_the_quota_that_was_hit(
    test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """The error says which free-tier limit was reached (requests or tokens per minute)."""
    client = LLMClient(gemini_settings(test_settings), client=FakeGemini(quota_error("Minute", "1s")))

    with pytest.raises(LLMUnavailableError, match=r"per-minute limit \(GenerateRequestsPerMinutePerProjectPerModel"):
        await client.text(stage="s", system="s", messages=[{"role": "user", "content": "q"}])


async def test_daily_quota_is_remembered_and_the_model_skipped(test_settings: Settings) -> None:
    """After a per-day 429 the model is skipped for later calls - no wasted requests."""
    fake = FakeGemini(quota_error("Day"), fake_gemini_response("first"), fake_gemini_response("second"))
    client = LLMClient(gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash"), client=fake)

    assert await client.text(stage="s", system="s", messages=[{"role": "user", "content": "1"}]) == "first"
    assert await client.text(stage="s", system="s", messages=[{"role": "user", "content": "2"}]) == "second"

    assert [request["model"] for request in fake.requests] == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.6-flash"]
    assert QUOTAS.is_exhausted("gemini-3.8-flash")
    assert not QUOTAS.is_exhausted("gemini-3.6-flash")


async def test_quota_memory_is_shared_between_clients(test_settings: Settings) -> None:
    """A new client (e.g. the next evaluation question) also skips the exhausted model."""
    settings = gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash")
    await LLMClient(settings, client=FakeGemini(quota_error("Day"), fake_gemini_response("a"))).text(
        stage="s", system="s", messages=[{"role": "user", "content": "q"}]
    )
    fresh = FakeGemini(fake_gemini_response("b"))

    await LLMClient(settings, client=fresh).text(stage="s", system="s", messages=[{"role": "user", "content": "q"}])

    assert [request["model"] for request in fresh.requests] == ["gemini-3.6-flash"]


async def test_when_all_models_are_busy_the_client_waits_for_the_retry_delay(
    test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """Every model rate limited -> wait the shortest retryDelay, then the next round succeeds."""
    fake = FakeGemini(quota_error("Minute", "9s"), quota_error("Minute", "7s"), fake_gemini_response("done"))
    client = LLMClient(gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash"), client=fake)

    assert await client.text(stage="s", system="s", messages=[{"role": "user", "content": "q"}]) == "done"
    assert recorded_sleeps == [7.0]
    assert [request["model"] for request in fake.requests] == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.8-flash"]


async def test_a_single_rate_limited_model_is_waited_out_longer_than_an_overloaded_one(
    test_settings: Settings, recorded_sleeps: list[float]
) -> None:
    """Per-minute limits clear within a minute, so one model may wait up to 8 rounds; 503s stop after 3."""
    busy = [quota_error("Minute", "20s")] * 5
    patient = LLMClient(gemini_settings(test_settings, model="gemma-4-31b-it"),
                        client=FakeGemini(*busy, fake_gemini_response("finally")))
    assert await patient.text(stage="s", system="s", messages=[{"role": "user", "content": "q"}]) == "finally"
    assert recorded_sleeps == [20.0] * 5

    always_limited = FakeGemini(quota_error("Minute", "1s"))
    with pytest.raises(LLMUnavailableError, match="after 8 rounds"):
        await LLMClient(gemini_settings(test_settings), client=always_limited).text(
            stage="s", system="s", messages=[{"role": "user", "content": "q"}]
        )
    assert len(always_limited.requests) == 8

    overloaded = FakeGemini(errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}))
    with pytest.raises(LLMUnavailableError, match="after 3 rounds"):
        await LLMClient(gemini_settings(test_settings), client=overloaded).text(
            stage="s", system="s", messages=[{"role": "user", "content": "q"}]
        )
    assert len(overloaded.requests) == 3


async def test_a_retired_fallback_model_is_skipped_but_a_retired_main_model_is_an_error(
    test_settings: Settings,
) -> None:
    """A 404 ("no longer available to new users") on one pool model skips it for good; alone it is fatal."""
    retired = client_error(404, "This model is no longer available to new users.", "NOT_FOUND")
    fake = FakeGemini(retired, fake_gemini_response("first"), fake_gemini_response("second"))
    pool = LLMClient(gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash"), client=fake)

    assert await pool.text(stage="s", system="s", messages=[{"role": "user", "content": "1"}]) == "first"
    assert await pool.text(stage="s", system="s", messages=[{"role": "user", "content": "2"}]) == "second"
    assert [request["model"] for request in fake.requests] == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.6-flash"]
    assert QUOTAS.is_missing("gemini-3.8-flash")

    with pytest.raises(LLMConfigurationError, match="LLM_FALLBACK_MODEL"):
        await LLMClient(gemini_settings(test_settings, model="gemini-2.5-flash-lite"), client=FakeGemini(retired)).text(
            stage="s", system="s", messages=[{"role": "user", "content": "q"}]
        )


async def test_all_daily_quotas_used_up_gives_a_clear_message(test_settings: Settings) -> None:
    """When every model is out of quota for the day, the error says so and when it resets."""
    fake = FakeGemini(quota_error("Day"))
    client = LLMClient(gemini_settings(test_settings, llm_fallback_model="gemini-3.6-flash"), client=fake)

    with pytest.raises(LLMUnavailableError, match="used up its free daily quota.*midnight Pacific"):
        await client.text(stage="s", system="s", messages=[{"role": "user", "content": "q"}])
    assert len(fake.requests) == 2  # each model tried once, then no more wasted requests


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
