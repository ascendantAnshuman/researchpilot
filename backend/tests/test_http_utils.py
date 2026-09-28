"""Tests for retries, backoff, Retry-After handling and rate limiting (all HTTP mocked)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from app.services.http_utils import (
    ExternalServiceError,
    RateLimiter,
    backoff_delay,
    parse_retry_after,
    request_with_retry,
)

URL = "https://api.example.org/items"


def test_parse_retry_after_seconds() -> None:
    """Numeric Retry-After values are seconds."""
    assert parse_retry_after("5") == 5.0


def test_parse_retry_after_is_capped() -> None:
    """A huge Retry-After never stalls a research run for more than a minute."""
    assert parse_retry_after("600") == 60.0


def test_parse_retry_after_http_date() -> None:
    """HTTP-date Retry-After values are converted into a delay from now."""
    in_thirty_seconds = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)

    delay = parse_retry_after(in_thirty_seconds)

    assert delay is not None and 25 <= delay <= 30


@pytest.mark.parametrize("value", [None, "", "soon"])
def test_parse_retry_after_ignores_missing_or_garbage(value: str | None) -> None:
    """Unparseable headers fall back to normal backoff."""
    assert parse_retry_after(value) is None


def test_backoff_delay_grows_exponentially_with_jitter_and_cap() -> None:
    """Delays double per attempt, include 50-100% jitter, and never exceed the cap."""
    for attempt in range(8):
        ceiling = min(30.0, 2.0**attempt)
        assert 0.5 * ceiling <= backoff_delay(attempt, base_delay=1.0, max_delay=30.0) <= ceiling


async def test_rate_limited_request_is_retried_after_retry_after(
    respx_mock: respx.MockRouter, recorded_sleeps: list[float]
) -> None:
    """HTTP 429 is retried after exactly the delay the server asked for."""
    route = respx_mock.get(URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={"ok": True})]
    )

    async with httpx.AsyncClient() as client:
        response = await request_with_retry(client, "GET", URL, service="Example", max_retries=3)

    assert response.json() == {"ok": True}
    assert route.call_count == 2
    assert recorded_sleeps == [2.0]


async def test_server_errors_are_retried_with_growing_delays(
    respx_mock: respx.MockRouter, recorded_sleeps: list[float]
) -> None:
    """5xx responses are retried with exponential backoff."""
    respx_mock.get(URL).mock(side_effect=[httpx.Response(503), httpx.Response(502), httpx.Response(200)])

    async with httpx.AsyncClient() as client:
        response = await request_with_retry(client, "GET", URL, service="Example", max_retries=3)

    assert response.status_code == 200
    assert len(recorded_sleeps) == 2
    assert 0.5 <= recorded_sleeps[0] <= 1.0
    assert 1.0 <= recorded_sleeps[1] <= 2.0


async def test_network_timeouts_are_retried(respx_mock: respx.MockRouter, recorded_sleeps: list[float]) -> None:
    """A connection timeout is treated as transient."""
    route = respx_mock.get(URL).mock(side_effect=[httpx.ConnectTimeout("slow network"), httpx.Response(200)])

    async with httpx.AsyncClient() as client:
        response = await request_with_retry(client, "GET", URL, service="Example", max_retries=2)

    assert response.status_code == 200
    assert route.call_count == 2
    assert len(recorded_sleeps) == 1


async def test_client_errors_are_not_retried(respx_mock: respx.MockRouter, recorded_sleeps: list[float]) -> None:
    """A 404 will not fix itself, so it fails immediately."""
    route = respx_mock.get(URL).mock(return_value=httpx.Response(404, text="not found"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ExternalServiceError) as error:
            await request_with_retry(client, "GET", URL, service="Example", max_retries=3)

    assert error.value.status_code == 404
    assert route.call_count == 1
    assert recorded_sleeps == []


async def test_gives_up_after_max_retries(respx_mock: respx.MockRouter, recorded_sleeps: list[float]) -> None:
    """After max_retries retries, an ExternalServiceError carries the last status code."""
    route = respx_mock.get(URL).mock(return_value=httpx.Response(429))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ExternalServiceError, match="gave up after 3 attempts") as error:
            await request_with_retry(client, "GET", URL, service="Example", max_retries=2)

    assert error.value.status_code == 429
    assert route.call_count == 3
    assert len(recorded_sleeps) == 2


async def test_rate_limiter_spaces_out_calls(recorded_sleeps: list[float]) -> None:
    """Three back-to-back calls with a 3 s limiter wait 0 s, ~3 s and ~6 s."""
    limiter = RateLimiter(3.0)

    for _ in range(3):
        await limiter.wait()

    assert recorded_sleeps == [pytest.approx(3.0, abs=0.2), pytest.approx(6.0, abs=0.2)]


async def test_request_uses_rate_limiter(respx_mock: respx.MockRouter, recorded_sleeps: list[float]) -> None:
    """request_with_retry waits on the limiter before each attempt."""
    respx_mock.get(URL).mock(return_value=httpx.Response(200))
    limiter = RateLimiter(1.0)

    async with httpx.AsyncClient() as client:
        await request_with_retry(client, "GET", URL, service="Example", max_retries=0, rate_limiter=limiter)
        await request_with_retry(client, "GET", URL, service="Example", max_retries=0, rate_limiter=limiter)

    assert recorded_sleeps == [pytest.approx(1.0, abs=0.2)]
