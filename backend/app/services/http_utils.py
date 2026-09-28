"""Shared HTTP plumbing: client factory, exponential-backoff retries and rate limiting.

Every external call (arXiv, Semantic Scholar, PDF downloads) goes through
:func:`retry_async`, so timeouts, dropped connections, HTTP 429 (rate limited)
and 5xx errors are retried with exponential backoff plus jitter:

    delay = min(max_delay, base_delay * 2**attempt) * random(0.5 .. 1.0)

A ``Retry-After`` header from the server always takes precedence.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0


class ExternalServiceError(RuntimeError):
    """An external API failed permanently (non-retryable error or retries exhausted)."""

    def __init__(self, service: str, message: str, status_code: int | None = None) -> None:
        """Store which service failed and, if known, the HTTP status code."""
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status_code = status_code


class RetryableHTTPError(Exception):
    """A response that should be retried (429 or 5xx), optionally with a server-given delay."""

    def __init__(self, status_code: int, retry_after: float | None = None) -> None:
        """Remember the status code and the parsed ``Retry-After`` delay."""
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


async def _sleep(seconds: float) -> None:
    """Indirection over ``asyncio.sleep`` so tests can skip real waiting."""
    await asyncio.sleep(seconds)


class RateLimiter:
    """Process-wide minimum spacing between calls to one service.

    Uses a thread lock (not an asyncio lock) so a single limiter works across
    event loops, e.g. several pipeline runs or tests.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        """Create a limiter allowing at most one call per ``min_interval_seconds``."""
        self.min_interval = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    async def wait(self) -> None:
        """Wait until this caller's reserved slot arrives."""
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if delay > 0:
            await _sleep(delay)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP date) into seconds, capped at 60."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with jitter for the given 0-based retry attempt."""
    return min(max_delay, base_delay * (2**attempt)) * random.uniform(0.5, 1.0)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    service: str,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Run ``operation`` and retry it on transient failures.

    Retried: network errors and timeouts (``httpx.TransportError``) and
    :class:`RetryableHTTPError`. Anything else propagates immediately.

    Raises:
        ExternalServiceError: When all ``max_retries`` retries are used up.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except (httpx.TransportError, RetryableHTTPError) as exc:
            status_code = exc.status_code if isinstance(exc, RetryableHTTPError) else None
            if attempt >= max_retries:
                raise ExternalServiceError(
                    service, f"gave up after {attempt + 1} attempts ({exc!r})", status_code
                ) from exc
            server_delay = exc.retry_after if isinstance(exc, RetryableHTTPError) else None
            delay = server_delay if server_delay is not None else backoff_delay(attempt, base_delay, max_delay)
            logger.warning(
                "%s request failed (%r); retry %d/%d in %.1fs", service, exc, attempt + 1, max_retries, delay
            )
            await _sleep(delay)
            attempt += 1


def raise_for_status(response: httpx.Response, service: str) -> None:
    """Classify an HTTP error response.

    Raises:
        RetryableHTTPError: For 408/425/429/5xx, so :func:`retry_async` retries.
        ExternalServiceError: For any other 4xx/5xx status.
    """
    if response.status_code < 400:
        return
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise RetryableHTTPError(response.status_code, parse_retry_after(response.headers.get("Retry-After")))
    snippet = response.text[:200].replace("\n", " ") if response.content else ""
    raise ExternalServiceError(service, f"HTTP {response.status_code}: {snippet}", response.status_code)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    service: str,
    max_retries: int,
    rate_limiter: RateLimiter | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Send an HTTP request with rate limiting and retries; return the successful response."""

    async def _once() -> httpx.Response:
        if rate_limiter is not None:
            await rate_limiter.wait()
        response = await client.request(method, url, **kwargs)
        raise_for_status(response, service)
        return response

    return await retry_async(_once, service=service, max_retries=max_retries)


def create_http_client(settings: Settings | None = None) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with the configured timeout and User-Agent."""
    settings = settings or get_settings()
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.http_timeout_seconds, connect=min(10.0, settings.http_timeout_seconds)),
        headers={"User-Agent": settings.http_user_agent},
        follow_redirects=True,
    )
