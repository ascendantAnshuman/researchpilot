"""Shared pytest fixtures.

``ENVIRONMENT=test`` is set *before* the application is imported so that
:func:`app.config.get_settings` ignores the developer's ``.env`` file and every
test runs against predictable, isolated settings. Each test gets its own
temporary data directory, so its SQLite database starts empty.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

os.environ["ENVIRONMENT"] = "test"

from chromadb.api.shared_system_client import SharedSystemClient  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.db import create_tables, dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.llm import reset_llm_state  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.vector_store import VectorStore, release_store_directory  # noqa: E402
from tests.api_harness import Api, build_api  # noqa: E402
from tests.fakes import HashingEmbedder  # noqa: E402

TEST_JWT_SECRET = "test-only-jwt-secret-that-is-long-enough-0123456789"
TEST_PASSWORD = "CorrectHorse42"


@pytest.fixture(autouse=True)
def _fresh_llm_state() -> Iterator[None]:
    """Each test starts with no remembered model quotas or pacing limiters."""
    reset_llm_state()
    yield
    reset_llm_state()


@pytest.fixture()
def test_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings that point at a throw-away data directory and contain no real API keys."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)
    monkeypatch.setenv("BCRYPT_ROUNDS", "4")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture()
def client(test_settings: Settings) -> Iterator[TestClient]:
    """A TestClient around a freshly built app; the ``with`` block runs start-up/shut-down."""
    with TestClient(create_app(test_settings)) as test_client:
        yield test_client


@pytest.fixture()
def db_session(test_settings: Settings) -> Iterator[Session]:
    """A session bound to a fresh, empty test database (no web app involved)."""
    test_settings.ensure_data_dirs()
    init_engine(test_settings.sqlalchemy_database_url)
    create_tables()
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
        dispose_engine()


@pytest.fixture()
def register_user(client: TestClient) -> Callable[..., dict[str, Any]]:
    """Factory that registers a user through the API and returns the JSON response."""

    def _register(
        email: str = "student@university.edu",
        password: str = TEST_PASSWORD,
        full_name: str | None = "Test Student",
    ) -> dict[str, Any]:
        response = client.post(
            "/api/auth/register",
            json={"email": email, "password": password, "full_name": full_name},
        )
        assert response.status_code == 201, response.text
        return response.json()

    return _register


@pytest.fixture()
def auth_headers(register_user: Callable[..., dict[str, Any]]) -> dict[str, str]:
    """``Authorization`` header for a freshly registered user."""
    token = register_user()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make retry/rate-limit waits instant and record how long they would have been."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("app.services.http_utils._sleep", fake_sleep)
    return delays


@pytest.fixture()
def keyed_settings(test_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Test settings with a (fake) Anthropic key, so Claude-dependent routes are enabled."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture()
def api(keyed_settings: Settings, vector_store: VectorStore) -> Iterator[Api]:
    """A running app whose research runs use the offline fake pipeline."""
    app, chat_llm = build_api(keyed_settings, vector_store)
    with TestClient(app) as test_client:
        yield Api(client=test_client, app=app, chat_llm=chat_llm, store=vector_store)


@pytest.fixture()
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    """A real httpx client (tests mock its traffic with respx where needed)."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture()
def vector_store(tmp_path: Path) -> Iterator[VectorStore]:
    """A ChromaDB store in a temp folder using the offline hashing embedder."""
    store = VectorStore(tmp_path / "chroma", HashingEmbedder())
    yield store
    # Release Chroma's cached client and our process lock so Windows can delete the temp folder.
    SharedSystemClient.clear_system_cache()
    release_store_directory(tmp_path / "chroma")
