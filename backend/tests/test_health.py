"""Tests for the health endpoint and the basic FastAPI wiring (CORS, docs, start-up)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings, get_settings
from app.main import create_app


def test_health_returns_ok(client: TestClient, test_settings: Settings) -> None:
    """The health check answers 200 with non-secret configuration details."""
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "ResearchPilot"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["database"] == "ok"
    assert body["llm_model"] == test_settings.llm_model
    assert body["llm_configured"] is False
    assert body["critic_enabled"] is True
    assert "server_time" in body


def test_health_returns_503_when_database_is_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the database check fails, the endpoint says so and returns 503."""
    monkeypatch.setattr("app.routes.health.check_database", lambda: False)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "unavailable"


def test_health_never_leaks_secrets(client: TestClient) -> None:
    """No key or secret value appears anywhere in the health payload."""
    text = client.get("/api/health").text.lower()

    assert "secret" not in text
    assert "api_key" not in text


def test_health_reports_llm_configured_when_key_present(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting ANTHROPIC_API_KEY flips ``llm_configured`` to true."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["llm_configured"] is True


def test_root_points_to_docs(client: TestClient) -> None:
    """Opening the bare API URL tells you where the docs and health check live."""
    body = client.get("/").json()

    assert body["docs"] == "/docs"
    assert body["health"] == "/api/health"


def test_openapi_schema_lists_health_route(client: TestClient) -> None:
    """The auto-generated OpenAPI schema (used by /docs) includes the health route."""
    schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    """A browser pre-flight from the Next.js dev server is allowed."""
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unknown_origin(client: TestClient) -> None:
    """Origins that are not listed in CORS_ORIGINS get no CORS permission."""
    response = client.options(
        "/api/health",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_startup_creates_data_directories(client: TestClient, test_settings: Settings) -> None:
    """The lifespan hook creates the data, Chroma and PDF-cache folders."""
    assert test_settings.data_dir.is_dir()
    assert test_settings.chroma_dir.is_dir()
    assert test_settings.pdf_cache_dir.is_dir()
