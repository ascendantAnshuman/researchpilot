"""Tests for settings parsing, derived values and safety validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import BACKEND_DIR, Settings


def make_settings(**overrides: Any) -> Settings:
    """Build settings without reading any ``.env`` file."""
    return Settings(_env_file=None, **overrides)


def test_defaults_match_project_spec() -> None:
    """Defaults follow the project brief: 800/100 chunks, top 10 papers, MiniLM, critic on."""
    settings = make_settings()

    assert settings.chunk_size_tokens == 800
    assert settings.chunk_overlap_tokens == 100
    assert settings.top_n_papers == 10
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.enable_critic is True
    assert settings.max_revisions == 1


def test_relative_data_dir_resolves_against_backend_folder() -> None:
    """A relative DATA_DIR is anchored at backend/, whatever the current directory is."""
    assert make_settings(data_dir="mydata").data_dir == (BACKEND_DIR / "mydata").resolve()


def test_absolute_data_dir_is_kept(tmp_path: Path) -> None:
    """An absolute DATA_DIR is used unchanged."""
    assert make_settings(data_dir=tmp_path).data_dir == tmp_path


def test_blank_optional_values_become_none() -> None:
    """Empty ``KEY=`` lines count as unset rather than as empty strings."""
    settings = make_settings(
        anthropic_api_key="",
        database_url="   ",
        llm_input_cost_per_mtok="",
        semantic_scholar_api_key="",
    )

    assert settings.anthropic_api_key is None
    assert settings.database_url is None
    assert settings.llm_input_cost_per_mtok is None
    assert settings.semantic_scholar_api_key is None
    assert settings.llm_configured is False


def test_cors_origins_are_split_and_trimmed() -> None:
    """Comma-separated origins become a clean list with no trailing slashes or blanks."""
    settings = make_settings(cors_origins="http://a.com, https://b.app/ ,")

    assert settings.cors_origin_list == ["http://a.com", "https://b.app"]


def test_api_prefix_is_normalised() -> None:
    """``api/`` and ``/api/`` both become ``/api``."""
    assert make_settings(api_prefix="api/").api_prefix == "/api"
    assert make_settings(api_prefix="/api/").api_prefix == "/api"


def test_default_database_is_sqlite_inside_data_dir(tmp_path: Path) -> None:
    """Without DATABASE_URL the SQLite file lives in DATA_DIR."""
    settings = make_settings(data_dir=tmp_path)

    assert settings.sqlalchemy_database_url == f"sqlite:///{(tmp_path / 'researchpilot.db').as_posix()}"


def test_explicit_database_url_wins(tmp_path: Path) -> None:
    """DATABASE_URL overrides the default SQLite location."""
    settings = make_settings(data_dir=tmp_path, database_url="sqlite:///./other.db")

    assert settings.sqlalchemy_database_url == "sqlite:///./other.db"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    """Overlap >= chunk size would make the chunker loop forever, so it is rejected."""
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make_settings(chunk_size_tokens=200, chunk_overlap_tokens=200)


def test_rank_weights_cannot_all_be_zero() -> None:
    """Ranking needs at least one positive weight."""
    with pytest.raises(ValidationError, match="RANK_WEIGHT"):
        make_settings(rank_weight_relevance=0, rank_weight_citations=0, rank_weight_recency=0)


def test_production_requires_jwt_secret() -> None:
    """Production refuses to start without a JWT secret."""
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        make_settings(environment="production", jwt_secret_key=None)


def test_production_rejects_short_jwt_secret() -> None:
    """Production refuses a guessable, short JWT secret."""
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        make_settings(environment="production", jwt_secret_key="short")


def test_production_accepts_long_jwt_secret() -> None:
    """A long enough secret passes the production check."""
    settings = make_settings(environment="production", jwt_secret_key="x" * 48)

    assert settings.is_production is True
    assert settings.jwt_secret_is_ephemeral is False
    assert settings.jwt_secret == "x" * 48


def test_development_generates_stable_ephemeral_secret() -> None:
    """In development a random secret is generated once and reused for the process."""
    settings = make_settings(environment="development", jwt_secret_key=None)

    assert settings.jwt_secret_is_ephemeral is True
    first = settings.jwt_secret
    assert len(first) >= 32
    assert settings.jwt_secret == first


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values come from environment variables, including booleans and numbers."""
    monkeypatch.setenv("TOP_N_PAPERS", "7")
    monkeypatch.setenv("ENABLE_CRITIC", "false")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")

    settings = make_settings()

    assert settings.top_n_papers == 7
    assert settings.enable_critic is False
    assert settings.llm_model == "claude-sonnet-5"


def test_invalid_effort_is_rejected() -> None:
    """LLM_EFFORT only accepts the levels the Claude API understands."""
    with pytest.raises(ValidationError):
        make_settings(llm_effort="extreme")


def test_ensure_data_dirs_creates_all_folders(tmp_path: Path) -> None:
    """ensure_data_dirs builds the data, chroma and pdf folders."""
    settings = make_settings(data_dir=tmp_path / "nested" / "data")

    settings.ensure_data_dirs()

    assert settings.chroma_dir.is_dir()
    assert settings.pdf_cache_dir.is_dir()
