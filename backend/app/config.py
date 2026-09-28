"""Typed application settings for ResearchPilot.

Every tunable value (API keys, model name, limits, file locations) is read from
environment variables or a ``.env`` file via *pydantic-settings*, so no secret or
magic number needs to be hard-coded anywhere else in the codebase.

Precedence (highest wins):
    1. Keyword arguments passed to ``Settings(...)`` (used by tests).
    2. Real environment variables (set by your shell, Docker or Render).
    3. ``backend/.env``, then ``<project-root>/.env``.
    4. The defaults declared on :class:`Settings`.

Environment variable names are the upper-cased field names, e.g. the field
``llm_model`` is configured with ``LLM_MODEL=...``.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Path = BACKEND_DIR.parent

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

MIN_PRODUCTION_SECRET_LENGTH = 32

_OPTIONAL_FIELDS: tuple[str, ...] = (
    "anthropic_api_key",
    "semantic_scholar_api_key",
    "jwt_secret_key",
    "database_url",
    "llm_input_cost_per_mtok",
    "llm_output_cost_per_mtok",
)


class Settings(BaseSettings):
    """All runtime configuration for the API, the agents and the evaluation harness."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ---------- Application ----------
    app_name: str = "ResearchPilot"
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000"

    # ---------- Storage ----------
    data_dir: Path = Path("data")
    database_url: str | None = None

    # ---------- Auth ----------
    jwt_secret_key: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=1440, ge=1)
    bcrypt_rounds: int = Field(default=12, ge=4, le=16)
    allow_registration: bool = True

    # ---------- LLM (Anthropic Claude) ----------
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5"
    llm_effort: Effort = "high"
    llm_max_tokens: int = Field(default=32000, ge=256, le=128000)
    llm_timeout_seconds: float = Field(default=300.0, gt=0)
    llm_max_retries: int = Field(default=5, ge=0, le=10)
    llm_max_concurrency: int = Field(default=3, ge=1, le=16)
    llm_refusal_fallback: bool = True
    llm_input_cost_per_mtok: float | None = Field(default=None, ge=0)
    llm_output_cost_per_mtok: float | None = Field(default=None, ge=0)

    # ---------- Paper search ----------
    semantic_scholar_api_key: SecretStr | None = None
    search_results_per_query: int = Field(default=10, ge=1, le=100)
    top_n_papers: int = Field(default=10, ge=1, le=50)
    rank_weight_relevance: float = Field(default=0.5, ge=0)
    rank_weight_citations: float = Field(default=0.3, ge=0)
    rank_weight_recency: float = Field(default=0.2, ge=0)
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=4, ge=0, le=10)
    http_user_agent: str = "ResearchPilot/0.1 (academic research assistant)"

    # ---------- Reader / retrieval ----------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = Field(default=800, ge=100)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    max_pdf_pages: int = Field(default=40, ge=1)
    max_pdf_mb: float = Field(default=25.0, gt=0)
    reader_concurrency: int = Field(default=3, ge=1, le=16)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)

    # ---------- Pipeline ----------
    enable_critic: bool = True
    max_revisions: int = Field(default=1, ge=0, le=3)
    max_evidence_chunks: int = Field(default=24, ge=4, le=80)
    max_claims: int = Field(default=40, ge=5, le=200)
    unsupported_claim_policy: Literal["remove", "flag"] = "remove"

    # ---------- API ----------
    max_concurrent_runs: int = Field(default=2, ge=1, le=10)
    max_active_runs_per_user: int = Field(default=2, ge=1, le=10)
    chat_history_turns: int = Field(default=6, ge=0, le=20)

    _ephemeral_jwt_secret: str | None = PrivateAttr(default=None)

    # ---------- Validators ----------

    @field_validator(*_OPTIONAL_FIELDS, mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Treat empty ``KEY=`` lines in ``.env`` as "not set" instead of an empty string."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("data_dir", mode="after")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        """Resolve a relative data directory against ``backend/`` rather than the CWD."""
        path = value.expanduser()
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @field_validator("api_prefix", mode="after")
    @classmethod
    def _normalise_api_prefix(cls, value: str) -> str:
        """Ensure the API prefix looks like ``/api`` (leading slash, no trailing slash)."""
        cleaned = value.strip().strip("/")
        return f"/{cleaned}" if cleaned else ""

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Reject combinations of values that would break the pipeline or be insecure."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        weights = (self.rank_weight_relevance, self.rank_weight_citations, self.rank_weight_recency)
        if sum(weights) <= 0:
            raise ValueError("At least one RANK_WEIGHT_* value must be greater than zero")
        if self.environment == "production":
            secret = self.jwt_secret_key.get_secret_value() if self.jwt_secret_key else ""
            if len(secret) < MIN_PRODUCTION_SECRET_LENGTH:
                raise ValueError(
                    "JWT_SECRET_KEY must be set to at least "
                    f"{MIN_PRODUCTION_SECRET_LENGTH} characters when ENVIRONMENT=production"
                )
        return self

    # ---------- Derived values ----------

    @property
    def is_production(self) -> bool:
        """Whether the app runs with production safety checks."""
        return self.environment == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        """``CORS_ORIGINS`` split on commas, trimmed, without trailing slashes."""
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlalchemy_database_url(self) -> str:
        """The SQLAlchemy URL: ``DATABASE_URL`` if given, else SQLite inside ``DATA_DIR``."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'researchpilot.db').as_posix()}"

    @property
    def chroma_dir(self) -> Path:
        """Directory holding the persistent ChromaDB index."""
        return self.data_dir / "chroma"

    @property
    def pdf_cache_dir(self) -> Path:
        """Directory where downloaded paper PDFs are cached."""
        return self.data_dir / "pdfs"

    @property
    def llm_configured(self) -> bool:
        """Whether an Anthropic API key is present."""
        return self.anthropic_api_key is not None and bool(self.anthropic_api_key.get_secret_value().strip())

    @property
    def jwt_secret_is_ephemeral(self) -> bool:
        """True when no ``JWT_SECRET_KEY`` is configured and a random one is used instead."""
        return self.jwt_secret_key is None

    @property
    def jwt_secret(self) -> str:
        """The JWT signing secret.

        Falls back to a random per-process secret in development so the app starts
        without configuration; tokens then stop working after a restart.
        """
        if self.jwt_secret_key is not None:
            return self.jwt_secret_key.get_secret_value()
        if self._ephemeral_jwt_secret is None:
            self._ephemeral_jwt_secret = secrets.token_urlsafe(48)
        return self._ephemeral_jwt_secret

    def ensure_data_dirs(self) -> None:
        """Create the data, ChromaDB and PDF cache directories if they are missing."""
        for directory in (self.data_dir, self.chroma_dir, self.pdf_cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (created once, then cached).

    When ``ENVIRONMENT=test`` is set in the real environment, ``.env`` files are
    ignored so the test-suite never picks up a developer's real keys or paths.
    """
    if os.getenv("ENVIRONMENT", "").strip().lower() == "test":
        return Settings(_env_file=None)
    return Settings()
