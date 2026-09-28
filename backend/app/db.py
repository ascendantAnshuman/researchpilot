"""Database engine, session management and table creation (SQLAlchemy 2.x).

The engine is created from ``Settings.sqlalchemy_database_url`` when the app starts
and can be re-initialised (tests point it at a temporary SQLite file).

* Request handlers get a session through the :func:`get_db` FastAPI dependency.
* Background jobs and CLI scripts use the :func:`session_scope` context manager,
  which commits on success and rolls back on error.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _configure_sqlite_connection(dbapi_connection: Any, _connection_record: Any) -> None:
    """Enable foreign keys, WAL journaling and a busy timeout on each SQLite connection.

    * ``foreign_keys`` makes ``ON DELETE CASCADE`` work (SQLite disables it by default).
    * ``WAL`` lets the API read while a background research run is writing.
    * ``busy_timeout`` waits up to 5 s for a lock instead of failing immediately.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_engine(database_url: str | None = None) -> Engine:
    """Create (or re-create) the global engine and session factory.

    Args:
        database_url: SQLAlchemy URL. Defaults to the configured database.

    Returns:
        The newly created engine.
    """
    global _engine, _session_factory
    dispose_engine()

    url = database_url or get_settings().sqlalchemy_database_url
    is_sqlite = url.startswith("sqlite")
    connect_args: dict[str, Any] = {"check_same_thread": False} if is_sqlite else {}

    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    if is_sqlite:
        event.listen(engine, "connect", _configure_sqlite_connection)

    _engine = engine
    _session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    logger.info("Database engine ready: %s", engine.url.render_as_string(hide_password=True))
    return engine


def get_engine() -> Engine:
    """Return the global engine, creating it from settings on first use."""
    if _engine is None:
        return init_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the global session factory, creating the engine on first use."""
    if _session_factory is None:
        init_engine()
    return cast(sessionmaker[Session], _session_factory)


def dispose_engine() -> None:
    """Close all pooled connections and forget the current engine."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def create_tables() -> None:
    """Create every table that does not exist yet (safe to call on each start-up)."""
    import app.models  # noqa: F401  (importing registers the models on Base.metadata)

    Base.metadata.create_all(get_engine())
    logger.info("Database tables ensured: %s", ", ".join(sorted(Base.metadata.tables)))


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a session and always closes it afterwards."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background jobs: commit on success, roll back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> bool:
    """Return True if a trivial ``SELECT 1`` succeeds against the database."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        return False
