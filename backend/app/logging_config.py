"""Central logging configuration.

Call :func:`setup_logging` once at process start; the FastAPI app factory and the
CLI entry points do this. Every other module simply uses::

    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
HANDLER_NAME = "researchpilot"

NOISY_LOGGERS: tuple[str, ...] = (
    "anthropic",
    "chromadb",
    "google_genai",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "multipart",
    "passlib",
    "sentence_transformers",
    "transformers",
    "urllib3",
)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with one consistently formatted stderr handler.

    Safe to call repeatedly: the handler is added only once and later calls just
    update the level. Chatty third-party libraries are capped at WARNING unless
    ``level`` is DEBUG.

    Args:
        level: A standard logging level name such as ``"INFO"`` or ``"DEBUG"``.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not any(handler.get_name() == HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(handler)

    library_level = logging.DEBUG if numeric_level <= logging.DEBUG else logging.WARNING
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(library_level)
