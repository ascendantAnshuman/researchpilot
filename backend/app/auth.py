"""Authentication helpers: bcrypt password hashing, JWT access tokens and the
``get_current_user`` FastAPI dependency.

How it works:
    1. On register, the password is hashed with bcrypt (a slow, salted hash) and
       only the hash is stored.
    2. On login, the password is checked against the hash and, if correct, the
       server returns a signed JWT: ``{"sub": "<user id>", "exp": ..., "type": "access"}``.
    3. The browser sends ``Authorization: Bearer <token>`` on every request, and
       :func:`get_current_user` verifies the signature and expiry, then loads the user.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import User

logger = logging.getLogger(__name__)

ACCESS_TOKEN_TYPE = "access"

# auto_error=False so a missing token produces the same JSON 401 as a bad token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{get_settings().api_prefix}/auth/token", auto_error=False)


class TokenError(Exception):
    """Raised when a JWT is malformed, expired, wrongly signed or of the wrong type."""


@lru_cache(maxsize=4)
def _password_context(rounds: int) -> CryptContext:
    """Build (once per cost factor) the passlib context used for hashing."""
    return CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=rounds)


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt using the configured cost factor."""
    return _password_context(get_settings().bcrypt_rounds).hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Check a plain-text password against a stored bcrypt hash.

    Returns False (instead of raising) if the stored hash is malformed.
    """
    try:
        return _password_context(get_settings().bcrypt_rounds).verify(password, hashed_password)
    except (ValueError, TypeError):
        logger.warning("Stored password hash could not be parsed")
        return False


def create_access_token(
    subject: str | int,
    settings: Settings | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, int]:
    """Create a signed JWT access token.

    Args:
        subject: The user id stored in the ``sub`` claim.
        settings: Settings providing the secret and algorithm (defaults to global).
        expires_delta: Token lifetime (defaults to ``ACCESS_TOKEN_EXPIRE_MINUTES``).

    Returns:
        A ``(token, lifetime_in_seconds)`` tuple.
    """
    settings = settings or get_settings()
    lifetime = expires_delta if expires_delta is not None else timedelta(minutes=settings.access_token_expire_minutes)
    issued_at = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": str(subject),
        "type": ACCESS_TOKEN_TYPE,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + lifetime).timestamp()),
    }
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, max(int(lifetime.total_seconds()), 0)


def decode_access_token(token: str, settings: Settings | None = None) -> str:
    """Verify a JWT access token and return its subject (the user id as a string).

    Raises:
        TokenError: If the token is expired, tampered with, signed with another
            secret, not an access token, or has no subject.
    """
    settings = settings or get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except JWTError as exc:
        raise TokenError("Invalid token") from exc

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise TokenError("Invalid token type")
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise TokenError("Token has no subject")
    return subject


def normalise_email(email: str) -> str:
    """Lower-case and trim an email so lookups are case-insensitive."""
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> User | None:
    """Fetch a user by email (case-insensitive), or None if there is no such user."""
    return db.scalar(select(User).where(User.email == normalise_email(email)))


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Return the user if the email/password pair is correct, else None.

    For unknown emails a dummy hash check still runs, so response time does not
    reveal which email addresses have accounts.
    """
    user = get_user_by_email(db, email)
    if user is None:
        _password_context(get_settings().bcrypt_rounds).dummy_verify()
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def _unauthorized(detail: str) -> HTTPException:
    """A 401 response carrying the standard ``WWW-Authenticate: Bearer`` header."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    """FastAPI dependency: resolve the ``Authorization: Bearer`` token to an active user.

    Raises:
        HTTPException: 401 if the token is missing/invalid or the user no longer
            exists; 403 if the account is disabled.
    """
    if not token:
        raise _unauthorized("Not authenticated")
    try:
        subject = decode_access_token(token, settings)
    except TokenError as exc:
        raise _unauthorized(str(exc)) from exc

    user = db.get(User, int(subject)) if subject.isdigit() else None
    if user is None:
        raise _unauthorized("User no longer exists")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    return user
