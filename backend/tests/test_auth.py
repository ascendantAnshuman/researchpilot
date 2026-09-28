"""Unit tests for password hashing, JWT handling and credential checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt
from sqlalchemy.orm import Session

from app.auth import (
    TokenError,
    authenticate_user,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.config import Settings
from app.models import User


def test_hash_password_is_salted_bcrypt() -> None:
    """Hashes are bcrypt, never the plain password, and differ for the same input."""
    first = hash_password("CorrectHorse42")
    second = hash_password("CorrectHorse42")

    assert first.startswith("$2b$")
    assert "CorrectHorse42" not in first
    assert first != second


def test_verify_password_accepts_right_and_rejects_wrong() -> None:
    """Only the original password verifies."""
    hashed = hash_password("CorrectHorse42")

    assert verify_password("CorrectHorse42", hashed) is True
    assert verify_password("correcthorse42", hashed) is False


def test_verify_password_with_malformed_hash_returns_false() -> None:
    """A corrupted stored hash fails closed instead of crashing."""
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_access_token_round_trip(test_settings: Settings) -> None:
    """A token decodes back to the user id it was issued for."""
    token, expires_in = create_access_token(42, test_settings)

    assert decode_access_token(token, test_settings) == "42"
    assert expires_in == test_settings.access_token_expire_minutes * 60


def test_expired_token_is_rejected(test_settings: Settings) -> None:
    """Tokens past their ``exp`` are refused."""
    token, _ = create_access_token(1, test_settings, expires_delta=timedelta(seconds=-5))

    with pytest.raises(TokenError, match="expired"):
        decode_access_token(token, test_settings)


def test_token_signed_with_another_secret_is_rejected(test_settings: Settings) -> None:
    """A token forged with a different secret does not verify."""
    forged = jwt.encode(
        {"sub": "1", "type": "access", "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())},
        "some-other-secret-key-that-is-long-enough-000",
        algorithm="HS256",
    )

    with pytest.raises(TokenError, match="Invalid token"):
        decode_access_token(forged, test_settings)


def test_tampered_token_is_rejected(test_settings: Settings) -> None:
    """Changing a character in the middle of the signature breaks verification."""
    token, _ = create_access_token(1, test_settings)
    position = len(token) - 10  # well inside the signature, where every bit is significant
    replacement = "A" if token[position] != "A" else "B"
    tampered = token[:position] + replacement + token[position + 1 :]

    with pytest.raises(TokenError):
        decode_access_token(tampered, test_settings)


def test_token_of_wrong_type_is_rejected(test_settings: Settings) -> None:
    """Only ``type=access`` tokens are accepted as access tokens."""
    token = jwt.encode(
        {"sub": "1", "type": "refresh", "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())},
        test_settings.jwt_secret,
        algorithm=test_settings.jwt_algorithm,
    )

    with pytest.raises(TokenError, match="type"):
        decode_access_token(token, test_settings)


def test_token_without_subject_is_rejected(test_settings: Settings) -> None:
    """A token must say which user it belongs to."""
    token = jwt.encode(
        {"type": "access", "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())},
        test_settings.jwt_secret,
        algorithm=test_settings.jwt_algorithm,
    )

    with pytest.raises(TokenError, match="subject"):
        decode_access_token(token, test_settings)


def test_authenticate_user(db_session: Session) -> None:
    """Correct credentials return the user; wrong or unknown ones return None."""
    db_session.add(User(email="ada@university.edu", hashed_password=hash_password("Analytical1843")))
    db_session.commit()

    user = authenticate_user(db_session, "ada@university.edu", "Analytical1843")
    assert user is not None and user.email == "ada@university.edu"
    assert authenticate_user(db_session, "  ADA@University.EDU ", "Analytical1843") is not None
    assert authenticate_user(db_session, "ada@university.edu", "wrong-password1") is None
    assert authenticate_user(db_session, "nobody@university.edu", "Analytical1843") is None
