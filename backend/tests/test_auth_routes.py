"""API tests for /api/auth: register, login, OAuth2 token and /me."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.auth import create_access_token
from app.config import Settings
from app.db import session_scope
from app.models import User
from tests.conftest import TEST_PASSWORD

RegisterUser = Callable[..., dict[str, Any]]


def bearer(token: str) -> dict[str, str]:
    """Build an Authorization header."""
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def test_register_returns_token_and_user(client: TestClient, register_user: RegisterUser) -> None:
    """Registering logs the user in straight away."""
    body = register_user()

    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["user"]["email"] == "student@university.edu"
    assert body["user"]["full_name"] == "Test Student"
    assert client.get("/api/auth/me", headers=bearer(body["access_token"])).status_code == 200


def test_register_never_returns_password_material(register_user: RegisterUser) -> None:
    """Neither the password nor its hash appears in the response."""
    body = register_user()

    assert "password" not in str(body).lower()
    assert TEST_PASSWORD not in str(body)


def test_register_stores_a_hash_not_the_password(register_user: RegisterUser) -> None:
    """The database holds a bcrypt hash."""
    register_user()

    with session_scope() as db:
        user = db.scalar(select(User).where(User.email == "student@university.edu"))
        assert user is not None
        assert user.hashed_password.startswith("$2b$")
        assert user.hashed_password != TEST_PASSWORD


def test_register_normalises_email(register_user: RegisterUser) -> None:
    """Emails are stored lower-case."""
    body = register_user(email="  Student.Name@University.EDU ")

    assert body["user"]["email"] == "student.name@university.edu"


def test_register_rejects_duplicate_email_case_insensitively(
    client: TestClient, register_user: RegisterUser
) -> None:
    """The same address with different capitalisation is still a duplicate."""
    register_user(email="student@university.edu")

    response = client.post(
        "/api/auth/register",
        json={"email": "STUDENT@university.edu", "password": TEST_PASSWORD},
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.parametrize(
    ("password", "reason"),
    [
        ("short1", "too short"),
        ("onlyletters", "no digit"),
        ("1234567890", "no letter"),
        ("a1" + "x" * 80, "longer than bcrypt's 72-byte limit"),
    ],
)
def test_register_rejects_weak_passwords(client: TestClient, password: str, reason: str) -> None:
    """Weak or over-long passwords get a 422 validation error."""
    response = client.post("/api/auth/register", json={"email": "weak@university.edu", "password": password})

    assert response.status_code == 422, reason


def test_register_rejects_invalid_email(client: TestClient) -> None:
    """Malformed emails get a 422 validation error."""
    response = client.post("/api/auth/register", json={"email": "not-an-email", "password": TEST_PASSWORD})

    assert response.status_code == 422


def test_registration_can_be_closed(test_settings: Settings) -> None:
    """ALLOW_REGISTRATION=false blocks new accounts (protects a public deployment's API budget)."""
    from app.main import create_app

    closed = test_settings.model_copy(update={"allow_registration": False})
    with TestClient(create_app(closed)) as closed_client:
        response = closed_client.post(
            "/api/auth/register", json={"email": "late@university.edu", "password": TEST_PASSWORD}
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Registration is closed on this server"


def test_register_without_full_name(client: TestClient) -> None:
    """The name is optional."""
    response = client.post("/api/auth/register", json={"email": "anon@university.edu", "password": TEST_PASSWORD})

    assert response.status_code == 201
    assert response.json()["user"]["full_name"] is None


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def test_login_with_correct_credentials(client: TestClient, register_user: RegisterUser) -> None:
    """A correct email/password pair returns a working token."""
    register_user()

    response = client.post(
        "/api/auth/login", json={"email": "Student@University.edu", "password": TEST_PASSWORD}
    )

    assert response.status_code == 200
    token = response.json()["access_token"]
    assert client.get("/api/auth/me", headers=bearer(token)).json()["email"] == "student@university.edu"


def test_login_wrong_password_and_unknown_email_look_identical(
    client: TestClient, register_user: RegisterUser
) -> None:
    """Both failures give the same 401 message, so attackers cannot probe which emails exist."""
    register_user()

    wrong_password = client.post(
        "/api/auth/login", json={"email": "student@university.edu", "password": "WrongPassword9"}
    )
    unknown_email = client.post(
        "/api/auth/login", json={"email": "ghost@university.edu", "password": TEST_PASSWORD}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json() == {"detail": "Incorrect email or password"}
    assert wrong_password.headers["www-authenticate"] == "Bearer"


def test_oauth2_token_endpoint_accepts_form_data(client: TestClient, register_user: RegisterUser) -> None:
    """The form-based endpoint used by Swagger's Authorize button works."""
    register_user()

    response = client.post(
        "/api/auth/token", data={"username": "student@university.edu", "password": TEST_PASSWORD}
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_disabled_account_cannot_log_in(client: TestClient, register_user: RegisterUser) -> None:
    """Deactivated users get 403 on login and on /me."""
    token = register_user()["access_token"]
    with session_scope() as db:
        user = db.scalar(select(User).where(User.email == "student@university.edu"))
        assert user is not None
        user.is_active = False

    login = client.post("/api/auth/login", json={"email": "student@university.edu", "password": TEST_PASSWORD})
    me = client.get("/api/auth/me", headers=bearer(token))

    assert login.status_code == 403
    assert me.status_code == 403


# ---------------------------------------------------------------------------
# /me and token validation
# ---------------------------------------------------------------------------


def test_me_requires_a_token(client: TestClient) -> None:
    """No token means 401 with a Bearer challenge."""
    response = client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_rejects_garbage_token(client: TestClient) -> None:
    """A random string is not a valid token."""
    response = client.get("/api/auth/me", headers=bearer("definitely.not.a-jwt"))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_me_rejects_expired_token(client: TestClient, register_user: RegisterUser, test_settings: Settings) -> None:
    """An expired token for a real user is refused."""
    user_id = register_user()["user"]["id"]
    expired, _ = create_access_token(user_id, test_settings, expires_delta=timedelta(seconds=-1))

    response = client.get("/api/auth/me", headers=bearer(expired))

    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"


def test_token_of_deleted_user_is_rejected(client: TestClient, register_user: RegisterUser) -> None:
    """A valid token stops working once its user is deleted."""
    token = register_user()["access_token"]
    with session_scope() as db:
        user = db.scalar(select(User).where(User.email == "student@university.edu"))
        assert user is not None
        db.delete(user)

    response = client.get("/api/auth/me", headers=bearer(token))

    assert response.status_code == 401
    assert response.json()["detail"] == "User no longer exists"


def test_me_returns_the_owner_of_the_token(client: TestClient, auth_headers: dict[str, str]) -> None:
    """/me identifies the user behind the token."""
    body = client.get("/api/auth/me", headers=auth_headers).json()

    assert body["email"] == "student@university.edu"
    assert set(body) == {"id", "email", "full_name", "created_at"}
    assert body["created_at"].endswith(("Z", "+00:00"))
