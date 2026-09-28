"""Authentication routes: register, log in (JSON or OAuth2 form) and current user."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user_by_email,
    hash_password,
)
from app.config import Settings, get_settings
from app.db import get_db
from app.models import User
from app.schemas import LoginRequest, TokenResponse, UserCreate, UserRead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_CREDENTIALS = "Incorrect email or password"


def _token_response(user: User, settings: Settings) -> TokenResponse:
    """Issue a fresh access token for ``user``."""
    token, expires_in = create_access_token(user.id, settings)
    return TokenResponse(access_token=token, expires_in=expires_in, user=UserRead.model_validate(user))


def _login(db: Session, settings: Settings, email: str, password: str) -> TokenResponse:
    """Shared login logic for the JSON and OAuth2-form endpoints."""
    user = authenticate_user(db, email, password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_CREDENTIALS,
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    logger.info("User id=%s logged in", user.id)
    return _token_response(user, settings)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account and receive an access token",
)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Register a new user. Returns a token straight away so the user is logged in."""
    if not settings.allow_registration:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Registration is closed on this server")
    if get_user_by_email(db, payload.email) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")

    user = User(email=payload.email, hashed_password=hash_password(payload.password), full_name=payload.full_name)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        # Two simultaneous registrations with the same email: the unique index wins.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists"
        ) from exc
    db.refresh(user)
    logger.info("Registered user id=%s", user.id)
    return _token_response(user, settings)


@router.post("/login", response_model=TokenResponse, summary="Log in with email and password (JSON)")
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Exchange email + password for an access token. Used by the frontend."""
    return _login(db, settings, payload.email, payload.password)


@router.post("/token", response_model=TokenResponse, summary="OAuth2 password flow (Swagger 'Authorize' button)")
def token(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Same as ``/login`` but takes form fields, as the OAuth2 standard requires.

    In the Swagger UI (``/docs``) click **Authorize** and enter your email in the
    *username* field.
    """
    return _login(db, settings, form.username, form.password)


@router.get("/me", response_model=UserRead, summary="The currently logged-in user")
def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the account that owns the bearer token."""
    return current_user
