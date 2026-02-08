"""Authentication routes: register, login (magic link), verify, logout."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    create_jwt_token,
    create_magic_link_token,
    get_current_user,
    verify_magic_link_token,
)
from app.database import get_db
from app.email_service import send_magic_link
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: str = ""
    linkedin_public_id: str = ""
    linkedin_profile_url: str = ""


class LoginRequest(BaseModel):
    email: EmailStr


@router.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user. Returns JWT token."""
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == req.email))
    existing = result.scalar_one_or_none()

    if existing:
        # User exists -> send magic link instead
        token = create_magic_link_token(req.email)
        await send_magic_link(req.email, token)
        return {
            "status": "existing_user",
            "message": "Ya tenes cuenta. Te enviamos un enlace de acceso a tu email.",
        }

    user = User(
        email=req.email,
        first_name=req.first_name,
        last_name=req.last_name,
        phone=req.phone,
        linkedin_public_id=req.linkedin_public_id,
        linkedin_profile_url=req.linkedin_profile_url,
        created_at=datetime.utcnow(),
        last_login_at=datetime.utcnow(),
    )
    db.add(user)
    await db.flush()

    jwt_token = create_jwt_token(user.id, user.email)

    return {
        "status": "ok",
        "token": jwt_token,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Send a magic link to the user's email."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user:
        return {"status": "error", "message": "No encontramos una cuenta con ese email."}

    token = create_magic_link_token(req.email)
    await send_magic_link(req.email, token)

    return {"status": "ok", "message": "Te enviamos un enlace de acceso a tu email."}


@router.get("/verify")
async def verify_magic_link(token: str, db: AsyncSession = Depends(get_db)):
    """Verify a magic link token and set session cookie."""
    email = verify_magic_link_token(token)
    if not email:
        return RedirectResponse(url="/login?error=invalid_token")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        return RedirectResponse(url="/login?error=user_not_found")

    user.last_login_at = datetime.utcnow()

    jwt_token = create_jwt_token(user.id, user.email)

    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key="session_token",
        value=jwt_token,
        httponly=True,
        max_age=7 * 24 * 3600,  # 7 days
        samesite="lax",
    )
    return response


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    """Return current authenticated user info."""
    return {
        "status": "ok",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone or "",
            "linkedin_public_id": user.linkedin_public_id or "",
            "linkedin_profile_url": user.linkedin_profile_url or "",
        },
    }


@router.post("/logout")
async def logout():
    """Clear session cookie."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("session_token")
    return response
