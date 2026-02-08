"""Authentication routes: register, login (magic link), verify, logout."""

import logging
from datetime import datetime

import httpx
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
from app.crypto_utils import decrypt_cookies
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


class CheckLinkedInRequest(BaseModel):
    linkedin_public_id: str


@router.post("/check-linkedin")
async def check_linkedin(req: CheckLinkedInRequest, db: AsyncSession = Depends(get_db)):
    """Check if a LinkedIn public_id already exists. If so, auto-login."""
    if not req.linkedin_public_id:
        return {"status": "new_user"}

    result = await db.execute(
        select(User).where(User.linkedin_public_id == req.linkedin_public_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        return {"status": "new_user"}

    user.last_login_at = datetime.utcnow()
    jwt_token = create_jwt_token(user.id, user.email)

    return {
        "status": "existing_user",
        "token": jwt_token,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }


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


class SaveCookiesRequest(BaseModel):
    encrypted_cookies: str


@router.post("/save-linkedin-cookies")
async def save_linkedin_cookies(
    req: SaveCookiesRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save encrypted LinkedIn cookies to the user record for reuse."""
    user.encrypted_linkedin_cookies = req.encrypted_cookies
    user.linkedin_cookies_updated_at = datetime.utcnow()
    return {"status": "ok"}


@router.get("/check-stored-cookies")
async def check_stored_cookies(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check if the user has stored LinkedIn cookies and if they're still valid."""
    if not user.encrypted_linkedin_cookies:
        return {"status": "no_cookies"}

    try:
        cookies = decrypt_cookies(user.encrypted_linkedin_cookies)
    except Exception:
        user.encrypted_linkedin_cookies = None
        return {"status": "no_cookies"}

    li_at = cookies.get("li_at", "")
    jsessionid = cookies.get("jsessionid", "")
    if not li_at:
        user.encrypted_linkedin_cookies = None
        return {"status": "no_cookies"}

    # Validate against LinkedIn API
    csrf_value = jsessionid.strip('"') if jsessionid else "ajax:0"
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "csrf-token": csrf_value,
        "x-restli-protocol-version": "2.0.0",
    }
    cookies_dict = {"li_at": li_at, "JSESSIONID": jsessionid or '"ajax:0"'}

    try:
        async with httpx.AsyncClient(
            cookies=cookies_dict, headers=headers, follow_redirects=False, timeout=10.0
        ) as client:
            resp = await client.get("https://www.linkedin.com/voyager/api/me")
            if resp.status_code == 200:
                return {
                    "status": "valid",
                    "encrypted_cookies": user.encrypted_linkedin_cookies,
                }
    except Exception:
        pass

    # Cookies expired or invalid
    user.encrypted_linkedin_cookies = None
    user.linkedin_cookies_updated_at = None
    return {"status": "expired"}


@router.post("/logout")
async def logout():
    """Clear session cookie."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("session_token")
    return response
