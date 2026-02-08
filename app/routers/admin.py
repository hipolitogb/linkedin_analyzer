"""Admin routes for managing invitation codes and viewing users."""

import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import InvitationCode, Payment, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin-dev-secret")


def _verify_admin(request: Request):
    """Verify admin secret from X-Admin-Secret header."""
    secret = request.headers.get("X-Admin-Secret", "")
    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")


class CreateCodeRequest(BaseModel):
    code: str
    name: str = ""
    max_uses: int = 1
    expires_at: str | None = None


@router.post("/codes")
async def create_code(
    req: CreateCodeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create an invitation code."""
    _verify_admin(request)

    expires = None
    if req.expires_at:
        expires = datetime.fromisoformat(req.expires_at)

    code = InvitationCode(
        code=req.code,
        name=req.name,
        max_uses=req.max_uses,
        expires_at=expires,
        created_at=datetime.utcnow(),
    )
    db.add(code)
    await db.flush()

    return {
        "status": "ok",
        "code_id": str(code.id),
        "code": code.code,
        "max_uses": code.max_uses,
    }


@router.get("/codes")
async def list_codes(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """List all invitation codes with usage stats."""
    _verify_admin(request)

    result = await db.execute(
        select(InvitationCode).order_by(InvitationCode.created_at.desc())
    )
    codes = result.scalars().all()

    return {
        "status": "ok",
        "codes": [
            {
                "id": str(c.id),
                "code": c.code,
                "name": c.name or "",
                "max_uses": c.max_uses,
                "current_uses": c.current_uses,
                "remaining": c.max_uses - c.current_uses,
                "is_active": c.is_active,
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in codes
        ],
    }


@router.delete("/codes/{code_id}")
async def deactivate_code(
    code_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Deactivate an invitation code."""
    _verify_admin(request)

    result = await db.execute(select(InvitationCode).where(InvitationCode.id == code_id))
    code = result.scalar_one_or_none()

    if not code:
        raise HTTPException(status_code=404, detail="Code not found")

    code.is_active = False
    return {"status": "ok", "message": f"Code '{code.code}' deactivated"}


@router.get("/users")
async def list_users(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """List all users with payment and scrape counts."""
    _verify_admin(request)

    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()

    user_list = []
    for u in users:
        # Count payments
        pay_result = await db.execute(
            select(func.count()).where(Payment.user_id == u.id)
        )
        payment_count = pay_result.scalar() or 0

        user_list.append({
            "id": str(u.id),
            "email": u.email,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "linkedin_public_id": u.linkedin_public_id or "",
            "payment_count": payment_count,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        })

    return {"status": "ok", "users": user_list}
