"""Payment and invitation code routes."""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import InvitationCode, Payment, ScrapeSession, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payments", tags=["payments"])


class RedeemCodeRequest(BaseModel):
    code: str


class MockPaymentRequest(BaseModel):
    card_last_four: str = "4242"
    amount_usd: float = 5.0


@router.post("/redeem-code")
async def redeem_code(
    req: RedeemCodeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Redeem an invitation code for free scraping access."""
    result = await db.execute(
        select(InvitationCode).where(
            InvitationCode.code == req.code,
            InvitationCode.is_active == True,
        )
    )
    code = result.scalar_one_or_none()

    if not code:
        raise HTTPException(status_code=400, detail="Codigo de invitacion invalido o inactivo.")

    if code.current_uses >= code.max_uses:
        raise HTTPException(status_code=400, detail="Este codigo ya alcanzo su limite de usos.")

    if code.expires_at and code.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Este codigo ha expirado.")

    # Increment usage
    code.current_uses += 1

    # Create payment record
    payment = Payment(
        user_id=user.id,
        amount_usd=0.0,
        payment_method="invitation_code",
        invitation_code_id=code.id,
        status="completed",
        max_days_back=90,
        created_at=datetime.utcnow(),
    )
    db.add(payment)
    await db.flush()

    return {
        "status": "ok",
        "payment_id": str(payment.id),
        "max_days_back": payment.max_days_back,
        "message": "Codigo canjeado exitosamente. Podes scrapear hasta 3 meses de datos.",
    }


@router.post("/mock-pay")
async def mock_payment(
    req: MockPaymentRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Process a mock payment of $5 USD."""
    # Mock: always succeeds
    payment = Payment(
        user_id=user.id,
        amount_usd=req.amount_usd,
        payment_method="mock_payment",
        status="completed",
        max_days_back=90,
        created_at=datetime.utcnow(),
    )
    db.add(payment)
    await db.flush()

    return {
        "status": "ok",
        "payment_id": str(payment.id),
        "max_days_back": payment.max_days_back,
        "message": f"Pago de USD {req.amount_usd:.2f} procesado (mock). Podes scrapear hasta 3 meses de datos.",
    }


@router.get("/history")
async def payment_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all payments for the current user."""
    result = await db.execute(
        select(Payment)
        .where(Payment.user_id == user.id)
        .order_by(Payment.created_at.desc())
    )
    payments = result.scalars().all()

    return {
        "status": "ok",
        "payments": [
            {
                "id": str(p.id),
                "amount_usd": p.amount_usd,
                "payment_method": p.payment_method,
                "status": p.status,
                "max_days_back": p.max_days_back,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in payments
        ],
    }


@router.get("/active")
async def active_payment(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Find the most recent completed payment without an associated scrape session."""
    # Get all completed payments for user
    result = await db.execute(
        select(Payment)
        .where(
            Payment.user_id == user.id,
            Payment.status == "completed",
        )
        .order_by(Payment.created_at.desc())
    )
    payments = result.scalars().all()

    # Find one that doesn't have a scrape session yet
    for payment in payments:
        session_result = await db.execute(
            select(ScrapeSession).where(ScrapeSession.payment_id == payment.id)
        )
        if not session_result.scalar_one_or_none():
            return {
                "status": "ok",
                "has_active_payment": True,
                "payment_id": str(payment.id),
                "max_days_back": payment.max_days_back,
            }

    return {"status": "ok", "has_active_payment": False}
