"""Multi-tenant scraping routes with payment validation and date enforcement."""

import logging
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.crypto_utils import decrypt_cookies
from app.database import async_session_factory, get_db
from app.models import Payment, ScrapeSession, User
from app.services.scrape_service import run_scrape

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scrape", tags=["scraping"])

MAX_DAYS_RANGE = 93  # ~3 months
MAX_LOOKBACK_DAYS = 365


class StartScrapeRequest(BaseModel):
    payment_id: str
    from_date: str  # YYYY-MM-DD
    to_date: str  # YYYY-MM-DD
    encrypted_cookies: str
    public_id: str


async def _run_scrape_background(
    scrape_session_id: UUID,
    user_id: UUID,
    li_at: str,
    jsessionid: str,
    public_id: str,
    from_date: str,
    to_date: str,
):
    """Background task to run scrape with its own DB session."""
    async with async_session_factory() as db:
        try:
            await run_scrape(
                db=db,
                scrape_session_id=scrape_session_id,
                user_id=user_id,
                li_at=li_at,
                jsessionid=jsessionid,
                public_id=public_id,
                from_date=from_date,
                to_date=to_date,
            )
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Background scrape failed: {e}", exc_info=True)


@router.post("/start")
async def start_scrape(
    req: StartScrapeRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a scrape session. Validates payment, enforces date limits."""
    # Validate payment
    payment = await db.get(Payment, req.payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Pago no encontrado.")
    if str(payment.user_id) != str(user.id):
        raise HTTPException(status_code=403, detail="Este pago no te pertenece.")
    if payment.status != "completed":
        raise HTTPException(status_code=400, detail="El pago no esta completado.")

    # Check payment not already used
    existing_session = await db.execute(
        select(ScrapeSession).where(ScrapeSession.payment_id == payment.id)
    )
    if existing_session.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Este pago ya fue utilizado para un scrape.")

    # Parse and validate dates
    try:
        from_dt = datetime.strptime(req.from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(req.to_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha invalido. Usa YYYY-MM-DD.")

    if from_dt > to_dt:
        raise HTTPException(status_code=400, detail="La fecha 'desde' debe ser anterior a 'hasta'.")

    # Enforce 3-month limit
    delta = (to_dt - from_dt).days
    if delta > MAX_DAYS_RANGE:
        raise HTTPException(
            status_code=400,
            detail=f"Rango maximo de {MAX_DAYS_RANGE} dias (~3 meses). Seleccionaste {delta} dias.",
        )

    # Enforce 365-day lookback
    now = datetime.utcnow()
    lookback = (now - from_dt).days
    if lookback > MAX_LOOKBACK_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Solo podes ir hasta {MAX_LOOKBACK_DAYS} dias atras desde hoy.",
        )

    # Decrypt cookies
    try:
        cookies = decrypt_cookies(req.encrypted_cookies)
    except Exception:
        raise HTTPException(status_code=400, detail="Cookies invalidas o expiradas. Reconecta LinkedIn.")

    li_at = cookies.get("li_at", "")
    jsessionid = cookies.get("jsessionid", "")
    if not li_at:
        raise HTTPException(status_code=400, detail="Cookie li_at no encontrada.")

    # Create scrape session
    scrape_session = ScrapeSession(
        user_id=user.id,
        payment_id=payment.id,
        from_date=from_dt,
        to_date=to_dt,
        status="pending",
        created_at=datetime.utcnow(),
    )
    db.add(scrape_session)
    await db.flush()

    # Update user's linkedin_public_id if not set
    if not user.linkedin_public_id:
        user.linkedin_public_id = req.public_id

    # Launch background scrape
    background_tasks.add_task(
        _run_scrape_background,
        scrape_session.id,
        user.id,
        li_at,
        jsessionid,
        req.public_id,
        req.from_date,
        req.to_date,
    )

    return {
        "status": "ok",
        "scrape_session_id": str(scrape_session.id),
        "message": "Scraping iniciado. Podes consultar el estado con /api/scrape/status.",
    }


@router.get("/status/{session_id}")
async def scrape_status(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the status of a scrape session."""
    session = await db.get(ScrapeSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesion de scrape no encontrada.")
    if str(session.user_id) != str(user.id):
        raise HTTPException(status_code=403, detail="No tenes acceso a esta sesion.")

    return {
        "status": "ok",
        "scrape_status": session.status,
        "posts_scraped": session.posts_scraped or 0,
        "error_message": session.error_message,
        "from_date": session.from_date.strftime("%Y-%m-%d") if session.from_date else None,
        "to_date": session.to_date.strftime("%Y-%m-%d") if session.to_date else None,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }
