"""Multi-tenant dashboard routes: metrics, analysis, posts from DB."""

import asyncio
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.analyzer import classify_posts, compute_metrics, deep_pattern_analysis
from app.auth import get_current_user
from app.database import get_db
from app.models import AnalysisResult, Post, ScrapeSession, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def db_posts_to_legacy_dicts(posts: list[Post]) -> list[dict]:
    """Convert Post ORM objects to the dict format expected by compute_metrics()."""
    return [
        {
            "text": p.text or "",
            "date": p.date,
            "content_type": p.content_type or "text",
            "reactions": p.reactions or 0,
            "comments": p.comments or 0,
            "shares": p.shares or 0,
            "engagement": p.engagement or 0,
            "impressions": p.impressions or 0,
            "has_image": p.has_image or False,
            "image_urls": p.image_urls or [],
            "text_length": p.text_length or 0,
            "url": p.url or "",
            "is_repost": p.is_repost or False,
            "original_author": p.original_author or "",
            "reshare_comment": p.reshare_comment or "",
            "category": p.category,
            "sentiment": p.sentiment,
            "topics": p.topics or [],
            "image_type": p.image_type,
            "activity_id": p.linkedin_activity_id or "",
        }
        for p in posts
    ]


async def _load_user_posts(
    db: AsyncSession, user_id, from_date: str | None = None, to_date: str | None = None
) -> list[Post]:
    """Load posts from DB for a user, optionally filtered by date."""
    query = select(Post).where(Post.user_id == user_id).order_by(Post.date.desc())

    if from_date:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        query = query.where(Post.date >= from_dt)
    if to_date:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        query = query.where(Post.date <= to_dt)

    result = await db.execute(query)
    return list(result.scalars().all())


class MetricsRequest(BaseModel):
    from_date: str | None = None
    to_date: str | None = None


class AnalyzeRequest(BaseModel):
    from_date: str | None = None
    to_date: str | None = None


@router.get("/posts")
async def get_posts(
    from_date: str | None = None,
    to_date: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's posts from DB, optionally filtered by date."""
    posts = await _load_user_posts(db, user.id, from_date, to_date)
    post_dicts = db_posts_to_legacy_dicts(posts)

    return {
        "status": "ok",
        "count": len(post_dicts),
        "posts": [
            {
                "text": p["text"][:200],
                "date": p["date"].isoformat() if isinstance(p["date"], datetime) else str(p["date"]),
                "content_type": p["content_type"],
                "engagement": p["engagement"],
                "reactions": p["reactions"],
                "comments": p["comments"],
                "impressions": p["impressions"],
            }
            for p in post_dicts
        ],
    }


@router.post("/metrics")
async def get_metrics(
    req: MetricsRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compute dashboard metrics instantly (no AI). Pure Python computation."""
    posts = await _load_user_posts(db, user.id, req.from_date, req.to_date)
    if not posts:
        return {"status": "error", "message": "No hay posts para este rango de fechas."}

    post_dicts = db_posts_to_legacy_dicts(posts)

    # Ensure classification defaults for unclassified posts
    for p in post_dicts:
        p.setdefault("category", p.get("category") or "other")
        p.setdefault("sentiment", p.get("sentiment") or "neutral")
        p.setdefault("topics", p.get("topics") or [])
        p.setdefault("image_type", p.get("image_type") or "none")

    # compute_metrics() is UNCHANGED from the original codebase
    metrics = compute_metrics(post_dicts)

    return {"status": "ok", "metrics": metrics}


@router.post("/analyze")
async def analyze_posts(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run AI classification + pattern analysis. SSE streaming."""
    body = await request.json()
    from_date = body.get("from_date")
    to_date = body.get("to_date")

    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    posts = await _load_user_posts(db, user.id, from_date, to_date)
    if not posts:
        return {"status": "error", "message": "No hay posts para analizar."}

    post_dicts = db_posts_to_legacy_dicts(posts)

    # Check for cached analysis
    cached = await db.execute(
        select(AnalysisResult).where(
            AnalysisResult.user_id == user.id,
            AnalysisResult.analysis_type == "full_dashboard",
            AnalysisResult.post_count == len(post_dicts),
        ).order_by(AnalysisResult.created_at.desc())
    )
    cached_result = cached.scalar_one_or_none()

    if cached_result:
        # Check if same date range
        cached_from = cached_result.from_date.strftime("%Y-%m-%d") if cached_result.from_date else None
        cached_to = cached_result.to_date.strftime("%Y-%m-%d") if cached_result.to_date else None
        if cached_from == from_date and cached_to == to_date:
            return {"status": "ok", "metrics": cached_result.result_data, "cached": True}

    if not openai_api_key or openai_api_key.startswith("sk-PONE"):
        # No AI: just compute metrics with defaults
        for p in post_dicts:
            p.setdefault("category", p.get("category") or "other")
            p.setdefault("sentiment", p.get("sentiment") or "neutral")
            p.setdefault("topics", p.get("topics") or [])
            p.setdefault("image_type", p.get("image_type") or "none")
        metrics = compute_metrics(post_dicts)
        return {"status": "ok", "metrics": metrics}

    # SSE streaming with AI analysis
    # We need post IDs to update classifications later
    post_id_map = {p.linkedin_activity_id: p.id for p in posts}

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue = asyncio.Queue()

        def progress_cb(current, total, from_cache=False):
            try:
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"current": current, "total": total, "cached": from_cache},
                )
            except Exception:
                pass

        # Phase 1: Classify
        yield f"data: {json.dumps({'phase': 'classify', 'message': 'Fase 1: Clasificando posts...'})}\n\n"

        classify_task = loop.run_in_executor(
            None, classify_posts, post_dicts, openai_api_key, progress_cb
        )

        done = False
        while not done:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["current"] >= msg["total"]:
                    done = True
            except asyncio.TimeoutError:
                if classify_task.done():
                    done = True
                else:
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"

        classified = await classify_task

        # Save classifications back to DB
        try:
            async with db.begin_nested():
                for p in classified:
                    aid = p.get("activity_id", "")
                    post_id = post_id_map.get(aid)
                    if post_id:
                        await db.execute(
                            update(Post).where(Post.id == post_id).values(
                                category=p.get("category"),
                                sentiment=p.get("sentiment"),
                                topics=p.get("topics"),
                                image_type=p.get("image_type"),
                            )
                        )
                await db.flush()
        except Exception as e:
            logger.warning(f"Failed to save classifications to DB: {e}")

        # Compute metrics
        metrics = compute_metrics(classified)

        # Phase 2: Deep pattern analysis
        yield f"data: {json.dumps({'phase': 'patterns', 'message': 'Fase 2: Analizando patrones...'})}\n\n"

        pattern_task = loop.run_in_executor(
            None, deep_pattern_analysis, classified, metrics, openai_api_key
        )

        while not pattern_task.done():
            await asyncio.sleep(1)
            yield f"data: {json.dumps({'heartbeat': True})}\n\n"

        pattern_results = await pattern_task
        metrics["pattern_analysis"] = pattern_results

        # Cache the complete analysis
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d") if from_date else None
            to_dt = datetime.strptime(to_date, "%Y-%m-%d") if to_date else None
            analysis = AnalysisResult(
                user_id=user.id,
                analysis_type="full_dashboard",
                from_date=from_dt,
                to_date=to_dt,
                post_count=len(classified),
                result_data=metrics,
                created_at=datetime.utcnow(),
            )
            db.add(analysis)
            await db.flush()
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")

        yield f"data: {json.dumps({'status': 'ok', 'metrics': metrics})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/scrape-history")
async def scrape_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List past scrape sessions for the user."""
    result = await db.execute(
        select(ScrapeSession)
        .where(ScrapeSession.user_id == user.id)
        .order_by(ScrapeSession.created_at.desc())
    )
    sessions = result.scalars().all()

    return {
        "status": "ok",
        "sessions": [
            {
                "id": str(s.id),
                "from_date": s.from_date.strftime("%Y-%m-%d") if s.from_date else None,
                "to_date": s.to_date.strftime("%Y-%m-%d") if s.to_date else None,
                "status": s.status,
                "posts_scraped": s.posts_scraped or 0,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in sessions
        ],
    }
