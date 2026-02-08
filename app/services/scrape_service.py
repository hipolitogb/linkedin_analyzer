"""Business logic for multi-tenant scraping: run scrape, store posts in DB."""

import logging
from datetime import datetime
from typing import Callable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Post, ScrapeSession
from app.scraper import process_scraped_posts, scrape_profile_posts

logger = logging.getLogger(__name__)


def posts_to_db_rows(
    user_id: UUID,
    scrape_session_id: UUID,
    posts: list[dict],
) -> list[dict]:
    """Convert processed post dicts to dicts compatible with Post model columns."""
    rows = []
    for p in posts:
        activity_id = p.get("activity_id", "")
        if not activity_id:
            # Fallback: extract from URL if available
            url = p.get("url", "")
            if "activity:" in url:
                activity_id = url.split("activity:")[-1].rstrip("/")
            elif p.get("text"):
                # Last resort: hash of text + date
                import hashlib
                date_str = p["date"].isoformat() if isinstance(p["date"], datetime) else str(p["date"])
                activity_id = hashlib.md5((p["text"][:80] + date_str).encode()).hexdigest()

        if not activity_id:
            continue  # Skip posts without identifiable ID

        date_val = p["date"]
        if isinstance(date_val, str):
            date_val = datetime.fromisoformat(date_val)

        rows.append({
            "user_id": user_id,
            "scrape_session_id": scrape_session_id,
            "linkedin_activity_id": activity_id,
            "text": p.get("text", ""),
            "date": date_val,
            "content_type": p.get("content_type", "text"),
            "reactions": p.get("reactions", 0),
            "comments": p.get("comments", 0),
            "shares": p.get("shares", 0),
            "engagement": p.get("engagement", 0),
            "impressions": p.get("impressions", 0),
            "has_image": p.get("has_image", False),
            "image_urls": p.get("image_urls", []),
            "text_length": p.get("text_length", 0),
            "url": p.get("url", ""),
            "is_repost": p.get("is_repost", False),
            "original_author": p.get("original_author", ""),
            "reshare_comment": p.get("reshare_comment", ""),
        })
    return rows


async def upsert_posts(db: AsyncSession, rows: list[dict]) -> int:
    """Bulk upsert posts into the database. Returns count of new posts."""
    if not rows:
        return 0

    new_count = 0
    for row in rows:
        # Check if post already exists
        result = await db.execute(
            text(
                "SELECT id FROM posts WHERE user_id = :user_id AND linkedin_activity_id = :activity_id"
            ),
            {"user_id": row["user_id"], "activity_id": row["linkedin_activity_id"]},
        )
        existing = result.first()

        if existing:
            # Update engagement metrics and repost classification
            await db.execute(
                text("""
                    UPDATE posts SET
                        reactions = :reactions,
                        comments = :comments,
                        shares = :shares,
                        engagement = :engagement,
                        impressions = :impressions,
                        is_repost = :is_repost,
                        original_author = :original_author,
                        reshare_comment = :reshare_comment,
                        updated_at = :now
                    WHERE user_id = :user_id AND linkedin_activity_id = :activity_id
                """),
                {
                    "reactions": row["reactions"],
                    "comments": row["comments"],
                    "shares": row["shares"],
                    "engagement": row["engagement"],
                    "impressions": row["impressions"],
                    "is_repost": row.get("is_repost", False),
                    "original_author": row.get("original_author", ""),
                    "reshare_comment": row.get("reshare_comment", ""),
                    "now": datetime.utcnow(),
                    "user_id": row["user_id"],
                    "activity_id": row["linkedin_activity_id"],
                },
            )
        else:
            post = Post(**row)
            db.add(post)
            new_count += 1

    await db.flush()
    return new_count


async def run_scrape(
    db: AsyncSession,
    scrape_session_id: UUID,
    user_id: UUID,
    li_at: str,
    jsessionid: str,
    public_id: str,
    from_date: str,
    to_date: str,
    progress_callback: Callable | None = None,
) -> int:
    """Run a full scrape and store results in DB. Returns count of new posts."""
    # Update session status
    session = await db.get(ScrapeSession, scrape_session_id)
    if not session:
        raise ValueError(f"ScrapeSession {scrape_session_id} not found")

    session.status = "scraping"
    session.started_at = datetime.utcnow()
    await db.flush()

    try:
        # Call existing scraper (UNCHANGED)
        scrape_result = await scrape_profile_posts(
            li_at,
            public_id,
            jsessionid=jsessionid,
            from_date=from_date,
            to_date=to_date,
        )

        raw_posts = scrape_result["posts"]
        posts = process_scraped_posts(raw_posts)

        # Apply date filter on processed posts
        if from_date or to_date:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d") if from_date else datetime.min
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            ) if to_date else datetime.max
            posts = [p for p in posts if from_dt <= p["date"] <= to_dt]

        # Convert to DB rows and upsert
        rows = posts_to_db_rows(user_id, scrape_session_id, posts)
        new_count = await upsert_posts(db, rows)

        # Update session
        session.status = "completed"
        session.posts_scraped = len(rows)
        session.completed_at = datetime.utcnow()
        await db.flush()

        logger.info(
            f"Scrape session {scrape_session_id}: {len(rows)} posts stored ({new_count} new)"
        )
        return new_count

    except Exception as e:
        session.status = "failed"
        session.error_message = str(e)[:500]
        session.completed_at = datetime.utcnow()
        await db.flush()
        logger.error(f"Scrape session {scrape_session_id} failed: {e}", exc_info=True)
        raise
