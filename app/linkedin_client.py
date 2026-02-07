import csv
import io
import re
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _parse_date(date_str: str) -> datetime | None:
    """Parse dates like 'You uploaded a feed document on March 11, 2025 at 6:45 PM (GMT)'"""
    match = re.search(
        r"on (\w+) (\d+), (\d{4}) at (\d+):(\d+) (AM|PM)",
        date_str,
    )
    if not match:
        return None

    month_name, day, year, hour, minute, ampm = match.groups()
    month = MONTH_MAP.get(month_name)
    if not month:
        return None

    hour = int(hour)
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    return datetime(int(year), month, int(day), hour, int(minute))


def _detect_media_type(date_str: str, media_link: str) -> str:
    """Detect content type from the date description and media URL."""
    date_lower = date_str.lower()
    link_lower = media_link.lower()

    if "profile photo" in date_lower or "profile-displayphoto" in link_lower:
        return "profile_photo"
    if "feed document" in date_lower or "document" in link_lower:
        return "carousel"
    if "feed photo" in date_lower or "feedshare-shrink" in link_lower:
        return "image"
    if "video" in date_lower:
        return "video"
    return "other"


def load_posts_from_backup(backup_dir: str) -> list[dict]:
    """Load posts from LinkedIn data export CSV files."""
    backup_path = Path(backup_dir)
    posts = []

    # Load Rich_Media.csv (main post data with media)
    rich_media_path = backup_path / "Rich_Media.csv"
    if rich_media_path.exists():
        content = rich_media_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))

        # Group feed photos by date to detect multi-image posts
        for row in reader:
            date_str = row.get("Date/Time", "")
            text = row.get("Media Description", "").strip()
            media_link = row.get("Media Link", "").strip()

            if text == "-":
                text = ""

            date = _parse_date(date_str)
            if not date:
                continue

            media_type = _detect_media_type(date_str, media_link)

            # Skip profile photos
            if media_type == "profile_photo":
                continue

            posts.append({
                "text": text,
                "date": date,
                "content_type": media_type,
                "media_link": media_link,
                "has_image": media_type in ("image", "carousel"),
                "text_length": len(text),
                # No engagement data in backup
                "reactions": 0,
                "comments": 0,
                "shares": 0,
                "engagement": 0,
                "image_urls": [media_link] if media_type in ("image", "carousel") else [],
            })

    # Merge feed photos posted at the same time (multi-image posts)
    posts = _merge_same_time_posts(posts)

    # Sort by date descending
    posts.sort(key=lambda p: p["date"], reverse=True)

    logger.info(f"Loaded {len(posts)} posts from backup")
    return posts


def _merge_same_time_posts(posts: list[dict]) -> list[dict]:
    """Merge multiple feed photos uploaded at the exact same time into one post."""
    merged = {}
    for p in posts:
        key = p["date"].isoformat()
        if key in merged:
            existing = merged[key]
            # Append image URLs
            existing["image_urls"].extend(p["image_urls"])
            # Keep the longer text
            if len(p["text"]) > len(existing["text"]):
                existing["text"] = p["text"]
                existing["text_length"] = p["text_length"]
        else:
            merged[key] = p

    return list(merged.values())
