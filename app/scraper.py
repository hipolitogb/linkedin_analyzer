"""Scrape LinkedIn profile posts using Voyager API (profileUpdatesV2 endpoint)."""

import asyncio
import json
import logging
import random
import re
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://www.linkedin.com/voyager/api"

REQUEST_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/83.0.4103.116 Safari/537.36"
    ),
    "accept-language": "en-AU,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
    "x-li-lang": "en_US",
    "x-restli-protocol-version": "2.0.0",
}

MAX_POSTS_PER_PAGE = 100
TZ_BA = ZoneInfo("America/Argentina/Buenos_Aires")


async def scrape_profile_posts(
    li_at: str,
    public_id: str,
    jsessionid: str = "",
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:
    """Fetch LinkedIn posts via Voyager API profileUpdatesV2 endpoint."""
    now = datetime.now(tz=TZ_BA)
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=TZ_BA) if from_date else now - timedelta(days=365)
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=TZ_BA, hour=23, minute=59, second=59) if to_date else now

    jsession_value = jsessionid.strip() if jsessionid else '"ajax:0"'
    csrf_value = jsession_value.strip('"')

    cookies = {
        "li_at": li_at.strip(),
        "JSESSIONID": jsession_value,
        "lang": "v=2&lang=en-us",
    }
    headers = {
        **REQUEST_HEADERS,
        "csrf-token": csrf_value,
    }

    logger.info(f"Using JSESSIONID: {jsession_value[:30]}...")
    logger.info(f"Using csrf-token: {csrf_value[:30]}...")

    async with httpx.AsyncClient(
        cookies=cookies,
        headers=headers,
        follow_redirects=False,
        timeout=30.0,
    ) as client:

        # Step 1: Validate session
        logger.info("Validating session via /voyager/api/me ...")
        resp = await client.get(f"{API_BASE}/me")
        (DEBUG_DIR / "api_me.json").write_text(resp.text[:50000], encoding="utf-8")

        if resp.status_code == 401:
            raise Exception("LinkedIn rejected the cookie (401). Please provide a fresh li_at + JSESSIONID.")
        if resp.status_code != 200:
            raise Exception(f"/api/me returned {resp.status_code}. Session may be invalid.")

        logger.info("Session valid.")

        # Step 2: Get profile URN
        me_data = resp.json()
        profile_urn = _extract_urn(me_data, public_id)

        if not profile_urn:
            # Try fetching profile by public_id
            profile_urn = await _get_profile_urn(client, public_id)

        if not profile_urn:
            raise Exception(f"Could not find profile URN for '{public_id}'.")

        logger.info(f"Profile URN: {profile_urn}")

        # Step 3: Fetch posts via /identity/profileUpdatesV2
        result = await _fetch_posts(client, profile_urn, from_dt, to_dt)

    logger.info(f"Scraped {len(result['posts'])} posts in requested range (fetched {result['total_fetched']} total)")
    return result


async def _get_profile_urn(client: httpx.AsyncClient, public_id: str) -> str | None:
    """Get profile URN by fetching profile data."""
    url = f"{API_BASE}/identity/dash/profiles?q=memberIdentity&memberIdentity={public_id}&decorationId=com.linkedin.voyager.dash.deco.identity.profile.WebTopCardCore-16"
    try:
        await asyncio.sleep(random.randint(2, 5))
        resp = await client.get(url)
        (DEBUG_DIR / "profile_lookup.json").write_text(resp.text[:50000], encoding="utf-8")
        if resp.status_code == 200:
            data = resp.json()
            return _extract_urn(data, public_id)
    except Exception as e:
        logger.warning(f"Profile lookup failed: {e}")
    return None


def _extract_urn(data: dict, public_id: str) -> str | None:
    """Extract fsd_profile URN from API response."""
    if not data or not isinstance(data, dict):
        return None

    urns = []

    # From miniProfile (common in /api/me response)
    mini = data.get("miniProfile", {})
    if isinstance(mini, dict):
        for key in ("dashEntityUrn", "entityUrn", "objectUrn"):
            v = mini.get(key, "")
            if v:
                urns.append(v)

    # From included array (normalized JSON responses)
    for item in data.get("included", []):
        if isinstance(item, dict):
            for key in ("dashEntityUrn", "entityUrn"):
                v = item.get(key, "")
                if v and "profile" in v:
                    urns.append(v)

    # From data.elements
    data_obj = data.get("data", {})
    if isinstance(data_obj, dict):
        for item in data_obj.get("elements", []):
            if isinstance(item, dict):
                for key in ("dashEntityUrn", "entityUrn"):
                    v = item.get(key, "")
                    if v:
                        urns.append(v)

    # Prefer fsd_profile URN
    for urn in urns:
        if "fsd_profile" in urn:
            return urn

    # Convert fs_miniProfile -> fsd_profile
    for urn in urns:
        if "fs_miniProfile" in urn:
            converted = urn.replace("fs_miniProfile", "fsd_profile")
            logger.info(f"Converted URN: {urn} -> {converted}")
            return converted

    return urns[0] if urns else None


async def _fetch_posts(client: httpx.AsyncClient, profile_urn: str, from_dt: datetime, to_dt: datetime) -> list[dict]:
    """Fetch all posts using /identity/profileUpdatesV2 with pagination."""
    all_posts = []
    pagination_token = ""
    start = 0
    max_pages = 30

    for page in range(max_pages):
        # Rate limiting: longer delay to reduce chance of LinkedIn cutting us off
        if page > 0:
            delay = random.randint(5, 10)
            logger.info(f"Rate limit delay: {delay}s...")
            await asyncio.sleep(delay)

        params = {
            "count": MAX_POSTS_PER_PAGE,
            "q": "memberShareFeed",
            "moduleKey": "member-shares:phone",
            "includeLongTermHistory": True,
            "profileUrn": profile_urn,
        }
        if pagination_token:
            # Token already encodes the position — don't send start offset
            params["paginationToken"] = pagination_token
        else:
            params["start"] = start

        logger.info(f"Fetching page {page} (start={start}, token={'yes' if pagination_token else 'no'})...")

        resp = await client.get(f"{API_BASE}/identity/profileUpdatesV2", params=params)

        if resp.status_code == 429:
            logger.warning("Rate limited (429). Waiting 30s...")
            await asyncio.sleep(30)
            resp = await client.get(f"{API_BASE}/identity/profileUpdatesV2", params=params)

        if resp.status_code == 401:
            raise Exception("Session expired (401). Please provide fresh cookies.")

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location", "")
            logger.warning(f"Page {page}: redirect ({resp.status_code}) — likely rate limited. Waiting 30s and retrying...")
            await asyncio.sleep(30)
            resp = await client.get(f"{API_BASE}/identity/profileUpdatesV2", params=params)
            if resp.status_code in (301, 302, 303, 307, 308):
                logger.warning(f"Page {page}: redirect again after retry — stopping pagination with {len(all_posts)} posts. "
                               f"LinkedIn may limit how far back you can paginate in one session.")
                break

        (DEBUG_DIR / f"posts_page_{page}.json").write_text(
            resp.text, encoding="utf-8"
        )

        if resp.status_code != 200:
            logger.error(f"Page {page}: got status {resp.status_code}")
            break

        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            logger.error(f"Page {page}: non-JSON response ({content_type})")
            (DEBUG_DIR / f"posts_page_{page}_raw.html").write_text(resp.text[:50000], encoding="utf-8")
            break

        data = resp.json()

        # Check for API error
        if isinstance(data, dict) and "status" in data and data["status"] != 200:
            logger.error(f"API error: {data.get('message', data)}")
            break

        # Parse posts from response
        new_posts = _parse_posts(data)
        logger.info(f"Page {page}: parsed {len(new_posts)} posts")

        if not new_posts:
            logger.info("No more posts found.")
            break

        # Deduplicate and add
        existing_keys = {(p.get("text", "")[:80], p.get("timestamp", 0)) for p in all_posts}
        for post in new_posts:
            key = (post.get("text", "")[:80], post.get("timestamp", 0))
            if key not in existing_keys:
                all_posts.append(post)
                existing_keys.add(key)

        # Check cutoff — stop paginating when oldest post is before from_dt
        timestamps = [_parse_timestamp(p.get("timestamp", 0)) for p in new_posts]
        timestamps = [t for t in timestamps if t]
        if timestamps and min(timestamps) < from_dt.replace(tzinfo=None):
            logger.info(f"Reached from_date cutoff. Total: {len(all_posts)} posts")
            break

        # Pagination: get next token from metadata
        metadata = data.get("metadata", {})
        next_token = metadata.get("paginationToken", "")
        if not next_token:
            logger.info(f"No more pagination token. Total: {len(all_posts)} posts")
            break

        pagination_token = next_token
        start += MAX_POSTS_PER_PAGE

    # Compute actual date range of scraped posts
    all_timestamps = [_parse_timestamp(p.get("timestamp", 0)) for p in all_posts]
    all_timestamps = [t for t in all_timestamps if t]
    actual_oldest = min(all_timestamps) if all_timestamps else None
    actual_newest = max(all_timestamps) if all_timestamps else None

    logger.info(f"Scraped {len(all_posts)} posts total. "
                f"Actual date range: {actual_oldest.strftime('%Y-%m-%d') if actual_oldest else '?'} "
                f"to {actual_newest.strftime('%Y-%m-%d') if actual_newest else '?'}")
    logger.info(f"Requested date range: {from_dt.date()} to {to_dt.date()}")

    if actual_oldest and from_dt.replace(tzinfo=None) < actual_oldest:
        logger.warning(f"LinkedIn did not return posts before {actual_oldest.strftime('%Y-%m-%d')}. "
                       f"Requested from {from_dt.date()} but oldest post is {actual_oldest.strftime('%Y-%m-%d')}. "
                       f"LinkedIn limits how far back you can paginate in one session.")

    # Filter to [from_dt, to_dt] range
    filtered = []
    from_naive = from_dt.replace(tzinfo=None)
    to_naive = to_dt.replace(tzinfo=None)
    for p in all_posts:
        dt = _parse_timestamp(p.get("timestamp", 0))
        if dt and from_naive <= dt <= to_naive:
            filtered.append(p)
        elif not dt:
            filtered.append(p)  # keep posts without parseable timestamp
    logger.info(f"Date filter: {len(all_posts)} -> {len(filtered)} posts in [{from_dt.date()}, {to_dt.date()}]")

    return {
        "posts": filtered,
        "total_fetched": len(all_posts),
        "requested_from": from_dt.date().isoformat(),
        "requested_to": to_dt.date().isoformat(),
        "actual_from": actual_oldest.strftime("%Y-%m-%d") if actual_oldest else None,
        "actual_to": actual_newest.strftime("%Y-%m-%d") if actual_newest else None,
    }


def _parse_posts(data: dict) -> list[dict]:
    """Parse posts from profileUpdatesV2 response.

    Response structure:
    - data["elements"]: list of post update objects
    - data["included"]: list of referenced entities (normalized JSON)
    - Each element has commentary, actor, content, socialDetail, etc.
    """
    if not data or not isinstance(data, dict):
        return []

    elements = data.get("elements", [])
    included = data.get("included", [])

    # Build entity lookup from included
    entity_map = {}
    for item in included:
        if isinstance(item, dict):
            urn = item.get("entityUrn", "") or item.get("$id", "")
            if urn:
                entity_map[urn] = item

    posts = []

    # Parse each element (top-level post updates)
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        post = _extract_post(elem, entity_map)
        if post and (post.get("text") or post.get("timestamp")):
            posts.append(post)

    # If elements is empty, try parsing from included (some responses use this)
    if not posts:
        for item in included:
            if not isinstance(item, dict):
                continue
            etype = item.get("$type", "")
            eurn = item.get("entityUrn", "")
            if "Update" in etype or "activity" in eurn.lower() or "ugcPost" in eurn:
                post = _extract_post(item, entity_map)
                if post and (post.get("text") or post.get("timestamp")):
                    posts.append(post)

    return posts


def _parse_relative_time(text: str) -> datetime | None:
    """Parse LinkedIn relative time like '2 days ago', '1w •', '3mo •' etc."""
    if not text:
        return None
    text = text.strip().lower()
    now = datetime.now()

    # "X days/weeks/months/years ago" format (accessibilityText)
    match = re.search(r'(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago', text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == "second":
            return now - timedelta(seconds=num)
        if unit == "minute":
            return now - timedelta(minutes=num)
        if unit == "hour":
            return now - timedelta(hours=num)
        if unit == "day":
            return now - timedelta(days=num)
        if unit == "week":
            return now - timedelta(weeks=num)
        if unit == "month":
            return now - timedelta(days=num * 30)
        if unit == "year":
            return now - timedelta(days=num * 365)

    # Short format: "2d •", "1w •", "3mo •", "1yr •"
    match = re.search(r'(\d+)(s|m|h|d|w|mo|yr)', text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            return now - timedelta(seconds=num)
        if unit == "m":
            return now - timedelta(minutes=num)
        if unit == "h":
            return now - timedelta(hours=num)
        if unit == "d":
            return now - timedelta(days=num)
        if unit == "w":
            return now - timedelta(weeks=num)
        if unit == "mo":
            return now - timedelta(days=num * 30)
        if unit == "yr":
            return now - timedelta(days=num * 365)

    return None


def _timestamp_from_activity_urn(urn: str) -> int:
    """Extract timestamp from LinkedIn activity URN.
    Activity IDs encode the creation time: activity_id >> 22 = Unix timestamp in ms.
    """
    if not urn:
        return 0
    match = re.search(r'activity:(\d+)', urn)
    if match:
        try:
            activity_id = int(match.group(1))
            ts_ms = activity_id >> 22
            # Sanity check: should be between 2015 and 2030
            if 1420000000000 < ts_ms < 1900000000000:
                return ts_ms
        except (ValueError, OverflowError):
            pass
    return 0


def _resolve_ref(ref_or_obj, entity_map: dict):
    """Resolve a reference string to an entity, or return the object itself."""
    if isinstance(ref_or_obj, str) and ref_or_obj in entity_map:
        return entity_map[ref_or_obj]
    if isinstance(ref_or_obj, dict):
        return ref_or_obj
    return {}


def _extract_post(elem: dict, entity_map: dict) -> dict | None:
    """Extract structured post data from a profileUpdatesV2 element."""
    text = ""
    timestamp = 0
    reactions = 0
    comments = 0
    shares = 0
    impressions = 0
    content_type = "text"
    is_repost = False
    original_author = ""
    reshare_comment = ""

    # --- TEXT ---
    # Path 1: commentary.text.text (standard for profileUpdatesV2)
    commentary = elem.get("commentary") or elem.get("*commentary", "")
    commentary = _resolve_ref(commentary, entity_map)
    if isinstance(commentary, dict):
        text_obj = commentary.get("text", "")
        if isinstance(text_obj, dict):
            text = text_obj.get("text", "")
        elif isinstance(text_obj, str):
            text = text_obj

    # Path 2: direct text/title
    if not text:
        for field in ("text", "title", "headline"):
            val = elem.get(field)
            if isinstance(val, str) and len(val) > 5:
                text = val
                break
            if isinstance(val, dict):
                text = val.get("text", "")
                if text:
                    break

    # Path 3: specificContent (older format)
    if not text:
        sc = elem.get("specificContent", {})
        if isinstance(sc, dict):
            share_content = sc.get("com.linkedin.ugc.ShareContent", {})
            if isinstance(share_content, dict):
                text = share_content.get("shareCommentary", {}).get("text", "")

    # --- REPOST / RESHARE DETECTION ---
    # Pattern 1: resharedUpdate field present (actor = profile owner, nested = original)
    reshare = elem.get("resharedUpdate") or elem.get("*resharedUpdate", "")
    reshare_obj = _resolve_ref(reshare, entity_map)
    if isinstance(reshare_obj, dict) and reshare_obj:
        is_repost = True
        reshare_comment = text
        orig_actor = reshare_obj.get("actor") or reshare_obj.get("*actor", "")
        orig_actor = _resolve_ref(orig_actor, entity_map)
        if isinstance(orig_actor, dict):
            name_obj = orig_actor.get("name", "")
            if isinstance(name_obj, dict):
                original_author = name_obj.get("text", "") or name_obj.get("accessibilityText", "")
            elif isinstance(name_obj, str):
                original_author = name_obj
        nested = _extract_post(reshare_obj, entity_map)
        if nested and nested.get("text"):
            text = nested["text"]

    # Pattern 2: header says "X reposted this" (actor = original author, header = profile owner)
    if not is_repost:
        header = elem.get("header") or elem.get("*header", "")
        header = _resolve_ref(header, entity_map)
        if isinstance(header, dict):
            header_text_obj = header.get("text", "")
            if isinstance(header_text_obj, dict):
                header_text_str = header_text_obj.get("text", "")
            elif isinstance(header_text_obj, str):
                header_text_str = header_text_obj
            else:
                header_text_str = ""
            if "reposted this" in header_text_str.lower():
                is_repost = True
                # In this pattern, actor is the original author
                actor = elem.get("actor") or elem.get("*actor", "")
                actor = _resolve_ref(actor, entity_map)
                if isinstance(actor, dict):
                    name_obj = actor.get("name", "")
                    if isinstance(name_obj, dict):
                        original_author = name_obj.get("text", "") or name_obj.get("accessibilityText", "")
                    elif isinstance(name_obj, str):
                        original_author = name_obj

    # --- ACTIVITY ID & URL ---
    activity_id_str = ""
    for urn_source in (elem.get("entityUrn", ""), (elem.get("updateMetadata") or {}).get("urn", "") if isinstance(elem.get("updateMetadata"), dict) else ""):
        match = re.search(r'activity:(\d+)', urn_source)
        if match:
            activity_id_str = match.group(1)
            break
    post_url = f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id_str}/" if activity_id_str else ""

    # --- TIMESTAMP ---
    # Primary source: activity URN (activity_id >> 22 = exact creation time in ms)
    timestamp = 0
    if activity_id_str:
        timestamp = _timestamp_from_activity_urn(f"activity:{activity_id_str}")

    # Fallback 1: createdAt / publishedAt fields
    if not timestamp:
        timestamp = elem.get("createdAt", 0) or elem.get("publishedAt", 0) or elem.get("lastModifiedAt", 0)

    # Fallback 2: actor.subDescription (relative time like "2d", "1w", "1mo")
    date_text = ""
    if not timestamp:
        actor = elem.get("actor") or elem.get("*actor", "")
        actor = _resolve_ref(actor, entity_map)
        if isinstance(actor, dict):
            timestamp = actor.get("timestamp", 0)
            sub_desc = actor.get("subDescription", {})
            if isinstance(sub_desc, dict):
                date_text = sub_desc.get("accessibilityText", "") or sub_desc.get("text", "")

    # Fallback 3: updateMetadata
    if not timestamp:
        meta = elem.get("updateMetadata") or elem.get("*updateMetadata", "")
        meta = _resolve_ref(meta, entity_map)
        if isinstance(meta, dict):
            timestamp = meta.get("publishedAt", 0) or meta.get("createdAt", 0)

    # --- ENGAGEMENT ---
    social = elem.get("socialDetail") or elem.get("*socialDetail", "")
    social = _resolve_ref(social, entity_map)
    if isinstance(social, dict):
        counts = social.get("totalSocialActivityCounts") or social.get("*totalSocialActivityCounts", "")
        counts = _resolve_ref(counts, entity_map)
        if isinstance(counts, dict):
            reactions = counts.get("numLikes", 0) or counts.get("reactionCount", 0) or 0
            comments = counts.get("numComments", 0) or counts.get("commentCount", 0) or 0
            shares = counts.get("numShares", 0) or counts.get("shareCount", 0) or 0
            impressions = counts.get("numImpressions", 0) or counts.get("numViews", 0) or 0

    # Fallback: direct fields
    if not reactions:
        reactions = elem.get("numLikes", 0) or elem.get("reactionCount", 0) or 0
    if not comments:
        comments = elem.get("numComments", 0) or elem.get("commentCount", 0) or 0
    if not shares:
        shares = elem.get("numShares", 0) or elem.get("shareCount", 0) or 0

    # --- CONTENT TYPE ---
    content = elem.get("content") or elem.get("*content", "")
    content = _resolve_ref(content, entity_map)
    if isinstance(content, dict):
        # Check all keys — API uses Java-style keys like "com.linkedin.voyager.feed.render.ImageComponent"
        all_keys = " ".join(content.keys()).lower()
        ctype_str = (content.get("$type", "") + " " + all_keys).lower()

        if "image" in ctype_str:
            content_type = "image"
        elif "video" in ctype_str:
            content_type = "video"
        elif "document" in ctype_str:
            content_type = "carousel"
        elif "article" in ctype_str:
            content_type = "article"
        elif "poll" in ctype_str:
            content_type = "poll"

        if content_type == "text":
            if content.get("image") or content.get("images"):
                content_type = "image"
            elif content.get("video") or content.get("linkedInVideo"):
                content_type = "video"
            elif content.get("document"):
                content_type = "carousel"
            elif content.get("article") or content.get("linkedInArticle"):
                content_type = "article"
            elif content.get("poll"):
                content_type = "poll"

    if not text and not timestamp and not date_text:
        return None

    date_str = ""
    if timestamp:
        try:
            dt = datetime.fromtimestamp(timestamp / 1000) if timestamp > 1e12 else datetime.fromtimestamp(timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            pass

    # If no numeric timestamp, convert relative time text to approximate date
    if not timestamp and date_text:
        parsed = _parse_relative_time(date_text)
        if parsed:
            date_str = parsed.strftime("%Y-%m-%d %H:%M")
            timestamp = int(parsed.timestamp() * 1000)

    return {
        "text": text,
        "dateStr": date_str,
        "timestamp": timestamp,
        "reactions": reactions,
        "comments": comments,
        "shares": shares,
        "engagement": reactions + comments + shares,
        "impressions": impressions,
        "contentType": content_type,
        "hasImage": content_type in ("image", "carousel"),
        "imageUrls": [],
        "textLength": len(text),
        "url": post_url,
        "activity_id": activity_id_str,
        "is_repost": is_repost,
        "original_author": original_author,
        "reshare_comment": reshare_comment,
    }


def parse_relative_date(date_str: str) -> datetime | None:
    """Parse LinkedIn relative dates or absolute dates."""
    if not date_str:
        return None
    date_str = date_str.strip()

    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    date_lower = date_str.lower()
    now = datetime.now()
    match = re.search(r"(\d+)\s*(m(?:in)?|h(?:r)?|d|w|mo|yr|y)\b", date_lower)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit in ("m", "min"):
            return now - timedelta(minutes=num)
        if unit in ("h", "hr"):
            return now - timedelta(hours=num)
        if unit == "d":
            return now - timedelta(days=num)
        if unit == "w":
            return now - timedelta(weeks=num)
        if unit == "mo":
            return now - timedelta(days=num * 30)
        if unit in ("yr", "y"):
            return now - timedelta(days=num * 365)

    return None


def _parse_timestamp(ts: int) -> datetime | None:
    """Parse a Unix timestamp (ms or seconds)."""
    if not ts:
        return None
    try:
        if ts > 1e12:
            return datetime.fromtimestamp(ts / 1000)
        return datetime.fromtimestamp(ts)
    except (ValueError, OSError):
        return None


def process_scraped_posts(raw_posts: list[dict]) -> list[dict]:
    """Convert raw API posts into our standard format.

    Timestamps are converted to America/Argentina/Buenos_Aires (GMT-3).
    """
    processed = []
    for raw in raw_posts:
        ts = raw.get("timestamp", 0)
        date = None
        if ts:
            try:
                utc_dt = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=timezone.utc)
                date = utc_dt.astimezone(TZ_BA).replace(tzinfo=None)
            except (ValueError, OSError):
                pass
        if not date:
            date = parse_relative_date(raw.get("dateStr", ""))
        if not date:
            date = datetime.now()

        processed.append({
            "text": raw.get("text", ""),
            "date": date,
            "content_type": raw.get("contentType", "text"),
            "reactions": raw.get("reactions", 0),
            "comments": raw.get("comments", 0),
            "shares": raw.get("shares", 0),
            "engagement": raw.get("engagement", 0),
            "impressions": raw.get("impressions", 0),
            "has_image": raw.get("hasImage", False),
            "image_urls": raw.get("imageUrls", []),
            "text_length": raw.get("textLength", 0),
            "url": raw.get("url", ""),
            "is_repost": raw.get("is_repost", False),
            "original_author": raw.get("original_author", ""),
            "reshare_comment": raw.get("reshare_comment", ""),
        })

    return processed
