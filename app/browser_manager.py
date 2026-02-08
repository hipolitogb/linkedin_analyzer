"""Playwright browser session manager for remote LinkedIn login."""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


@dataclass
class BrowserSession:
    """Represents an active Playwright browser session."""
    session_id: str
    browser: object = None
    context: object = None
    page: object = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    cookies_extracted: bool = False
    _viewport_width: int = 1280
    _viewport_height: int = 800


class BrowserSessionManager:
    """Manages Playwright browser instances for LinkedIn login sessions."""

    def __init__(self):
        self.sessions: dict[str, BrowserSession] = {}
        self._playwright = None
        self._browser_type = None

    async def _ensure_playwright(self):
        """Lazy-init Playwright."""
        if self._playwright is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser_type = self._playwright.chromium

    async def create_session(self) -> BrowserSession:
        """Create a new browser session and navigate to LinkedIn login."""
        await self._ensure_playwright()

        session_id = str(uuid4())
        browser = await self._browser_type.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Navigate to LinkedIn login
        try:
            await page.goto("https://www.linkedin.com/login", wait_until="networkidle", timeout=30000)
        except Exception as e:
            logger.warning(f"LinkedIn login page load timeout (may be ok): {e}")

        session = BrowserSession(
            session_id=session_id,
            browser=browser,
            context=context,
            page=page,
        )
        self.sessions[session_id] = session
        logger.info(f"Browser session {session_id} created")
        return session

    async def take_screenshot(self, session_id: str) -> str:
        """Take a screenshot and return as base64 string."""
        session = self.sessions.get(session_id)
        if not session or not session.page:
            raise ValueError(f"Session {session_id} not found")

        screenshot_bytes = await session.page.screenshot(type="jpeg", quality=70)
        return base64.b64encode(screenshot_bytes).decode()

    async def send_click(self, session_id: str, x: int, y: int):
        """Click at coordinates."""
        session = self.sessions.get(session_id)
        if not session or not session.page:
            raise ValueError(f"Session {session_id} not found")
        await session.page.mouse.click(x, y)

    async def send_type(self, session_id: str, text: str):
        """Type text."""
        session = self.sessions.get(session_id)
        if not session or not session.page:
            raise ValueError(f"Session {session_id} not found")
        await session.page.keyboard.type(text, delay=50)

    async def send_key(self, session_id: str, key: str):
        """Press a key."""
        session = self.sessions.get(session_id)
        if not session or not session.page:
            raise ValueError(f"Session {session_id} not found")
        await session.page.keyboard.press(key)

    async def scroll(self, session_id: str, delta_y: int):
        """Scroll the page."""
        session = self.sessions.get(session_id)
        if not session or not session.page:
            raise ValueError(f"Session {session_id} not found")
        await session.page.mouse.wheel(0, delta_y)

    async def extract_cookies(self, session_id: str) -> dict | None:
        """Extract li_at and JSESSIONID cookies if present."""
        session = self.sessions.get(session_id)
        if not session or not session.context:
            raise ValueError(f"Session {session_id} not found")

        cookies = await session.context.cookies()
        li_at = None
        jsessionid = None

        for cookie in cookies:
            if cookie["name"] == "li_at":
                li_at = cookie["value"]
            elif cookie["name"] == "JSESSIONID":
                jsessionid = cookie["value"]

        if not li_at:
            return None

        session.cookies_extracted = True
        return {
            "li_at": li_at,
            "jsessionid": jsessionid or "",
        }

    async def extract_profile(self, session_id: str) -> dict | None:
        """Extract LinkedIn profile info using captured cookies."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        cookie_data = await self.extract_cookies(session_id)
        if not cookie_data:
            return None

        # Call LinkedIn Voyager API /me to get profile
        li_at = cookie_data["li_at"]
        jsessionid = cookie_data["jsessionid"]
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
        cookies_dict = {
            "li_at": li_at,
            "JSESSIONID": jsessionid or '"ajax:0"',
        }

        try:
            async with httpx.AsyncClient(
                cookies=cookies_dict,
                headers=headers,
                follow_redirects=False,
                timeout=15.0,
            ) as client:
                resp = await client.get("https://www.linkedin.com/voyager/api/me")
                if resp.status_code != 200:
                    logger.warning(f"/api/me returned {resp.status_code}")
                    return None

                data = resp.json()
                mini_profile = data.get("miniProfile", {})
                first_name = mini_profile.get("firstName", "")
                last_name = mini_profile.get("lastName", "")
                public_id = mini_profile.get("publicIdentifier", "")
                profile_url = f"https://www.linkedin.com/in/{public_id}/" if public_id else ""

                return {
                    "first_name": first_name,
                    "last_name": last_name,
                    "public_id": public_id,
                    "profile_url": profile_url,
                    "email": "",  # Not available via this endpoint
                }
        except Exception as e:
            logger.error(f"Profile extraction failed: {e}")
            return None

    async def close_session(self, session_id: str):
        """Close a browser session and free resources."""
        session = self.sessions.pop(session_id, None)
        if not session:
            return

        try:
            if session.page:
                await session.page.close()
            if session.context:
                await session.context.close()
            if session.browser:
                await session.browser.close()
        except Exception as e:
            logger.warning(f"Error closing session {session_id}: {e}")

        logger.info(f"Browser session {session_id} closed")

    async def cleanup_stale_sessions(self, max_age_seconds: int = 600):
        """Remove sessions older than max_age_seconds."""
        now = datetime.utcnow()
        stale = [
            sid for sid, sess in self.sessions.items()
            if (now - sess.created_at).total_seconds() > max_age_seconds
        ]
        for sid in stale:
            await self.close_session(sid)
            logger.info(f"Cleaned up stale session {sid}")


# Global singleton
browser_manager = BrowserSessionManager()
