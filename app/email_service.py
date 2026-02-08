"""Email sending service for magic link authentication."""

import logging
import os

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


async def send_magic_link(email: str, token: str, base_url: str | None = None) -> bool:
    """Send a magic link login email.

    For MVP: logs the link to console. Future: SMTP or transactional email API.
    """
    url = base_url or BASE_URL
    login_url = f"{url}/auth/verify?token={token}"

    # MVP: Log to console
    logger.info("=" * 60)
    logger.info(f"MAGIC LINK for {email}")
    logger.info(f"  {login_url}")
    logger.info("=" * 60)

    # TODO: Replace with actual email sending (SendGrid, Resend, SMTP)
    # smtp_host = os.getenv("SMTP_HOST")
    # if smtp_host:
    #     ... send actual email ...

    return True
