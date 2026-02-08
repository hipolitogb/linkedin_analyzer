"""Temporary cookie encryption using Fernet symmetric encryption."""

import json
import os

from cryptography.fernet import Fernet

COOKIE_ENCRYPTION_KEY = os.getenv("COOKIE_ENCRYPTION_KEY", "")

# Auto-generate key for dev if not set
if not COOKIE_ENCRYPTION_KEY:
    COOKIE_ENCRYPTION_KEY = Fernet.generate_key().decode()

_fernet = Fernet(COOKIE_ENCRYPTION_KEY.encode() if isinstance(COOKIE_ENCRYPTION_KEY, str) else COOKIE_ENCRYPTION_KEY)


def encrypt_cookies(cookies: dict) -> str:
    """Encrypt a cookie dict to a string for temporary storage."""
    data = json.dumps(cookies).encode()
    return _fernet.encrypt(data).decode()


def decrypt_cookies(encrypted: str) -> dict:
    """Decrypt an encrypted cookie string back to a dict."""
    data = _fernet.decrypt(encrypted.encode())
    return json.loads(data.decode())
