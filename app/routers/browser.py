"""Browser WebSocket endpoint for remote LinkedIn login via Playwright."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.browser_manager import browser_manager
from app.crypto_utils import encrypt_cookies

logger = logging.getLogger(__name__)

router = APIRouter(tags=["browser"])


@router.post("/api/browser/start")
async def start_browser_session():
    """Start a new Playwright browser session for LinkedIn login."""
    try:
        session = await browser_manager.create_session()
        return {"status": "ok", "session_id": session.session_id}
    except Exception as e:
        logger.error(f"Failed to start browser session: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.post("/api/browser/stop/{session_id}")
async def stop_browser_session(session_id: str):
    """Stop and clean up a browser session."""
    await browser_manager.close_session(session_id)
    return {"status": "ok"}


@router.websocket("/ws/browser/{session_id}")
async def browser_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for interacting with the Playwright browser.

    Protocol:
      Client -> Server:
        {"type": "click", "x": 500, "y": 300}
        {"type": "type", "text": "hello"}
        {"type": "key", "key": "Enter"}
        {"type": "scroll", "deltaY": 300}
        {"type": "screenshot"}
        {"type": "extract_cookies"}
        {"type": "extract_profile"}

      Server -> Client:
        {"type": "screenshot", "data": "<base64 JPEG>"}
        {"type": "cookies", "status": "ok", "li_at": "...", "jsessionid": "...", "encrypted": "..."}
        {"type": "cookies", "status": "not_ready", "message": "..."}
        {"type": "profile", "status": "ok", ...}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()

    if session_id not in browser_manager.sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    # Send initial screenshot
    try:
        screenshot = await browser_manager.take_screenshot(session_id)
        await websocket.send_json({"type": "screenshot", "data": screenshot})
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

    # Start screenshot streaming task
    screenshot_interval = 0.5  # seconds
    stop_streaming = asyncio.Event()

    async def stream_screenshots():
        while not stop_streaming.is_set():
            try:
                await asyncio.sleep(screenshot_interval)
                if stop_streaming.is_set():
                    break
                screenshot = await browser_manager.take_screenshot(session_id)
                await websocket.send_json({"type": "screenshot", "data": screenshot})
            except Exception:
                break

    streaming_task = asyncio.create_task(stream_screenshots())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            try:
                if msg_type == "click":
                    await browser_manager.send_click(
                        session_id,
                        int(msg.get("x", 0)),
                        int(msg.get("y", 0)),
                    )
                    # Send screenshot after click
                    await asyncio.sleep(0.3)
                    screenshot = await browser_manager.take_screenshot(session_id)
                    await websocket.send_json({"type": "screenshot", "data": screenshot})

                elif msg_type == "type":
                    await browser_manager.send_type(session_id, msg.get("text", ""))
                    await asyncio.sleep(0.2)
                    screenshot = await browser_manager.take_screenshot(session_id)
                    await websocket.send_json({"type": "screenshot", "data": screenshot})

                elif msg_type == "key":
                    await browser_manager.send_key(session_id, msg.get("key", ""))
                    await asyncio.sleep(0.3)
                    screenshot = await browser_manager.take_screenshot(session_id)
                    await websocket.send_json({"type": "screenshot", "data": screenshot})

                elif msg_type == "scroll":
                    await browser_manager.scroll(session_id, int(msg.get("deltaY", 0)))
                    await asyncio.sleep(0.2)
                    screenshot = await browser_manager.take_screenshot(session_id)
                    await websocket.send_json({"type": "screenshot", "data": screenshot})

                elif msg_type == "screenshot":
                    screenshot = await browser_manager.take_screenshot(session_id)
                    await websocket.send_json({"type": "screenshot", "data": screenshot})

                elif msg_type == "check_login":
                    cookies = await browser_manager.extract_cookies(session_id)
                    if cookies:
                        encrypted = encrypt_cookies(cookies)
                        await websocket.send_json({
                            "type": "login_detected",
                            "encrypted": encrypted,
                        })
                    else:
                        await websocket.send_json({
                            "type": "login_status",
                            "status": "waiting",
                        })

                elif msg_type == "extract_cookies":
                    cookies = await browser_manager.extract_cookies(session_id)
                    if cookies:
                        encrypted = encrypt_cookies(cookies)
                        await websocket.send_json({
                            "type": "cookies",
                            "status": "ok",
                            "li_at": cookies["li_at"][:10] + "...",  # Truncated for security
                            "jsessionid": cookies["jsessionid"][:10] + "..." if cookies["jsessionid"] else "",
                            "encrypted": encrypted,
                        })
                    else:
                        await websocket.send_json({
                            "type": "cookies",
                            "status": "not_ready",
                            "message": "No se detectaron cookies de LinkedIn. Iniciá sesion primero.",
                        })

                elif msg_type == "extract_profile":
                    profile = await browser_manager.extract_profile(session_id)
                    if profile:
                        await websocket.send_json({
                            "type": "profile",
                            "status": "ok",
                            **profile,
                        })
                    else:
                        await websocket.send_json({
                            "type": "profile",
                            "status": "error",
                            "message": "No se pudo extraer el perfil. Intentá de nuevo.",
                        })

                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

            except Exception as e:
                logger.error(f"Browser action error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        stop_streaming.set()
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass
