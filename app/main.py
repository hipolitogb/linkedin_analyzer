import asyncio
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.linkedin_client import load_posts_from_backup
from app.scraper import scrape_profile_posts, process_scraped_posts
from app.analyzer import classify_posts, deep_pattern_analysis, compute_metrics, save_dashboard, load_dashboard

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LinkedIn Post Analyzer")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = Path("/tmp/linkedin_backups")
UPLOAD_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# In-memory cache
_cache: dict[str, list[dict]] = {}


class ScrapeRequest(BaseModel):
    li_at: str
    public_id: str
    jsessionid: str = ""
    from_date: str = ""
    to_date: str = ""


def _save_posts(posts: list[dict], public_id: str, source: str) -> Path:
    """Save posts to a JSON file in data/ directory."""
    serializable = []
    for p in posts:
        row = {**p}
        if isinstance(row.get("date"), datetime):
            row["date"] = row["date"].isoformat()
        serializable.append(row)

    filename = f"{public_id}_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = DATA_DIR / filename
    filepath.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(posts)} posts to {filepath}")
    return filepath


def _load_posts_from_json(filepath: Path) -> list[dict]:
    """Load posts from a saved JSON file."""
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    for p in raw:
        if isinstance(p.get("date"), str):
            p["date"] = datetime.fromisoformat(p["date"])
    return raw


def _find_latest_json(public_id: str = "") -> Path | None:
    """Find the most recent saved JSON file, optionally filtered by public_id."""
    files = sorted(DATA_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if public_id:
        files = [f for f in files if f.name.startswith(public_id)]
    return files[0] if files else None


def _list_saved_files() -> list[dict]:
    """List all saved JSON files with metadata."""
    results = []
    for f in sorted(DATA_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            count = len(raw)
        except Exception:
            count = 0
        results.append({
            "filename": f.name,
            "count": count,
            "date": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return results


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/config")
async def get_config():
    """Return non-secret config: whether we have an API key in .env."""
    key = os.getenv("OPENAI_API_KEY", "")
    has_key = bool(key and not key.startswith("sk-PONE"))
    return {"has_openai_key": has_key}


@app.get("/api/saved-data")
async def saved_data():
    """List all saved JSON data files."""
    return {"status": "ok", "files": _list_saved_files()}


@app.post("/api/load-saved")
async def load_saved(request: Request):
    """Load posts from a previously saved JSON file."""
    body = await request.json()
    filename = body.get("filename", "")

    filepath = DATA_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".json":
        return {"status": "error", "message": f"File not found: {filename}"}

    try:
        posts = _load_posts_from_json(filepath)
        _cache["posts"] = posts
        return {
            "status": "ok",
            "count": len(posts),
            "source": filename,
        }
    except Exception as e:
        logger.error(f"Load saved failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/scrape-posts")
async def scrape_posts(req: ScrapeRequest):
    """Scrape posts from LinkedIn using Voyager API."""
    try:
        scrape_result = await scrape_profile_posts(
            req.li_at,
            req.public_id,
            jsessionid=req.jsessionid,
            from_date=req.from_date or None,
            to_date=req.to_date or None,
        )
        raw_posts = scrape_result["posts"]
        posts = process_scraped_posts(raw_posts)

        # Apply date filter on processed posts (timezone-converted dates)
        if req.from_date or req.to_date:
            from_dt = datetime.strptime(req.from_date, "%Y-%m-%d") if req.from_date else datetime.min
            to_dt = datetime.strptime(req.to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59) if req.to_date else datetime.max
            before = len(posts)
            posts = [p for p in posts if from_dt <= p["date"] <= to_dt]
            if len(posts) < before:
                logger.info(f"Post-process date filter: {before} -> {len(posts)} posts in [{req.from_date}, {req.to_date}]")

        _cache["posts"] = posts

        # Save to disk
        saved_path = _save_posts(posts, req.public_id, "scrape")

        # Build warning if LinkedIn didn't return the full requested range
        warning = None
        actual_from = scrape_result.get("actual_from")
        requested_from = scrape_result.get("requested_from")
        if actual_from and requested_from and actual_from > requested_from:
            warning = (f"LinkedIn solo devolvió posts desde {actual_from}. "
                       f"Se solicitó desde {requested_from} pero la API no entregó posts más antiguos. "
                       f"Intentá de nuevo más tarde con cookies frescas para obtener más historial.")

        return {
            "status": "ok",
            "count": len(posts),
            "total_fetched": scrape_result["total_fetched"],
            "saved_to": saved_path.name,
            "requested_range": {"from": requested_from, "to": scrape_result.get("requested_to")},
            "actual_range": {"from": actual_from, "to": scrape_result.get("actual_to")},
            "warning": warning,
            "preview": [
                {
                    "text": p["text"][:120],
                    "date": p["date"].isoformat(),
                    "engagement": p["engagement"],
                    "content_type": p["content_type"],
                }
                for p in posts[:5]
            ],
        }
    except Exception as e:
        logger.error(f"Scrape failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/upload-backup")
async def upload_backup(file: UploadFile = File(...)):
    """Upload a LinkedIn data export ZIP."""
    try:
        session_dir = UPLOAD_DIR / "current"
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True)

        file_path = session_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)

        extract_dir = session_dir / "extracted"
        if file.filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(extract_dir)
            csv_dir = _find_csv_dir(extract_dir)
        else:
            csv_dir = None

        if not csv_dir:
            return {"status": "error", "message": "Could not find CSV files in the upload."}

        posts = load_posts_from_backup(str(csv_dir))
        _cache["posts"] = posts

        saved_path = _save_posts(posts, "backup", "csv")

        return {
            "status": "ok",
            "count": len(posts),
            "saved_to": saved_path.name,
            "preview": [
                {
                    "text": p["text"][:120],
                    "date": p["date"].isoformat(),
                    "content_type": p["content_type"],
                }
                for p in posts[:5]
            ],
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/load-local-backup")
async def load_local_backup():
    """Load backup from the local backupp/ directory."""
    try:
        candidates = [
            Path("backupp"),
            Path("/app/backupp"),
            Path(__file__).resolve().parent.parent / "backupp",
        ]
        backup_dir = None
        for c in candidates:
            if c.exists() and (c / "Rich_Media.csv").exists():
                backup_dir = c
                break

        if not backup_dir:
            tried = ", ".join(str(c) for c in candidates)
            return {"status": "error", "message": f"No 'backupp' directory found. Tried: {tried}"}

        posts = load_posts_from_backup(str(backup_dir))
        _cache["posts"] = posts

        saved_path = _save_posts(posts, "backup", "local")

        return {
            "status": "ok",
            "count": len(posts),
            "saved_to": saved_path.name,
            "preview": [
                {
                    "text": p["text"][:120],
                    "date": p["date"].isoformat(),
                    "content_type": p["content_type"],
                }
                for p in posts[:5]
            ],
        }
    except Exception as e:
        logger.error(f"Load failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/api/load-dashboard")
async def api_load_dashboard():
    """Load previously saved dashboard from disk."""
    cached = load_dashboard()
    if cached:
        return {"status": "ok", "metrics": cached}
    return {"status": "error", "message": "No saved dashboard found."}


@app.post("/api/analyze")
async def analyze(request: Request):
    """Analyze posts with OpenAI. Two phases: classify + deep pattern analysis."""
    body = await request.json()
    openai_api_key = body.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")

    posts = _cache.get("posts")
    if not posts:
        return {"status": "error", "message": "No posts loaded. Scrape or upload first."}

    if not openai_api_key or openai_api_key.startswith("sk-PONE"):
        for p in posts:
            p.setdefault("category", "other")
            p.setdefault("sentiment", "neutral")
            p.setdefault("topics", [])
            p.setdefault("image_type", "none")
        metrics = compute_metrics(posts)
        save_dashboard(metrics)
        return {"status": "ok", "metrics": metrics}

    # Stream progress via SSE
    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue = asyncio.Queue()

        def progress_cb(current, total, from_cache=False):
            try:
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"current": current, "total": total, "cached": from_cache}
                )
            except Exception:
                pass

        # Phase 1: Classify each post (cached, fast if already done)
        yield f"data: {json.dumps({'phase': 'classify', 'message': 'Phase 1: Classifying posts...'})}\n\n"

        classify_task = loop.run_in_executor(
            None, classify_posts, posts, openai_api_key, progress_cb
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
        _cache["posts"] = classified

        # Compute metrics (needed for pattern analysis input)
        metrics = compute_metrics(classified)

        # Phase 2: Deep pattern analysis (one big call with all data)
        yield f"data: {json.dumps({'phase': 'patterns', 'message': 'Phase 2: Analyzing patterns across all posts...'})}\n\n"

        pattern_task = loop.run_in_executor(
            None, deep_pattern_analysis, classified, metrics, openai_api_key
        )

        # Wait for pattern analysis
        while not pattern_task.done():
            await asyncio.sleep(1)
            yield f"data: {json.dumps({'heartbeat': True})}\n\n"

        pattern_results = await pattern_task
        metrics["pattern_analysis"] = pattern_results

        # Save complete dashboard
        save_dashboard(metrics)

        yield f"data: {json.dumps({'status': 'ok', 'metrics': metrics})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _find_csv_dir(base: Path) -> Path | None:
    if (base / "Rich_Media.csv").exists():
        return base
    for d in base.rglob("Rich_Media.csv"):
        return d.parent
    for d in base.rglob("*.csv"):
        return d.parent
    return None
