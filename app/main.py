"""Booth web app.

    uvicorn app.main:app --reload            # empty gallery; register at /enroll
    GALLERY_CSV=gallery/gallery.csv uvicorn app.main:app   # preload a gallery
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import face
from app.matcher import (
    GalleryEntry,
    add_to_gallery,
    build_gallery,
    decide,
    empty_gallery,
    load_gallery,
    match,
    wipe_gallery,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gallery = empty_gallery()
    csv_path = os.environ.get("GALLERY_CSV")
    if csv_path:
        gallery = build_gallery(load_gallery(csv_path))
        for entry, reason in gallery.skipped:
            print(f"gallery: skipped {entry.photo.name}: {reason}", flush=True)
        print(f"gallery: {len(gallery)} students loaded", flush=True)
        app.state.gallery = gallery
    try:
        yield
    finally:
        wipe_gallery(app.state.gallery)


app = FastAPI(title="Headshot Auto-Match", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    # Make browsers re-check pages and scripts on every load, so an updated
    # booth.js is never mixed with an old cached one.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def booth_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/enroll")
def enroll_page():
    return FileResponse(STATIC_DIR / "enroll.html")


async def read_photo(request: Request):
    # The photo arrives as the raw request body rather than multipart form
    # data: multipart uploads over 1 MB get spooled to a temp file on disk,
    # and photos must never touch disk.
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "photo too large")
    try:
        img = face.decode_image(data)
    except face.FaceError as e:
        raise HTTPException(400, str(e))
    return face.check_quality(img, face.detect_faces(img))


@app.post("/capture")
async def capture_photo(request: Request):
    quality = await read_photo(request)
    if not quality.ok:
        return {"ok": False, "reason": quality.reason}

    decision = decide(match(quality.face.normed_embedding, request.app.state.gallery))
    if decision.entry is None:
        return {"ok": True, "match": None, "fallback": decision.reason}
    # Only the name goes to the browser; the email stays on the server.
    return {"ok": True, "match": {"name": decision.entry.name}}


@app.post("/enroll")
async def enroll(request: Request):
    """Register someone at the booth for testing. Only the embedding is kept,
    in memory, until the app stops; the photo itself is discarded.

    Name and email come in headers (URL-encoded) rather than the query string,
    which would put them in the server's access log.
    """
    name = unquote(request.headers.get("X-Name", "")).strip()
    email = unquote(request.headers.get("X-Email", "")).strip()
    if not name or "@" not in email:
        raise HTTPException(400, "Please enter a name and a valid email.")
    quality = await read_photo(request)
    if not quality.ok:
        return {"ok": False, "reason": quality.reason}

    gallery = request.app.state.gallery
    add_to_gallery(gallery, GalleryEntry(name, email, None), quality.face.normed_embedding)
    return {"ok": True, "name": name, "gallery_size": len(gallery)}
