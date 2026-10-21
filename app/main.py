"""Booth web app.

    uvicorn app.main:app --reload            # capture page only
    GALLERY_CSV=gallery/gallery.csv uvicorn app.main:app   # also load gallery
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import face
from app.matcher import build_gallery, load_gallery, wipe_gallery

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gallery = None
    csv_path = os.environ.get("GALLERY_CSV")
    if csv_path:
        gallery = build_gallery(load_gallery(csv_path))
        for entry, reason in gallery.skipped:
            print(f"gallery: skipped {entry.photo.name}: {reason}")
        print(f"gallery: {len(gallery)} students loaded")
        app.state.gallery = gallery
    try:
        yield
    finally:
        if app.state.gallery is not None:
            wipe_gallery(app.state.gallery)
            app.state.gallery = None


app = FastAPI(title="Headshot Auto-Match", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def booth_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/capture")
async def capture_photo(request: Request):
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

    faces = face.detect_faces(img)
    height, width = img.shape[:2]
    return {"faces": len(faces), "width": width, "height": height}
