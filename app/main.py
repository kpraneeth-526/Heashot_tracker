"""Booth web app.

    uvicorn app.main:app --reload            # empty gallery; register at /enroll
    GALLERY_CSV=gallery/gallery.csv uvicorn app.main:app   # preload a gallery

    # LFW gallery, reusing the embeddings saved by eval.evaluate (fast start)
    GALLERY_CSV=gallery/lfw_gallery.csv GALLERY_CACHE=data/lfw_embeddings.npz \\
        uvicorn app.main:app
"""

import os
import secrets
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import face, roster
from app.matcher import (
    GalleryEntry,
    embed_photo,
    add_to_gallery,
    build_gallery,
    cached_embedder,
    decide,
    empty_gallery,
    load_gallery,
    match,
    normalize_id,
    wipe_gallery,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PENDING = 50  # captures waiting for Confirm; the oldest is dropped beyond this


@dataclass
class Pending:
    """A booth photo waiting for the student to confirm who they are. The
    browser only holds the ticket, so emails and UGA IDs never leave the
    server. (Week 3: the headshot itself will be held here until it's sent.)"""

    match: GalleryEntry | None
    candidates: list[GalleryEntry] = field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gallery = empty_gallery()
    app.state.pending = OrderedDict()
    csv_path = os.environ.get("GALLERY_CSV")
    if csv_path:
        cache_path = os.environ.get("GALLERY_CACHE")
        embed = cached_embedder(cache_path) if cache_path else embed_photo
        gallery = build_gallery(load_gallery(csv_path), embed)
        for entry, reason in gallery.skipped:
            print(f"gallery: skipped {entry.photo.name}: {reason}", flush=True)
        print(f"gallery: {len(gallery)} students loaded", flush=True)
        app.state.gallery = gallery
    try:
        yield
    finally:
        app.state.pending.clear()
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
    ticket = _new_ticket(request.app, Pending(decision.entry))
    if decision.entry is None:
        return {"ok": True, "ticket": ticket, "match": None, "fallback": decision.reason}
    # Only the name goes to the browser; the email stays on the server.
    return {"ok": True, "ticket": ticket, "match": {"name": decision.entry.name}}


def _new_ticket(app: FastAPI, pending: Pending) -> str:
    ticket = secrets.token_urlsafe(16)
    app.state.pending[ticket] = pending
    while len(app.state.pending) > MAX_PENDING:
        app.state.pending.popitem(last=False)
    return ticket


def _pending(request: Request, ticket: str) -> Pending:
    pending = request.app.state.pending.get(ticket)
    if pending is None:
        raise HTTPException(404, "This photo has expired. Please retake it.")
    return pending


# Lookup and confirm take JSON bodies rather than query strings so names, IDs
# and emails stay out of the server's access log.
class LookupRequest(BaseModel):
    ticket: str
    query: str


@app.post("/lookup")
def lookup(body: LookupRequest, request: Request):
    """Find a student by UGA ID or name. Returns names only; the student
    picks one by its position in the list."""
    pending = _pending(request, body.ticket)
    pending.candidates = roster.search(roster.roster(request.app.state.gallery), body.query)
    return {"results": [{"choice": i, "name": e.name} for i, e in enumerate(pending.candidates)]}


class ConfirmRequest(BaseModel):
    ticket: str
    match: bool = False  # "Yes, that's me" on the auto-match
    choice: int | None = None  # picked from the /lookup results
    email: str | None = None  # typed by hand: not on the roster


@app.post("/confirm")
def confirm(body: ConfirmRequest, request: Request):
    pending = _pending(request, body.ticket)
    if sum((body.match, body.choice is not None, body.email is not None)) != 1:
        raise HTTPException(400, "Choose exactly one of match, choice or email.")

    if body.match:
        if pending.match is None:
            raise HTTPException(400, "There was no match to confirm.")
        entry, how = pending.match, "auto-match"
    elif body.choice is not None:
        if not 0 <= body.choice < len(pending.candidates):
            raise HTTPException(400, "Please search again and pick your name.")
        entry, how = pending.candidates[body.choice], "lookup"
    else:
        email = body.email.strip()
        if "@" not in email:
            raise HTTPException(400, "Please enter a valid email.")
        entry, how = GalleryEntry("", email, None), "typed email"

    del request.app.state.pending[body.ticket]
    # Week 3: send the headshot to entry.email here. Emails are never logged.
    print(f"confirm: {entry.name or '(not on roster)'} via {how}", flush=True)
    return {"ok": True, "name": entry.name or None}


@app.post("/enroll")
async def enroll(request: Request):
    """Register someone at the booth for testing. Only the embedding is kept,
    in memory, until the app stops; the photo itself is discarded.

    Name and email come in headers (URL-encoded) rather than the query string,
    which would put them in the server's access log.
    """
    name = unquote(request.headers.get("X-Name", "")).strip()
    email = unquote(request.headers.get("X-Email", "")).strip()
    uga_id = normalize_id(unquote(request.headers.get("X-Uga-Id", "")))
    if not name or "@" not in email:
        raise HTTPException(400, "Please enter a name and a valid email.")
    if uga_id and not uga_id.isdigit():
        raise HTTPException(400, "A UGA ID can only contain digits.")
    quality = await read_photo(request)
    if not quality.ok:
        return {"ok": False, "reason": quality.reason}

    gallery = request.app.state.gallery
    add_to_gallery(gallery, GalleryEntry(name, email, None, uga_id), quality.face.normed_embedding)
    return {"ok": True, "name": name, "gallery_size": len(gallery)}
