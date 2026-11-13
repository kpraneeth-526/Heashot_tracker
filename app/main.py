"""Booth web app.

    uvicorn app.main:app --reload            # empty gallery; register at /enroll
    GALLERY_CSV=gallery/gallery.csv uvicorn app.main:app   # preload a gallery

    # LFW gallery, reusing the embeddings saved by eval.evaluate (fast start)
    GALLERY_CSV=gallery/lfw_gallery.csv GALLERY_CACHE=data/lfw_embeddings.npz \\
        uvicorn app.main:app

    # Demo mode: volunteers who tick "save" at /enroll are kept in this folder
    # and loaded again at the next start. Without it, nothing is ever saved.
    VOLUNTEERS_DIR=gallery/volunteers uvicorn app.main:app
"""

import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import face, roster, volunteers
from app.cleanup import log_event, purge
from app.mailer import describe_server, load_env_file, send_email

# Mail settings (e.g. a Gmail login) can live in a git-ignored .env file.
load_env_file()
from app.matcher import (
    MIN_MARGIN,
    MIN_SCORE,
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
PENDING_TTL_S = 10 * 60  # an unconfirmed photo is forgotten after this long


@dataclass
class Pending:
    """A booth photo waiting for the student to confirm who they are. The
    browser only holds the ticket, so emails and UGA IDs never leave the
    server. The headshot is held here, in memory, until it's emailed."""

    match: GalleryEntry | None
    photo: bytearray = field(default_factory=bytearray)
    candidates: list[GalleryEntry] = field(default_factory=list)
    created: float = field(default_factory=time.monotonic)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gallery = empty_gallery()
    app.state.pending = OrderedDict()
    app.state.volunteers_dir = os.environ.get("VOLUNTEERS_DIR")
    app.state.audit_db = os.environ.get("AUDIT_DB", "audit.db")
    print(f"email: sending through {describe_server()}", flush=True)
    # Show match scores on the booth screen while testing; SHOW_SCORES=0 hides them.
    app.state.show_scores = os.environ.get("SHOW_SCORES", "1") != "0"
    entries = []
    if csv_path := os.environ.get("GALLERY_CSV"):
        entries += load_gallery(csv_path)
    saved = volunteers.csv_path(app.state.volunteers_dir) if app.state.volunteers_dir else None
    if saved:
        saved_entries = load_gallery(saved)
        print(f"gallery: {len(saved_entries)} saved volunteer(s)", flush=True)
        entries += saved_entries
    if entries:
        cache_path = os.environ.get("GALLERY_CACHE")
        embed = cached_embedder(cache_path) if cache_path else embed_photo
        gallery = build_gallery(entries, embed)
        for entry, reason in gallery.skipped:
            print(f"gallery: skipped {entry.photo.name}: {reason}", flush=True)
        print(f"gallery: {len(gallery)} students loaded", flush=True)
        app.state.gallery = gallery
    try:
        yield
    finally:
        for pending in app.state.pending.values():
            purge(pending.photo)
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
    ticket = _new_ticket(request.app, Pending(decision.entry, bytearray(await request.body())))
    scores = _scores(decision.result)
    # Scores only, no names: the log must not record who was at the booth.
    if scores:
        lead = "n/a" if scores["margin"] is None else f"{scores['margin']:.2f}"
        print(f"capture: score {scores['best']:.2f}, lead {lead} -> {decision.reason}", flush=True)

    if decision.entry is None:
        reply = {"ok": True, "ticket": ticket, "match": None, "fallback": decision.reason}
    else:
        # Only the name goes to the browser; the email stays on the server.
        reply = {"ok": True, "ticket": ticket, "match": {"name": decision.entry.name}}
    if request.app.state.show_scores and scores:
        reply["scores"] = scores
    return reply


def _scores(result) -> dict | None:
    """Best score, lead over the runner-up, and what each needs to pass."""
    if result.best is None:
        return None
    return {
        "best": round(result.best_score, 3),
        "margin": round(result.margin, 3) if result.second_score > -1 else None,
        "min_score": MIN_SCORE,
        "min_margin": MIN_MARGIN,
    }


def _new_ticket(app: FastAPI, pending: Pending) -> str:
    waiting = app.state.pending
    ticket = secrets.token_urlsafe(16)
    waiting[ticket] = pending
    # Forget abandoned photos: anything too old, and the oldest beyond MAX_PENDING.
    while waiting:
        oldest = next(iter(waiting.values()))
        if len(waiting) <= MAX_PENDING and time.monotonic() - oldest.created < PENDING_TTL_S:
            break
        purge(waiting.popitem(last=False)[1].photo)
    return ticket


def _pending(request: Request, ticket: str) -> Pending:
    pending = request.app.state.pending.get(ticket)
    if pending is None or time.monotonic() - pending.created >= PENDING_TTL_S:
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
    try:
        send_email(entry.email, entry.name, bytes(pending.photo))
        status = "sent"
    except Exception as e:
        status = "failed"
        print(f"email: failed ({type(e).__name__})", flush=True)
    finally:
        purge(pending.photo)
    log_event(request.app.state.audit_db, status)
    # How they were identified and whether it sent; never who they are.
    print(f"confirm: via {how} -> {status}", flush=True)
    return {"ok": True, "name": entry.name or None, "sent": status == "sent"}


@app.post("/enroll")
async def enroll(request: Request):
    """Register someone at the booth for testing. Normally only the embedding
    is kept, in memory, until the app stops, and the photo is discarded.
    In demo mode (VOLUNTEERS_DIR set) a person who ticks "save" (X-Save: 1)
    has their photo saved there so they're still registered next time.

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
    save = request.headers.get("X-Save") == "1"
    folder = request.app.state.volunteers_dir
    if save and not folder:
        raise HTTPException(400, "Saving is turned off on this booth. Untick the box to register "
                                 "for this session only.")
    quality = await read_photo(request)
    if not quality.ok:
        return {"ok": False, "reason": quality.reason}

    photo = None
    if save:
        photo = volunteers.save(folder, name, email, uga_id, await request.body())
    gallery = request.app.state.gallery
    add_to_gallery(gallery, GalleryEntry(name, email, photo, uga_id), quality.face.normed_embedding)
    return {"ok": True, "name": name, "saved": save, "gallery_size": len(gallery)}
