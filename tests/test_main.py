import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import face
from app.main import app
from app.matcher import Gallery, GalleryEntry

ADA = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ALAN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
STRANGER = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@pytest.fixture
def client(monkeypatch):
    """Booth app with the face model replaced: every photo has one good face
    whose embedding is `client.embedding` (default: Ada's)."""
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.setattr(face, "detect_faces", lambda img: [object()])
    with TestClient(app) as c:
        c.embedding = ADA
        monkeypatch.setattr(face, "check_quality", lambda img, faces: face.Quality(
            True, face=SimpleNamespace(normed_embedding=c.embedding)))
        yield c


def with_gallery(client):
    client.app.state.gallery = Gallery(
        [GalleryEntry("Ada Lovelace", "ada@example.edu", Path("ada.jpg")),
         GalleryEntry("Alan Turing", "alan@example.edu", Path("alan.jpg"))],
        np.stack([ADA, ALAN]),
    )


def post_photo(client, data=None):
    return client.post("/capture", content=data or jpeg_bytes(),
                       headers={"Content-Type": "image/jpeg"})


def jpeg_bytes(width=640, height=480):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(buf, format="JPEG")
    return buf.getvalue()


def test_booth_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Headshot Booth" in res.text


def test_capture_reports_quality_problem(client, monkeypatch):
    monkeypatch.setattr(face, "check_quality",
                        lambda img, faces: face.Quality(False, "Please hold still."))
    assert post_photo(client).json() == {"ok": False, "reason": "Please hold still."}


def test_capture_without_gallery_falls_back(client):
    assert post_photo(client).json() == {
        "ok": True, "match": None, "fallback": "no gallery loaded"}


def test_capture_matches_and_sends_only_the_name(client):
    with_gallery(client)
    res = post_photo(client)
    assert res.json() == {"ok": True, "match": {"name": "Ada Lovelace"}}
    assert "@" not in res.text


def test_capture_unknown_person_falls_back(client):
    with_gallery(client)
    client.embedding = STRANGER
    assert post_photo(client).json() == {
        "ok": True, "match": None, "fallback": "score too low"}


def test_capture_rejects_empty_and_garbage(client):
    assert client.post("/capture", content=b"").status_code == 400
    assert client.post("/capture", content=b"not an image").status_code == 400


def test_capture_writes_nothing_to_disk(client):
    tmpdir = tempfile.gettempdir()
    before = set(os.listdir(tmpdir))
    cwd_before = set(os.listdir("."))

    # Larger than the 1 MB threshold at which multipart uploads would spool to disk.
    big = jpeg_bytes(4000, 3000) + os.urandom(2 * 1024 * 1024)
    res = post_photo(client, big)

    assert res.status_code == 200
    assert set(os.listdir(tmpdir)) <= before
    assert set(os.listdir(".")) <= cwd_before
