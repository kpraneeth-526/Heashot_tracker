import io
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import face
from app.main import app
from app.matcher import Gallery, GalleryEntry


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    good_face = SimpleNamespace(normed_embedding=np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(face, "check_quality", lambda img: good_face)
    with TestClient(app) as c:
        yield c


def jpeg_bytes(width=640, height=480):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(buf, format="JPEG")
    return buf.getvalue()


def test_booth_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Headshot Booth" in res.text


def post_photo(client):
    return client.post("/capture", content=jpeg_bytes(), headers={"Content-Type": "image/jpeg"})


def test_capture_accepts_good_photo(client):
    res = post_photo(client)
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_capture_reports_quality_problem(client, monkeypatch):
    def blurry(img):
        raise face.QualityError("The photo is blurry. Hold still and retake.")

    monkeypatch.setattr(face, "check_quality", blurry)
    res = post_photo(client)
    assert res.status_code == 200
    assert res.json() == {"ok": False, "reason": "The photo is blurry. Hold still and retake."}


def test_capture_logs_matched_name_not_email(client, capsys):
    entries = [
        GalleryEntry("Ada Lovelace", "ada@example.edu", None),
        GalleryEntry("Alan Turing", "alan@example.edu", None),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    client.app.state.gallery = Gallery(entries, embeddings)

    assert post_photo(client).json() == {"ok": True}
    out = capsys.readouterr().out
    assert "matched Ada Lovelace" in out
    assert "@" not in out


def test_capture_rejects_empty_and_garbage(client):
    assert client.post("/capture", content=b"").status_code == 400
    assert client.post("/capture", content=b"not an image").status_code == 400


def test_capture_writes_nothing_to_disk(client):
    tmpdir = tempfile.gettempdir()
    before = set(os.listdir(tmpdir))
    cwd_before = set(os.listdir("."))

    # Larger than the 1 MB threshold at which multipart uploads would spool to disk.
    big = jpeg_bytes(4000, 3000) + os.urandom(2 * 1024 * 1024)
    res = client.post("/capture", content=big, headers={"Content-Type": "image/jpeg"})

    assert res.status_code == 200
    assert set(os.listdir(tmpdir)) <= before
    assert set(os.listdir(".")) <= cwd_before
