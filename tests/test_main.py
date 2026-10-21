import io
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import face
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.setattr(face, "detect_faces", lambda img: [object()])
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


def test_capture_counts_faces(client):
    res = client.post("/capture", content=jpeg_bytes(), headers={"Content-Type": "image/jpeg"})
    assert res.status_code == 200
    assert res.json() == {"faces": 1, "width": 640, "height": 480}


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
