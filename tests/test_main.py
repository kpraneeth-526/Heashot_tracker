import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import face, main
from app.cleanup import read_events
from app.main import app
from app.matcher import Gallery, GalleryEntry

ADA = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ALAN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
STRANGER = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own audit log, and email is captured, never sent."""
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "audit.db"))
    outbox = []
    monkeypatch.setattr(main, "send_email",
                        lambda to, name, photo: outbox.append((to, name, photo)))
    return SimpleNamespace(audit_db=tmp_path / "audit.db", outbox=outbox)


@pytest.fixture
def client(monkeypatch):
    """Booth app with the face model replaced: every photo has one good face
    whose embedding is `client.embedding` (default: Ada's)."""
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.delenv("VOLUNTEERS_DIR", raising=False)
    monkeypatch.setattr(face, "detect_faces", lambda img: [object()])
    with TestClient(app) as c:
        c.embedding = ADA
        monkeypatch.setattr(face, "check_quality", lambda img, faces: face.Quality(
            True, face=SimpleNamespace(normed_embedding=c.embedding)))
        yield c


def with_gallery(client):
    client.app.state.gallery = Gallery(
        [GalleryEntry("Ada Lovelace", "ada@example.edu", Path("ada.jpg"), "8110000001"),
         GalleryEntry("Alan Turing", "alan@example.edu", Path("alan.jpg"), "8110000002")],
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


def without_ticket(body):
    """The reply minus its random ticket and the scores (tested separately)."""
    assert body.pop("ticket")
    body.pop("scores", None)
    return body


def test_capture_with_empty_gallery_falls_back(client):
    assert without_ticket(post_photo(client).json()) == {
        "ok": True, "match": None, "fallback": "empty gallery"}


def enroll(client, name, email):
    return client.post("/enroll", content=jpeg_bytes(), headers={
        "Content-Type": "image/jpeg", "X-Name": name, "X-Email": email})


def test_enroll_then_match(client):
    assert enroll(client, "Grace%20Hopper", "grace%40example.edu").json() == {
        "ok": True, "name": "Grace Hopper", "saved": False, "gallery_size": 1}
    assert without_ticket(post_photo(client).json()) == {
        "ok": True, "match": {"name": "Grace Hopper"}}
    assert client.app.state.gallery.entries[0].email == "grace@example.edu"


def test_enroll_requires_name_and_email(client):
    assert enroll(client, "", "grace@example.edu").status_code == 400
    assert enroll(client, "Grace", "not-an-email").status_code == 400


def test_enroll_reports_quality_problem(client, monkeypatch):
    monkeypatch.setattr(face, "check_quality",
                        lambda img, faces: face.Quality(False, "Please step closer to the camera."))
    assert enroll(client, "Grace", "grace@example.edu").json() == {
        "ok": False, "reason": "Please step closer to the camera."}
    assert len(client.app.state.gallery) == 0


def test_capture_matches_and_sends_only_the_name(client):
    with_gallery(client)
    res = post_photo(client)
    assert without_ticket(res.json()) == {"ok": True, "match": {"name": "Ada Lovelace"}}
    assert "@" not in res.text


def test_capture_unknown_person_falls_back(client):
    with_gallery(client)
    client.embedding = STRANGER
    assert without_ticket(post_photo(client).json()) == {
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


def test_pages_and_scripts_are_not_cached(client):
    for path in ("/", "/enroll", "/static/booth.js", "/static/camera.js"):
        assert client.get(path).headers["Cache-Control"] == "no-cache"



# --- Confirm / Not me / lookup / typed email ---------------------------------

def capture_ticket(client, photo=None):
    """Take a photo and return its ticket."""
    return post_photo(client, photo).json()["ticket"]


def test_confirm_auto_match_emails_the_photo(client, isolated, capsys):
    with_gallery(client)
    photo = jpeg_bytes()
    ticket = capture_ticket(client, photo)
    pending = client.app.state.pending[ticket]
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).json() == {
        "ok": True, "name": "Ada Lovelace", "sent": True}
    assert isolated.outbox == [("ada@example.edu", "Ada Lovelace", photo)]
    assert pending.photo == bytearray()  # wiped after sending
    assert [status for _, status in read_events(isolated.audit_db)] == ["sent"]
    out = capsys.readouterr().out
    assert "confirm: via auto-match -> sent" in out
    assert "@" not in out and "Ada" not in out


def test_ticket_works_once(client):
    with_gallery(client)
    ticket = capture_ticket(client)
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).status_code == 200
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).status_code == 404


def test_unknown_ticket(client):
    assert client.post("/lookup", json={"ticket": "nope", "query": "ada"}).status_code == 404
    assert client.post("/confirm", json={"ticket": "nope", "match": True}).status_code == 404


def test_cannot_confirm_a_match_that_did_not_happen(client):
    with_gallery(client)
    client.embedding = STRANGER
    ticket = capture_ticket(client)
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).status_code == 400


def test_not_me_then_lookup_by_name(client, isolated):
    with_gallery(client)
    ticket = capture_ticket(client)  # matched Ada, but it's really Alan
    res = client.post("/lookup", json={"ticket": ticket, "query": "alan tu"})
    assert res.json() == {"results": [{"choice": 0, "name": "Alan Turing"}]}
    assert "@" not in res.text and "811" not in res.text
    assert client.post("/confirm", json={"ticket": ticket, "choice": 0}).json() == {
        "ok": True, "name": "Alan Turing", "sent": True}
    assert [to for to, _, _ in isolated.outbox] == ["alan@example.edu"]


def test_lookup_by_uga_id(client):
    with_gallery(client)
    client.embedding = STRANGER
    ticket = capture_ticket(client)
    res = client.post("/lookup", json={"ticket": ticket, "query": "811-000-0002"})
    assert res.json() == {"results": [{"choice": 0, "name": "Alan Turing"}]}


def test_confirm_choice_must_come_from_lookup(client):
    with_gallery(client)
    ticket = capture_ticket(client)
    assert client.post("/confirm", json={"ticket": ticket, "choice": 0}).status_code == 400


def test_confirm_typed_email(client, capsys):
    client.embedding = STRANGER
    ticket = capture_ticket(client)
    assert client.post("/confirm", json={"ticket": ticket, "email": "not-an-email"}).status_code == 400
    assert client.post("/confirm", json={"ticket": ticket, "email": "new@example.edu"}).json() == {
        "ok": True, "name": None, "sent": True}
    out = capsys.readouterr().out
    assert "via typed email" in out
    assert "new@example.edu" not in out


def test_confirm_needs_exactly_one_answer(client):
    with_gallery(client)
    ticket = capture_ticket(client)
    assert client.post("/confirm", json={"ticket": ticket}).status_code == 400
    assert client.post("/confirm", json={
        "ticket": ticket, "match": True, "email": "a@example.edu"}).status_code == 400


def test_old_tickets_are_dropped(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PENDING", 2)
    first = capture_ticket(client)
    capture_ticket(client)
    capture_ticket(client)
    assert first not in client.app.state.pending
    assert len(client.app.state.pending) == 2


def test_enroll_with_uga_id(client):
    res = client.post("/enroll", content=jpeg_bytes(), headers={
        "Content-Type": "image/jpeg", "X-Name": "Grace", "X-Email": "grace@example.edu",
        "X-Uga-Id": "811-555-0000"})
    assert res.json()["ok"]
    assert client.app.state.gallery.entries[0].uga_id == "8115550000"
    bad = client.post("/enroll", content=jpeg_bytes(), headers={
        "Content-Type": "image/jpeg", "X-Name": "Grace", "X-Email": "grace@example.edu",
        "X-Uga-Id": "abc"})
    assert bad.status_code == 400


def fake_face(monkeypatch, embedding):
    monkeypatch.setattr(face, "detect_faces", lambda img: [object()])
    monkeypatch.setattr(face, "check_quality", lambda img, faces: face.Quality(
        True, face=SimpleNamespace(normed_embedding=embedding)))


def enroll_and_save(client, name, email, save="1"):
    return client.post("/enroll", content=jpeg_bytes(), headers={
        "Content-Type": "image/jpeg", "X-Name": name, "X-Email": email, "X-Save": save})


def test_saved_volunteer_is_back_after_restart(tmp_path, monkeypatch):
    folder = tmp_path / "volunteers"
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.setenv("VOLUNTEERS_DIR", str(folder))
    monkeypatch.setattr(main, "embed_photo", lambda path: ALAN)  # model stand-in at startup
    fake_face(monkeypatch, ALAN)

    with TestClient(app) as c:
        res = enroll_and_save(c, "Alan Turing", "alan@example.edu")
        assert res.json()["saved"] is True
        assert len(list(folder.glob("*.jpg"))) == 1

    with TestClient(app) as c:  # a fresh start: memory is empty, the folder is not
        assert len(c.app.state.gallery) == 1
        assert post_photo(c).json()["match"] == {"name": "Alan Turing"}


def test_unticked_volunteer_is_not_saved(tmp_path, monkeypatch):
    folder = tmp_path / "volunteers"
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.setenv("VOLUNTEERS_DIR", str(folder))
    fake_face(monkeypatch, ADA)
    with TestClient(app) as c:
        assert enroll_and_save(c, "Ada", "ada@example.edu", save="0").json()["saved"] is False
    assert not folder.exists()


def test_save_refused_when_booth_not_in_demo_mode(client):
    res = enroll_and_save(client, "Ada", "ada@example.edu")
    assert res.status_code == 400
    assert "Saving is turned off" in res.json()["detail"]
    assert len(client.app.state.gallery) == 0


def test_capture_includes_scores(client, capsys):
    with_gallery(client)
    body = post_photo(client).json()
    assert body["scores"] == {"best": 1.0, "margin": 1.0, "min_score": main.MIN_SCORE,
                              "min_margin": main.MIN_MARGIN}
    log = capsys.readouterr().out
    assert "capture: score 1.00, lead 1.00 -> match" in log
    assert "Ada" not in log  # scores only, never who it was


def test_scores_can_be_hidden(monkeypatch):
    monkeypatch.setenv("SHOW_SCORES", "0")
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.delenv("VOLUNTEERS_DIR", raising=False)
    fake_face(monkeypatch, ADA)
    with TestClient(app) as c:
        with_gallery(c)
        assert "scores" not in post_photo(c).json()


def test_no_scores_with_empty_gallery(client):
    assert "scores" not in post_photo(client).json()


def test_failed_email_is_logged_and_photo_still_wiped(client, isolated, monkeypatch, capsys):
    with_gallery(client)
    def broken(to, name, photo):
        raise ConnectionRefusedError("ada@example.edu unreachable")
    monkeypatch.setattr(main, "send_email", broken)
    ticket = capture_ticket(client, jpeg_bytes())
    pending = client.app.state.pending[ticket]
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).json()["sent"] is False
    assert pending.photo == bytearray()
    assert [status for _, status in read_events(isolated.audit_db)] == ["failed"]
    out = capsys.readouterr().out
    assert "email: failed (ConnectionRefusedError)" in out
    assert "@" not in out  # the error text contains the address; it must not be logged


def test_audit_log_has_only_time_and_status(client, isolated):
    with_gallery(client)
    ticket = capture_ticket(client, jpeg_bytes())
    client.post("/confirm", json={"ticket": ticket, "match": True})
    [(at, status)] = read_events(isolated.audit_db)
    assert status == "sent" and at.endswith("+00:00")
    raw = isolated.audit_db.read_bytes()
    assert b"ada" not in raw.lower() and b"Lovelace" not in raw


def test_abandoned_photo_expires_and_is_wiped(client, monkeypatch):
    with_gallery(client)
    ticket = capture_ticket(client, jpeg_bytes())
    pending = client.app.state.pending[ticket]
    monkeypatch.setattr(main, "PENDING_TTL_S", 0)
    assert client.post("/confirm", json={"ticket": ticket, "match": True}).status_code == 404
    capture_ticket(client, jpeg_bytes())  # the next photo clears old ones out
    assert ticket not in client.app.state.pending
    assert pending.photo == bytearray()


def test_dropped_tickets_wipe_their_photos(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PENDING", 1)
    first = capture_ticket(client, jpeg_bytes())
    pending = client.app.state.pending[first]
    capture_ticket(client, jpeg_bytes())
    assert pending.photo == bytearray()


def test_shutdown_wipes_waiting_photos(monkeypatch):
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.delenv("VOLUNTEERS_DIR", raising=False)
    fake_face(monkeypatch, ADA)
    with TestClient(app) as c:
        pending = c.app.state.pending[capture_ticket(c, jpeg_bytes())]
        assert len(pending.photo) > 0
    assert pending.photo == bytearray()
