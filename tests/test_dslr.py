import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import dslr, main
from app.main import app
from tests.test_main import isolated, jpeg_bytes  # noqa: F401


class FakeDslr:
    def __init__(self):
        self.problem = None
        self.closed = False
        self.shutter = []

    def shutter_waiting(self):
        return bool(self.shutter)

    def shutter_photo(self):
        return self.shutter.pop(0) if self.shutter else None

    def connect(self):
        return "iso Auto"

    def frame(self):
        if self.problem:
            raise dslr.CameraError(self.problem)
        return b"\xff\xd8frame"

    def take(self):
        if self.problem:
            raise dslr.CameraError(self.problem)
        return jpeg_bytes(5184, 3456)

    def close(self):
        self.closed = True


@pytest.fixture
def camera(monkeypatch):
    fake = FakeDslr()
    monkeypatch.setenv("DSLR", "1")
    monkeypatch.delenv("GALLERY_CSV", raising=False)
    monkeypatch.delenv("VOLUNTEERS_DIR", raising=False)
    monkeypatch.setattr(dslr, "Dslr", lambda: fake)
    with TestClient(app) as c:
        c.fake = fake
        yield c
    assert fake.closed


def test_dslr_turned_off():
    with TestClient(app) as c:  # DSLR=0 (see conftest.py)
        assert c.get("/camera").json() == {"dslr": False}
        assert c.get("/camera/frame").status_code == 404
        assert c.post("/camera/take").status_code == 404


def test_live_view_frame(camera):
    assert camera.get("/camera").json() == {"dslr": True}
    res = camera.get("/camera/frame")
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content == b"\xff\xd8frame"


def test_take_returns_full_size_photo(camera):
    res = camera.post("/camera/take")
    assert Image.open(io.BytesIO(res.content)).size == (5184, 3456)


def test_camera_problem_is_explained(camera):
    camera.fake.problem = "No camera found."
    for res in (camera.get("/camera/frame"), camera.post("/camera/take")):
        assert res.status_code == 503
        assert res.json()["detail"] == "No camera found."


def test_booth_starts_without_camera(monkeypatch, capsys):
    class Missing(FakeDslr):
        def connect(self):
            raise dslr.CameraError("No camera found.")
    monkeypatch.setenv("DSLR", "1")
    monkeypatch.setattr(dslr, "Dslr", Missing)
    with TestClient(app) as c:
        # The page falls back to the webcam.
        assert c.get("/camera").json() == {"dslr": False, "reason": "No camera found."}
    assert "no DSLR found, so the booth will use the webcam" in capsys.readouterr().out


def test_exposure_from_exif():
    img = Image.new("RGB", (10, 10))
    exif = Image.Exif()
    exif_ifd = exif.get_ifd(dslr.ExifTags.IFD.Exif)
    exif_ifd[dslr.ExifTags.Base.ExposureTime] = 1 / 60
    exif_ifd[dslr.ExifTags.Base.FNumber] = 5.6
    exif_ifd[dslr.ExifTags.Base.ISOSpeedRatings] = 1600
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert dslr.exposure(buf.getvalue()) == "1/60s f/5.6 ISO 1600"
    assert dslr.exposure(b"junk") == "unknown exposure"


def test_full_size_dslr_photo_accepted():
    assert main.MAX_UPLOAD_BYTES >= 18 * 1024 * 1024


def test_refused_photo_is_retried_once(monkeypatch):
    class GPhoto2Error(Exception):
        pass

    class Path:
        folder, name = "/", "capt0000.jpg"

    class File:
        def get_data_and_size(self):
            return b"\xff\xd8photo"

    class Camera:
        refusals = 1

        def capture(self, kind):
            if Camera.refusals:
                Camera.refusals -= 1
                raise GPhoto2Error("[-110] I/O in progress")
            return Path()

        def file_get(self, folder, name, kind):
            return File()

        def file_delete(self, folder, name):
            pass

        def exit(self):
            pass

    fake_gp = type("gp", (), {"GPhoto2Error": GPhoto2Error, "GP_CAPTURE_IMAGE": 0,
                              "GP_FILE_TYPE_NORMAL": 1})
    monkeypatch.setattr(dslr, "gp", fake_gp)
    camera = dslr.Dslr()
    connections = []
    monkeypatch.setattr(camera, "_connected", lambda: connections.append(1) or Camera())
    assert camera.take() == b"\xff\xd8photo"
    assert len(connections) == 2  # reconnected for the second try

    Camera.refusals = 2
    with pytest.raises(dslr.CameraError, match="couldn't take the photo"):
        camera.take()


def test_shutter_button_photo(camera):
    assert "x-shutter" not in camera.get("/camera/frame").headers
    assert camera.post("/camera/shutter").status_code == 204

    camera.fake.shutter.append(jpeg_bytes(5184, 3456))
    assert camera.get("/camera/frame").headers["x-shutter"] == "1"
    res = camera.post("/camera/shutter")
    assert Image.open(io.BytesIO(res.content)).size == (5184, 3456)
    assert "x-shutter" not in camera.get("/camera/frame").headers


def test_shutter_photos_collected_from_camera_events(monkeypatch):
    class Path:
        def __init__(self, name):
            self.folder, self.name = "/", name

    class File:
        def __init__(self, name):
            self.name = name

        def get_data_and_size(self):
            return self.name.encode()

    class Camera:
        deleted = []

        def __init__(self):
            self.events = [(5, None), (2, Path("capt0000.jpg")), (2, Path("capt0001.cr2")),
                           (2, Path("capt0002.JPG"))]

        def capture_preview(self):
            return File("frame")

        def wait_for_event(self, ms):
            return self.events.pop(0) if self.events else (1, None)

        def file_get(self, folder, name, kind):
            return File(name)

        def file_delete(self, folder, name):
            Camera.deleted.append(name)

    fake_gp = type("gp", (), {"GPhoto2Error": Exception, "GP_FILE_TYPE_NORMAL": 1,
                              "GP_EVENT_TIMEOUT": 1, "GP_EVENT_FILE_ADDED": 2})
    monkeypatch.setattr(dslr, "gp", fake_gp)
    camera = dslr.Dslr()
    fake = Camera()
    monkeypatch.setattr(camera, "_connected", lambda: fake)
    assert camera.frame() == b"frame"
    assert camera.shutter_waiting()
    assert camera.shutter_photo() == b"capt0000.jpg"
    assert camera.shutter_photo() == b"capt0002.JPG"  # the RAW file is skipped
    assert camera.shutter_photo() is None
    # Every file is deleted from the camera, RAW included.
    assert Camera.deleted == ["capt0000.jpg", "capt0001.cr2", "capt0002.JPG"]


def test_dslr_used_automatically_without_setting(monkeypatch):
    monkeypatch.delenv("DSLR")
    monkeypatch.setattr(dslr, "Dslr", FakeDslr)
    with TestClient(app) as c:
        assert c.get("/camera").json() == {"dslr": True}


def test_camera_plugged_in_after_start(monkeypatch):
    class Late(FakeDslr):
        plugged_in = False

        def connect(self):
            if not Late.plugged_in:
                raise dslr.CameraError("No camera found.")
            return "iso Auto"
    monkeypatch.setenv("DSLR", "1")
    monkeypatch.setattr(dslr, "Dslr", Late)
    with TestClient(app) as c:
        assert c.get("/camera").json()["dslr"] is False  # webcam
        Late.plugged_in = True
        assert c.get("/camera").json() == {"dslr": True}  # after reloading the page
