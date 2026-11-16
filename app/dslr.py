"""A DSLR plugged in by USB, controlled by the booth (via gphoto2).

    DSLR=1 uvicorn app.main:app

The booth page shows the camera's live view so people can centre themselves,
and "Take photo" fires the camera. Photos taken with the camera's own shutter
button are picked up too (checked with every live view frame). Either way
the photo is downloaded straight into
memory: the camera keeps it in its own RAM rather than on the SD card, and
the Mac never writes it to disk. Tested with a Canon T3i (EOS 600D).
"""

import io
import subprocess
import threading
import time
from collections import deque

from PIL import ExifTags, Image

try:
    import gphoto2 as gp
except ImportError:  # the booth still runs with a webcam without it
    gp = None

# Applied when the camera connects. ISO Auto lets the camera raise the ISO
# indoors instead of using slow shutter speeds, which blur photos.
SETTINGS = {
    "capturetarget": "Internal RAM",  # not the SD card
    "imageformat": "L",  # JPEG, Large, Fine (RAW can't be read or emailed)
    "iso": "Auto",
}
CONNECT_ATTEMPTS = 5  # macOS sometimes grabs the camera back first
MAX_SHUTTER_PHOTOS = 3  # shutter-button photos waiting for the booth screen
REPORTED = ("autoexposuremode", "iso", "imageformat", "focusmode", "lensname", "batterylevel")


class CameraError(Exception):
    """Something the booth staff can fix: plugged out, switched off, ..."""


class Dslr:
    def __init__(self):
        self._camera = None
        self._lock = threading.Lock()  # one camera command at a time
        self._shutter = deque(maxlen=MAX_SHUTTER_PHOTOS)

    def connect(self) -> str:
        """Connect (if not already) and describe the camera for the log."""
        with self._lock:
            camera = self._connected()
            config = camera.get_config()
            found = {}
            for name in REPORTED:
                ok, widget = gp.gp_widget_get_child_by_name(config, name)
                if ok >= gp.GP_OK:
                    found[name] = widget.get_value()
            return ", ".join(f"{k} {v}" for k, v in found.items())

    def frame(self) -> bytes:
        """One live view frame (a small JPEG, about 1056x704 on a T3i).
        Also collects any photos taken with the camera's shutter button."""
        def preview(camera):
            frame = _data(camera.capture_preview())
            self._collect_shutter_photos(camera)
            return frame
        return self._run(preview, "The camera stopped responding")

    def shutter_waiting(self) -> bool:
        return bool(self._shutter)

    def shutter_photo(self) -> bytes | None:
        """The oldest photo taken with the shutter button, or None."""
        with self._lock:
            return self._shutter.popleft() if self._shutter else None

    def take(self) -> bytes:
        """Take a full-size photo and return the JPEG."""
        def shoot(camera):
            path = camera.capture(gp.GP_CAPTURE_IMAGE)
            return _download(camera, path.folder, path.name)
        # The T3i sometimes refuses the first shot after a stretch of live
        # view; a fresh connection then works, so try once more.
        return self._run(shoot, "The camera couldn't take the photo. Stand at least "
                                "1.5 m (5 ft) from the camera, with light on your face, "
                                "and try again", attempts=2)

    def close(self) -> None:
        with self._lock:
            self._shutter.clear()
            self._disconnect()

    def _collect_shutter_photos(self, camera):
        # The camera reports each photo it takes as a "file added" event.
        for _ in range(50):  # never spin forever on a chatty camera
            kind, path = camera.wait_for_event(1)
            if kind == gp.GP_EVENT_TIMEOUT:
                return
            if kind != gp.GP_EVENT_FILE_ADDED:
                continue
            try:
                photo = _download(camera, path.folder, path.name)
            except gp.GPhoto2Error as e:
                print(f"camera: couldn't download a shutter photo: {e}", flush=True)
                continue
            if path.name.lower().endswith((".jpg", ".jpeg")):  # not RAW
                self._shutter.append(photo)

    def _run(self, command, problem: str, attempts: int = 1):
        with self._lock:
            for attempt in range(1, attempts + 1):
                camera = self._connected()
                try:
                    return command(camera)
                except gp.GPhoto2Error as e:
                    # Start afresh: this also recovers after a cable is
                    # replugged or the camera wakes from sleep.
                    self._disconnect()
                    print(f"camera: {e} (attempt {attempt} of {attempts})", flush=True)
                    if attempt == attempts:
                        raise CameraError(f"{problem}.") from e

    def _connected(self):
        if self._camera is not None:
            return self._camera
        if gp is None:
            raise CameraError("gphoto2 isn't installed: run .venv/bin/pip install gphoto2")
        camera = None
        for _ in range(CONNECT_ATTEMPTS):
            # macOS's camera helper (ptpcamerad, for Photos and Image Capture)
            # grabs any camera that's plugged in, and macOS restarts it the
            # moment it's stopped. Stop it hard and claim the camera at once.
            subprocess.run(["killall", "-9", "ptpcamerad", "PTPCamera"], capture_output=True)
            try:
                camera = gp.Camera()
                camera.init()
                break
            except gp.GPhoto2Error:
                camera = None
                time.sleep(0.3)
        if camera is None:
            raise CameraError("No camera found. Check it's plugged in by USB and "
                              "switched on.")
        self._camera = camera
        self._apply_settings()
        return camera

    def _apply_settings(self):
        config = self._camera.get_config()
        for name, value in SETTINGS.items():
            ok, widget = gp.gp_widget_get_child_by_name(config, name)
            if ok < gp.GP_OK:
                continue
            try:
                widget.set_value(value)
                self._camera.set_single_config(name, widget)
            except gp.GPhoto2Error:
                print(f"camera: couldn't set {name} to {value}", flush=True)

    def _disconnect(self):
        if self._camera is not None:
            try:
                self._camera.exit()
            except gp.GPhoto2Error:
                pass
            self._camera = None


def _download(camera, folder: str, name: str) -> bytes:
    """Copy a photo off the camera, then delete it there."""
    photo = _data(camera.file_get(folder, name, gp.GP_FILE_TYPE_NORMAL))
    try:
        camera.file_delete(folder, name)
    except gp.GPhoto2Error:
        pass  # photos in the camera's RAM are usually gone already
    return photo


def _data(camera_file) -> bytes:
    return bytes(memoryview(camera_file.get_data_and_size()))


def exposure(photo: bytes) -> str:
    """Shutter speed, aperture and ISO from the photo's EXIF, e.g.
    "1/60s f/5.6 ISO 1600", for spotting why a photo is dark or blurry."""
    try:
        exif = Image.open(io.BytesIO(photo)).getexif().get_ifd(ExifTags.IFD.Exif)
    except Exception:
        return "unknown exposure"
    parts = []
    if t := exif.get(ExifTags.Base.ExposureTime):
        t = float(t)
        parts.append(f"1/{round(1 / t)}s" if 0 < t < 1 else f"{t:g}s")
    if f := exif.get(ExifTags.Base.FNumber):
        parts.append(f"f/{float(f):g}")
    if iso := exif.get(ExifTags.Base.ISOSpeedRatings):
        parts.append(f"ISO {iso}")
    return " ".join(parts) or "unknown exposure"
