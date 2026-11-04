"""Face detection and embedding using InsightFace (buffalo_l)."""

import io
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
# Faces smaller than this fraction of the largest face are treated as
# background (people walking past the booth) and ignored.
BACKGROUND_FACE_RATIO = 0.5


@dataclass(frozen=True)
class QualityThresholds:
    min_det_score: float = 0.6
    min_face_px: int = 80  # width of the face box in pixels
    max_yaw: float = 0.25  # nose offset from eye midpoint, in eye-distances
    max_roll_deg: float = 15.0  # head tilt
    min_brightness: float = 50.0  # mean grey level of the face, 0-255
    max_brightness: float = 210.0
    min_sharpness: float = 40.0  # variance of the Laplacian on a 112x112 face crop


@dataclass
class Quality:
    ok: bool
    reason: str | None = None  # short message to show at the booth
    face: object | None = None  # the face that was checked
    metrics: dict = field(default_factory=dict)


class FaceError(ValueError):
    """The image can't be used: undecodable, or not exactly one face."""


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so the rest of the app (and the tests) don't need the
    # model loaded. First use downloads buffalo_l to ~/.insightface/models.
    from insightface.app import FaceAnalysis

    model = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    model.prepare(ctx_id=-1, det_size=DET_SIZE)
    return model


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR array, honouring EXIF rotation."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise FaceError(f"could not decode image: {e}") from e
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def load_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def detect_faces(img: np.ndarray) -> list:
    """Return InsightFace Face objects (bbox, det_score, normed_embedding, ...)."""
    return _model().get(img)


def foreground_faces(faces: list) -> list:
    """Drop faces much smaller than the largest one (people in the background)."""
    if not faces:
        return []
    widths = [f.bbox[2] - f.bbox[0] for f in faces]
    biggest = max(widths)
    return [f for f, w in zip(faces, widths) if w >= biggest * BACKGROUND_FACE_RATIO]


def face_pose(face) -> tuple[float, float]:
    """(yaw, roll_degrees) estimated from the five landmarks.

    yaw is how far the nose sits from the midpoint between the eyes, measured
    in eye-distances: 0 when facing the camera, about +/-0.5 at a 45 degree turn.
    """
    left_eye, right_eye, nose = (np.asarray(p, dtype=float) for p in face.kps[:3])
    eye_vec = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_vec)) or 1.0
    yaw = (nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist
    roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))
    return yaw, roll


def _face_crop(img: np.ndarray, face) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    return img[y1:y2, x1:x2]


def check_quality(
    img: np.ndarray, faces: list, t: QualityThresholds = QualityThresholds()
) -> Quality:
    """Is this photo good enough to match? Checks run in the order a person
    can fix them, and the first failure is reported."""
    faces = foreground_faces(faces)
    if not faces:
        return Quality(False, "No face found. Please look at the camera.")
    if len(faces) > 1:
        return Quality(False, "Only one person in the photo, please.")

    face = faces[0]
    yaw, roll = face_pose(face)
    crop = cv2.cvtColor(_face_crop(img, face), cv2.COLOR_BGR2GRAY)
    width = float(face.bbox[2] - face.bbox[0])
    metrics = {
        "det_score": float(face.det_score),
        "face_px": width,
        "yaw": yaw,
        "roll_deg": roll,
        "brightness": float(crop.mean()) if crop.size else 0.0,
        # Resize first so sharpness doesn't depend on how big the face is.
        "sharpness": float(cv2.Laplacian(cv2.resize(crop, (112, 112)), cv2.CV_64F).var())
        if crop.size
        else 0.0,
    }

    def fail(reason):
        return Quality(False, reason, face, metrics)

    if metrics["det_score"] < t.min_det_score:
        return fail("Face not clear. Please look at the camera.")
    if width < t.min_face_px:
        return fail("Please step closer to the camera.")
    if abs(yaw) > t.max_yaw:
        return fail("Please face the camera directly.")
    if abs(roll) > t.max_roll_deg:
        return fail("Please hold your head straight.")
    if metrics["brightness"] < t.min_brightness:
        return fail("Too dark. Please step into the light.")
    if metrics["brightness"] > t.max_brightness:
        return fail("Too bright. Please step out of direct light.")
    if metrics["sharpness"] < t.min_sharpness:
        return fail("Photo is blurry. Please hold still.")
    return Quality(True, None, face, metrics)


def compute_embedding(img: np.ndarray) -> np.ndarray:
    """L2-normalised 512-d embedding of the single (foreground) face in `img`."""
    faces = foreground_faces(detect_faces(img))
    if len(faces) != 1:
        raise FaceError(f"expected exactly one face, found {len(faces)}")
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)
