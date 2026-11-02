"""Face detection and embedding using InsightFace (buffalo_l)."""

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)

# Booth photo quality gates. Starting values; tune in the Week 2 evaluation.
MIN_FACE_PX = 80  # faces narrower than this are ignored (people in the background)
MIN_SHARPNESS = 40.0  # Laplacian variance of the face crop at 112x112
MAX_YAW = 0.25  # nose offset from the eye midpoint, as a fraction of eye distance
PITCH_RANGE = (0.25, 0.85)  # nose height between the eyes (0) and mouth (1)


class FaceError(ValueError):
    """The image can't be used: undecodable, or not exactly one face."""


class QualityError(FaceError):
    """A face was found but the photo isn't good enough to match. The message
    is shown to the student, so it says what to do."""


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


def compute_embedding(img: np.ndarray) -> np.ndarray:
    """L2-normalised 512-d embedding of the single face in `img`."""
    faces = detect_faces(img)
    if len(faces) != 1:
        raise FaceError(f"expected exactly one face, found {len(faces)}")
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)


def _face_width(face) -> float:
    x1, _, x2, _ = face.bbox
    return float(x2 - x1)


def sharpness(img: np.ndarray, bbox) -> float:
    """Laplacian variance of the face crop. The crop is resized to a fixed
    size first so the score doesn't depend on how close the student stands."""
    import cv2

    height, width = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def head_pose(kps) -> tuple[float, float]:
    """Rough (yaw, pitch) from the five detector landmarks: eyes, nose, mouth
    corners. Yaw is ~0 and pitch ~0.5 when looking straight at the camera."""
    left_eye, right_eye, nose, left_mouth, right_mouth = np.asarray(kps, dtype=np.float64)
    eye_mid = (left_eye + right_eye) / 2
    mouth_mid = (left_mouth + right_mouth) / 2
    eye_dist = np.linalg.norm(right_eye - left_eye)
    eye_to_mouth = mouth_mid[1] - eye_mid[1]
    if eye_dist <= 0 or eye_to_mouth <= 0:
        return float("inf"), float("inf")
    yaw = (nose[0] - eye_mid[0]) / eye_dist
    pitch = (nose[1] - eye_mid[1]) / eye_to_mouth
    return float(yaw), float(pitch)


def check_quality(img: np.ndarray):
    """Return the one usable face in a booth photo, or raise QualityError.

    Checks: exactly one face (ignoring small faces in the background), sharp,
    and facing the camera.
    """
    faces = [f for f in detect_faces(img) if _face_width(f) >= MIN_FACE_PX]
    if not faces:
        raise QualityError("No face found. Step closer and look at the camera.")
    if len(faces) > 1:
        raise QualityError("More than one person in the photo. Please retake with just you.")
    face = faces[0]

    if sharpness(img, face.bbox) < MIN_SHARPNESS:
        raise QualityError("The photo is blurry. Hold still and retake.")

    yaw, pitch = head_pose(face.kps)
    if abs(yaw) > MAX_YAW or not PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1]:
        raise QualityError("Please face the camera straight on and retake.")
    return face
