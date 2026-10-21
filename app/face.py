"""Face detection and embedding using InsightFace (buffalo_l)."""

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)


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


def compute_embedding(img: np.ndarray) -> np.ndarray:
    """L2-normalised 512-d embedding of the single face in `img`."""
    faces = detect_faces(img)
    if len(faces) != 1:
        raise FaceError(f"expected exactly one face, found {len(faces)}")
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)
