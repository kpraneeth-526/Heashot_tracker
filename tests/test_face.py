from types import SimpleNamespace

import numpy as np
import pytest

from app.face import QualityThresholds, check_quality, face_pose, foreground_faces

RNG = np.random.default_rng(0)


def textured(size=400, mean=128.0, spread=60.0):
    """A noisy grey image: sharp, with the given average brightness."""
    img = RNG.normal(mean, spread, (size, size)).clip(0, 255).astype(np.uint8)
    return np.dstack([img] * 3)


def fake_face(x=100, y=100, width=200, yaw=0.0, roll_deg=0.0, det_score=0.9):
    """Face box at (x, y) with landmarks placed for the given pose."""
    eye_dist = width * 0.4
    cx, cy = x + width / 2, y + width * 0.4
    a = np.radians(roll_deg)
    half = np.array([np.cos(a), np.sin(a)]) * eye_dist / 2
    left_eye, right_eye = np.array([cx, cy]) - half, np.array([cx, cy]) + half
    nose = np.array([cx + yaw * eye_dist, cy + width * 0.2])
    return SimpleNamespace(
        bbox=np.array([x, y, x + width, y + width], dtype=float),
        kps=np.array([left_eye, right_eye, nose, nose, nose]),
        det_score=det_score,
    )


def test_good_photo_passes():
    q = check_quality(textured(), [fake_face()])
    assert q.ok, q.reason
    assert q.metrics["face_px"] == 200


def test_no_face():
    q = check_quality(textured(), [])
    assert not q.ok and "No face" in q.reason


def test_two_people():
    faces = [fake_face(x=0), fake_face(x=200, width=180)]
    q = check_quality(textured(), faces)
    assert not q.ok and "one person" in q.reason


def test_small_background_face_is_ignored():
    faces = [fake_face(x=100, width=200), fake_face(x=320, y=10, width=40)]
    assert foreground_faces(faces) == [faces[0]]
    assert check_quality(textured(), faces).ok


@pytest.mark.parametrize("face, message", [
    (fake_face(det_score=0.3), "not clear"),
    (fake_face(width=50), "step closer"),
    (fake_face(yaw=0.4), "face the camera"),
    (fake_face(yaw=-0.4), "face the camera"),
    (fake_face(roll_deg=25), "head straight"),
])
def test_pose_and_size_failures(face, message):
    q = check_quality(textured(), [face])
    assert not q.ok
    assert message in q.reason


def test_too_dark_and_too_bright():
    assert "Too dark" in check_quality(textured(mean=20, spread=10), [fake_face()]).reason
    assert "Too bright" in check_quality(textured(mean=240, spread=10), [fake_face()]).reason


def test_blurry():
    flat = np.full((400, 400, 3), 128, dtype=np.uint8)
    assert "blurry" in check_quality(flat, [fake_face()]).reason


def test_thresholds_are_adjustable():
    lenient = QualityThresholds(max_yaw=0.5)
    assert check_quality(textured(), [fake_face(yaw=0.4)], lenient).ok


def test_face_pose_estimates():
    yaw, roll = face_pose(fake_face(yaw=0.2, roll_deg=10))
    assert yaw == pytest.approx(0.2, abs=0.05)
    assert roll == pytest.approx(10, abs=0.5)
