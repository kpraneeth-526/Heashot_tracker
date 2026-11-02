from types import SimpleNamespace

import numpy as np
import pytest

from app import face

# Landmarks of a face looking straight at the camera (ArcFace 112x112 template).
FRONTAL_KPS = [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]]


def fake_face(bbox=(0, 0, 112, 112), kps=FRONTAL_KPS):
    return SimpleNamespace(bbox=np.array(bbox, dtype=np.float32), kps=np.array(kps))


def sharp_image(size=200):
    return np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.fixture
def faces(monkeypatch):
    found = []
    monkeypatch.setattr(face, "detect_faces", lambda img: found)
    return found


def test_accepts_one_sharp_frontal_face(faces):
    faces.append(fake_face())
    assert face.check_quality(sharp_image()) is faces[0]


def test_rejects_no_face(faces):
    with pytest.raises(face.QualityError, match="No face"):
        face.check_quality(sharp_image())


def test_rejects_two_people(faces):
    faces.extend([fake_face(), fake_face(bbox=(90, 0, 200, 110))])
    with pytest.raises(face.QualityError, match="More than one person"):
        face.check_quality(sharp_image())


def test_ignores_small_faces_in_background(faces):
    faces.extend([fake_face(), fake_face(bbox=(150, 150, 190, 190))])
    assert face.check_quality(sharp_image()) is faces[0]


def test_rejects_blurry_photo(faces):
    faces.append(fake_face())
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    with pytest.raises(face.QualityError, match="blurry"):
        face.check_quality(flat)


@pytest.mark.parametrize("nose", [[80.0, 71.7], [56.0, 55.0], [56.0, 90.0]])
def test_rejects_face_turned_away(faces, nose):
    kps = [FRONTAL_KPS[0], FRONTAL_KPS[1], nose, FRONTAL_KPS[3], FRONTAL_KPS[4]]
    faces.append(fake_face(kps=kps))
    with pytest.raises(face.QualityError, match="face the camera"):
        face.check_quality(sharp_image())


def test_head_pose_frontal():
    yaw, pitch = face.head_pose(FRONTAL_KPS)
    assert abs(yaw) < 0.05
    assert 0.45 < pitch < 0.55
