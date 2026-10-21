import numpy as np
import pytest

from app.matcher import build_gallery, cosine_similarity, load_gallery, wipe_gallery


def write_gallery(tmp_path, rows, photos=("a.jpg", "b.jpg")):
    for name in photos:
        (tmp_path / name).write_bytes(b"fake")
    csv_path = tmp_path / "gallery.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0]), np.array([0, 3.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(-1.0)


def test_load_gallery_reads_rows_and_resolves_paths(tmp_path):
    csv_path = write_gallery(tmp_path, [
        "name,email,filename",
        " Ada Lovelace , ada@example.edu ,a.jpg",
        ",,",
        "Alan Turing,alan@example.edu,b.jpg",
    ])
    entries = load_gallery(csv_path)
    assert [(e.name, e.email) for e in entries] == [
        ("Ada Lovelace", "ada@example.edu"),
        ("Alan Turing", "alan@example.edu"),
    ]
    assert entries[0].photo == tmp_path / "a.jpg"


def test_load_gallery_missing_column(tmp_path):
    csv_path = write_gallery(tmp_path, ["name,filename", "Ada,a.jpg"])
    with pytest.raises(ValueError, match="missing column.*email"):
        load_gallery(csv_path)


def test_load_gallery_reports_every_bad_row(tmp_path):
    csv_path = write_gallery(tmp_path, [
        "name,email,filename",
        "Ada,ada@example.edu,a.jpg",
        "Ada Again,ADA@example.edu,b.jpg",
        "Nobody,nobody@example.edu,missing.jpg",
        "No Email,,a.jpg",
        "Bad Email,not-an-email,a.jpg",
    ])
    with pytest.raises(ValueError) as exc:
        load_gallery(csv_path)
    msg = str(exc.value)
    assert "line 3: duplicate email" in msg
    assert "line 4: photo not found" in msg
    assert "line 5: name, email and filename are all required" in msg
    assert "line 6: invalid email" in msg


def test_build_gallery_normalises_and_skips_failures(tmp_path):
    csv_path = write_gallery(tmp_path, [
        "name,email,filename",
        "Ada,ada@example.edu,a.jpg",
        "Alan,alan@example.edu,b.jpg",
    ])

    def fake_embed(path):
        if path.name == "b.jpg":
            raise ValueError("expected exactly one face, found 0")
        return np.array([3.0, 4.0])

    gallery = build_gallery(load_gallery(csv_path), embed=fake_embed)
    assert [e.name for e in gallery.entries] == ["Ada"]
    np.testing.assert_allclose(gallery.embeddings, [[0.6, 0.8]])
    assert gallery.skipped[0][0].name == "Alan"
    assert "found 0" in gallery.skipped[0][1]


def test_wipe_gallery_zeroes_embeddings(tmp_path):
    csv_path = write_gallery(tmp_path, ["name,email,filename", "Ada,ada@example.edu,a.jpg"])
    gallery = build_gallery(load_gallery(csv_path), embed=lambda p: np.ones(4))
    buffer = gallery.embeddings

    wipe_gallery(gallery)

    assert not buffer.any()
    assert len(gallery) == 0
    assert gallery.embeddings.size == 0
