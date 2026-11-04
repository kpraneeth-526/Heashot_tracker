from pathlib import Path

import numpy as np
import pytest

from app.matcher import (
    Gallery,
    GalleryEntry,
    build_gallery,
    cosine_similarity,
    decide,
    load_gallery,
    match,
    wipe_gallery,
)


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


def make_gallery(*vectors):
    entries = [GalleryEntry(f"P{i}", f"p{i}@example.edu", Path(f"p{i}.jpg"))
               for i in range(len(vectors))]
    if not vectors:
        return Gallery([], np.empty((0, 0), dtype=np.float32))
    vecs = np.array(vectors, dtype=np.float32)
    return Gallery(entries, vecs / np.linalg.norm(vecs, axis=1, keepdims=True))


def test_match_returns_best_and_runner_up():
    gallery = make_gallery([1, 0, 0], [0.8, 0.6, 0], [0, 0, 1])
    result = match(np.array([1.0, 0.1, 0]), gallery)
    assert result.best.name == "P0"
    assert result.best_score > result.second_score
    assert result.second_score == pytest.approx(cosine_similarity(
        np.array([1.0, 0.1, 0]), np.array([0.8, 0.6, 0])))


def test_match_unnormalised_query():
    gallery = make_gallery([1, 0], [0, 1])
    assert match(np.array([0.0, 5.0]), gallery).best_score == pytest.approx(1.0)


def test_match_small_galleries():
    assert match(np.array([1.0, 0]), make_gallery()).best is None
    one = match(np.array([1.0, 0]), make_gallery([1, 0]))
    assert one.best.name == "P0" and one.second_score == -1.0


def test_decide_needs_score_and_margin():
    gallery = make_gallery([1, 0, 0], [0.9, 0.436, 0], [0, 0, 1])
    clear = match(np.array([0.0, 0, 1]), gallery)
    assert decide(clear, 0.5, 0.1).entry.name == "P2"

    ambiguous = match(np.array([1.0, 0.2, 0]), gallery)  # P0 and P1 both close
    d = decide(ambiguous, 0.5, 0.1)
    assert d.entry is None and d.reason == "too close to another person"

    weak = match(np.array([1.0, 1.0, 1.0]), make_gallery([1, 0, 0], [0, 0, -1]))
    d = decide(weak, 0.9, 0.0)
    assert d.entry is None and d.reason == "score too low"


def test_decide_empty_gallery():
    d = decide(match(np.array([1.0]), make_gallery()))
    assert d.entry is None and d.reason == "empty gallery"
