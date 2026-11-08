from pathlib import Path

import numpy as np

from app.matcher import Gallery, GalleryEntry
from app.roster import MAX_RESULTS, roster, search

ADA = GalleryEntry("Ada Lovelace", "ada@example.edu", Path("a.jpg"), "8110000001")
ADAM = GalleryEntry("Adam Smith", "adam@example.edu", Path("b.jpg"), "8110000002")
ALAN = GalleryEntry("Alan Turing", "alan@example.edu", Path("c.jpg"))
EVERYONE = [ADA, ADAM, ALAN]


def test_search_by_name_prefixes():
    assert search(EVERYONE, "ada") == [ADA, ADAM]
    assert search(EVERYONE, "ada lov") == [ADA]
    assert search(EVERYONE, "  LOVELACE ") == [ADA]
    assert search(EVERYONE, "tur al") == [ALAN]
    assert search(EVERYONE, "bob") == []


def test_search_needs_a_couple_of_letters():
    assert search(EVERYONE, "a") == []
    assert search(EVERYONE, "") == []


def test_search_by_uga_id_is_exact():
    assert search(EVERYONE, "811-000-0002") == [ADAM]
    assert search(EVERYONE, "811 000 0001") == [ADA]
    assert search(EVERYONE, "811") == []  # no partial ID matches


def test_search_limits_results():
    many = [GalleryEntry(f"Sam {i}", f"s{i}@example.edu", None) for i in range(20)]
    assert len(search(many, "sam")) == MAX_RESULTS


def test_roster_includes_students_whose_photo_was_skipped():
    gallery = Gallery([ADA], np.ones((1, 2)), skipped=[(ALAN, "no face found")])
    assert roster(gallery) == [ADA, ALAN]
