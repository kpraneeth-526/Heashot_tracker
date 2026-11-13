from app import volunteers
from app.matcher import load_gallery


def test_save_writes_photo_and_loadable_csv(tmp_path):
    photo = volunteers.save(tmp_path, "Ada Lovelace", "ada@example.edu", "8110000001", b"jpeg")
    assert photo.read_bytes() == b"jpeg"
    entries = load_gallery(volunteers.csv_path(tmp_path))
    assert [(e.name, e.email, e.photo, e.uga_id) for e in entries] == [
        ("Ada Lovelace", "ada@example.edu", photo, "8110000001")]


def test_saving_same_person_again_replaces_old_photo(tmp_path):
    old = volunteers.save(tmp_path, "Ada", "ada@example.edu", "", b"old")
    new = volunteers.save(tmp_path, "Ada L", "ADA@example.edu", "", b"new")
    assert not old.exists() and new.exists()
    assert [e.name for e in load_gallery(volunteers.csv_path(tmp_path))] == ["Ada L"]


def test_same_uga_id_replaces_too(tmp_path):
    old = volunteers.save(tmp_path, "Ada", "ada@example.edu", "8110000001", b"old")
    volunteers.save(tmp_path, "Ada", "ada@uga.edu", "8110000001", b"new")
    assert not old.exists()
    assert [e.email for e in load_gallery(volunteers.csv_path(tmp_path))] == ["ada@uga.edu"]


def test_remove_deletes_photo_and_row(tmp_path):
    ada = volunteers.save(tmp_path, "Ada", "ada@example.edu", "", b"a")
    volunteers.save(tmp_path, "Alan", "alan@example.edu", "", b"b")
    assert volunteers.remove(tmp_path, "Ada@Example.edu")
    assert not ada.exists()
    assert [e.name for e in load_gallery(volunteers.csv_path(tmp_path))] == ["Alan"]
    assert not volunteers.remove(tmp_path, "nobody@example.edu")


def test_cli_list_and_remove(tmp_path, capsys):
    volunteers.save(tmp_path, "Ada", "ada@example.edu", "", b"a")
    assert volunteers.main(["list", str(tmp_path)]) == 0
    assert "Ada  <ada@example.edu>" in capsys.readouterr().out
    assert volunteers.main(["remove", str(tmp_path), "ada@example.edu"]) == 0
    assert volunteers.main(["remove", str(tmp_path), "ada@example.edu"]) == 1


def test_no_csv_until_someone_is_saved(tmp_path):
    assert volunteers.csv_path(tmp_path) is None
