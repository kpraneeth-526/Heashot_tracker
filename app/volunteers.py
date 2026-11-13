"""Volunteers saved for demos, so they stay registered between booth runs.

Only used when the booth is started with VOLUNTEERS_DIR set, and only for
people who tick "save my photo" when registering. Everything lives in that
folder: one JPEG per person plus volunteers.csv (the normal gallery format).
Keep the folder inside gallery/, which git ignores.

    python -m app.volunteers list   gallery/volunteers
    python -m app.volunteers remove gallery/volunteers someone@example.com
"""

import csv
import secrets
import sys
from pathlib import Path

CSV_NAME = "volunteers.csv"
COLUMNS = ("name", "email", "filename", "uga_id")


def _read(folder: Path) -> list[dict]:
    path = folder / CSV_NAME
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(folder: Path, rows: list[dict]) -> None:
    tmp = folder / (CSV_NAME + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({c: row.get(c, "") for c in COLUMNS} for row in rows)
    tmp.replace(folder / CSV_NAME)


def _drop(folder: Path, rows: list[dict], gone: list[dict]) -> list[dict]:
    for row in gone:
        (folder / row["filename"]).unlink(missing_ok=True)
    return [r for r in rows if r not in gone]


def csv_path(folder: str | Path) -> Path | None:
    """The folder's volunteers.csv, or None if nobody has been saved yet."""
    path = Path(folder) / CSV_NAME
    return path if path.exists() else None


def save(folder: str | Path, name: str, email: str, uga_id: str, photo: bytes) -> Path:
    """Save a volunteer's reference photo. Anyone already saved with the same
    email or UGA ID is replaced (their old photo is deleted)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = _read(folder)
    gone = [r for r in rows if r["email"].lower() == email.lower()
            or (uga_id and r.get("uga_id") == uga_id)]
    rows = _drop(folder, rows, gone)

    photo_path = folder / f"{secrets.token_hex(8)}.jpg"
    photo_path.write_bytes(photo)
    rows.append({"name": name, "email": email, "filename": photo_path.name, "uga_id": uga_id})
    _write(folder, rows)
    return photo_path


def remove(folder: str | Path, email: str) -> bool:
    """Delete a volunteer's photo and row. True if someone was removed."""
    folder = Path(folder)
    rows = _read(folder)
    gone = [r for r in rows if r["email"].lower() == email.lower()]
    if gone:
        _write(folder, _drop(folder, rows, gone))
    return bool(gone)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "list":
        rows = _read(Path(args[1]))
        for row in rows:
            print(f"{row['name']}  <{row['email']}>")
        print(f"{len(rows)} saved volunteer(s)")
        return 0
    if len(args) == 3 and args[0] == "remove":
        if remove(args[1], args[2]):
            print(f"removed {args[2]}")
            return 0
        print(f"no saved volunteer with email {args[2]}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
