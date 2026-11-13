"""Forget photos once they're sent, and keep a minimal audit log.

The audit log is a SQLite file with one row per email attempt: a timestamp
and "sent" or "failed". No names, emails, IDs or photos.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("sent", "failed")


def purge(photo: bytearray) -> None:
    """Overwrite the photo in place, then empty it. Best effort: Python may
    have made other copies, but this one can't be read back afterwards."""
    photo[:] = bytes(len(photo))
    photo.clear()


def log_event(db_path: str | Path, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS events (at TEXT NOT NULL, status TEXT NOT NULL)")
        db.execute(
            "INSERT INTO events (at, status) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), status),
        )
    # sqlite3's context manager commits but doesn't close
    db.close()


def read_events(db_path: str | Path) -> list[tuple[str, str]]:
    if not Path(db_path).exists():
        return []
    db = sqlite3.connect(db_path)
    try:
        return db.execute("SELECT at, status FROM events ORDER BY rowid").fetchall()
    finally:
        db.close()
