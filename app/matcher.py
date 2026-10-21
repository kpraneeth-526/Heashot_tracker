"""In-memory gallery of registered students and similarity scoring."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REQUIRED_COLUMNS = ("name", "email", "filename")


@dataclass(frozen=True)
class GalleryEntry:
    name: str
    email: str
    photo: Path


@dataclass
class Gallery:
    entries: list[GalleryEntry]
    embeddings: np.ndarray  # shape (len(entries), dim), rows L2-normalised
    skipped: list[tuple[GalleryEntry, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_gallery(csv_path: str | Path) -> list[GalleryEntry]:
    """Read gallery.csv (name, email, filename) and validate every row.

    Filenames are resolved relative to the CSV's folder. All problems are
    collected and raised together so the whole file can be fixed in one pass.
    """
    csv_path = Path(csv_path)
    folder = csv_path.parent
    entries: list[GalleryEntry] = []
    errors: list[str] = []
    seen_emails: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing column(s): {', '.join(missing)}")

        for line, row in enumerate(reader, start=2):
            name, email, filename = (row[c].strip() for c in REQUIRED_COLUMNS)
            if not (name or email or filename):
                continue
            if not (name and email and filename):
                errors.append(f"line {line}: name, email and filename are all required")
                continue
            if "@" not in email:
                errors.append(f"line {line}: invalid email {email!r}")
                continue
            if email.lower() in seen_emails:
                errors.append(f"line {line}: duplicate email {email!r}")
                continue
            photo = folder / filename
            if not photo.is_file():
                errors.append(f"line {line}: photo not found: {filename}")
                continue
            seen_emails.add(email.lower())
            entries.append(GalleryEntry(name, email, photo))

    if errors:
        raise ValueError(f"{csv_path}:\n  " + "\n  ".join(errors))
    return entries


def _embed_photo(path: Path) -> np.ndarray:
    from app.face import compute_embedding, load_image

    return compute_embedding(load_image(path))


def build_gallery(
    entries: list[GalleryEntry],
    embed: Callable[[Path], np.ndarray] = _embed_photo,
) -> Gallery:
    """Embed each reference photo. Photos that fail (no face, two faces, ...)
    are skipped and reported; those students fall back to manual entry."""
    kept: list[GalleryEntry] = []
    vectors: list[np.ndarray] = []
    skipped: list[tuple[GalleryEntry, str]] = []
    for entry in entries:
        try:
            vec = np.asarray(embed(entry.photo), dtype=np.float32)
        except Exception as e:
            skipped.append((entry, str(e)))
            continue
        kept.append(entry)
        vectors.append(vec / np.linalg.norm(vec))

    embeddings = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
    return Gallery(kept, embeddings, skipped)


def wipe_gallery(gallery: Gallery) -> None:
    """Zero the embedding buffer and drop all references.

    Best effort: Python can't guarantee every copy is overwritten, but this
    stops the gallery being reachable from the app once the event ends.
    """
    gallery.embeddings.fill(0)
    gallery.embeddings = np.empty((0, 0), dtype=np.float32)
    gallery.entries.clear()
    gallery.skipped.clear()
