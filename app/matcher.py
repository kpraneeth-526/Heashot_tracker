"""In-memory gallery of registered students and similarity scoring."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REQUIRED_COLUMNS = ("name", "email", "filename")

# A match needs a high enough score AND a clear lead over the runner-up.
# Starting points; tune with `python -m eval.evaluate --sweep`.
MIN_SCORE = 0.45
MIN_MARGIN = 0.10

# NumPy 2.x on Apple Silicon (Accelerate BLAS) emits bogus divide-by-zero /
# overflow warnings from matmul even though the results are correct.
def quiet_matmul():
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


@dataclass(frozen=True)
class GalleryEntry:
    name: str
    email: str
    photo: Path | None  # None for people registered at the booth (photo not kept)


@dataclass
class Gallery:
    entries: list[GalleryEntry]
    embeddings: np.ndarray  # shape (len(entries), dim), rows L2-normalised
    skipped: list[tuple[GalleryEntry, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def empty_gallery() -> Gallery:
    return Gallery([], np.empty((0, 0), dtype=np.float32))


def add_to_gallery(gallery: Gallery, entry: GalleryEntry, embedding: np.ndarray) -> None:
    """Add one person, replacing anyone already registered with the same email."""
    vec = np.asarray(embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    keep = [i for i, e in enumerate(gallery.entries) if e.email.lower() != entry.email.lower()]
    old = gallery.embeddings
    rows = [old[i] for i in keep] + [vec]
    gallery.entries[:] = [gallery.entries[i] for i in keep] + [entry]
    gallery.embeddings = np.stack(rows)
    if old.size:
        old.fill(0)  # don't leave the previous copy of everyone's embedding around


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


def embed_photo(path: Path) -> np.ndarray:
    from app.face import compute_embedding, load_image

    return compute_embedding(load_image(path))


def cached_embedder(npz_path: str | Path) -> Callable[[Path], np.ndarray]:
    """An `embed` function for build_gallery that reuses saved embeddings
    (the file eval.evaluate writes, keyed "Person/Person_0001.jpg") instead of
    running the face model again. Photos missing from the file are embedded
    as usual. Only for public test data such as LFW: real students' face data
    is never written to disk.
    """
    data = np.load(npz_path)
    saved = {str(p): e for p, e, has in zip(data["paths"], data["emb"], data["has_face"]) if has}

    def embed(path: Path) -> np.ndarray:
        key = f"{path.parent.name}/{path.name}"
        return saved[key] if key in saved else embed_photo(path)

    return embed


def build_gallery(
    entries: list[GalleryEntry],
    embed: Callable[[Path], np.ndarray] = embed_photo,
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


@dataclass(frozen=True)
class MatchResult:
    best: GalleryEntry | None
    best_score: float
    second_score: float  # -1.0 when the gallery has fewer than two people

    @property
    def margin(self) -> float:
        return self.best_score - self.second_score


@dataclass(frozen=True)
class Decision:
    entry: GalleryEntry | None  # None means fall back to manual email entry
    reason: str
    result: MatchResult


def match(embedding: np.ndarray, gallery: Gallery) -> MatchResult:
    """Best and second-best cosine similarity against the gallery."""
    if len(gallery) == 0:
        return MatchResult(None, -1.0, -1.0)
    query = np.asarray(embedding, dtype=np.float32)
    with quiet_matmul():
        scores = gallery.embeddings @ (query / np.linalg.norm(query))
    if len(scores) == 1:
        return MatchResult(gallery.entries[0], float(scores[0]), -1.0)
    second, best = np.argpartition(scores, -2)[-2:]
    if scores[second] > scores[best]:
        best, second = second, best
    return MatchResult(gallery.entries[best], float(scores[best]), float(scores[second]))


def decide(
    result: MatchResult, min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN
) -> Decision:
    if result.best is None:
        return Decision(None, "empty gallery", result)
    if result.best_score < min_score:
        return Decision(None, "score too low", result)
    if result.margin < min_margin:
        return Decision(None, "too close to another person", result)
    return Decision(result.best, "match", result)


def wipe_gallery(gallery: Gallery) -> None:
    """Zero the embedding buffer and drop all references.

    Best effort: Python can't guarantee every copy is overwritten, but this
    stops the gallery being reachable from the app once the event ends.
    """
    gallery.embeddings.fill(0)
    gallery.embeddings = np.empty((0, 0), dtype=np.float32)
    gallery.entries.clear()
    gallery.skipped.clear()
