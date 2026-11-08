"""Find a student on the roster by UGA ID or name: the booth's fallback when
face matching can't pick someone (or picks the wrong person)."""

from app.matcher import Gallery, GalleryEntry, normalize_id

MAX_RESULTS = 5
MIN_NAME_CHARS = 2


def roster(gallery: Gallery) -> list[GalleryEntry]:
    """Everyone who can be looked up: students with a usable reference photo,
    plus those whose photo was skipped (they can still be found by ID/name)."""
    return gallery.entries + [entry for entry, _ in gallery.skipped]


def search(entries: list[GalleryEntry], query: str) -> list[GalleryEntry]:
    """A query with digits is an exact UGA ID lookup. Otherwise every word of
    the query must start a word of the name: "ada lov" finds "Ada Lovelace"."""
    query = query.strip()
    if any(c.isdigit() for c in query):
        wanted = normalize_id(query)
        return [e for e in entries if e.uga_id and e.uga_id == wanted][:1]

    words = query.lower().split()
    if len("".join(words)) < MIN_NAME_CHARS:
        return []
    found = []
    for entry in entries:
        name_words = entry.name.lower().split()
        if all(any(n.startswith(w) for n in name_words) for w in words):
            found.append(entry)
            if len(found) == MAX_RESULTS:
                break
    return found
