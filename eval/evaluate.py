"""Measure matching accuracy on LFW (Labeled Faces in the Wild).

    python -m eval.evaluate                       # default thresholds
    python -m eval.evaluate --sweep               # table of thresholds
    python -m eval.evaluate --distractors         # Week 4: add 1-photo people to gallery

Protocol:
  * People with 2+ photos are split: most are "registered" (their first photo
    goes in the gallery, the rest are booth photos), the rest are "unregistered"
    (not in the gallery; all their photos should fall back to manual entry).
  * --distractors also puts every 1-photo person in the gallery, which makes
    the gallery bigger and wrong matches more likely.

Outcomes for each booth photo:
  correct   registered, matched to the right person
  wrong     matched to the WRONG person (registered or not) -> goal ~0%
  missed    registered, but fell back to manual entry
  fallback  unregistered, fell back to manual entry (the right outcome)

Embeddings are cached in data/lfw_embeddings.npz; the first run embeds every
photo it needs (expect roughly 10-20 minutes for all of LFW on a laptop CPU).
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from app.face import check_quality, detect_faces, load_image
from app.matcher import MIN_MARGIN, MIN_SCORE, quiet_matmul

DEFAULT_LFW = Path("data/lfw")
DEFAULT_CACHE = Path("data/lfw_embeddings.npz")
SWEEP_SCORES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
SWEEP_MARGINS = [0.0, 0.05, 0.10, 0.15]


def list_people(lfw_dir: Path) -> dict[str, list[str]]:
    """person -> sorted photo paths relative to lfw_dir."""
    people = {}
    for d in sorted(p for p in lfw_dir.iterdir() if p.is_dir()):
        photos = sorted(f"{d.name}/{f.name}" for f in d.glob("*.jpg"))
        if photos:
            people[d.name] = photos
    return people


def embed_photo(path: Path) -> tuple[np.ndarray | None, str]:
    """Embedding of the main face, plus "ok" or the quality failure message.

    LFW photos are centred on the labelled person, so when there are several
    faces the one nearest the centre is used for the embedding. The quality
    check still sees every face, like the booth would.
    """
    img = load_image(path)
    faces = detect_faces(img)
    if not faces:
        return None, "No face found."
    h, w = img.shape[:2]
    centre = np.array([w / 2, h / 2])
    main = min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))
    quality = check_quality(img, faces)
    if not quality.ok:
        status = quality.reason
    elif quality.face is not main:
        status = "Main face is not the centred one."
    else:
        status = "ok"
    return np.asarray(main.normed_embedding, dtype=np.float32), status


def load_embeddings(lfw_dir: Path, cache: Path, wanted: list[str]):
    """Embed every path in `wanted`, reusing and extending the cache."""
    store: dict[str, tuple[np.ndarray | None, str]] = {}
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        for p, e, has, s in zip(data["paths"], data["emb"], data["has_face"], data["status"]):
            store[str(p)] = (e if has else None, str(s))

    todo = [p for p in wanted if p not in store]
    for i, rel in enumerate(todo, 1):
        store[rel] = embed_photo(lfw_dir / rel)
        if i % 250 == 0 or i == len(todo):
            print(f"  embedded {i}/{len(todo)}", file=sys.stderr)
            save_cache(cache, store)
    return store


def save_cache(cache: Path, store) -> None:
    paths = sorted(store)
    emb = np.zeros((len(paths), 512), dtype=np.float32)
    has = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        if store[p][0] is not None:
            emb[i], has[i] = store[p][0], True
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, paths=np.array(paths), emb=emb, has_face=has,
             status=np.array([store[p][1] for p in paths]))
    tmp.replace(cache)


def build_split(people, store, args):
    rng = random.Random(args.seed)
    multi = [p for p, photos in people.items() if len(photos) >= 2]
    singles = [p for p, photos in people.items() if len(photos) == 1]
    rng.shuffle(multi)
    n_unreg = int(len(multi) * args.unregistered)
    unregistered, registered = multi[:n_unreg], multi[n_unreg:]

    def usable(photos):
        return [p for p in photos if store[p][0] is not None]

    def probes_of(photos):
        ok = [p for p in photos if args.no_quality or store[p][1] == "ok"]
        return ok[: args.max_probes]

    gallery_names, gallery_vecs = [], []
    probes = []  # (path, true person or None if unregistered)
    for person in registered:
        photos = usable(people[person])
        if len(photos) < 2:
            continue
        gallery_names.append(person)
        gallery_vecs.append(store[photos[0]][0])
        probes += [(p, person) for p in probes_of(photos[1:])]
    for person in unregistered:
        probes += [(p, None) for p in probes_of(usable(people[person]))]
    if args.distractors:
        for person in singles:
            photos = usable(people[person])
            if photos:
                gallery_names.append(person)
                gallery_vecs.append(store[photos[0]][0])

    return gallery_names, np.stack(gallery_vecs), probes


def top_two(probe_vecs: np.ndarray, gallery: np.ndarray, chunk: int = 1024):
    best_idx = np.empty(len(probe_vecs), dtype=int)
    best = np.empty(len(probe_vecs))
    second = np.empty(len(probe_vecs))
    for start in range(0, len(probe_vecs), chunk):
        with quiet_matmul():
            scores = probe_vecs[start:start + chunk] @ gallery.T
        top = np.argpartition(scores, -2, axis=1)[:, -2:]
        vals = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-vals, axis=1)
        top, vals = np.take_along_axis(top, order, 1), np.take_along_axis(vals, order, 1)
        best_idx[start:start + chunk] = top[:, 0]
        best[start:start + chunk] = vals[:, 0]
        second[start:start + chunk] = vals[:, 1]
    return best_idx, best, second


def outcomes(truth_idx, best_idx, best, second, min_score, min_margin) -> Counter:
    accepted = (best >= min_score) & (best - second >= min_margin)
    registered = truth_idx >= 0
    return Counter({
        "correct": int(np.sum(accepted & registered & (best_idx == truth_idx))),
        "wrong": int(np.sum(accepted & ~(registered & (best_idx == truth_idx)))),
        "missed": int(np.sum(~accepted & registered)),
        "fallback": int(np.sum(~accepted & ~registered)),
    })


def pct(n, total):
    return f"{100 * n / total:6.2f}%" if total else "   n/a"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate matching accuracy on LFW.")
    parser.add_argument("--lfw", type=Path, default=DEFAULT_LFW)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    parser.add_argument("--sweep", action="store_true", help="print a table of thresholds")
    parser.add_argument("--distractors", action="store_true",
                        help="add every 1-photo person to the gallery")
    parser.add_argument("--unregistered", type=float, default=0.2,
                        help="fraction of 2+-photo people left out of the gallery")
    parser.add_argument("--max-probes", type=int, default=5,
                        help="booth photos per person (stops George W Bush dominating)")
    parser.add_argument("--max-people", type=int, default=None,
                        help="only use this many people (quick test runs)")
    parser.add_argument("--no-quality", action="store_true",
                        help="match booth photos even if they fail the quality check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.lfw.is_dir():
        print(f"LFW not found at {args.lfw}. See README for the download.", file=sys.stderr)
        return 1

    people = list_people(args.lfw)
    if args.max_people:
        names = sorted(people)
        random.Random(args.seed).shuffle(names)
        people = {n: people[n] for n in sorted(names[: args.max_people])}
    wanted = [p for photos in people.values()
              if len(photos) >= 2 or args.distractors for p in photos]
    store = load_embeddings(args.lfw, args.cache, wanted)

    names, gallery, probes = build_split(people, store, args)
    index = {n: i for i, n in enumerate(names)}
    truth = np.array([index[t] if t else -1 for _, t in probes])
    best_idx, best, second = top_two(np.stack([store[p][0] for p, _ in probes]), gallery)

    n_reg, n_unreg = int(np.sum(truth >= 0)), int(np.sum(truth < 0))
    print(f"gallery: {len(names)} people | booth photos: {len(probes)} "
          f"({n_reg} registered, {n_unreg} unregistered)")
    statuses = Counter(store[p][1] for p in wanted if store[p][1] != "ok")
    if statuses:
        print("photos failing the quality check (excluded as booth photos"
              f"{'' if not args.no_quality else ', but --no-quality is on'}):")
        for reason, n in statuses.most_common():
            print(f"  {n:6d}  {reason}")

    if args.sweep:
        print(f"\n{'score':>5} {'margin':>6} | {'correct':>8} {'wrong':>8} "
              f"{'missed':>8} {'fallback':>8}   (correct/missed out of registered, "
              "fallback out of unregistered, wrong out of all)")
        for s in SWEEP_SCORES:
            for m in SWEEP_MARGINS:
                c = outcomes(truth, best_idx, best, second, s, m)
                print(f"{s:5.2f} {m:6.2f} | {pct(c['correct'], n_reg)} "
                      f"{pct(c['wrong'], len(probes))} {pct(c['missed'], n_reg)} "
                      f"{pct(c['fallback'], n_unreg)}")
        return 0

    c = outcomes(truth, best_idx, best, second, args.min_score, args.min_margin)
    print(f"\nthresholds: min score {args.min_score}, min margin {args.min_margin}")
    print(f"  correct   {c['correct']:6d}  {pct(c['correct'], n_reg)} of registered")
    print(f"  wrong     {c['wrong']:6d}  {pct(c['wrong'], len(probes))} of all booth photos")
    print(f"  missed    {c['missed']:6d}  {pct(c['missed'], n_reg)} of registered")
    print(f"  fallback  {c['fallback']:6d}  {pct(c['fallback'], n_unreg)} of unregistered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
