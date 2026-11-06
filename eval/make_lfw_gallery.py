"""Write a gallery CSV from LFW so the booth can run against LFW people.

    python -m eval.make_lfw_gallery --people 200     # or --people 0 for everyone
    GALLERY_CSV=gallery/lfw_gallery.csv GALLERY_CACHE=data/lfw_embeddings.npz \
        uvicorn app.main:app

Each person's first photo is their reference. Emails are made up on the
reserved example.edu domain, so nothing can ever reach a real inbox.
Add your own row to the CSV afterwards to test the booth with your face.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from eval.evaluate import DEFAULT_LFW, LFW_MISLABELLED, list_people


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lfw", type=Path, default=DEFAULT_LFW)
    parser.add_argument("--out", type=Path, default=Path("gallery/lfw_gallery.csv"))
    parser.add_argument("--people", type=int, default=200,
                        help="how many people; 0 means everyone")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    people = list_people(args.lfw)
    names = sorted(n for n in people if n not in LFW_MISLABELLED)
    random.Random(args.seed).shuffle(names)
    if args.people:
        names = names[: args.people]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "email", "filename"])
        for person in sorted(names):
            photo = os.path.relpath(args.lfw / people[person][0], args.out.parent)
            email = person.lower().replace("-", "_") + "@example.edu"
            writer.writerow([person.replace("_", " "), email, photo])
    print(f"wrote {len(names)} people to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
