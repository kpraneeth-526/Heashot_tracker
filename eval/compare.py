"""Print the similarity score of the faces in two photos.

    python -m eval.compare photo_a.jpg photo_b.jpg
"""

import argparse
import sys

from app.face import FaceError, compute_embedding, load_image
from app.matcher import cosine_similarity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photo_a")
    parser.add_argument("photo_b")
    args = parser.parse_args(argv)

    embeddings = []
    for path in (args.photo_a, args.photo_b):
        try:
            embeddings.append(compute_embedding(load_image(path)))
        except (FaceError, OSError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            return 1

    print(f"{cosine_similarity(*embeddings):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
