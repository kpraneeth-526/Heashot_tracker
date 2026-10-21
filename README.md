# Headshot Auto-Match

Prototype booth that matches a headshot to a registered student and emails it
to them. Volunteers and consented photos only.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

The first face-model call downloads InsightFace `buffalo_l` (~280 MB) to
`~/.insightface/models`.

## Run

    .venv/bin/pytest                                   # tests
    .venv/bin/python -m eval.compare a.jpg b.jpg       # similarity of two photos
    .venv/bin/uvicorn app.main:app --reload            # booth at http://localhost:8000
    GALLERY_CSV=gallery/gallery.csv .venv/bin/uvicorn app.main:app   # with gallery

See `gallery/README.md` for the gallery format.
