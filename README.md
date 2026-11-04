# Headshot Auto-Match

Prototype booth that matches a headshot to a registered student and emails it
to them. Development uses the public LFW dataset (Labeled Faces in the Wild)
instead of real students.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

The first face-model call downloads InsightFace `buffalo_l` (~280 MB) to
`~/.insightface/models`.

Download LFW (~180 MB) into `data/`, which git ignores:

    mkdir -p data
    curl -L -o data/lfw.tgz https://ndownloader.figshare.com/files/5976018
    tar -xzf data/lfw.tgz -C data && rm data/lfw.tgz

## Run

    .venv/bin/pytest                                   # tests
    .venv/bin/python -m eval.compare a.jpg b.jpg       # similarity of two photos
    .venv/bin/uvicorn app.main:app --reload            # booth at http://localhost:8000
    GALLERY_CSV=gallery/gallery.csv .venv/bin/uvicorn app.main:app   # with gallery

See `gallery/README.md` for the gallery format.

To test with your own face, open http://localhost:8000/enroll, enter a name and
email, and take a reference photo. Then use the booth at http://localhost:8000.
Registrations are kept in memory only and are gone when the app stops.

## Accuracy on LFW

    .venv/bin/python -m eval.evaluate                  # default thresholds
    .venv/bin/python -m eval.evaluate --sweep          # compare thresholds
    .venv/bin/python -m eval.evaluate --distractors    # bigger, harder gallery

The first run embeds every photo and caches the result in
`data/lfw_embeddings.npz` (about 40 minutes on a laptop); later runs take seconds.

To try the booth against LFW people:

    .venv/bin/python -m eval.make_lfw_gallery --people 200
    GALLERY_CSV=gallery/lfw_gallery.csv .venv/bin/uvicorn app.main:app
