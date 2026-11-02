"""The Essentia model files DJ Tagger needs, and `djtagger setup` to fetch them.

Kept free of Essentia/TensorFlow imports so `setup` and the missing-model
check start instantly and work before the heavy stack has ever loaded.
"""

import json
import os
import urllib.request
from typing import Callable

from .config import MODEL_DIR

BASE_URL = "https://essentia.upf.edu/models/"

# (filename, path under BASE_URL, required, what it is for)
MODELS: list[tuple[str, str, bool, str]] = [
    ("discogs-effnet-bs64-1.pb", "feature-extractors/discogs-effnet/", True,
     "EffNet audio embeddings (every other head reads these)"),
    ("genre_discogs400-discogs-effnet-1.pb", "classification-heads/genre_discogs400/", True,
     "400-class genre"),
    ("mood_happy-discogs-effnet-1.pb", "classification-heads/mood_happy/", True, "mood: happy"),
    ("mood_sad-discogs-effnet-1.pb", "classification-heads/mood_sad/", True, "mood: sad"),
    ("mood_aggressive-discogs-effnet-1.pb", "classification-heads/mood_aggressive/", True,
     "mood: aggressive"),
    ("mood_relaxed-discogs-effnet-1.pb", "classification-heads/mood_relaxed/", True,
     "mood: relaxed"),
    ("genre_electronic-discogs-effnet-1.pb", "classification-heads/genre_electronic/", False,
     "5-class electronic genre"),
    ("danceability-discogs-effnet-1.pb", "classification-heads/danceability/", False,
     "danceability"),
    ("voice_instrumental-discogs-effnet-1.pb", "classification-heads/voice_instrumental/", False,
     "vocal presence (set role)"),
    ("msd-musicnn-1.pb", "feature-extractors/musicnn/", False,
     "MusicNN embeddings (arousal/valence)"),
    ("emomusic-msd-musicnn-2.pb", "classification-heads/emomusic/", False, "arousal/valence"),
    ("deepsquare-k16-3.pb", "tempo/tempocnn/", False, "TempoCNN BPM (--detect-bpm-key only)"),
]

# Genre label lists: (local cache name, URL of the model's metadata JSON).
# The cache holds just the "classes" list, the format analyzer._fetch_labels reads.
LABELS: list[tuple[str, str]] = [
    ("genre_discogs400_labels.json",
     BASE_URL + "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"),
    ("genre_electronic_labels.json",
     BASE_URL + "classification-heads/genre_electronic/genre_electronic-discogs-effnet-1.json"),
]


class ModelsMissingError(RuntimeError):
    """Required model files are not in the model directory."""


def missing_models(model_dir: str | None = None, required_only: bool = False) -> list[str]:
    """Filenames from MODELS that are absent (or empty) in *model_dir*."""
    d = model_dir or MODEL_DIR
    return [
        name for name, _, required, _ in MODELS
        if (required or not required_only)
        and not (os.path.isfile(os.path.join(d, name)) and os.path.getsize(os.path.join(d, name)) > 0)
    ]


def require_models(model_dir: str | None = None) -> None:
    """Raise ModelsMissingError with a fix-it message if a required model is absent."""
    d = model_dir or MODEL_DIR
    missing = missing_models(d, required_only=True)
    if missing:
        raise ModelsMissingError(
            f"Required model files are missing from {d}: {', '.join(missing)}. "
            f"Run `djtagger setup` to download them "
            f"(set DJTAGGER_MODEL_DIR to use another folder)."
        )


def _download(url: str, dest: str, timeout: int = 60) -> None:
    """Download *url* to *dest* atomically (via a .part file)."""
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "djtagger-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    if os.path.getsize(part) == 0:
        os.remove(part)
        raise RuntimeError(f"empty download: {url}")
    os.replace(part, dest)


def download_all(
    model_dir: str | None = None,
    force: bool = False,
    on_item: Callable[[str, str], None] | None = None,
) -> list[tuple[str, str]]:
    """Fetch every model and label file not already present.

    *on_item(name, status)* is called per file with status "downloaded",
    "present" or "failed: <reason>". Returns the list of (name, status).
    """
    d = model_dir or MODEL_DIR
    os.makedirs(d, exist_ok=True)
    results: list[tuple[str, str]] = []

    def report(name: str, status: str) -> None:
        results.append((name, status))
        if on_item:
            on_item(name, status)

    for name, sub, _, _ in MODELS:
        dest = os.path.join(d, name)
        if not force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            report(name, "present")
            continue
        try:
            _download(BASE_URL + sub + name, dest)
            report(name, "downloaded")
        except Exception as ex:
            report(name, f"failed: {ex}")

    for name, url in LABELS:
        dest = os.path.join(d, name)
        if not force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            report(name, "present")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "djtagger-setup"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                classes = json.loads(resp.read())["classes"]
            with open(dest, "w") as f:
                json.dump(classes, f)
            report(name, "downloaded")
        except Exception as ex:
            report(name, f"failed: {ex}")

    return results
