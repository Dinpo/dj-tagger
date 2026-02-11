"""Essentia ML audio analysis — load models, analyze track."""

import os
import sys
import json
import warnings
import logging
import urllib.request

import numpy as np

from .config import MODEL_DIR, GENRE_MIN_PROB

_GENRE_LABELS_CACHE = os.path.join(MODEL_DIR, "genre_discogs400_labels.json")

# ─── Suppress TF / Essentia warning spam ────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("essentia").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import essentia  # noqa: E402
essentia.log.warningActive = False
essentia.log.infoActive = False
import essentia.standard as es  # noqa: E402

# ─── Genre Labels (fetched once) ────────────────────────────

_genre_labels: list[str] | None = None


def get_genre_labels() -> list[str]:
    global _genre_labels
    if _genre_labels is not None:
        return _genre_labels

    # Try local cache first
    if os.path.isfile(_GENRE_LABELS_CACHE):
        try:
            with open(_GENRE_LABELS_CACHE) as f:
                _genre_labels = json.load(f)
            return _genre_labels
        except Exception:
            pass

    # Fetch from network with timeout
    url = (
        "https://essentia.upf.edu/models/classification-heads/"
        "genre_discogs400/genre_discogs400-discogs-effnet-1.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            _genre_labels = json.loads(resp.read())["classes"]
    except Exception as ex:
        raise RuntimeError(
            f"Cannot load genre labels: network fetch failed ({ex}) "
            f"and no local cache at {_GENRE_LABELS_CACHE}. "
            f"Run once with network access to cache the labels."
        ) from ex

    # Cache locally for offline use
    try:
        os.makedirs(os.path.dirname(_GENRE_LABELS_CACHE), exist_ok=True)
        with open(_GENRE_LABELS_CACHE, "w") as f:
            json.dump(_genre_labels, f)
    except Exception:
        pass

    return _genre_labels


# ─── Load Models ────────────────────────────────────────────

def load_models(model_dir: str | None = None) -> dict:
    """Load all Essentia TensorFlow models. Returns a dict of model objects."""
    d = model_dir or MODEL_DIR
    models: dict = {}

    models["embed"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=f"{d}/discogs-effnet-bs64-1.pb",
        output="PartitionedCall:1",
    )
    models["genre"] = es.TensorflowPredict2D(
        graphFilename=f"{d}/genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    models["moods"] = {}
    for mood in ("happy", "sad", "aggressive", "relaxed"):
        models["moods"][mood] = es.TensorflowPredict2D(
            graphFilename=f"{d}/mood_{mood}-discogs-effnet-1.pb",
            input="model/Placeholder",
            output="model/Softmax",
        )
    return models


# ─── Analyze a Single Track ─────────────────────────────────

def analyze_track(filepath: str, models: dict) -> dict:
    """Run ML analysis on an audio file.

    Returns dict with keys: genres, moods, valence, energy, raw_energy, duration.
    """
    audio = es.MonoLoader(filename=filepath, sampleRate=16000)()
    embeddings = models["embed"](audio)

    # Genre predictions
    genre_preds = models["genre"](embeddings)
    genre_avg = np.mean(genre_preds, axis=0)
    labels = get_genre_labels()
    top_genres = sorted(zip(labels, genre_avg), key=lambda x: -x[1])[:5]
    genres = []
    for label, prob in top_genres:
        if prob < GENRE_MIN_PROB:
            break
        clean = label.split("---")[-1]
        genres.append((clean, round(float(prob), 3)))

    # Mood predictions
    moods = {}
    for mood_name, model in models["moods"].items():
        preds = model(embeddings)
        moods[mood_name] = round(float(np.mean(preds, axis=0)[0]), 3)

    valence = round(
        float(np.clip((moods["happy"] - moods["sad"] + 1) / 2, 0, 1)), 3
    )
    raw_energy = float(
        np.clip((moods["aggressive"] + (1 - moods["relaxed"])) / 2, 0, 1)
    )
    energy = round(min(1.0, raw_energy * 1.5 + 0.3), 3)

    return {
        "genres": genres,
        "moods": moods,
        "valence": valence,
        "energy": energy,
        "raw_energy": round(raw_energy, 3),
        "duration": len(audio) / 16000,
    }
