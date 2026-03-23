"""Essentia ML audio analysis — load models, analyze track."""

import os
import sys
import json
import warnings
import logging
import urllib.request

import numpy as np

from .config import (
    MODEL_DIR,
    GENRE_MIN_PROB,
    ENERGY_W_DANCEABILITY,
    ENERGY_W_AROUSAL,
    ENERGY_W_AGGRESSIVE,
    ENERGY_W_RELAXED,
    ENERGY_SCALE,
    ENERGY_OFFSET,
    SEGMENT_LENGTH_SEC,
    SEGMENT_HOP_SEC,
    CAMELOT_MAP,
)

# ─── Genre label caches ──────────────────────────────────────

_GENRE_LABELS_CACHE = os.path.join(MODEL_DIR, "genre_discogs400_labels.json")
_ELEC_LABELS_CACHE = os.path.join(MODEL_DIR, "genre_electronic_labels.json")

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

# ─── Genre Labels (fetched once, cached locally) ────────────

_genre_labels: list[str] | None = None
_elec_labels: list[str] | None = None


def _fetch_labels(cache_path: str, url: str) -> list[str]:
    """Fetch label list from network, with local file cache."""
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            labels = json.loads(resp.read())["classes"]
    except Exception as ex:
        raise RuntimeError(
            f"Cannot load labels: network fetch failed ({ex}) "
            f"and no local cache at {cache_path}. "
            f"Run once with network access to cache the labels."
        ) from ex

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(labels, f)
    except Exception:
        pass

    return labels


def get_genre_labels() -> list[str]:
    global _genre_labels
    if _genre_labels is None:
        _genre_labels = _fetch_labels(
            _GENRE_LABELS_CACHE,
            "https://essentia.upf.edu/models/classification-heads/"
            "genre_discogs400/genre_discogs400-discogs-effnet-1.json",
        )
    return _genre_labels


def get_electronic_genre_labels() -> list[str]:
    global _elec_labels
    if _elec_labels is None:
        _elec_labels = _fetch_labels(
            _ELEC_LABELS_CACHE,
            "https://essentia.upf.edu/models/classification-heads/"
            "genre_electronic/genre_electronic-discogs-effnet-1.json",
        )
    return _elec_labels


# ─── Load Models ────────────────────────────────────────────

def _try_load(loader, path: str, **kwargs):
    """Load a model file, returning None if the file is missing."""
    if not os.path.isfile(path):
        print(
            f"[djtagger] Warning: model not found: {os.path.basename(path)} — "
            f"some features will use fallback formulas.",
            file=sys.stderr,
        )
        return None
    return loader(graphFilename=path, **kwargs)


def load_models(model_dir: str | None = None) -> dict:
    """Load all Essentia TensorFlow models. Returns a dict of model objects."""
    d = model_dir or MODEL_DIR
    models: dict = {}

    # EffNet embedding model (required)
    models["embed"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=f"{d}/discogs-effnet-bs64-1.pb",
        output="PartitionedCall:1",
    )

    # Genre heads
    models["genre"] = es.TensorflowPredict2D(
        graphFilename=f"{d}/genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    models["genre_electronic"] = _try_load(
        es.TensorflowPredict2D,
        f"{d}/genre_electronic-discogs-effnet-1.pb",
        input="model/Placeholder",
        output="model/Softmax",
    )

    # Mood heads (original 4)
    models["moods"] = {}
    for mood in ("happy", "sad", "aggressive", "relaxed"):
        models["moods"][mood] = es.TensorflowPredict2D(
            graphFilename=f"{d}/mood_{mood}-discogs-effnet-1.pb",
            input="model/Placeholder",
            output="model/Softmax",
        )

    # New EffNet-based heads
    models["danceability"] = _try_load(
        es.TensorflowPredict2D,
        f"{d}/danceability-discogs-effnet-1.pb",
        input="model/Placeholder",
        output="model/Softmax",
    )
    models["mood_party"] = _try_load(
        es.TensorflowPredict2D,
        f"{d}/mood_party-discogs-effnet-1.pb",
        input="model/Placeholder",
        output="model/Softmax",
    )

    # MusicNN pipeline (separate embedding model + emomusic head)
    models["musicnn_embed"] = _try_load(
        es.TensorflowPredictMusiCNN,
        f"{d}/msd-musicnn-1.pb",
        output="model/dense/BiasAdd",
    )
    models["emomusic"] = _try_load(
        es.TensorflowPredict2D,
        f"{d}/emomusic-msd-musicnn-2.pb",
        input="model/Placeholder",
        output="model/Identity",
    )

    return models


# ─── Analyze a Single Track ─────────────────────────────────

def analyze_track(filepath: str, models: dict) -> dict:
    """Run ML analysis on an audio file.

    Returns dict with keys: genres, electronic_genres, moods, danceability,
    mood_party, arousal, valence, energy, raw_energy, peak_energy,
    intro_energy, energy_variance, duration.
    """
    audio = es.MonoLoader(filename=filepath, sampleRate=16000)()
    audio_44k = es.MonoLoader(filename=filepath, sampleRate=44100)()
    embeddings = models["embed"](audio)

    # ─── Genre predictions (Discogs 400-class) ──────────────
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

    # ─── Electronic genre predictions ───────────────────────
    electronic_genres: list[tuple[str, float]] = []
    if models.get("genre_electronic") is not None:
        elec_preds = models["genre_electronic"](embeddings)
        elec_avg = np.mean(elec_preds, axis=0)
        elec_labels = get_electronic_genre_labels()
        top_elec = sorted(zip(elec_labels, elec_avg), key=lambda x: -x[1])[:5]
        for label, prob in top_elec:
            if prob < GENRE_MIN_PROB:
                break
            electronic_genres.append((label, round(float(prob), 3)))

    # ─── Original mood predictions ──────────────────────────
    moods = {}
    for mood_name, model in models["moods"].items():
        preds = model(embeddings)
        moods[mood_name] = round(float(np.mean(preds, axis=0)[0]), 3)

    # ─── Danceability ───────────────────────────────────────
    if models.get("danceability") is not None:
        dance_preds = models["danceability"](embeddings)
        danceability = round(float(np.mean(dance_preds, axis=0)[0]), 3)
    else:
        # Fallback: estimate from moods
        danceability = round(float(np.clip(
            (moods["happy"] + moods["aggressive"] + (1 - moods["sad"])) / 3, 0, 1
        )), 3)

    # ─── Party mood ─────────────────────────────────────────
    if models.get("mood_party") is not None:
        party_preds = models["mood_party"](embeddings)
        mood_party = round(float(np.mean(party_preds, axis=0)[0]), 3)
    else:
        mood_party = round(float(np.clip(
            (moods["happy"] + moods["aggressive"]) / 2, 0, 1
        )), 3)

    # ─── Arousal/Valence from emomusic (MusicNN) ───────────
    has_emomusic = (
        models.get("musicnn_embed") is not None
        and models.get("emomusic") is not None
    )
    if has_emomusic:
        musicnn_emb = models["musicnn_embed"](audio)
        emo_preds = models["emomusic"](musicnn_emb)
        # emomusic outputs on ~1-9 scale, normalize to 0-1
        arousal_norm = round(float(np.clip(
            (np.mean(emo_preds[:, 0]) - 1) / 8, 0, 1
        )), 3)
        valence_norm = round(float(np.clip(
            (np.mean(emo_preds[:, 1]) - 1) / 8, 0, 1
        )), 3)
    else:
        # Fallback to v4-style heuristics
        arousal_norm = round(float(np.clip(
            (moods["aggressive"] + (1 - moods["relaxed"])) / 2, 0, 1
        )), 3)
        valence_norm = round(float(np.clip(
            (moods["happy"] - moods["sad"] + 1) / 2, 0, 1
        )), 3)

    # ─── BPM + Key detection (DSP, no model needed) ──────
    try:
        rhythm = es.RhythmExtractor2013(method="multifeature")
        bpm_val, _, bpm_confidence, _, _ = rhythm(audio_44k)
        bpm = round(float(bpm_val))
    except Exception:
        bpm = 0
        bpm_confidence = 0.0

    try:
        key_extractor = es.KeyExtractor()
        key_name, scale, key_strength = key_extractor(audio_44k)
        standard_key = f"{key_name}{'m' if scale == 'minor' else ''}"
        key_str = CAMELOT_MAP.get(standard_key, standard_key)
        key_strength = round(float(key_strength), 3)
    except Exception:
        key_str = ""
        key_strength = 0.0

    # ─── Energy helpers ───────────────────────────────────
    def _raw_energy(dance_v: float, arousal_v: float, agg_v: float, rel_v: float) -> float:
        """Compute raw energy from weighted signals, then scale to useful range."""
        raw = (dance_v * ENERGY_W_DANCEABILITY
               + arousal_v * ENERGY_W_AROUSAL
               + agg_v * ENERGY_W_AGGRESSIVE
               + (1 - rel_v) * ENERGY_W_RELAXED)
        return float(np.clip(raw * ENERGY_SCALE + ENERGY_OFFSET, 0, 1))

    # ─── Energy formula (whole-track average) ───────────────
    raw_energy = round(_raw_energy(
        danceability, arousal_norm, moods["aggressive"], moods["relaxed"],
    ), 3)

    # ─── Segment analysis ──────────────────────────────────
    num_frames = embeddings.shape[0]
    segment_energies: list[float] = []

    if num_frames >= SEGMENT_LENGTH_SEC:
        for start in range(0, num_frames, SEGMENT_HOP_SEC):
            end = min(start + SEGMENT_LENGTH_SEC, num_frames)
            if end - start < 10:
                break
            seg_emb = embeddings[start:end]

            # Per-segment danceability
            if models.get("danceability") is not None:
                seg_dance = float(np.mean(models["danceability"](seg_emb), axis=0)[0])
            else:
                seg_dance = danceability

            # Per-segment moods (aggressive + relaxed)
            seg_agg = float(np.mean(models["moods"]["aggressive"](seg_emb), axis=0)[0])
            seg_rel = float(np.mean(models["moods"]["relaxed"](seg_emb), axis=0)[0])

            seg_e = _raw_energy(seg_dance, arousal_norm, seg_agg, seg_rel)
            segment_energies.append(seg_e)

    if segment_energies:
        peak_energy = round(max(segment_energies), 3)
        intro_energy = round(segment_energies[0], 3)
        energy_variance = round(float(np.var(segment_energies)), 4)
    else:
        peak_energy = raw_energy
        intro_energy = raw_energy
        energy_variance = 0.0

    # Blend: 70% average + 30% peak
    energy = round(float(np.clip(raw_energy * 0.7 + peak_energy * 0.3, 0, 1)), 3)

    return {
        "genres": genres,
        "electronic_genres": electronic_genres,
        "moods": moods,
        "danceability": danceability,
        "mood_party": mood_party,
        "arousal": arousal_norm,
        "valence": valence_norm,
        "energy": energy,
        "raw_energy": raw_energy,
        "peak_energy": peak_energy,
        "intro_energy": intro_energy,
        "energy_variance": energy_variance,
        "bpm": bpm,
        "key": key_str,
        "key_strength": key_strength,
        "duration": len(audio_44k) / 44100,
    }
