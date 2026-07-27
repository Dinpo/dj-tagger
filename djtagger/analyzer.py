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
from . import classify

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

    # Voice/instrumental head (v7): vocal presence feeds the set-role split
    models["voice"] = _try_load(
        es.TensorflowPredict2D,
        f"{d}/voice_instrumental-discogs-effnet-1.pb",
        input="model/Placeholder",
        output="model/Softmax",
    )

    # TempoCNN for precise BPM detection
    models["tempo_cnn"] = _try_load(
        es.TensorflowPredictTempoCNN,
        f"{d}/deepsquare-k16-3.pb",
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

def analyze_track(
    filepath: str,
    models: dict,
    detect_bpm_key: bool = False,
) -> dict:
    """Run ML analysis on an audio file.

    When *detect_bpm_key* is False (default), BPM/key detection is skipped
    and the returned ``bpm`` / ``key`` fields are 0 / "". This avoids
    overwriting values a DJ tool (Serato etc.) has already written, and
    also skips the 44.1 kHz audio load that only those algorithms need.

    Returns dict with keys: genres, electronic_genres, moods, danceability,
    arousal, valence, energy, raw_energy, peak_energy, intro_energy,
    energy_variance, spectral_centroid, onset_rate, dynamic_range, sub_bass,
    flux, pulse_reg, vocal, intro_db, outro_db, arc_slope, drop_db, peak_pos,
    arc_level, arc_momentum, drive, emo, set_role, bpm, key, key_strength,
    duration.
    """
    audio = es.MonoLoader(filename=filepath, sampleRate=16000)()
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
    # Per-frame predictions are kept: the segment analysis below derives
    # per-segment scores by slicing them (the heads are frame-wise, so a
    # slice mean is identical to re-running the model on the segment).
    moods = {}
    mood_frame_preds: dict[str, np.ndarray] = {}
    for mood_name, model in models["moods"].items():
        preds = model(embeddings)
        mood_frame_preds[mood_name] = preds
        moods[mood_name] = round(float(np.mean(preds, axis=0)[0]), 3)

    # ─── Danceability ───────────────────────────────────────
    dance_frame_preds = None
    if models.get("danceability") is not None:
        dance_frame_preds = models["danceability"](embeddings)
        danceability = round(float(np.mean(dance_frame_preds, axis=0)[0]), 3)
    else:
        # Fallback: estimate from moods
        danceability = round(float(np.clip(
            (moods["happy"] + moods["aggressive"] + (1 - moods["sad"])) / 3, 0, 1
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

    # ─── BPM + Key (opt-in: DJ software usually owns these) ──
    bpm = 0
    key_str = ""
    key_strength = 0.0
    if detect_bpm_key:
        audio_44k = None  # loaded lazily — only if something below needs it

        try:
            if models.get("tempo_cnn") is not None:
                # TempoCNN: load at 11025 Hz, probability over 256 BPM bins (30-286)
                audio_11k = es.MonoLoader(filename=filepath, sampleRate=11025)()
                tempo_preds = models["tempo_cnn"](audio_11k)
                avg_preds = np.mean(tempo_preds, axis=0)
                peak_bin = int(np.argmax(avg_preds))
                # Weighted average around peak for sub-BPM precision
                window = 5
                lo = max(0, peak_bin - window)
                hi = min(len(avg_preds), peak_bin + window + 1)
                bpm = round(float(np.average(
                    np.arange(lo, hi) + 30, weights=avg_preds[lo:hi]
                )), 2)
            else:
                # DSP fallback
                audio_44k = es.MonoLoader(filename=filepath, sampleRate=44100)()
                rhythm = es.RhythmExtractor2013(method="multifeature")
                bpm_val, _, _, _, _ = rhythm(audio_44k)
                bpm = round(float(bpm_val), 2)
        except Exception:
            bpm = 0

        try:
            if audio_44k is None:
                audio_44k = es.MonoLoader(filename=filepath, sampleRate=44100)()
            key_extractor = es.KeyExtractor(profileType="edmm")
            key_name, scale, ks = key_extractor(audio_44k)
            standard_key = f"{key_name}{'m' if scale == 'minor' else ''}"
            key_str = CAMELOT_MAP.get(standard_key, standard_key)
            key_strength = round(float(ks), 3)
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

            # Per-segment scores from the stored whole-track per-frame
            # predictions: no model re-runs per segment (saves ~3 model
            # calls per 15 s of audio, the analysis hot path).
            if dance_frame_preds is not None:
                seg_dance = float(np.mean(dance_frame_preds[start:end], axis=0)[0])
            else:
                seg_dance = danceability
            seg_agg = float(np.mean(mood_frame_preds["aggressive"][start:end], axis=0)[0])
            seg_rel = float(np.mean(mood_frame_preds["relaxed"][start:end], axis=0)[0])

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

    # ─── Vocal presence (v7, reuses the EffNet embeddings) ──
    vocal = 0.0
    try:
        if models.get("voice") is not None:
            voice_preds = models["voice"](embeddings)
            # classes are [instrumental, voice]; index 1 = voice probability
            vocal = round(float(np.mean(voice_preds, axis=0)[1]), 3)
    except Exception:
        vocal = 0.0

    # Set-role classification (v7). Reuses the 16 kHz audio already in memory.
    try:
        arc = classify.compute_arc(
            audio, 16000, segment_energies, energy, valence_norm, vocal,
        )
        arc["arc_ok"] = True
    except Exception as ex:
        # Do NOT fabricate a role from neutral values: an empty role is
        # honest (comment omits it, callers can detect it via arc_ok), and
        # the failure is surfaced instead of silently mislabeling the track.
        print(
            f"[djtagger] Warning: arc analysis failed for "
            f"{os.path.basename(filepath)}: {ex}",
            file=sys.stderr,
        )
        arc = {
            "spectral_centroid": 0.0, "onset_rate": 0.0, "dynamic_range": 0.0,
            "sub_bass": 0.0, "flux": 0.0, "pulse_reg": 0.0, "vocal": vocal,
            "intro_db": 0.0, "outro_db": 0.0, "arc_slope": 0.0,
            "drop_db": 0.0, "peak_pos": 0.5,
            "arc_level": energy, "arc_momentum": 0.0,
            "drive": 0.0, "emo": 0.0,
            "set_role": "",
            "arc_ok": False,
        }

    return {
        "genres": genres,
        "electronic_genres": electronic_genres,
        "moods": moods,
        "danceability": danceability,
        "arousal": arousal_norm,
        "valence": valence_norm,
        "energy": energy,
        "raw_energy": raw_energy,
        "peak_energy": peak_energy,
        "intro_energy": intro_energy,
        "energy_variance": energy_variance,
        "spectral_centroid": arc["spectral_centroid"],
        "onset_rate": arc["onset_rate"],
        "dynamic_range": arc["dynamic_range"],
        "sub_bass": arc["sub_bass"],
        "flux": arc["flux"],
        "pulse_reg": arc["pulse_reg"],
        "vocal": arc["vocal"],
        "intro_db": arc["intro_db"],
        "outro_db": arc["outro_db"],
        "arc_slope": arc["arc_slope"],
        "drop_db": arc["drop_db"],
        "peak_pos": arc["peak_pos"],
        "arc_level": arc["arc_level"],
        "arc_momentum": arc["arc_momentum"],
        "drive": arc["drive"],
        "emo": arc["emo"],
        "set_role": arc["set_role"],
        "arc_ok": arc["arc_ok"],
        "bpm": bpm,
        "key": key_str,
        "key_strength": key_strength,
        "duration": len(audio) / 16000,
    }
