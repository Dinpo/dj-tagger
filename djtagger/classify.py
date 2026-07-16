"""Set-role classification v7: energy bands + drive/emo character split.

Roles follow the DJ set taxonomy:
  Opener  (energy 3-5): hypnotic, deep, atmospheric; low activity.
  Builder (energy 6-7): driving, urgent, rising; promises something bigger.
  Peak    (energy 8-10): maximum intensity, drops, hooks.
  Closer  (energy 5-7): emotional, anthemic, melodic, fades out.

Energy bands decide Opener vs Peak. The shared mid band splits into
Builder vs Closer by comparing a "drive" index (spectral flux, onset rate,
rising loudness) against an "emo" index (valence, brightness, vocals,
fading outro). When a genre-energy stats file exists, bands come from the
track's energy percentile WITHIN its genre instead of the global scale.

Pure Python (numpy + config + dsp only), no Essentia, so it is unit-testable.
"""

import json
import os

import numpy as np

from . import dsp
from .config import (
    FEATURE_RANGE,
    GENRE_STATS_FILE,
    MOMENTUM_SCALE,
    ROLE_THRESHOLDS,
    ROLE_BUILDER,
    ROLE_CLOSER,
    ROLE_OPENER,
    ROLE_PEAK,
)

__all__ = [
    "ROLE_OPENER", "ROLE_BUILDER", "ROLE_PEAK", "ROLE_CLOSER",
    "arc_momentum", "drive_emo", "classify_role", "compute_arc",
    "load_genre_stats", "genre_energy_percentile", "decide_role",
]


def _normalize(x: float, lo: float, hi: float) -> float:
    """Scale x from [lo, hi] into [0, 1], clipped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def arc_momentum(segment_energies) -> float:
    """Normalized internal energy slope in [-1, 1].

    Kept as an informational tag: the mood-segment slope is nearly flat on
    most tracks, so the v7 role decision uses the loudness-arc slope instead.
    """
    if segment_energies is None or len(segment_energies) < 2:
        return 0.0
    y = np.asarray(segment_energies, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.clip(slope * MOMENTUM_SCALE, -1.0, 1.0))


def drive_emo(flux, onset_rate, slope, valence, centroid, outro_db, vocal):
    """Compute the (drive, emo) character indices, each in [0, 1].

    drive: musical activity and forward push (Builder character).
    emo: brightness, positivity, vocals, and a fading finale (Closer character).
    """
    drive = (
        _normalize(flux, *FEATURE_RANGE["flux"])
        + _normalize(onset_rate, *FEATURE_RANGE["onset_rate"])
        + _normalize(max(0.0, slope), *FEATURE_RANGE["rise"])
    ) / 3.0
    emo = (
        _normalize(valence, *FEATURE_RANGE["valence"])
        + _normalize(centroid, *FEATURE_RANGE["spectral_centroid"])
        + _normalize(vocal, *FEATURE_RANGE["vocal"])
        + _normalize(-outro_db, *FEATURE_RANGE["outro_fade"])
    ) / 4.0
    return round(float(drive), 3), round(float(emo), 3)


def classify_role(arc_level, drive, emo, thresholds=None, genre_pctl=None) -> str:
    """Map energy band + character indices to a discrete role.

    genre_pctl, when given, is the track's energy percentile within its own
    genre cohort (0..1) and replaces the global energy bands entirely.
    """
    t = thresholds or ROLE_THRESHOLDS
    if genre_pctl is not None:
        if genre_pctl >= t["peak_genre_pctl"]:
            return ROLE_PEAK
        if genre_pctl <= t["opener_genre_pctl"]:
            return ROLE_OPENER
    else:
        if arc_level >= t["peak_level"]:
            return ROLE_PEAK
        if arc_level <= t["opener_level"]:
            return ROLE_OPENER
    return ROLE_BUILDER if drive >= emo + t["drive_bias"] else ROLE_CLOSER


# ─── Genre-relative energy bands ────────────────────────────

_genre_stats_cache: dict | None = None
_genre_stats_mtime: float | None = None


def load_genre_stats(path: str | None = None) -> dict | None:
    """Load the per-genre energy quantile table, or None if absent/invalid.

    The table is produced by `djtagger genre-stats`. The default-path load
    is cached by file mtime, so a table (re)written mid-run, even by another
    process, is picked up on the next call instead of being latched forever.
    """
    global _genre_stats_cache, _genre_stats_mtime
    p = path or GENRE_STATS_FILE
    try:
        mtime = os.stat(p).st_mtime
    except OSError:
        if path is None:
            _genre_stats_cache, _genre_stats_mtime = None, None
        return None
    if path is None and mtime == _genre_stats_mtime and _genre_stats_cache is not None:
        return _genre_stats_cache
    stats = None
    try:
        with open(p) as f:
            stats = json.load(f)
    except Exception as ex:
        # A present-but-unreadable table is a real problem worth surfacing:
        # silently falling back to global bands hid this failure for weeks.
        import sys
        print(f"[djtagger] Warning: cannot read genre stats {p}: {ex}. "
              f"Falling back to global energy bands.", file=sys.stderr)
        stats = None
    if path is None:
        _genre_stats_cache, _genre_stats_mtime = stats, mtime
    return stats


def primary_genre(genre: str) -> str:
    """First genre segment, lowercased ('Tech House; Minimal' -> 'tech house')."""
    if not genre:
        return ""
    return genre.split(";")[0].strip().lower()


def genre_energy_percentile(energy, genre, stats, min_n=None) -> float | None:
    """Energy percentile (0..1) within the track's genre cohort.

    Returns None when stats are missing, the genre is unknown, or the cohort
    is too small to trust; callers then fall back to global energy bands.
    """
    if min_n is None:
        min_n = ROLE_THRESHOLDS["genre_min_n"]
    if not stats:
        return None
    entry = stats.get(primary_genre(genre))
    if not entry or entry.get("n", 0) < min_n:
        return None
    q = entry.get("q")
    if not q or len(q) < 2:
        return None
    try:
        if energy <= q[0]:
            return 0.0
        if energy >= q[-1]:
            return 1.0
        # q is a quantile grid p0..p100; interpolate energy's position in it.
        pos = float(np.interp(energy, q, np.linspace(0.0, 1.0, len(q))))
    except (TypeError, ValueError):
        # Malformed entry (non-numeric quantiles etc.): fall back to global bands.
        return None
    return round(pos, 3)


def decide_role(result: dict, genre: str = "", stats: dict | None = None) -> str:
    """Final role for an analyzed track, genre-aware when stats permit.

    `result` is the analyzer's result dict (needs arc_level, drive, emo).
    Used by the CLI after genre resolution; falls back to the global bands
    (identical to the role compute_arc assigned) when no cohort applies.
    """
    if stats is None:
        stats = load_genre_stats()
    pctl = genre_energy_percentile(result["arc_level"], genre, stats)
    return classify_role(result["arc_level"], result["drive"], result["emo"],
                         genre_pctl=pctl)


def compute_arc(audio, sr, segment_energies, energy, valence, vocal=0.0) -> dict:
    """Compute DSP features, character indices, and the (global-band) role."""
    centroid = dsp.spectral_centroid(audio, sr)
    onset = dsp.onset_density(audio, sr)
    dyn = dsp.dynamic_range(audio, sr)
    subb = dsp.sub_bass_ratio(audio, sr)
    flux = dsp.spectral_flux(audio, sr)
    arc = dsp.loudness_arc(audio, sr)

    level = float(np.clip(energy, 0.0, 1.0))
    momentum = arc_momentum(segment_energies)
    drive, emo = drive_emo(flux, onset, arc["slope"], valence, centroid,
                           arc["outro_db"], vocal)
    role = classify_role(level, drive, emo)

    return {
        "spectral_centroid": round(centroid, 2),
        "onset_rate": round(onset, 3),
        "dynamic_range": round(dyn, 2),
        "sub_bass": round(subb, 4),
        "flux": round(flux, 4),
        "vocal": round(float(vocal), 3),
        "intro_db": arc["intro_db"],
        "outro_db": arc["outro_db"],
        "arc_slope": arc["slope"],
        "drop_db": arc["drop_db"],
        "peak_pos": arc["peak_pos"],
        "arc_level": round(level, 3),
        "arc_momentum": round(momentum, 3),
        "drive": drive,
        "emo": emo,
        "set_role": role,
    }
