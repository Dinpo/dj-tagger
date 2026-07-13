"""Set-role classification: arc math plus the role decision.

Pure Python (numpy + config only), no Essentia, so it is unit-testable.
"""

import numpy as np

from . import dsp
from .config import (
    FEATURE_RANGE,
    MOMENTUM_SCALE,
    ROLE_THRESHOLDS,
    ROLE_BUILDER,
    ROLE_CLOSER,
    ROLE_PEAK,
    ROLE_WARMUP,
)

__all__ = [
    "ROLE_WARMUP", "ROLE_BUILDER", "ROLE_PEAK", "ROLE_CLOSER",
    "arc_momentum", "classify_role", "compute_arc",
]


def _normalize(x: float, lo: float, hi: float) -> float:
    """Scale x from [lo, hi] into [0, 1], clipped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def arc_momentum(segment_energies) -> float:
    """Normalized internal energy slope in [-1, 1].

    Positive means the track rises across its own segments (builder-like),
    negative means it falls (closer/outro-like), near zero means flat.
    """
    if segment_energies is None or len(segment_energies) < 2:
        return 0.0
    y = np.asarray(segment_energies, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.clip(slope * MOMENTUM_SCALE, -1.0, 1.0))


def classify_role(arc_level, arc_momentum_v, valence, brightness,
                  intensity_index, thresholds=None) -> str:
    """Map the two axes plus tie-break signals to a discrete role."""
    t = thresholds or ROLE_THRESHOLDS
    if arc_level >= t["peak_level"]:
        return ROLE_PEAK
    if arc_level >= t["peak_level_soft"] and intensity_index >= t["peak_intensity"]:
        return ROLE_PEAK
    if arc_momentum_v >= t["rising"]:
        return ROLE_BUILDER
    if arc_momentum_v <= t["falling"]:
        return ROLE_CLOSER
    # Flat momentum, non-peak: released/bright reads as a Closer, else Warm-up.
    if valence >= t["release_valence"] or brightness >= t["release_bright"]:
        return ROLE_CLOSER
    return ROLE_WARMUP


def compute_arc(audio, sr, segment_energies, energy, valence) -> dict:
    """Compute DSP features, the two axes, and the role for one track."""
    centroid = dsp.spectral_centroid(audio, sr)
    onset = dsp.onset_density(audio, sr)
    dyn = dsp.dynamic_range(audio, sr)
    subb = dsp.sub_bass_ratio(audio, sr)

    brightness = _normalize(centroid, *FEATURE_RANGE["spectral_centroid"])
    intensity_index = (
        _normalize(subb, *FEATURE_RANGE["sub_bass"])
        + _normalize(dyn, *FEATURE_RANGE["dynamic_range"])
        + _normalize(onset, *FEATURE_RANGE["onset_rate"])
    ) / 3.0

    level = float(np.clip(energy, 0.0, 1.0))
    momentum = arc_momentum(segment_energies)
    role = classify_role(level, momentum, valence, brightness, intensity_index)

    return {
        "spectral_centroid": round(centroid, 2),
        "onset_rate": round(onset, 3),
        "dynamic_range": round(dyn, 2),
        "sub_bass": round(subb, 4),
        "arc_level": round(level, 3),
        "arc_momentum": round(momentum, 3),
        "set_role": role,
    }
