import numpy as np

from djtagger import classify
from djtagger.classify import (
    ROLE_BUILDER,
    ROLE_CLOSER,
    ROLE_PEAK,
    ROLE_WARMUP,
)

SR = 16000
T = {
    "peak_level": 0.80,
    "peak_level_soft": 0.72,
    "peak_intensity": 0.65,
    "rising": 0.15,
    "falling": -0.15,
    "release_valence": 0.66,
    "release_bright": 0.60,
}


def test_momentum_rising():
    m = classify.arc_momentum([0.3, 0.4, 0.5, 0.6])
    assert m > 0.5


def test_momentum_falling():
    m = classify.arc_momentum([0.6, 0.5, 0.4, 0.3])
    assert m < -0.5


def test_momentum_flat_is_near_zero():
    assert abs(classify.arc_momentum([0.5, 0.5, 0.5, 0.5])) < 0.05


def test_momentum_single_segment_is_zero():
    assert classify.arc_momentum([0.5]) == 0.0
    assert classify.arc_momentum([]) == 0.0


def test_role_peak_from_high_level():
    assert classify.classify_role(0.9, 0.0, 0.5, 0.5, 0.5, T) == ROLE_PEAK


def test_role_peak_from_soft_gate():
    assert classify.classify_role(0.74, 0.0, 0.5, 0.5, 0.9, T) == ROLE_PEAK


def test_role_builder_when_rising():
    assert classify.classify_role(0.5, 0.5, 0.5, 0.4, 0.4, T) == ROLE_BUILDER


def test_role_closer_when_falling():
    assert classify.classify_role(0.5, -0.5, 0.5, 0.4, 0.4, T) == ROLE_CLOSER


def test_role_closer_flat_but_bright():
    # Flat momentum, released/bright -> Closer.
    assert classify.classify_role(0.5, 0.0, 0.75, 0.4, 0.4, T) == ROLE_CLOSER


def test_role_warmup_flat_dark_low():
    assert classify.classify_role(0.45, 0.0, 0.5, 0.3, 0.3, T) == ROLE_WARMUP


def test_builder_vs_closer_split_on_momentum_at_same_level():
    builder = classify.classify_role(0.5, 0.4, 0.5, 0.4, 0.4, T)
    closer = classify.classify_role(0.5, -0.4, 0.5, 0.4, 0.4, T)
    assert builder == ROLE_BUILDER
    assert closer == ROLE_CLOSER


def test_compute_arc_returns_all_keys_and_valid_role():
    audio = (0.3 * np.sin(2 * np.pi * 200 * np.arange(3 * SR) / SR)).astype(np.float32)
    seg = [0.4, 0.5, 0.6, 0.7]
    out = classify.compute_arc(audio, SR, seg, energy=0.55, valence=0.5)
    for k in ("spectral_centroid", "onset_rate", "dynamic_range", "sub_bass",
              "arc_level", "arc_momentum", "set_role"):
        assert k in out
    assert out["set_role"] in {ROLE_WARMUP, ROLE_BUILDER, ROLE_PEAK, ROLE_CLOSER}
    assert -1.0 <= out["arc_momentum"] <= 1.0
