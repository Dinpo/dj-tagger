import numpy as np

from djtagger import classify
from djtagger.classify import (
    ROLE_BUILDER,
    ROLE_CLOSER,
    ROLE_OPENER,
    ROLE_PEAK,
)

SR = 16000
T = {
    "peak_level": 0.80,
    "opener_level": 0.55,
    "peak_genre_pctl": 0.78,
    "opener_genre_pctl": 0.30,
    "genre_min_n": 30,
    "drive_bias": 0.0,
}


# ─── arc momentum (unchanged from v6, still written as a tag) ───

def test_momentum_rising():
    assert classify.arc_momentum([0.3, 0.4, 0.5, 0.6]) > 0.5


def test_momentum_falling():
    assert classify.arc_momentum([0.6, 0.5, 0.4, 0.3]) < -0.5


def test_momentum_single_segment_is_zero():
    assert classify.arc_momentum([0.5]) == 0.0
    assert classify.arc_momentum([]) == 0.0


# ─── drive / emo indices ────────────────────────────────────

def test_drive_emo_driving_track():
    # High flux/onset/rising slope, low valence/vocals, no outro fade.
    drive, emo = classify.drive_emo(
        flux=0.40, onset_rate=3.9, slope=5.0,
        valence=0.50, centroid=1400.0, outro_db=-1.0, vocal=0.05,
    )
    assert drive > emo


def test_drive_emo_emotional_track():
    # Bright, happy, vocal, long fading outro; little drive.
    drive, emo = classify.drive_emo(
        flux=0.27, onset_rate=2.5, slope=-4.0,
        valence=0.75, centroid=2340.0, outro_db=-20.0, vocal=0.55,
    )
    assert emo > drive


# ─── role decision ──────────────────────────────────────────

def test_role_peak_from_high_level():
    assert classify.classify_role(0.90, 0.3, 0.7, T) == ROLE_PEAK


def test_role_opener_from_low_level():
    assert classify.classify_role(0.45, 0.7, 0.3, T) == ROLE_OPENER


def test_role_builder_mid_band_drive_wins():
    assert classify.classify_role(0.65, 0.7, 0.3, T) == ROLE_BUILDER


def test_role_closer_mid_band_emo_wins():
    assert classify.classify_role(0.65, 0.3, 0.7, T) == ROLE_CLOSER


def test_role_genre_percentile_promotes_to_peak():
    # Globally mid energy, but top of its own genre -> promoted to Peak.
    assert classify.classify_role(0.60, 0.3, 0.7, T, genre_pctl=0.90) == ROLE_PEAK


def test_role_absolute_high_energy_is_peak_regardless_of_genre():
    # A 0.90-energy track sitting only mid in a hot genre is still a Peak:
    # genre banding must not demote an absolute banger.
    assert classify.classify_role(0.90, 0.7, 0.3, T, genre_pctl=0.50) == ROLE_PEAK
    assert classify.classify_role(0.90, 0.7, 0.3, T, genre_pctl=0.10) == ROLE_PEAK


def test_role_opener_requires_low_absolute_and_low_genre():
    # Low within its genre but absolutely mid -> NOT Opener (falls to the
    # drive/emo split); only low-on-both is an Opener.
    assert classify.classify_role(0.60, 0.7, 0.3, T, genre_pctl=0.10) == ROLE_BUILDER
    assert classify.classify_role(0.45, 0.7, 0.3, T, genre_pctl=0.10) == ROLE_OPENER
    # Absolutely low but NOT low within its (deep) genre -> not forced Opener.
    assert classify.classify_role(0.45, 0.3, 0.7, T, genre_pctl=0.60) == ROLE_CLOSER


# ─── genre stats helpers ────────────────────────────────────

def _stats():
    # Quantile grid p0..p100 in 5% steps for a fake genre.
    q = list(np.linspace(0.4, 1.0, 21))
    return {"tech house": {"n": 100, "q": q}}


def test_genre_energy_percentile_median():
    p = classify.genre_energy_percentile(0.7, "Tech House; Minimal", _stats(), min_n=30)
    assert p is not None and 0.45 < p < 0.55


def test_genre_energy_percentile_extremes():
    assert classify.genre_energy_percentile(0.39, "tech house", _stats(), min_n=30) == 0.0
    assert classify.genre_energy_percentile(1.0, "tech house", _stats(), min_n=30) == 1.0


def test_genre_energy_percentile_unknown_or_small():
    assert classify.genre_energy_percentile(0.7, "polka", _stats(), min_n=30) is None
    small = {"tech house": {"n": 5, "q": _stats()["tech house"]["q"]}}
    assert classify.genre_energy_percentile(0.7, "tech house", small, min_n=30) is None
    assert classify.genre_energy_percentile(0.7, "", _stats(), min_n=30) is None


# ─── compute_arc integration ────────────────────────────────

def test_compute_arc_returns_all_keys_and_valid_role():
    audio = (0.3 * np.sin(2 * np.pi * 200 * np.arange(3 * SR) / SR)).astype(np.float32)
    seg = [0.4, 0.5, 0.6, 0.7]
    out = classify.compute_arc(audio, SR, seg, energy=0.55, valence=0.5, vocal=0.2)
    for k in ("spectral_centroid", "onset_rate", "dynamic_range", "sub_bass",
              "flux", "vocal", "intro_db", "outro_db", "arc_slope", "drop_db",
              "peak_pos", "arc_level", "arc_momentum", "drive", "emo", "set_role"):
        assert k in out, k
    assert out["set_role"] in {ROLE_OPENER, ROLE_BUILDER, ROLE_PEAK, ROLE_CLOSER}
    assert -1.0 <= out["arc_momentum"] <= 1.0
    assert 0.0 <= out["drive"] <= 1.0
    assert 0.0 <= out["emo"] <= 1.0


def test_load_genre_stats_mtime_invalidation(tmp_path, monkeypatch):
    import json
    import os

    stats_file = tmp_path / "genre_energy.json"
    monkeypatch.setattr(classify, "GENRE_STATS_FILE", str(stats_file))
    # reset the module cache
    classify._genre_stats_cache = None
    classify._genre_stats_mtime = None

    # No file yet -> None, and NOT latched forever
    assert classify.load_genre_stats() is None

    stats_file.write_text(json.dumps({"house": {"n": 50, "q": [0.1, 0.9]}}))
    loaded = classify.load_genre_stats()
    assert loaded is not None and "house" in loaded

    # Rewrite with new content and a newer mtime -> picked up
    stats_file.write_text(json.dumps({"techno": {"n": 40, "q": [0.2, 1.0]}}))
    st = stats_file.stat()
    os.utime(stats_file, (st.st_atime, st.st_mtime + 5))
    reloaded = classify.load_genre_stats()
    assert reloaded is not None and "techno" in reloaded and "house" not in reloaded


def test_genre_energy_percentile_malformed_entry():
    bad = {"tech house": {"n": 100, "q": ["not", "numbers"]}}
    assert classify.genre_energy_percentile(0.7, "tech house", bad, min_n=30) is None
