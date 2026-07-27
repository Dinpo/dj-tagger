import numpy as np
import pytest

from djtagger import dsp

SR = 16000


def _sine(freq, seconds=2.0, sr=SR, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_spectral_centroid_tracks_tone_frequency():
    c = dsp.spectral_centroid(_sine(1000), SR)
    assert 850 < c < 1150


def test_spectral_centroid_silence_is_zero():
    assert dsp.spectral_centroid(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_sub_bass_ratio_high_for_low_tone():
    assert dsp.sub_bass_ratio(_sine(60), SR) > 0.8


def test_sub_bass_ratio_low_for_bright_tone():
    assert dsp.sub_bass_ratio(_sine(4000), SR) < 0.2


def test_dynamic_range_small_for_constant_amplitude():
    assert dsp.dynamic_range(_sine(440), SR) < 6.0


def test_dynamic_range_large_for_quiet_then_loud():
    quiet = _sine(440, seconds=1.0, amp=0.01)
    loud = _sine(440, seconds=1.0, amp=1.0)
    sig = np.concatenate([quiet, loud])
    assert dsp.dynamic_range(sig, SR) > 20.0


def test_onset_density_high_for_click_train():
    # 4 clicks per second for 3 seconds.
    sig = np.zeros(3 * SR, dtype=np.float32)
    for i in range(12):
        sig[int(i / 4 * SR)] = 1.0
    rate = dsp.onset_density(sig, SR)
    assert 2.5 < rate < 6.0


def test_onset_density_low_for_steady_tone():
    assert dsp.onset_density(_sine(440, seconds=3.0), SR) < 1.0


def test_spectral_flux_higher_for_changing_signal():
    # Click train changes spectrum constantly; steady tone barely changes.
    sig = np.zeros(3 * SR, dtype=np.float32)
    for i in range(12):
        sig[int(i / 4 * SR)] = 1.0
    assert dsp.spectral_flux(sig, SR) > dsp.spectral_flux(_sine(440, seconds=3.0), SR)


def test_spectral_flux_silence_is_zero():
    assert dsp.spectral_flux(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_spectral_flux_amplitude_invariant():
    sig = np.zeros(3 * SR, dtype=np.float32)
    for i in range(12):
        sig[int(i / 4 * SR)] = 1.0
    a = dsp.spectral_flux(sig, SR)
    b = dsp.spectral_flux(0.05 * sig, SR)
    assert abs(a - b) < 0.05 * max(a, 1e-9)


def test_loudness_arc_quiet_intro_and_rising():
    quiet = _sine(440, seconds=30.0, amp=0.02)
    loud = _sine(440, seconds=60.0, amp=0.8)
    arc = dsp.loudness_arc(np.concatenate([quiet, loud]), SR)
    assert arc["intro_db"] < -10.0     # intro much quieter than the peak
    assert arc["slope"] > 5.0          # loudness rises over the track
    assert arc["peak_pos"] > 0.3       # loudest moment is not at the start


def test_loudness_arc_flat_tone_is_flat():
    arc = dsp.loudness_arc(_sine(440, seconds=90.0, amp=0.5), SR)
    assert abs(arc["intro_db"]) < 2.0
    assert abs(arc["outro_db"]) < 2.0
    assert abs(arc["slope"]) < 1.0
    assert arc["drop_db"] < 3.0


def test_loudness_arc_detects_breakdown_drop():
    body1 = _sine(440, seconds=40.0, amp=0.8)
    breakdown = _sine(440, seconds=15.0, amp=0.05)
    body2 = _sine(440, seconds=40.0, amp=0.9)
    arc = dsp.loudness_arc(np.concatenate([body1, breakdown, body2]), SR)
    assert arc["drop_db"] > 15.0


def test_loudness_arc_short_audio_neutral():
    arc = dsp.loudness_arc(_sine(440, seconds=1.0), SR)
    assert arc["slope"] == 0.0
    assert arc["drop_db"] == 0.0


def test_precomputed_frames_and_mags_match_defaults():
    # Sharing frames/mags between features must not change any value.
    sig = np.zeros(3 * SR, dtype=np.float32)
    for i in range(12):
        sig[int(i / 4 * SR)] = 1.0
    sig += _sine(300, seconds=3.0, amp=0.1)

    m = dsp._magnitude_spectra(sig, 1024, 512)
    assert dsp.spectral_flux(sig, SR) == dsp.spectral_flux(sig, SR, mags=m)
    assert dsp.onset_density(sig, SR) == dsp.onset_density(sig, SR, mags=m)

    f = dsp._frames(sig, 2048, 1024)
    assert dsp.spectral_centroid(sig, SR) == dsp.spectral_centroid(sig, SR, frames=f)
    assert dsp.dynamic_range(sig, SR) == dsp.dynamic_range(sig, SR, frames=f)


def test_loudness_arc_short_track_edges_do_not_overlap():
    # 30 s track: fixed 20 s edges would overlap; the adaptive cap keeps
    # intro and outro windows distinct so a genuine ramp still registers.
    quiet = _sine(440, seconds=15.0, amp=0.02)
    loud = _sine(440, seconds=15.0, amp=0.8)
    arc = dsp.loudness_arc(np.concatenate([quiet, loud]), SR)
    assert arc["intro_db"] < arc["outro_db"] - 10.0


def test_pulse_regularity_high_for_regular_beat():
    # A 120 BPM click train (period 0.5s) has a strong regular pulse;
    # an irregular click train is much weaker; a steady tone (no transients)
    # is gated to zero.
    sig = np.zeros(8 * SR, dtype=np.float32)
    period = int(0.5 * SR)                 # 120 BPM
    for i in range(16):
        sig[i * period] = 1.0
    regular = dsp.pulse_regularity(sig, SR)

    steady = dsp.pulse_regularity(_sine(440, seconds=8.0), SR)

    rng = np.random.RandomState(0)
    irr = np.zeros(8 * SR, dtype=np.float32)
    pos = np.cumsum(rng.randint(int(0.15 * SR), int(0.9 * SR), size=30))
    for p in pos[pos < len(irr)]:
        irr[p] = 1.0
    irregular = dsp.pulse_regularity(irr, SR)

    assert regular > 0.4
    assert steady == 0.0          # tonal, no transients -> gated
    assert regular > irregular
