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
