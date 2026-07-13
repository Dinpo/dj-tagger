"""Pure-numpy low-level audio features for set-role classification.

Deliberately free of Essentia so it can be unit-tested with synthetic
signals. All functions take a 1-D float mono array plus a sample rate.
"""

import numpy as np


def _frames(audio: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """Slice audio into overlapping frames (frames x frame_size)."""
    if len(audio) < frame_size:
        if len(audio) == 0:
            return np.empty((0, frame_size), dtype=np.float32)
        pad = np.zeros(frame_size, dtype=np.float32)
        pad[: len(audio)] = audio
        return pad[None, :]
    n = 1 + (len(audio) - frame_size) // hop
    idx = np.arange(frame_size)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def _magnitude_spectra(audio: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """Windowed rFFT magnitude per frame (frames x bins)."""
    frames = _frames(audio, frame_size, hop)
    if frames.shape[0] == 0:
        return np.empty((0, frame_size // 2 + 1))
    window = np.hanning(frame_size).astype(np.float32)
    return np.abs(np.fft.rfft(frames * window, axis=1))


def spectral_centroid(audio, sr, frame_size=2048, hop=1024) -> float:
    """Magnitude-weighted mean frequency (Hz), averaged over frames."""
    mags = _magnitude_spectra(audio, frame_size, hop)
    if mags.shape[0] == 0:
        return 0.0
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    per_frame_sum = mags.sum(axis=1)
    active = per_frame_sum > 1e-8
    if not active.any():
        return 0.0
    centroids = (mags[active] @ freqs) / per_frame_sum[active]
    return float(np.mean(centroids))


def sub_bass_ratio(audio, sr, cutoff=120.0) -> float:
    """Fraction of spectral energy below `cutoff` Hz (0..1)."""
    frame_size = 4096
    mags = _magnitude_spectra(audio, frame_size, frame_size // 2)
    if mags.shape[0] == 0:
        return 0.0
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    energy = (mags ** 2).sum(axis=0)
    total = energy.sum()
    if total <= 1e-12:
        return 0.0
    return float(energy[freqs < cutoff].sum() / total)


def dynamic_range(audio, sr, frame_size=2048, hop=1024) -> float:
    """Spread of frame loudness in dB: p90 minus p10 of per-frame RMS."""
    frames = _frames(audio, frame_size, hop)
    if frames.shape[0] == 0:
        return 0.0
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-8)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def onset_density(audio, sr, frame_size=1024, hop=512) -> float:
    """Onsets per second via peak-picked spectral flux.

    Flux is normalised by the mean total magnitude per frame (an
    absolute, signal-scaled reference) rather than by its own max.
    Normalising by flux.max() is scale-invariant, so a steady tone's
    residual frame-to-frame spectral leakage (present even for a
    perfectly stationary signal, since its frequency rarely lands
    exactly on an FFT bin) gets stretched to fill the full 0..1 range
    and produces spurious "onsets" once picked against a mean+std
    threshold. Requiring the threshold to also clear an absolute
    floor keeps that leakage noise from being mistaken for real
    transients while still catching genuine onsets, whose relative
    flux is orders of magnitude larger.
    """
    mags = _magnitude_spectra(audio, frame_size, hop)
    if mags.shape[0] < 3:
        return 0.0
    mag_sum = mags.sum(axis=1)
    scale = mag_sum.mean()
    if scale <= 1e-8:
        return 0.0
    flux = np.maximum(0.0, np.diff(mags, axis=0)).sum(axis=1) / scale
    if flux.max() <= 1e-8:
        return 0.0
    threshold = max(flux.mean() + flux.std(), 0.05)
    onsets = 0
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            onsets += 1
    duration = len(audio) / sr
    return float(onsets / duration) if duration > 0 else 0.0
