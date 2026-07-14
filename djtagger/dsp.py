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


def spectral_flux(audio, sr, frame_size=1024, hop=512) -> float:
    """Mean positive spectral flux, normalized by mean frame magnitude.

    Measures how fast the spectrum changes over time (musical activity and
    drive). Normalizing by the mean total magnitude makes the value
    amplitude-invariant: numerator and denominator scale together.
    """
    mags = _magnitude_spectra(audio, frame_size, hop)
    if mags.shape[0] < 3:
        return 0.0
    denom = float(mags.sum(axis=1)[1:].mean())
    if denom <= 1e-8:
        return 0.0
    flux = np.maximum(0.0, np.diff(mags, axis=0)).sum(axis=1)
    return float(flux.mean() / denom)


def loudness_arc(audio, sr, frame_sec=1.0, hop_sec=0.5, edge_sec=20.0) -> dict:
    """Shape of the track's loudness envelope over time.

    Returns a dict with:
      intro_db / outro_db: mean dB below the track's loudest moment over the
        first / last edge_sec seconds (more negative = quieter edge)
      slope: linear dB trend across the track (positive = gets louder)
      drop_db: largest rise from a quiet moment to a loud one within an
        8 s look-back window (breakdown-to-drop height)
      peak_pos: position of the loudest frame, 0..1
    """
    neutral = {"intro_db": 0.0, "outro_db": 0.0, "slope": 0.0,
               "drop_db": 0.0, "peak_pos": 0.5}
    frame = int(frame_sec * sr)
    hop = int(hop_sec * sr)
    frames = _frames(audio, frame, hop)
    if frames.shape[0] < 8:
        return neutral
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
    rel = 20.0 * np.log10(rms)
    rel = rel - rel.max()

    fps = 1.0 / hop_sec
    edge = max(1, int(edge_sec * fps))
    intro_db = float(np.mean(rel[:edge]))
    outro_db = float(np.mean(rel[-edge:]))

    x = np.linspace(0.0, 1.0, len(rel))
    slope = float(np.polyfit(x, rel, 1)[0])

    look = max(1, int(8 * fps))
    drop_db = 0.0
    for t in range(look, len(rel)):
        drop_db = max(drop_db, float(rel[t] - rel[t - look:t].min()))

    peak_pos = float(int(np.argmax(rel)) / max(1, len(rel) - 1))
    return {"intro_db": round(intro_db, 2), "outro_db": round(outro_db, 2),
            "slope": round(slope, 2), "drop_db": round(drop_db, 2),
            "peak_pos": round(peak_pos, 3)}


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
    # absolute floor: steady-tone leakage sits near 0 while real onsets are ~1, so 0.05 excludes leakage with wide margin
    threshold = max(flux.mean() + flux.std(), 0.05)
    onsets = 0
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            onsets += 1
    duration = len(audio) / sr
    return float(onsets / duration) if duration > 0 else 0.0
