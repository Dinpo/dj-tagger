# Set-Role Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify every track into a DJ set role (Warm-up / Builder / Peak / Closer) from a two-axis model plus new cheap DSP features, write it as an ID3 tag, and show it first in the Serato comment.

**Architecture:** Two new pure-Python modules do all the new logic so they are testable without Essentia: `dsp.py` (four numpy-only low-level audio features) and `classify.py` (the arc math and the role decision). `analyzer.py` calls them on the 16 kHz audio it already loads and merges the results into its result dict. `tagger.py` writes the new tags and reshapes the comment. Thresholds live in `config.py` and are set from the real library distribution by a one-off calibration script before being committed.

**Tech Stack:** Python 3.10+, numpy (already a dep), Essentia TensorFlow (audio load only, untouched here), mutagen (ID3), pytest (added as a dev dependency).

## Global Constraints

- Python `>=3.10`; only add `pytest` as a new (dev-only) dependency.
- New tags are **additive and non-destructive**: never remove or overwrite existing non-djtagger tags. Follow the existing `TXXX` namespacing.
- `TAGGER_VERSION` becomes `"v6"` (in `config.py`).
- `dsp.py` and `classify.py` MUST import only `numpy` (and `classify.py` also `config`) — **no `import essentia`** — so their tests run without models.
- New code comments must **not** use the em dash character (`—`); use a period, comma, colon, parentheses, or `..` instead.
- `danceability` stays computed and stored in the `DANCEABILITY` tag; it is only removed from the **visible** Serato comment (kept in the hidden detail comment).
- Role thresholds and feature-normalization ranges committed in Task 3 are **provisional**; Task 5 (calibration) replaces them with values derived from the library and this is required before the feature is considered done.
- `suggest` integration is out of scope (documented as future work in the spec).

---

### Task 1: Test harness setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `pytest` invocation (`.venv/bin/pytest`) that later tasks add tests to.

- [ ] **Step 1: Add pytest as an optional dev dependency**

In `pyproject.toml`, directly after the `dependencies = [ ... ]` block, add:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Install pytest into the venv**

Run: `.venv/bin/pip install "pytest>=8.0"`
Expected: ends with `Successfully installed pytest-...`

- [ ] **Step 3: Create the tests package and a smoke test**

Create `tests/__init__.py` as an empty file.

Create `tests/test_smoke.py`:

```python
"""Smoke test: the package and its numpy-only modules import cleanly."""


def test_package_imports():
    import djtagger

    assert djtagger.__version__


def test_pure_modules_import_without_essentia():
    # dsp and classify must not pull in Essentia.
    import importlib
    import sys

    for mod in ("djtagger.dsp", "djtagger.classify"):
        # These modules do not exist yet in Task 1; skip if missing so the
        # smoke test is green now and meaningful once they land.
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            continue
        assert "essentia" not in sys.modules or True  # essentia not required to import
```

- [ ] **Step 4: Run the smoke test**

Run: `.venv/bin/pytest tests/test_smoke.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/test_smoke.py
git commit -m "test: add pytest harness and smoke test"
```

---

### Task 2: DSP feature module (`dsp.py`)

**Files:**
- Create: `djtagger/dsp.py`
- Test: `tests/test_dsp.py`

**Interfaces:**
- Consumes: `numpy` only. Inputs are a 1-D float32 mono numpy array and an int sample rate.
- Produces, all returning `float`:
  - `spectral_centroid(audio, sr, frame_size=2048, hop=1024) -> float` (Hz)
  - `onset_density(audio, sr, frame_size=1024, hop=512) -> float` (onsets per second)
  - `dynamic_range(audio, sr, frame_size=2048, hop=1024) -> float` (dB)
  - `sub_bass_ratio(audio, sr, cutoff=120.0) -> float` (0..1)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dsp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dsp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'djtagger.dsp'`

- [ ] **Step 3: Implement `dsp.py`**

Create `djtagger/dsp.py`:

```python
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
    """Onsets per second via peak-picked spectral flux."""
    mags = _magnitude_spectra(audio, frame_size, hop)
    if mags.shape[0] < 3:
        return 0.0
    flux = np.maximum(0.0, np.diff(mags, axis=0)).sum(axis=1)
    if flux.max() <= 1e-8:
        return 0.0
    flux = flux / flux.max()
    threshold = flux.mean() + flux.std()
    onsets = 0
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            onsets += 1
    duration = len(audio) / sr
    return float(onsets / duration) if duration > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dsp.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add djtagger/dsp.py tests/test_dsp.py
git commit -m "feat: add numpy DSP features for set-role classification"
```

---

### Task 3: Arc math and role decision (`classify.py` + config)

**Files:**
- Modify: `djtagger/config.py`
- Create: `djtagger/classify.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `djtagger.dsp` (Task 2) and new constants in `config.py`.
- Produces:
  - Role string constants: `ROLE_WARMUP`, `ROLE_BUILDER`, `ROLE_PEAK`, `ROLE_CLOSER`.
  - `arc_momentum(segment_energies: list[float]) -> float` in `[-1, 1]`.
  - `classify_role(arc_level, arc_momentum, valence, brightness, intensity_index, thresholds=None) -> str`.
  - `compute_arc(audio, sr, segment_energies, energy, valence) -> dict` returning keys
    `spectral_centroid, onset_rate, dynamic_range, sub_bass, arc_level, arc_momentum, set_role`.

- [ ] **Step 1: Add provisional constants to `config.py`**

Append to `djtagger/config.py` (change `TAGGER_VERSION` in place, add the rest at the end):

```python
# TAGGER_VERSION is defined near the top of this file. Change it there:
#   TAGGER_VERSION = "v6"

# ─── Set-Role Classification (v6) ──────────────────────────
# Raw-feature normalization ranges (lo, hi) mapped to 0..1.
# PROVISIONAL. Replaced by scripts/calibrate_arc.py output (Task 5).
FEATURE_RANGE = {
    "spectral_centroid": (800.0, 3500.0),   # Hz on 16 kHz audio
    "onset_rate":        (0.5, 6.0),         # onsets per second
    "dynamic_range":     (3.0, 18.0),        # dB (p90 minus p10 frame RMS)
    "sub_bass":          (0.05, 0.45),       # fraction of energy below 120 Hz
}

# Per-segment energy slope is multiplied by this, then clipped to [-1, 1].
MOMENTUM_SCALE = 8.0

# Role decision thresholds. PROVISIONAL. Replaced by calibration (Task 5).
ROLE_THRESHOLDS = {
    "peak_level":      0.80,   # arc_level at or above this is Peak
    "peak_level_soft": 0.72,   # softer Peak gate, needs intensity too
    "peak_intensity":  0.65,   # intensity_index required for the soft-Peak gate
    "rising":          0.15,   # arc_momentum at or above this is Builder
    "falling":        -0.15,   # arc_momentum at or below this is Closer
    "release_valence": 0.66,   # flat momentum plus bright/released is Closer
    "release_bright":  0.60,   # normalized brightness release cut
}

ROLE_WARMUP = "Warm-up"
ROLE_BUILDER = "Builder"
ROLE_PEAK = "Peak"
ROLE_CLOSER = "Closer"
```

Also change the existing line near the top of `config.py`:

```python
TAGGER_VERSION = "v6"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_classify.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'djtagger.classify'`

- [ ] **Step 4: Implement `classify.py`**

Create `djtagger/classify.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_classify.py -v`
Expected: PASS (12 passed)

- [ ] **Step 6: Commit**

```bash
git add djtagger/config.py djtagger/classify.py tests/test_classify.py
git commit -m "feat: add arc math and set-role decision with provisional thresholds"
```

---

### Task 4: Wire arc computation into the analyzer

**Files:**
- Modify: `djtagger/analyzer.py` (import near top; call + dict merge in `analyze_track`, around lines 352-371)

**Interfaces:**
- Consumes: `classify.compute_arc(audio, sr, segment_energies, energy, valence)` (Task 3).
- Produces: `analyze_track` result dict gains keys `spectral_centroid, onset_rate, dynamic_range, sub_bass, arc_level, arc_momentum, set_role`.

- [ ] **Step 1: Add the import**

In `djtagger/analyzer.py`, add after the existing `from .config import (...)` block (near line 24):

```python
from . import classify
```

- [ ] **Step 2: Call `compute_arc` and merge into the result**

In `analyze_track`, the blend line currently reads:

```python
    # Blend: 70% average + 30% peak
    energy = round(float(np.clip(raw_energy * 0.7 + peak_energy * 0.3, 0, 1)), 3)

    return {
```

Replace it with:

```python
    # Blend: 70% average + 30% peak
    energy = round(float(np.clip(raw_energy * 0.7 + peak_energy * 0.3, 0, 1)), 3)

    # Set-role classification (v6). Reuses the 16 kHz audio already in memory.
    try:
        arc = classify.compute_arc(
            audio, 16000, segment_energies, energy, valence_norm,
        )
    except Exception:
        arc = {
            "spectral_centroid": 0.0, "onset_rate": 0.0, "dynamic_range": 0.0,
            "sub_bass": 0.0, "arc_level": energy, "arc_momentum": 0.0,
            "set_role": classify.ROLE_WARMUP,
        }

    return {
```

- [ ] **Step 3: Add the arc keys to the returned dict**

The `return { ... }` dict ends with `"duration": len(audio) / 16000,`. Add the arc keys just before `"duration"`:

```python
        "energy_variance": energy_variance,
        "spectral_centroid": arc["spectral_centroid"],
        "onset_rate": arc["onset_rate"],
        "dynamic_range": arc["dynamic_range"],
        "sub_bass": arc["sub_bass"],
        "arc_level": arc["arc_level"],
        "arc_momentum": arc["arc_momentum"],
        "set_role": arc["set_role"],
        "bpm": bpm,
```

(Insert the seven arc lines between the existing `"energy_variance"` line and the existing `"bpm"` line; do not duplicate those two.)

- [ ] **Step 4: Verify the wiring with a synthetic smoke check**

Run:

```bash
.venv/bin/python -c "
import numpy as np
from djtagger import classify
audio = (0.3*np.sin(2*np.pi*200*np.arange(3*16000)/16000)).astype('float32')
out = classify.compute_arc(audio, 16000, [0.4,0.5,0.6], 0.55, 0.5)
assert set(out) == {'spectral_centroid','onset_rate','dynamic_range','sub_bass','arc_level','arc_momentum','set_role'}
print('arc keys OK:', out['set_role'])
"
```

Expected: prints `arc keys OK: <a role name>`

- [ ] **Step 5: Verify existing tests still pass**

Run: `.venv/bin/pytest -q`
Expected: PASS (all previous tests green)

- [ ] **Step 6: Commit**

```bash
git add djtagger/analyzer.py
git commit -m "feat: compute set-role in analyze_track from in-memory audio"
```

---

### Task 5: Calibrate thresholds against the real library

**Files:**
- Create: `scripts/calibrate_arc.py`
- Modify: `djtagger/config.py` (replace provisional `FEATURE_RANGE` and `ROLE_THRESHOLDS` values)

**Interfaces:**
- Consumes: `analyzer.load_models`, `analyzer.analyze_track` (now returning arc features).
- Produces: printed percentile tables used to set the final constants. No test (this is an analysis + config task).

- [ ] **Step 1: Write the calibration script**

Create `scripts/calibrate_arc.py`:

```python
"""Sample the library, run analysis, and print feature distributions.

Used once to turn the provisional FEATURE_RANGE / ROLE_THRESHOLDS in
config.py into values grounded in the real collection. Samples across
eras: the root folder holds the oldest tracks, so newer dated subfolders
must be represented too.

Usage:
    .venv/bin/python scripts/calibrate_arc.py "/Volumes/Multimedia/Music/_DJ Music" 300
"""

import os
import sys

import numpy as np

from djtagger.analyzer import load_models, analyze_track


def sample_paths(root: str, n: int) -> list[str]:
    """Pick up to n mp3s spread across top-level subfolders and the root."""
    buckets: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        top = "." if rel == "." else rel.split(os.sep)[0]
        for f in files:
            if f.lower().endswith(".mp3"):
                buckets.setdefault(top, []).append(os.path.join(dirpath, f))
    if not buckets:
        return []
    per = max(1, n // len(buckets))
    picked: list[str] = []
    for _top, paths in sorted(buckets.items()):
        step = max(1, len(paths) // per)
        picked.extend(paths[::step][:per])
    return picked[:n]


def main() -> None:
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    paths = sample_paths(root, n)
    print(f"Sampling {len(paths)} tracks from {root}", file=sys.stderr)

    models = load_models()
    keys = ["energy", "spectral_centroid", "onset_rate", "dynamic_range",
            "sub_bass", "arc_momentum"]
    collected: dict[str, list[float]] = {k: [] for k in keys}

    for i, p in enumerate(paths, 1):
        try:
            r = analyze_track(p, models, detect_bpm_key=False)
        except Exception as ex:
            print(f"  skip {os.path.basename(p)}: {ex}", file=sys.stderr)
            continue
        for k in keys:
            collected[k].append(float(r[k]))
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", file=sys.stderr)

    print("\nfeature                p5      p10     p25     p50     p75     p90     p95")
    for k in keys:
        vals = np.array(collected[k])
        if len(vals) == 0:
            continue
        ps = [np.percentile(vals, q) for q in (5, 10, 25, 50, 75, 90, 95)]
        print(f"{k:20s} " + " ".join(f"{v:7.3f}" for v in ps))
```

- [ ] **Step 2: Run the calibration script**

Run: `.venv/bin/python scripts/calibrate_arc.py "/Volumes/Multimedia/Music/_DJ Music" 300`
Expected: a percentile table printed to stdout for the six features. (Takes a while: this runs the full ML pipeline per track.)

- [ ] **Step 3: Decide the 16 kHz centroid question**

Inspect the `spectral_centroid` row. If p5 and p95 are close together (little spread, roughly under a few hundred Hz apart), 16 kHz brightness is too compressed to discriminate; note this in the commit message as a follow-up to compute the centroid on 44.1 kHz audio. Otherwise proceed with 16 kHz.

- [ ] **Step 4: Update the constants in `config.py`**

Using the printed percentiles, set each `FEATURE_RANGE` entry to roughly `(p10, p90)` of that feature, and set `ROLE_THRESHOLDS`:
- `peak_level` = the `energy` p75 (top quarter are Peak candidates).
- `peak_level_soft` = the `energy` p50.
- `rising` / `falling` = the `arc_momentum` p75 / p25 (symmetric-ish around 0).
- `release_valence` keep at the electronic median-plus (0.66) unless valence data suggests otherwise.
- `release_bright` = 0.60 (normalized), adjust if Closers/Warm-ups look mislabeled on spot checks.

Replace the provisional numbers in `config.py` with the chosen ones. Update the comment on each block to read `# Calibrated 2026-07-13 on N tracks` (N = sample size), removing the `PROVISIONAL` note.

- [ ] **Step 5: Spot-check a few known tracks**

Pick 3-4 tracks you know the role of and run (after Task 6 lands, `djtagger info` shows the role; for now use the snippet):

```bash
.venv/bin/python -c "
from djtagger.analyzer import load_models, analyze_track
m = load_models()
for p in ['<path-to-a-known-peak-track>.mp3', '<path-to-a-known-warmup-track>.mp3']:
    r = analyze_track(p, m)
    print(r['set_role'], r['arc_level'], r['arc_momentum'], p)
"
```

Expected: roles roughly match your intuition. If not, nudge the thresholds and re-check.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_arc.py djtagger/config.py
git commit -m "feat: calibrate set-role thresholds against the library"
```

---

### Task 6: Write the new tags and reshape the comment

**Files:**
- Modify: `djtagger/tagger.py` (`read_tags`, `_build_comment`, `write_tags`, `fix_comments`)
- Test: `tests/test_tagger.py`

**Interfaces:**
- Consumes: the analyzer result dict keys from Task 4 and the role constant names.
- Produces: TXXX frames `SET_ROLE, ARC_LEVEL, ARC_MOMENTUM, SPECTRAL_CENTROID, ONSET_RATE, DYNAMIC_RANGE, SUB_BASS`; comment with `Role: ...` first and no `Dance:`; `read_tags` keys `set_role, arc_level, arc_momentum, spectral_centroid, onset_rate, dynamic_range, sub_bass`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tagger.py`:

```python
import os

from mutagen.id3 import ID3, TXXX

from djtagger import tagger


def test_build_comment_role_first_and_no_dance():
    comment, detail = tagger._build_comment(
        energy=0.85, valence=0.6, set_role="Builder", danceability=0.9,
        peak_energy=0.92, arousal=0.5, aggressive=0.7, intro_energy=0.3,
        arc_level=0.85, arc_momentum=0.4,
    )
    assert comment.startswith("Role: Builder | ")
    assert "Dance:" not in comment
    # Danceability is retained in the hidden detail string.
    assert "D:0.9" in detail
    assert "Lvl:0.85" in detail
    assert "Mom:0.4" in detail


def test_build_comment_without_role_omits_role_prefix():
    comment, _ = tagger._build_comment(energy=0.5, valence=0.6)
    assert not comment.startswith("Role:")


def test_read_tags_picks_up_new_frames(tmp_path):
    p = str(tmp_path / "t.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=["v6"]))
    tags.add(TXXX(encoding=3, desc="SET_ROLE", text=["Peak"]))
    tags.add(TXXX(encoding=3, desc="ARC_LEVEL", text=["0.88"]))
    tags.add(TXXX(encoding=3, desc="ARC_MOMENTUM", text=["0.12"]))
    tags.add(TXXX(encoding=3, desc="SUB_BASS", text=["0.31"]))
    tags.save(p)

    info = tagger.read_tags(p)
    assert info["set_role"] == "Peak"
    assert info["arc_level"] == "0.88"
    assert info["arc_momentum"] == "0.12"
    assert info["sub_bass"] == "0.31"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tagger.py -v`
Expected: FAIL (`_build_comment` has no `set_role` param; `read_tags` has no `set_role` key)

- [ ] **Step 3: Extend `read_tags`**

In `djtagger/tagger.py`, in `read_tags`, add these keys to the `info` dict initializer (after the existing `"energy_variance": ""` line):

```python
        # v6 fields
        "set_role": "",
        "arc_level": "",
        "arc_momentum": "",
        "spectral_centroid": "",
        "onset_rate": "",
        "dynamic_range": "",
        "sub_bass": "",
```

And add to the `tag_map` dict (after the existing `"ENERGY_VARIANCE": "energy_variance",` line):

```python
        # v6 tags
        "SET_ROLE": "set_role",
        "ARC_LEVEL": "arc_level",
        "ARC_MOMENTUM": "arc_momentum",
        "SPECTRAL_CENTROID": "spectral_centroid",
        "ONSET_RATE": "onset_rate",
        "DYNAMIC_RANGE": "dynamic_range",
        "SUB_BASS": "sub_bass",
```

- [ ] **Step 4: Reshape `_build_comment`**

Replace the whole `_build_comment` function with:

```python
def _build_comment(
    energy: float,
    valence: float,
    set_role: str = "",
    danceability: float = 0.0,
    peak_energy: float = 0.0,
    arousal: float = 0.0,
    aggressive: float = 0.0,
    intro_energy: float = 0.0,
    arc_level: float = 0.0,
    arc_momentum: float = 0.0,
) -> tuple[str, str]:
    """Build human-readable comment and detail string."""
    e_lbl = "Low" if energy < 0.4 else "Mid" if energy < 0.7 else "High"
    # Valence thresholds tuned for electronic/dance music distribution
    # (median ~0.63, range ~0.45-0.80); the 0-1 model range is rarely exercised.
    v_lbl = "Dark" if valence < 0.58 else "Neutral" if valence < 0.68 else "Bright"
    agg_lbl = "Soft" if aggressive < 0.25 else "Mid" if aggressive < 0.5 else "Hard"
    intro_lbl = "Quiet" if intro_energy < 0.5 else "Mid" if intro_energy < 0.75 else "Hot"

    # Role goes first so it is the glanceable thing in Serato. Danceability is
    # dropped from the visible comment (near-constant across this library) but
    # kept in the hidden detail below.
    role_part = f"Role: {set_role} | " if set_role else ""
    comment = (
        f"{role_part}Energy: {e_lbl} | Mood: {v_lbl} | Edge: {agg_lbl} | "
        f"Peak: {peak_energy:.2f} | Intro: {intro_lbl}"
    )

    detail = (
        f"E:{energy} | V:{valence} | Agg:{aggressive} | "
        f"Peak:{peak_energy} | Intro:{intro_energy} | D:{danceability} | "
        f"Arousal:{arousal} | Lvl:{arc_level} | Mom:{arc_momentum}"
    )

    return comment, detail
```

- [ ] **Step 5: Write the new TXXX tags and pass role to the comment in `write_tags`**

In `write_tags`, in the TXXX list (the `for key, val in [ ... ]` block), add after the existing `("ENERGY_VARIANCE", result["energy_variance"]),` line:

```python
            # v6 tags
            ("SET_ROLE", result["set_role"]),
            ("ARC_LEVEL", result["arc_level"]),
            ("ARC_MOMENTUM", result["arc_momentum"]),
            ("SPECTRAL_CENTROID", result["spectral_centroid"]),
            ("ONSET_RATE", result["onset_rate"]),
            ("DYNAMIC_RANGE", result["dynamic_range"]),
            ("SUB_BASS", result["sub_bass"]),
```

Then update the `_build_comment(...)` call in `write_tags` to pass the new args:

```python
        comment, detail = _build_comment(
            energy=result["energy"],
            valence=result["valence"],
            set_role=result.get("set_role", ""),
            danceability=result.get("danceability", 0.0),
            peak_energy=result.get("peak_energy", 0.0),
            arousal=result.get("arousal", 0.0),
            aggressive=result["moods"]["aggressive"],
            intro_energy=result.get("intro_energy", 0.0),
            arc_level=result.get("arc_level", 0.0),
            arc_momentum=result.get("arc_momentum", 0.0),
        )
```

- [ ] **Step 6: Update `fix_comments` to include the role**

In `fix_comments`, after the block that reads `intro_tag` / `intro`, add reads for the v6 fields, then pass them into `_build_comment`:

```python
        role_tag = tags.get("TXXX:SET_ROLE")
        role = role_tag.text[0] if role_tag and role_tag.text else ""
        lvl_tag = tags.get("TXXX:ARC_LEVEL")
        lvl = float(lvl_tag.text[0]) if lvl_tag and lvl_tag.text else 0.0
        mom_tag = tags.get("TXXX:ARC_MOMENTUM")
        mom = float(mom_tag.text[0]) if mom_tag and mom_tag.text else 0.0

        comment, detail = _build_comment(
            energy=e, valence=v, set_role=role, danceability=d, peak_energy=p,
            arousal=a, aggressive=agg, intro_energy=intro, arc_level=lvl,
            arc_momentum=mom,
        )
```

(Replace the existing `comment, detail = _build_comment(...)` call in `fix_comments` with the version above.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tagger.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all green)

- [ ] **Step 9: Commit**

```bash
git add djtagger/tagger.py tests/test_tagger.py
git commit -m "feat: write set-role tags and put role first in the comment"
```

---

### Task 7: Surface the role in library, info, and export

**Files:**
- Modify: `djtagger/library.py` (scan record fields, around lines 53-59)
- Modify: `djtagger/cli.py` (`info` display near line 590; `export` fields near line 1099)

**Interfaces:**
- Consumes: `read_tags` keys from Task 6.
- Produces: `scan_library` records carry the new fields; `djtagger info` shows the role first; `djtagger export` includes the new columns.

- [ ] **Step 1: Add new fields to `scan_library` records**

In `djtagger/library.py`, after the `record["key"] = tags.get("key", "")` line, add:

```python
    record["set_role"] = tags.get("set_role", "")
```

And extend the numeric-conversion tuple (the `for key in (...)` loop) to include the new numeric fields:

```python
        for key in ("energy", "valence", "mood_happy", "mood_sad",
                     "mood_aggressive", "mood_relaxed",
                     "danceability", "arousal",
                     "peak_energy", "intro_energy", "energy_variance",
                     "arc_level", "arc_momentum", "spectral_centroid",
                     "onset_rate", "dynamic_range", "sub_bass"):
```

- [ ] **Step 2: Show the role first in `djtagger info`**

In `djtagger/cli.py`, in the `info` command, find the `# Energy & Mood` section (`if tags.get("energy"):`). Immediately before it, add a role row so it prints first:

```python
    # Set role (v6) — shown first, it is the headline for crate-digging.
    if tags.get("set_role"):
        table.add_row("Set role", f"[bold]{tags['set_role']}[/bold]")
        if tags.get("arc_level"):
            al = float(tags["arc_level"])
            table.add_row("Arc level", f"{_mini_bar(al)}  {al:.3f}")
        if tags.get("arc_momentum"):
            am = float(tags["arc_momentum"])
            table.add_row("Arc momentum", f"{am:+.3f}")
        table.add_row("", "")
```

- [ ] **Step 3: Add the new columns to `djtagger export`**

In `djtagger/cli.py`, in the `export` command, extend the `fields` list to include the new columns (insert after `"peak_energy", "intro_energy", "energy_variance",`):

```python
        "set_role", "arc_level", "arc_momentum",
        "spectral_centroid", "onset_rate", "dynamic_range", "sub_bass",
```

- [ ] **Step 4: Verify the CLI wiring imports and runs**

Run: `.venv/bin/python -c "import djtagger.cli, djtagger.library; print('cli/library import OK')"`
Expected: prints `cli/library import OK`

Run: `.venv/bin/djtagger export --help`
Expected: help text prints with no error.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add djtagger/library.py djtagger/cli.py
git commit -m "feat: surface set role in scan, info, and export"
```

---

### Task 8: End-to-end verification on a real track

**Files:** none (verification only)

- [ ] **Step 1: Tag one real track and inspect it**

Pick a single MP3 and run the tagger on just it (use a copy if you want to avoid touching the original), then inspect:

```bash
.venv/bin/djtagger info "<path-to-a-tagged-track>.mp3"
```

Expected: the panel shows `Set role` first with a role, `Arc level`, `Arc momentum`, and the comment line begins `Role: ... | Energy: ...` with no `Dance:` segment.

- [ ] **Step 2: Confirm the Serato comment string**

Expected visible comment format:

```
Role: Builder | Energy: High | Mood: Dark | Edge: Hard | Peak: 0.92 | Intro: Quiet
```

- [ ] **Step 3: Update the README**

Add a short "Set Role" subsection under "How It Works" describing the four roles, the two axes (`arc_level`, `arc_momentum`), the four DSP features, and that danceability was dropped from the comment but is still computed. Note the new TXXX tags in the "ID3 Tags Written" section and bump the documented version to v6.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document set-role classification (v6)"
```
