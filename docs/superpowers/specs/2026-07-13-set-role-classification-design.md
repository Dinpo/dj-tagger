# Set-Role Classification — Design

**Date:** 2026-07-13
**Status:** Approved for planning
**Tagger version bump:** v5 → v6

## Problem

DJ Tagger already estimates a track's `energy`, but energy alone does not tell a
DJ *where a track sits in the arc of an evening set*. A build-up track and a
closing track can share the same energy level yet play completely different
roles: one raises tension on the way up, the other releases it on the way down.

We want each track classified into a **set role** — Warm-up, Builder, Peak, or
Closer — written as an ID3 tag and surfaced first in the Serato comment, so the
role is glanceable while crate-digging.

## Goals

- Classify every track into one of four set roles: **Warm-up / Builder / Peak / Closer**.
- Base the classification on signals that actually vary across the arc, not on
  saturated ones. In this library `danceability` is high almost everywhere and
  `valence` is narrow (median ≈ 0.63), so both are weak discriminators on their own.
- Write the role (and the continuous values behind it) as ID3 tags, and put the
  role **first** in the human-readable Serato comment.
- Set all thresholds from the library's **actual feature distribution**, not from
  guessed constants (calibration-first).

## Non-Goals

- **`suggest` integration is out of scope for this iteration.** Ordering a whole
  night from these tags is recorded under *Future Work*, not built now.
- No learned/ML classifier. The raw features are written to tags so a learned
  model remains possible later, but v6 uses a transparent rule mapping.
- No change to genre resolution, BPM/key, album/year behaviour.

## The Model: two continuous axes → one role

Each track yields two calibrated continuous values; the discrete role is derived
from them.

- **`arc_level` ∈ [0, 1]** — overall intensity. Starts from the existing blended
  `energy` and is nudged by the new heavy/bright/dynamic features. This is the
  axis a future set-orderer would sort a night by.
- **`arc_momentum` ∈ [−1, +1]** — the track's *internal trajectory*: does energy
  rise, sit flat, or fall across its own segments. Computed as a normalized
  linear-fit slope over the per-segment energy array (already produced by the
  analyzer today but currently discarded after peak/intro/variance), sharpened by
  the tension features (drive, brightness).

### Role mapping

Boundaries are **determined during calibration** (see Phase 0), not fixed here.
The qualitative signature of each role:

| Role | Signature |
|------|-----------|
| **Peak** | high `arc_level` — hard, bright, heavy low end, sustained-hot internally |
| **Builder** | mid/low `arc_level` + **rising** `arc_momentum`, darker/tenser, driving |
| **Closer** | mid/low `arc_level` + **flat/falling** `arc_momentum` + brighter/released valence, melodic |
| **Warm-up** | low `arc_level` + flat `arc_momentum`, groovy/settled opener |

Builder and Closer deliberately overlap on energy; they split on **momentum**
first and **valence/brightness** as the tie-breaker (the classic "minor→major,
release at the end" move). Peak is separated primarily by level.

The mapping is implemented as a **pure function** `classify_role(features) -> (arc_level, arc_momentum, role)`
so it can be unit-tested in isolation and re-tuned without touching audio code.

## New DSP Features

Computed in `analyzer.py` immediately after embeddings, **reusing the 16 kHz mono
audio already loaded** — no second file decode:

| Feature | Captures | Role relevance |
|---------|----------|----------------|
| **spectral centroid** | brightness / "air" | Peak & Closer skew brighter; Warm-up/Builder darker |
| **onset rate** | percussive density / drive | Builder & Peak are driving; Closer is sparser/melodic |
| **RMS dynamic range** | quiet-loud contrast / "drop-iness" | Peak has big drops; Warm-up is flat |
| **sub-bass ratio** | low-end weight | Peak is heaviest; Warm-up/Closer lighter |

### 16 kHz caveat (resolved in calibration)

At 16 kHz the Nyquist limit is 8 kHz. Sub-bass, onset rate, and dynamic range are
fully intact. **Brightness above 8 kHz is lost**, so the spectral centroid is
compressed. Phase 0 calibration checks whether the 16 kHz centroid still *ranks*
tracks usefully. If it does not, we decode 44.1 kHz for the centroid only (the
analyzer already loads 44.1 kHz when BPM/key detection is on, so the pattern
exists).

## Phase 0: Calibration (do this first)

Before committing any threshold:

1. Run the new feature extraction over a sample of the library
   (`/Volumes/Multimedia/Music/_DJ Music`) — a few hundred tracks is enough.
2. Dump per-feature distributions: min/median/max, key percentiles (p10, p25,
   p50, p75, p90), and coarse histograms.
3. Decide the 16 kHz-vs-44.1 kHz centroid question from the spread.
4. Set `arc_level` / `arc_momentum` boundaries and the four role cutoffs from
   those percentiles (e.g. Peak = top-quartile `arc_level`), and record them as
   **named constants in `config.py`** with a comment noting they were calibrated
   and on what sample size.

This mirrors how the existing valence thresholds were tuned to the electronic
distribution rather than the model's nominal 0–1 range.

## Tags Written

New `TXXX` frames:

- `SET_ROLE` — one of `Warm-up` / `Builder` / `Peak` / `Closer`
- `ARC_LEVEL` — continuous [0,1]
- `ARC_MOMENTUM` — continuous [−1,+1]
- `SPECTRAL_CENTROID`, `ONSET_RATE`, `DYNAMIC_RANGE`, `SUB_BASS` — raw features,
  for transparency and future ML

`TAGGER_VERSION` becomes `v6`. `read_tags()` gains these keys; `write_tags()`
writes them.

### Serato comment

Role first, `Dance` removed from the **visible** comment (the `DANCEABILITY` tag
itself is still computed and written — only its comment field goes away):

```
Role: Builder | Energy: High | Mood: Dark | Edge: Hard | Peak: 0.92 | Intro: Quiet
```

The hidden `djtagger` detail comment keeps the numeric dump and gains the new
continuous values (`arc_level`, `arc_momentum`).

## Danceability

Unchanged in computation and in the `DANCEABILITY` tag; still used by
`find`/`stats`/`suggest`. Only removed from the visible Serato comment string in
`_build_comment()`. No filters or other consumers are touched.

## Files Touched

- `analyzer.py` — compute the 4 DSP features from the in-memory 16 kHz audio;
  compute `arc_level`, `arc_momentum` via the segment array; return them in the
  result dict.
- `classify.py` *(new, small)* — the pure `classify_role()` function + role
  constants. Kept separate from `analyzer.py` so it is testable without Essentia.
- `config.py` — calibrated threshold constants (filled in Phase 0), role labels.
- `tagger.py` — new TXXX tags in `read_tags()`/`write_tags()`; `_build_comment()`
  puts role first and drops `Dance`; `TAGGER_VERSION` → v6.
- `cli.py` — `info` output shows the role and new values; `export` includes the
  new columns.
- Tests — see below.

## Testing

- Unit-test `classify_role()` against synthetic feature dicts covering each of the
  four roles plus boundary cases (e.g. same level, opposite momentum → Builder vs
  Closer).
- Unit-test the momentum/slope math on hand-built segment arrays (rising, flat,
  falling, single-segment).
- Snapshot-test the new comment string (role first, no `Dance`).
- No network or Essentia needed for any of these — the pure function and the
  comment builder are isolated from audio loading.

## Error Handling

- Track shorter than one segment → fall back to whole-track features,
  `arc_momentum = 0`, role decided by `arc_level` alone.
- Any feature-extraction exception is caught and defaults to neutral values,
  matching the analyzer's existing defensive `try/except` style (BPM/key already
  do this).

## Future Work (not in this iteration)

- **Evening-arc `suggest` mode**: order a selected folder Warm-up → Builder →
  Peak → Closer using `arc_level` (ascending up, descending down) and the role
  label, so a whole night can be assembled in order. The continuous tags written
  here are exactly what such a mode would consume.
- **Learned role classifier** trained on hand-labeled tracks, using the raw
  feature tags as inputs.
