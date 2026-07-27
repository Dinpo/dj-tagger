# Hypnotic-groove detection and the pulse_regularity feature

**Date:** 2026-07-27
**Status:** feature captured (PULSE_REG tag), NOT used in the role decision.
**Verdict:** directional signal, not a clean separator; needs labels to use.

## The problem

Hypnotic melodic/techno grooves (e.g. *Alex Kennon, Joris Voorn - Blinding
Lights*) get tagged **Opener** because their *measured* energy (`arc_level`
~0.42) is genuinely low, even though they function as **peak-time** tracks.
User's insight: a hypnotic groove that runs relentlessly for an extended time
is characteristically peak-time (you can't open or close on it), so detect
"hypnotic groove" and treat it as peak-ish. This note records whether that is
achievable with audio features.

## What was tried

### 1. Sustainedness (energy_variance, loudness slope): REJECTED

Hypothesis: hypnotic = steady/flat over time. Measured on melodic house &
techno:

| Group | energy_variance |
|-------|-----------------|
| Peak (current)   | 0.0165 (dynamic, big drops) |
| Opener (current) | 0.0030 (steady) |
| Blinding Lights  | 0.0010 (very flat) |

Rejected: "steady/flat" is the *opener* profile, not the peak profile. Our
Peaks are the *dynamic* tracks. A hypnotic groove is flat like an opener, so
sustainedness cannot distinguish them.

### 2. Danceability: REJECTED

Saturated across the whole library (p10/p50/p90 = 0.95 / 0.99 / 1.0). No
discriminative power.

### 3. pulse_regularity (built: `dsp.pulse_regularity`)

Autocorrelation of the half-wave-rectified RMS-energy envelope: beat-band
(90-150 BPM) peak minus the off-beat baseline, gated on absolute transient
content (tonal material scores 0). Energy envelope chosen over spectral flux
because flux picks up tonal phase-drift artifacts that fake a pulse.

Melodic house & techno probe (audio-only, no ML), final metric:

| Group | pulse_reg mean | p25 | p75 |
|-------|----------------|-----|-----|
| Peak (current)                 | 0.803 | 0.725 | 0.917 |
| Opener: heavy+driving (hypno?) | **0.879** | 0.830 | 0.968 |
| Opener: light/sparse (atmos?)  | 0.748 | 0.571 | 0.890 |

Named checks: After Love 1.00, Elderbrook Numb 0.97, Bob Sinclar 0.92,
Blinding Lights 0.90, Undercatt Vegas 0.86, Angelov 0.77, Monkey Safari 0.75,
Sam Shure 0.52.

**Directional but not separating.** The suspected-hypnotic group does score
higher on average (0.879 vs 0.748), which is the right direction and stronger
than the spectral-flux version. But:
- Individual tracks overlap badly: a softer track like *Bob Sinclar - Take It
  Easy* (0.92) scores as high as *Blinding Lights* (0.90).
- pulse_regularity essentially measures "has a strong four-on-the-floor
  pulse" = danceable, which most of the library is. It is close to redundant
  with the already-saturated danceability.
- The heavy-vs-light split used here is a *proxy* for hypnotic-vs-atmospheric,
  not ground truth.

## Verdict and why it is not wired into roles

No threshold can be set or validated without ground-truth labels, and the
distributions overlap. Wiring an unvalidated, weakly-separating signal into
the role decision would risk corrupting the results that are currently good.
So `PULSE_REG` is **computed and stored** (available for future validation)
but does **not** affect `set_role`.

## The real root cause (known limitation)

For hypnotic/melodic grooves, **raw energy and set-role decouple**: a
moderate-intensity relentless groove can be a peak-time weapon. Our model is
energy-led, so it cannot fully resolve this class from audio alone. This is a
documented limitation, not a bug.

## What it would take to actually use pulse_regularity

1. **Labels.** Hand-label ~30-40 tracks into "hypnotic peak" vs "genuine
   opener" (the user's crate is NOT clean ground truth: it mixes openers and
   closers in).
2. **Backfill.** Populate `PULSE_REG` on the existing library: an audio-only
   pass (~30 min, no ML) or the next full `tag`/`--upgrade` run.
3. **Validate.** Test whether pulse_regularity, alone or combined with
   sub-bass heaviness, separates the two label sets. Only integrate if it
   genuinely does.
4. **Integrate.** If validated, add a term to `classify.decide_role` (and a
   threshold in `config.ROLE_THRESHOLDS`), then `rerole`.

Until step 1 exists, this stays parked. Everything short of labels is guessing.
