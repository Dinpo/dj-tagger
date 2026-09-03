# ENTRY / EXIT cue detection: investigated, parked

**Date:** 2026-09-03
**Status:** PARKED at the user's request. No code shipped into the package.
**Verdict:** three detector attempts, best 37% exact vs a 23% no-audio
baseline. Not accurate enough to build a feature on.

## The goal

The user marks tracks in Serato with **ENTRY** and **EXIT** cues. Because
most of the library is extended remixes, these separate the generic DJ-tool
intro/outro from the recognizable part of the track, so the good parts can be
stitched together while DJing. The question was whether those points can be
detected automatically.

Unlike the hypnotic-groove work, this had **real ground truth**: the user's
own existing markers.

## Ground truth extracted

`scripts/serato_cues.py` parses the base64 `GEOB:Serato Markers2` frame and
reads cue names + positions. Library scan found:

- **41 ENTRY + 40 EXIT markers across 53 tracks** (of 3,618)
- ENTRY snaps to power-of-two phrase boundaries, median exactly **16.0 bars**
- ENTRY distribution: bar 0 (21%), 4 (5%), 8 (14%), **16 (26%)**, 24 (7%),
  32 (14%), 48 (7%), 64 (7%)
- EXIT median ~265 s, typically 40-60 s before the end
- Other user cue names confirm phrase thinking: `-32`, `-64`, `-8X16`,
  `-2X32`, `-96 BEATS BREAKDOWN`, `NO EXIT`, `BASS SWAP`, `DROP`

Because the markers land on bar boundaries and Serato supplies BPM, the task
reduces to *picking which phrase boundary* (0/4/8/16/24/32/48/64) is the
entry, which is also trivially scoreable.

## What was tried, and the scores

Scoring = predicted phrase within 1 bar of the user's marker, n=43.

| Attempt | Approach | Exact |
|---------|----------|-------|
| baseline | always guess 16 bars (no audio at all) | **23%** |
| v1 | strongest arrangement step (sub-bass jump + rms jump + centroid drop) | 30% |
| v2 | first phrase boundary reaching "full" (rms/sub vs the track's own body level) | **37%** |
| v3 | chroma novelty vs the intro + mid-band ratio | 16% |

### Why v1 failed

It found the *biggest* change, which in dance music is the main drop after
the breakdown (bar 48-64), not the *first* point the track becomes
recognizable. Systematic pattern: true 8 -> predicted 32/48, true 16 ->
predicted 48/64. Wrong objective (maximum, not earliest-qualifying).

### Why v2 failed (the important finding)

Reformulated as "first time the arrangement is full". Still only 37%, and the
threshold sweep was **flat (28-37% across all 16 combinations)**, which is the
signature of a feature that carries no signal.

The diagnostic failures: tracks marked at bar 16 or 32 are **already at full
loudness and full sub-bass at bar 0** (Argy Omiki - WIND, Daft Punk - Around
The World, 7 SKIES, Corren Cavini, Fahlberg, Super Flu, Blue Boy). The
extended intro contains the entire groove: drums and bassline present.

**So the user's ENTRY is not an energy/loudness/bass event.** It is most
likely the arrival of the *hook / topline / vocal*, i.e. the track's melodic
identity, which none of these feature sets captured.

### Why v3 failed

Chroma (pitch-class) distance from the intro, folded 100-2000 Hz, plus
mid-band ratio. Worse than baseline behaviour at 16%. A crude
approximation of self-similarity segmentation, not the real algorithm.

## What is left to try (option A, not attempted)

1. **Proper structure segmentation**: full self-similarity matrix over
   chroma/MFCC with a checkerboard-kernel novelty curve, taking the first
   strong boundary. v3 was a shortcut; this is the actual textbook method.
2. **Per-window vocal detection** using the `voice_instrumental` model we
   already load. A vocal entering is a strong "recognizable" marker and is
   cheap (runs on existing EffNet embeddings).
3. **More labels.** 43 is thin. 100-150 ENTRY marks would both sharpen the
   target and make a learned model viable instead of hand-tuned thresholds.

## Cautions for whoever resumes

- **Do not tune thresholds until the number looks good on n=43** - that is
  fitting noise. Report accuracy across a threshold range; a rule that only
  works at one magic value is not real.
- **Always compare against the constant baseline (23%)**, not against zero.
- **21% of ENTRY markers are at bar 0.** If those mean "nothing to skip"
  rather than a detected boundary, they may be a different category than the
  rest, and they are a fifth of the label set.
- **Writing Serato cues is riskier than writing text tags**: the cue data
  lives in a binary GEOB frame alongside all the user's existing cues. Write
  to copies and verify in Serato before touching originals.
- Cache per-window features to JSON before iterating rules (as done here);
  audio passes are the slow part, rule evaluation is instant.

## Reusable artifact

`scripts/serato_cues.py` reads any Serato cue set (also useful for the Mixed
In Key "Energy N" cues). It is the enabler for resuming this work.
