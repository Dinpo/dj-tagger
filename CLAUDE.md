# CLAUDE.md: DJ Tagger

Guidance for AI sessions working on this repo. Read the [README](README.md)
for full user-facing docs; this file is the fast orientation plus the
non-obvious decisions and gotchas.

## What it is

A local CLI that analyzes MP3s with Essentia TensorFlow models and writes
ID3 tags: genre (Beatport → Last.fm → ML), energy, valence, moods, and a
**DJ set role**. Runs unattended over large libraries (~3,500 tracks here).
No cloud APIs for audio; only lightweight genre lookups hit the network.

- Python >= 3.10. Entry point: `djtagger.cli:app` (Typer). Venv at `.venv`.
- Tests: `.venv/bin/pytest` (currently 61, all green). `dsp.py` and
  `classify.py` are pure numpy (no Essentia import) so they test without models.
- Target library: `/Volumes/Multimedia/Music/_DJ Music` (see the user's
  auto-memory). Filenames follow `Artist - Title.mp3`.

## Module map

| File | Responsibility |
|------|----------------|
| `cli.py` | Typer commands + Rich UI. Commands: tag, info, stats, find, export, **genre-stats**, **rerole**, suggest, health, clean-genres, fix-audit, convert-keys, bench-bpm |
| `analyzer.py` | Loads TF models; `analyze_track()` returns the result dict (genre preds, moods, danceability, vocal, energy, arc features, set_role) |
| `dsp.py` | Pure-numpy audio features. NO Essentia. Shared framing/FFT via `_frames`/`_magnitude_spectra` |
| `classify.py` | Set-role model. `compute_arc()`, `drive_emo()`, `classify_role()`, `decide_role()`, genre-stats helpers. NO Essentia |
| `config.py` | Constants incl. `FEATURE_RANGE`, `ROLE_THRESHOLDS`, `TAGGER_VERSION`, `LEGACY_ROLES`, `GENRE_STATS_FILE` |
| `tagger.py` | ID3 read/write; `write_tags()` is the single role-decision point; `rerole_file()`; `fix_comments()` |
| `genres.py` | Beatport scrape, Last.fm, remix-aware match scoring |
| `scanner.py` | MP3 discovery; `filter_untagged` (resume), `filter_outdated` (--upgrade) |
| `library.py` | Reads all tracks into structured records for stats/find/export/suggest |

## The set-role model (v7): the active work

Four roles per the DJ set taxonomy: **Opener / Builder / Peak / Closer**.
Model: **energy bands set Opener vs Peak; the shared mid band splits
Builder vs Closer by character** via a `drive` index (spectral flux + onset +
rising loudness) vs an `emo` index (valence + brightness + vocals + fading
outro). See `classify.classify_role`.

- **Genre-relative bands (promote-only, "Fix B").** The absolute band always
  stands: `arc_level >= peak_level` (0.80) is Peak; Opener requires
  `arc_level <= opener_level` (0.55). On top of that, when `genre_energy.json`
  exists (built by `djtagger genre-stats`) and the cohort is >= 30, being high
  within its genre can additionally promote to Peak, and Opener also requires
  being low within its genre. Net: genre banding only ever PROMOTES to Peak or
  filters out of Opener; it never demotes an absolute banger. (It is NOT the
  old "genre replaces global bands" logic.)
- **Effective energy / heaviness lift ("Fix A").** `decide_role` and
  `compute_arc` band on an *effective* energy = `arc_level` plus a
  one-directional boost (max +0.15) from structural heaviness (`sub_bass` +
  `drop_db`), so smooth-but-heavy tracks the mood-energy underrates are not
  forced into Opener. Never lowers energy; the stored `arc_level` stays raw.
  Partial fix only: genuinely low-energy hypnotic grooves (~0.42) are a
  documented limitation (see the docs/superpowers note).
- **Single decision point.** The role is decided in `tagger.write_tags` via
  `classify.decide_role`, using the genre actually written to TCON (not the
  proposed genre). `compute_arc` only assigns a provisional global-band role.
  Do NOT re-add a second role decision in `cli.py`.
- **Tuning is decoupled from analysis.** `decide_role` reads only stored tags
  (`arc_level`, `drive`, `emo`, genre). So the loop is: edit `ROLE_THRESHOLDS`
  / `FEATURE_RANGE` in `config.py` → `djtagger rerole <lib>` (seconds, no ML)
  → re-check. Never re-run the ~3h analysis just to change role thresholds.

### Calibration provenance

`FEATURE_RANGE` and `ROLE_THRESHOLDS` are p10..p90 / percentile values from
sampling the library through the pipeline (dates in the config comments).
To recalibrate, sample the library, take each feature's p10/p90, and update
`config.py`. Genre bands come from `genre-stats` (uses the ENERGY tag).

## Key decisions and gotchas

- **Energy is validated, not guessed.** Our `energy` matched Mixed In Key's
  1-10 EnergyLevel 11/11 on a test folder. Do not "improve" energy without a
  reason; the mood-model-derived value is a good perceptual proxy. MIK's
  `TXXX:EnergyLevel` survives on ~99% of the library (we only overwrite the
  readable `COMM` comment, not that frame) if you ever want it as a reference.
- **Loudness is NOT energy.** Modern masters sit in a ~3 dB band; loudness
  does not track perceived energy. Loudness is used only for the *arc shape*
  (intro/outro/drop/slope), never as an energy value.
- **`arc_momentum` is informational only.** The mood-segment slope is nearly
  flat on this library (p5..p95 ~ -0.06..+0.11), so it is written as a tag but
  does NOT drive the role. The loudness-arc slope does.
- **Danceability** is computed and written to `DANCEABILITY` but deliberately
  omitted from the visible comment (near-constant across this library). Kept
  in the hidden detail comment. Do not remove the tag.
- **`FEATURE_RANGE` has no `dynamic_range`/`sub_bass` entries**: those raw
  features are written as tags but do not feed the role decision.
- **ENTRY/EXIT cue detection was investigated and PARKED** (2026-09-03).
  The user marks intro/outro boundaries as Serato ENTRY/EXIT cues; three
  detectors scored 30% / 37% / 16% exact against a 23% no-audio baseline, so
  nothing shipped. Key finding: those markers are NOT an energy/bass event
  (tracks marked at bar 16-32 are already full at bar 0), they track melodic
  identity. `scripts/serato_cues.py` reads the markers back as ground truth.
  Full write-up:
  `docs/superpowers/notes/2026-09-03-entry-exit-cue-detection.md`.
- **`pulse_regularity` (PULSE_REG) is captured but unused in roles.** It
  measures beat-pulse strength. Investigated for detecting hypnotic grooves
  (which read as false Openers because raw energy and set-role decouple for
  them); it only weakly separates hypnotic grooves from atmospheric openers,
  so it is stored for a future labeled validation but does NOT affect
  set_role. Full write-up:
  `docs/superpowers/notes/2026-07-27-hypnotic-groove-pulse-regularity.md`.
  Existing tracks have no PULSE_REG until a re-analysis/backfill.
- **Honest failures.** If `compute_arc` raises, `analyze_track` logs a
  warning, writes an EMPTY role, and sets `arc_ok=False`. Never fabricate a
  role from neutral values (that silently mislabels and poisons calibration).
- **Legacy roles.** v6 wrote `SET_ROLE="Warm-up"`; v7 renamed it to `Opener`.
  `config.LEGACY_ROLES` maps old→new; applied in `read_tags` and persisted by
  `fix_comments`. Add future renames there.
- **Version-aware upgrade.** `is_already_tagged` only checks `GENRE_SOURCE`,
  so a plain `tag` skips tagged files regardless of version. Use
  `tag --upgrade` to backfill new fields onto old-version tracks (keeps their
  resolved genres, no network). `filter_outdated` compares `TAGGER_VERSION`.
- **Performance.** `compute_arc` shares framing/FFT across features; the
  segment loop slices stored per-frame predictions instead of re-running
  models per segment. Wall-clock is ~3.4s/track dominated by the EffNet
  embedding pass, not our DSP. Any DSP refactor must stay output-identical
  (there is a bit-exact regression test approach: snapshot analyze_track
  output before/after and diff).

## Writing style (user rule)

Never use the em dash character (U+2014) anywhere (comments, commit
messages, docs). Use a period, comma, colon, parentheses, or `..` instead.
(Box-drawing `─` is fine.) This is enforced in code review.

## Current state (2026-10-26)

- `TAGGER_VERSION` is **v8** (bumped when PULSE_REG was added; bump again
  whenever the stored tag set changes so `tag --upgrade` can find stale files).
  All work committed and pushed to `origin/main`.
- **Library verified 100%:** all 3,690 tracks in `_DJ Music` are at v8 with a
  valid role, PULSE_REG, energy, a non-junk genre, and a `Role:` comment.
  Distribution roughly Opener 21% / Builder 20% / Peak 33% / Closer 25%.
- **ML-genre quality: fixed and applied.** The 5-class electronic genre head
  (ambient/dnb/house/techno/trance, chance = 0.20) may only lead when
  top >= 0.40 AND it leads the runner-up by >= 0.10; otherwise the finer
  Discogs-400 labels win. Same-source re-tags replace (not merge) their own
  prior genre. Apply resolver changes with `tag --retag-source ml`.
- **Verifying "properly tagged":** the resume check only looks for
  GENRE_SOURCE, so it does not prove completeness. To verify, read every
  file and check version, role, PULSE_REG, energy, genre and comment. Match
  `.mp3` case-insensitively (the library has `.MP3` / `.Mp3` files).
- **Pending, user-driven:** grading roles by ear, then tune
  `ROLE_THRESHOLDS` / `FEATURE_RANGE` and `rerole`. Parked: hypnotic-groove
  detection and ENTRY/EXIT cue detection (both need more labels; see notes).
