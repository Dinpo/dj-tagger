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
- Tests: `.venv/bin/pytest` (currently 45, all green). `dsp.py` and
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

- **Genre-relative bands.** When `genre_energy.json` exists (built by
  `djtagger genre-stats`) and the track's genre cohort is >= 30, the energy
  band comes from the track's percentile WITHIN its genre, so a melodic-house
  peak is not judged against the whole library's hardest tracks. Else global
  cutoffs (`opener_level` 0.55, `peak_level` 0.80 on `arc_level`) apply.
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

## Current state (2026-07-27)

- v7 set-role feature shipped; committed on `main` (unpushed unless the user
  asked). Design docs in `docs/superpowers/specs/` and `plans/` cover the v6
  origin; v7 evolved past them; this file + the README are current.
- **The whole library (~3,553 tracks) is tagged at v7.** Distribution:
  Opener 29.5% / Builder 19.7% / Peak 24.8% / Closer 26.0%. Zero arc failures.
- **Pending: user grading.** The user is validating roles by ear (roles are
  in the Serato comments). Next step once corrections arrive: adjust
  `ROLE_THRESHOLDS` / `FEATURE_RANGE` → `rerole` → re-check. Genre precision
  is deliberately NOT the priority (85% are already Beatport-curated).
