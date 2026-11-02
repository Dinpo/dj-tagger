# CLAUDE.md: DJ Tagger

Guidance for AI sessions working on this repo. Read the [README](README.md)
for full user-facing docs; this file is the fast orientation plus the
non-obvious decisions and gotchas.

## What it is

A local CLI that analyzes audio files (MP3, FLAC, M4A, AIFF, WAV, Ogg) with
Essentia TensorFlow models and writes tags: genre (Beatport → Last.fm → ML), energy, valence, moods, and a
**DJ set role**. Runs unattended over large libraries (~3,700 tracks here).
No cloud APIs for audio; only lightweight genre lookups hit the network.

- Python >= 3.10. Entry point: `djtagger.cli:app` (Typer). Venv at `.venv`.
- Tests: `.venv/bin/pytest` (currently 115, all green). `dsp.py` and
  `classify.py` are pure numpy (no Essentia import) so they test without models.
- Target library: `/Volumes/Multimedia/Music/_DJ Music` (see the user's
  auto-memory). Filenames follow `Artist - Title.<ext>`.

## Module map

| File | Responsibility |
|------|----------------|
| `cli.py` | Typer commands + Rich UI. Commands: **setup**, tag, info, stats, find, export, **genre-stats**, **rerole**, suggest, health, clean-genres, fix-audit, convert-keys, bench-bpm |
| `verify.py` | `djtagger verify` logic (full tag-set check per file; `scripts/verify_library.py` is now a thin wrapper) |
| `models.py` | Model registry (`MODELS`, `LABELS`), `download_all` for `djtagger setup`, `require_models` (friendly error instead of a TF crash). NO Essentia import |
| `analyzer.py` | Loads TF models; `analyze_track()` returns the result dict (genre preds, moods, danceability, vocal, energy, arc features, set_role) |
| `dsp.py` | Pure-numpy audio features. NO Essentia. Shared framing/FFT via `_frames`/`_magnitude_spectra` |
| `classify.py` | Set-role model. `compute_arc()`, `drive_emo()`, `classify_role()`, `decide_role()`, genre-stats helpers. NO Essentia |
| `config.py` | Constants incl. `FEATURE_RANGE`, `ROLE_THRESHOLDS`, `TAGGER_VERSION`, `LEGACY_ROLES`, `GENRE_STATS_FILE` |
| `tagfile.py` | Format-neutral tag access: `TagFile.open(path)` → ID3 (mp3/aif/wav), Vorbis (flac/ogg) or MP4 (m4a) backend; `AUDIO_EXTS`. The ONLY module that touches native frames |
| `tagger.py` | Tag read/write via `TagFile`; `write_tags()` is the single role-decision point; `rerole_file()`; `fix_comments()` |
| `genres.py` | Beatport scrape, Last.fm, remix-aware match scoring |
| `scanner.py` | Audio discovery (`find_audio_files`); `filter_untagged` (resume), `filter_outdated` (--upgrade), `filter_by_source` |
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
  one-directional boost from structural heaviness (`sub_bass` + `drop_db`),
  `HEAVINESS_BOOST * max(0, heaviness - 0.5)`, so the real max lift is ~0.075
  (not the 0.15 constant). Smooth-but-heavy tracks the mood-energy underrates
  are not forced into Opener. Never lowers energy; the stored `arc_level`
  stays raw. The boost applies to the absolute bands only: the genre
  percentile lookup uses RAW `arc_level`, because `genre_energy.json` is built
  from raw values (mixing them would compare boosted against raw).
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
  The whole library was backfilled via the v8 bump + `tag --upgrade`.
- **Honest failures.** If `compute_arc` raises, `analyze_track` logs a
  warning, writes an EMPTY role, and sets `arc_ok=False`. Never fabricate a
  role from neutral values (that silently mislabels and poisons calibration).
  On a re-tag where the arc fails, `write_tags` keeps the track's existing
  SET_ROLE rather than blanking it; an empty genre list likewise keeps the
  existing GENRE_DETECTED.
- **Legacy roles.** v6 wrote `SET_ROLE="Warm-up"`; v7 renamed it to `Opener`.
  `config.LEGACY_ROLES` maps old→new; applied in `read_tags` and persisted by
  `fix_comments`. Add future renames there.
- **Version-aware upgrade.** `is_already_tagged` only checks `GENRE_SOURCE`,
  so a plain `tag` skips tagged files regardless of version. Use
  `tag --upgrade` to backfill new fields onto old-version tracks (keeps their
  resolved genres, no network). `filter_outdated` compares `TAGGER_VERSION`,
  so adding a stored field WITHOUT bumping the version makes `--upgrade` find
  nothing (this happened with PULSE_REG; v8 fixed it).
- **Genre resolution rules** (`genres.resolve_metadata`, details in
  `docs/superpowers/notes/2026-10-26-ml-genre-quality.md`):
  - The 5-class electronic head (chance = 0.20) may lead the ML genre list
    only when top >= 0.40 AND it beats the runner-up by >= 0.10. A bare
    `> 0.20` gate fired on flat heads and prepended "ambient; dnb" noise.
  - `_merge_genres`: a re-tag from the SAME source replaces that source's own
    prior genre (ml→ml no longer accumulates). Cross-source still merges.
  - **Avoid `tag --force` for bulk re-tags.** If a Beatport lookup fails
    transiently, the ML fallback gets merged into curated Beatport genres.
    Use `--upgrade` (keeps genres, no network) or `--retag-source ml` (only
    touches ml-source tracks, and can upgrade them to Beatport).
- **Verify by reading tags, not run summaries.** `djtagger verify <lib>`
  reads every audio file (all formats, via `find_audio_files`) and checks
  version, role, PULSE_REG, energy, genre source, junk/wide genre, the
  comment role (when one is shown) and a non-zero header length; exit 1 on any
  issue. A run that reports "0 failed" can
  still leave stale tags (the ml→ml merge bug was found this way).
- **Operational notes.** The library is on an external volume that may be
  unmounted; check `/Volumes/Multimedia` first. Files renamed during a run
  fail with "No such file" (benign, re-run picks them up). `/tmp` is wiped on
  reboot, so keep long-run logs in the repo or scratchpad and re-verify from
  tags. `scripts/serato_cues.py` reads Serato cue markers (ground truth for
  any future ENTRY/EXIT work).
- **Multi-format (2026-11-02, GitHub issue #1).** All tag access goes
  through `tagfile.TagFile` (standard fields: genre/bpm/key/album/year/
  comment/detail; custom fields: our analysis names). Do not import
  `mutagen.id3` frames elsewhere. Key decisions:
  - MP3 output is byte-for-byte the same frames as before the refactor
    (verified by snapshotting write/rerole/fix-comments/clean-genre output on
    real files before and after), so no TAGGER_VERSION bump was needed.
  - Outside ID3, custom fields are `DJTAGGER_<NAME>` (Vorbis) and
    `----:com.apple.iTunes:DJTAGGER_<NAME>` (MP4). Reason: Vorbis keys are
    case-insensitive and Mixed In Key stores base64 blobs under `energy` and
    `key` in FLAC; a bare `ENERGY` would destroy them. FLAC key is read from
    `INITIALKEY` for the same reason.
  - FLAC `DESCRIPTION` is updated only when it mirrors `COMMENT` (MIK writes
    both); a different DESCRIPTION is left alone.
  - AIFF/WAV keep their existing ID3 version (v2.3 stays v2.3).
  - `.mp4` is excluded (video in music folders); `._*` AppleDouble files are
    skipped. Fixtures for every format live in `tests/fixtures/silent.*`
    (generated with ffmpeg; Ogg via `-c:a vorbis -strict -2`).
  - Essentia `MonoLoader` decodes every format via FFmpeg; no analyzer
    change was needed.
  - **Zero-length header guard.** `TagFile.save()` (non-MP3 backends) raises
    `BrokenHeaderError` when the header reports 0 length. Found on a real
    FFmpeg-exported FLAC (total_samples=0): audio intact, but after our tag
    write Serato re-read the header and loaded a 0-second track. Repair is a
    lossless re-encode + tag copy (verify decoded PCM identical). Tag writes
    were verified to leave decoded audio bit-identical in all 5 formats.
- **Usable by others (2026-11-02 review).** A fresh clone + venv was
  tested end to end. Gotchas found and fixed: macOS's built-in python3 is
  3.9 (pip then fails with a misleading "setup.py not found"), so the
  README says to use python3.12+; models download via `djtagger setup`
  (was 12 manual curl commands); missing models give a "run djtagger
  setup" error instead of a TensorFlow traceback; Last.fm uses curl_cffi
  (no curl binary needed). essentia-tensorflow only has wheels for macOS
  and Linux x86_64 (Windows: WSL2). `parse_filename` falls back to the
  file's artist/title tags when the filename has no artist or only a
  number the artist tag contradicts (track numbers like `01 - Title`;
  numeric artists like `1991` are kept). CI: `.github/workflows/tests.yml`
  installs as a new user would on Ubuntu + macOS, Python 3.10 and 3.13.
  Package version is 8.0.0 (matches TAGGER_VERSION v8; `__version__` reads
  the installed metadata).
- **Install + first run (2026-11-02).** Recommended install is
  `uv tool install git+https://github.com/Dinpo/dj-tagger` (uv picks a
  Python that fits `requires-python = ">=3.10,<3.15"`, capped at the newest
  essentia wheel). `tag` calls `cli._ensure_models()` first: if required
  models are missing and stdin is a TTY it runs the setup wizard
  (`_run_setup_wizard`: models, optional music folder, optional Last.fm
  key) and continues; non-TTY gets a "run djtagger setup" error. Answers
  live in `~/.config/djtagger/config.json` (0600; env vars win). Commands
  resolve their folder via `cli._music_path()`: argument, else saved
  folder, else an error. They NEVER fall back to the current directory
  (it used to be `.`, which made a bare `djtagger tag` tag wherever you
  stood). Testing the wizard needs a real pty: macOS `script` sends EOF
  before the first prompt; drive it with a Python `pty.fork` loop.
- **The visible comment is opt-in (user decision, 2026-11-02).** `tag` does
  not touch it without `--comment`; `--fix-comments` is also an explicit
  opt-in. `tagger._apply_summary` is the only writer: it puts our summary
  first and keeps the user's text after ` // `, replacing only our part on
  later runs (`_split_comment`, `_SUMMARY_RE`), so a comment that merely
  starts with "Role:" stays the user's. A comment that already holds a
  summary is always refreshed, flag or not, so `rerole` can't leave a stale
  role. The hidden detail (`COMM:djtagger:eng` / `DJTAGGER_DETAIL`) is ours
  and always written. ID3: the visible comment is `COMM::eng`, else any
  unnamed COMM (3 library files use lang XXX); named frames are not it.
  `COMM:ID3v1 Comment:eng` still holds the original text (e.g.
  `6B - Energy 6`) on ~3,500 MP3s, should it ever be wanted back.
  For this library: tag new tracks with `djtagger tag <lib> --comment`.
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

## Current state (2026-11-02)

- `TAGGER_VERSION` is **v8** (bumped when PULSE_REG was added; bump again
  whenever the stored tag set changes so `tag --upgrade` can find stale files).
  All work committed and pushed to `origin/main`.
- **Library fully tagged:** all 3,723 tracks in `_DJ Music` (3,690 MP3 +
  33 FLAC/M4A/AIFF/WAV, tagged 2026-11-02) are at v8 with a
  valid role, PULSE_REG, energy, a non-junk genre, and a `Role:` comment
  (comments were written before `--comment` became opt-in; new tracks need
  the flag, see below).
  Distribution roughly Opener 21% / Builder 20% / Peak 33% / Closer 25%.
- **ML-genre quality: fixed and applied.** The 5-class electronic genre head
  (ambient/dnb/house/techno/trance, chance = 0.20) may only lead when
  top >= 0.40 AND it leads the runner-up by >= 0.10; otherwise the finer
  Discogs-400 labels win. Same-source re-tags replace (not merge) their own
  prior genre. Apply resolver changes with `tag --retag-source ml`.
- **Verifying "properly tagged":** the resume check only looks for
  GENRE_SOURCE, so it does not prove completeness. Run
  `djtagger verify` (last run 2026-11-02, all formats: 3,723
  tracks, 3,722 clean; the one flag is a coherent 4-genre house list on an
  ML-only M4A, a false positive of the "4+ ML genres" heuristic).
- **Pending, user-driven:** grading roles by ear, then tune
  `ROLE_THRESHOLDS` / `FEATURE_RANGE` and `rerole`. Parked: hypnotic-groove
  detection and ENTRY/EXIT cue detection (both need more labels; see notes).
