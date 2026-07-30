# DJ Tagger

**Autonomous DJ music tagger powered by machine learning.**

Analyzes MP3 files using [Essentia](https://essentia.upf.edu/) TensorFlow models and enriches them with genre, energy, and mood metadata — written directly as ID3 tags. Genre detection combines three sources in priority order: Beatport (human-curated, track-level), Last.fm (artist tags), and local ML predictions. Beatport also fills empty album / year fields when a match is found.

Designed to run unattended on large collections. No LLM calls, no cloud APIs for audio — everything runs locally except lightweight genre lookups.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Tag files](#tag-files)
  - [Inspect a single track](#inspect-a-single-track)
  - [Library statistics](#library-statistics)
  - [Search and filter](#search-and-filter)
  - [Export library data](#export-library-data)
  - [Set-role tuning (genre-stats and rerole)](#set-role-tuning-genre-stats-and-rerole)
  - [Suggest tracks for sets](#suggest-tracks-for-sets)
  - [Collection health check](#collection-health-check)
- [Filename Convention](#filename-convention)
- [How It Works](#how-it-works)
  - [Audio Analysis Pipeline](#audio-analysis-pipeline)
  - [Energy and Valence Formulas](#energy-and-valence-formulas)
  - [Set Role](#set-role)
  - [3-Tier Genre Resolution](#3-tier-genre-resolution)
  - [Remix-Aware Matching](#remix-aware-matching)
- [ID3 Tags Written](#id3-tags-written)
  - [Genre Resolution Rules](#genre-resolution-rules)
  - [Comment Format](#comment-format)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [ML Thresholds](#ml-thresholds)
  - [Network Timeouts](#network-timeouts)
- [Monitoring](#monitoring)
  - [Log Files](#log-files)
  - [Status File](#status-file)
- [Architecture](#architecture)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **ML audio analysis** — Energy, valence, and 4 mood dimensions per track
- **3-tier genre resolution** — Beatport → Last.fm → Essentia ML fallback
- **Album + year** — filled from Beatport; empty / junk values (URL spam, non-year) are replaced, legit values preserved
- **Remix-aware matching** — Scores Beatport results to find the right version
- **Non-destructive** — Preserves existing genres, Serato cues, rekordbox data, BPM, key
- **Rich CLI** — Live progress, stats dashboard, and color-coded output via [Rich](https://rich.readthedocs.io/)
- **Resume support** — Skips already-tagged files automatically (checks for `TXXX:GENRE_SOURCE`)
- **Offline capable** — ML analysis is fully local; genre labels are cached after first network fetch
- **Status file** — JSON status at `/tmp/dj-tagger-status.json` for external monitoring

## Requirements

- **Python** >= 3.10
- **curl** — used for Beatport scraping and Last.fm API calls (must be on `PATH`)
- **Essentia TensorFlow models** — downloaded separately (see [Installation](#download-ml-models))

### Python Dependencies

| Package | Purpose |
|---------|---------|
| [essentia-tensorflow](https://essentia.upf.edu/) | Audio loading, embedding extraction, ML inference |
| [mutagen](https://mutagen.readthedocs.io/) | ID3 tag reading and writing |
| [numpy](https://numpy.org/) | Numerical operations on model outputs |
| [typer](https://typer.tiangolo.com/) >= 0.9 | CLI framework |
| [rich](https://rich.readthedocs.io/) >= 13.0 | Terminal formatting, progress bars, tables |

## Installation

```bash
git clone https://github.com/dinopatti/dj-tagger.git
cd dj-tagger
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Download ML Models

DJ Tagger needs Essentia TensorFlow models. Download them to `~/.local/essentia-models/` (or set `DJTAGGER_MODEL_DIR`):

```bash
mkdir -p ~/.local/essentia-models && cd ~/.local/essentia-models

# Embedding models
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -LO https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb

# Genre classification heads
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_electronic/genre_electronic-discogs-effnet-1.pb

# Mood classification heads
for mood in happy sad aggressive relaxed party; do
  curl -LO "https://essentia.upf.edu/models/classification-heads/mood_${mood}/mood_${mood}-discogs-effnet-1.pb"
done

# Danceability + Arousal/Valence
curl -LO https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb

# Voice/instrumental (v7 set-role: vocal presence)
curl -LO https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb
```

You should end up with 12 `.pb` files:

```
~/.local/essentia-models/
├── discogs-effnet-bs64-1.pb          # EffNet embedding model
├── msd-musicnn-1.pb                  # MusicNN embedding model
├── genre_discogs400-discogs-effnet-1.pb
├── genre_electronic-discogs-effnet-1.pb
├── mood_happy-discogs-effnet-1.pb
├── mood_sad-discogs-effnet-1.pb
├── mood_aggressive-discogs-effnet-1.pb
├── mood_relaxed-discogs-effnet-1.pb
├── danceability-discogs-effnet-1.pb
├── voice_instrumental-discogs-effnet-1.pb   # v7 vocal presence
└── emomusic-msd-musicnn-2.pb         # Arousal/Valence regression
```

If any model files are missing, DJ Tagger still works: it falls back to heuristics for the affected features and logs a warning. Without `voice_instrumental`, the set-role vocal signal is treated as 0.

### Last.fm API Key (Optional)

For Last.fm genre lookups (tier 2), get a free API key at [last.fm/api](https://www.last.fm/api/account/create) and set it:

```bash
export LASTFM_API_KEY="your_api_key_here"
```

Without this key, genre resolution falls back directly from Beatport to ML-only predictions.

## Usage

### Tag files

```bash
# Tag a folder (recursive)
djtagger tag /path/to/music

# Dry run — analyze without writing tags
djtagger tag /path/to/music --dry-run

# Force re-tag already tagged files
djtagger tag /path/to/music --force

# Skip Beatport lookups (Last.fm + ML-only genres)
djtagger tag /path/to/music --no-beatport

# Fix comments on already-tagged files (no re-analysis)
djtagger tag /path/to/music --fix-comments

# Upgrade older-version tracks to the current tagger (keeps their genres)
djtagger tag /path/to/music --upgrade
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Analyze files and resolve genres but don't write any tags |
| `--force` | Re-tag files even if they already have a `GENRE_SOURCE` tag |
| `--no-beatport` | Skip Beatport scraping; use Last.fm + ML only |
| `--detect-bpm-key` | Also detect BPM and key (off by default — DJ software usually owns these) |
| `--fix-comments` | Regenerate `COMM` tags from existing energy/valence values (no ML re-analysis) |
| `--upgrade` | Re-analyze files tagged by an older tagger version; keeps their resolved genres (no network lookups needed) |

If no path is given, it defaults to the current directory (or `DJTAGGER_MUSIC_PATH`).

### Inspect a single track

```bash
djtagger info /path/to/track.mp3
```

Shows all DJ Tagger tags with colored bars for energy, valence, and mood scores:

```
🎵 Friction & Skream — Teardrop (Friction & Subsonic Remix)

 Tag                  Value
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Genre                Drum and Bass
 Genre source         🟢 beatport

 Energy               ████████████████░░░░  0.812
 Valence              █████████░░░░░░░░░░░  0.467

 😊 Happy             █████████░░░░░░  0.591
 😢 Sad               ██████░░░░░░░░░  0.406
 🔥 Aggressive        ██████████░░░░░  0.673
 😌 Relaxed           ████░░░░░░░░░░░  0.258

 Comment              Role: Peak | Energy: High | Mood: Bright | Edge: Hard | Peak: 0.98 | Intro: Hot
 Tagger version       v7
```

### Library statistics

```bash
djtagger stats /path/to/music
```

Displays a full dashboard of your tagged collection: coverage, genre source breakdown, top 20 genres with bar charts, energy/valence histograms, genre-by-energy profiles, and tagger version counts.

```
📊 Library Statistics

╭──────────────────────────────────╮
│         Overview                 │
├──────────┬───────────────────────┤
│ Total    │                 3,312 │
│ Tagged   │                 3,307 │
│ Untagged │                     5 │
│ Coverage │                99.8%  │
╰──────────┴───────────────────────╯

╭──────────────────────────────────╮
│       Genre Sources              │
├──────────────────┬───────────────┤
│ 🟢 beatport      │         2,041 │
│ 🟡 lastfm+ml     │           893 │
│ 🔵 ml             │           373 │
╰──────────────────┴───────────────╯
```

### Version

```bash
djtagger --version
djtagger -v
```

### Search and filter

Query your tagged library by genre, energy, valence, mood scores, or genre source:

```bash
# Find high-energy Drum and Bass tracks
djtagger find /music --genre "Drum and Bass" --energy 0.8:1.0

# Find chill, happy tracks for a warm-up set
djtagger find /music --energy 0.3:0.6 --mood-relaxed 0.5: --valence 0.6:

# Find all tracks tagged by ML only (weakest confidence)
djtagger find /music --source ml

# Find untagged tracks
djtagger find /music --untagged

# Sort by valence descending, show top 20
djtagger find /music --sort valence --limit 20
```

| Flag | Description |
|------|-------------|
| `--genre`, `-g` | Filter by genre (case-insensitive substring match) |
| `--energy`, `-e` | Energy range as `MIN:MAX` (e.g., `0.7:1.0`, `0.5:`, `:0.3`) |
| `--valence` | Valence range as `MIN:MAX` |
| `--mood-happy` | Happy score range |
| `--mood-sad` | Sad score range |
| `--mood-aggressive` | Aggressive score range |
| `--mood-relaxed` | Relaxed score range |
| `--source`, `-s` | Filter by genre source (`beatport`, `lastfm+ml`, `ml`) |
| `--untagged` | Show only untagged tracks |
| `--sort` | Sort by: `energy`, `valence`, `genre`, `artist`, `title`, `path` (default: `energy`) |
| `--reverse`, `-r` | Reverse sort order |
| `--limit`, `-n` | Max results (default: 50, 0 = unlimited) |

### Export library data

Dump your entire library's tag data to CSV or JSON for use in spreadsheets, visualization tools, or other DJ software:

```bash
# Export as CSV
djtagger export /music > library.csv

# Export as JSON
djtagger export /music --format json > library.json

# Pipe to other tools
djtagger export /music --format json | jq '.[] | select(.energy > 0.8)'
```

Exported fields: `path`, `artist`, `title`, `folder`, `genre`, `genre_source`, `genre_detected`, `energy`, `valence`, `mood_happy`, `mood_sad`, `mood_aggressive`, `mood_relaxed`, `tagger_version`, `comment`, `tagged`.

Progress output goes to stderr so stdout stays clean for piping.

### Set-role tuning (genre-stats and rerole)

The set role (Opener / Builder / Peak / Closer) is decided from measured
features, but you can tune *how* those features map to roles without
re-analyzing any audio. Two commands support the loop:

```bash
# Build the per-genre energy table for genre-relative role bands.
# Reads existing ENERGY tags, writes genre_energy.json next to the models.
# Run once, and again after large library changes.
djtagger genre-stats /path/to/music

# Re-decide every track's role from its already-stored tags (no audio, no
# ML). Applies the current ROLE_THRESHOLDS / FEATURE_RANGE and the
# genre-relative bands, rewriting SET_ROLE and the comment in seconds.
djtagger rerole /path/to/music
```

Tuning workflow: listen and note wrong roles → edit `ROLE_THRESHOLDS` /
`FEATURE_RANGE` in `djtagger/config.py` → `djtagger rerole` → re-check.
The expensive ML analysis runs only once (via `tag`); role tuning after
that is a seconds-long `rerole`. See [Set Role](#set-role) for the model.

### Suggest tracks for sets

Build sets and find tracks that mix well together:

```bash
# Find tracks similar to a reference track (by energy, mood, genre)
djtagger suggest /music --like /music/DnB/Friction\ -\ Teardrop.mp3

# Build a rising-energy set (15 tracks, low to high)
djtagger suggest /music --energy-curve rising --count 15

# Build a falling-energy cooldown set
djtagger suggest /music --energy-curve falling --count 10

# Steady energy within a genre
djtagger suggest /music --genre "House" --energy-curve steady --count 20

# Diverse mix across your whole library
djtagger suggest /music
```

| Flag | Description |
|------|-------------|
| `--like`, `-l` | Path to a reference track — finds similar tracks by energy, valence, mood, and genre |
| `--genre`, `-g` | Filter suggestions to a specific genre |
| `--energy-curve`, `-c` | Build a set with energy progression: `rising`, `falling`, or `steady` |
| `--count`, `-n` | Number of suggestions (default: 15) |

**Similarity scoring** (for `--like`) weights energy similarity highest (most important for mixing), then valence, then individual mood scores, with a genre match bonus.

### Collection health check

Audit your library for quality issues and get an overall health score:

```bash
djtagger health /music
```

Example output:

```
🏥 Collection Health Report
/music — 3,312 files

  ⚠  142 tracks with ML-only genres (4% — weakest confidence)
  ⚠   23 tracks with no artist in filename
  ✓  3,165 tracks with Beatport/Last.fm genres
  ✓  All tracks tagged
  ✓  All tagged tracks have energy/valence scores
  ⚠  8 genres with only 1-2 tracks

  Health Score: 87/100 — Excellent
```

The health command shows:
- Untagged files (with a list if <= 20)
- ML-only genre tracks (weakest confidence — you may want to verify these)
- Missing artist names in filenames
- Rare genres (1-2 tracks — might indicate mis-tagging)
- An overall score from 0-100 based on coverage, genre source quality, artist info completeness, and genre completeness

## Filename Convention

DJ Tagger parses artist and title from the filename using this format:

```
Artist - Title.mp3
```

The ` - ` separator (space-dash-space) splits artist from title. If no separator is found, the entire filename (minus extension) is treated as the title with no artist.

**Artist cleaning:** A trailing country code in parentheses (e.g., `(UK)`, `(US)`) is stripped from the artist name before lookups.

**Examples:**

| Filename | Artist | Title |
|----------|--------|-------|
| `Friction & Skream - Teardrop (Friction & Subsonic Remix).mp3` | `Friction & Skream` | `Teardrop (Friction & Subsonic Remix)` |
| `Chase & Status (UK) - Blind Faith.mp3` | `Chase & Status` (cleaned) | `Blind Faith` |
| `Unknown Track.mp3` | *(empty)* | `Unknown Track` |

> **Note:** DJ Tagger only processes `.mp3` files. Other formats (FLAC, WAV, AAC, etc.) are silently skipped during scanning.

## How It Works

### Audio Analysis Pipeline

1. **Load audio** — Track is loaded at 16 kHz mono via Essentia's `MonoLoader`
2. **Extract embeddings** — Audio is passed through a [Discogs-EffNet](https://essentia.upf.edu/models.html) model, producing a time-series of embedding vectors
3. **Genre prediction** — Embeddings feed into a 400-class Discogs genre classification head; predictions are averaged across time frames, and the top 5 genres above a minimum probability threshold (default 0.05) are kept
4. **Mood prediction** — Embeddings feed into 4 independent mood classification heads (happy, sad, aggressive, relaxed), each producing a score from 0 to 1
5. **Composite metrics** — Energy and valence are derived from the mood scores (see formulas below)
6. **Genre resolution** — The 3-tier lookup (Beatport → Last.fm → ML) determines the final genre
7. **Tag writing** — Results are written as ID3 tags, preserving all existing non-DJ-Tagger tags

### Energy and Valence Formulas

**Energy** is biased toward the DJ-useful range (most tracks land between 0.5–1.0):

```
raw_energy = clamp((aggressive + (1 - relaxed)) / 2, 0, 1)
energy     = min(1.0, raw_energy × 1.5 + 0.3)
```

**Valence** maps to a 0–1 scale where 0 = dark/sad and 1 = bright/happy:

```
valence = clamp((happy - sad + 1) / 2, 0, 1)
```

### Set Role

Each track is classified into one of four DJ set roles and written as the first field of the visible comment:

- **Opener** (energy 3-5): hypnotic, deep, atmospheric; low activity, no heavy drops.
- **Builder** (energy 6-7): driving, urgent, rising; promises something bigger is coming.
- **Peak** (energy 8-10): maximum intensity; drops, aggressive rhythms, big hooks.
- **Closer** (energy 5-7): emotional, anthemic, melodic; sing-along vocals, fading finale.

Energy bands decide Opener vs Peak. Builder and Closer share the mid band and are split by character: a **drive** index (spectral flux, onset rate, rising loudness) against an **emo** index (valence, brightness, vocal presence, fading outro). Both indices are written as tags (`DRIVE`, `EMO`), alongside the raw features that feed them: spectral flux, vocal presence (voice/instrumental model on the existing EffNet embeddings), and a loudness-arc analysis of the track's envelope (intro/outro depth in dB, loudness slope, breakdown-to-drop height, peak position).

**Genre-relative bands:** run `djtagger genre-stats` once to build a per-genre energy table from the tagged library. The absolute energy band always stands (`arc_level` >= `peak_level` 0.80 is Peak; Opener requires energy at or below `opener_level` 0.55). On top of that, when a track's genre cohort is large enough (30+ tracks), being high *within its own genre* can additionally promote it to Peak, and Opener also requires being low within its genre. So genre banding only ever *promotes* a low-energy melodic peak or filters a track out of Opener; it never demotes an absolute banger.

**Heavy-track lift (effective energy):** the mood-based energy underrates smooth-but-driving production (melodic/hypnotic techno reads low even when it is a floor-filler). For the role decision only, a track's *effective* energy is lifted by its structural heaviness (sub-bass weight + drop height), one-directional (never lowered). The stored `arc_level` remains the raw measured energy. This is a partial fix; genuinely low-energy hypnotic grooves remain a known limitation (see `docs/superpowers/notes/`).

All thresholds and normalization ranges are calibrated against the library and stored in `config.py` (`ROLE_THRESHOLDS` / `FEATURE_RANGE`). On the full library this yields roughly Opener 20% / Builder 21% / Peak 33% / Closer 26%.

**Tuning:** roles are decided from stored tags, so after editing the thresholds you can re-apply them library-wide in seconds with `djtagger rerole` (no re-analysis). See [Set-role tuning](#set-role-tuning-genre-stats-and-rerole).

### 3-Tier Genre Resolution

Genre detection tries three sources in priority order, stopping at the first hit:

| Priority | Source | Method | Quality |
|----------|--------|--------|---------|
| 1st | **Beatport** | Scrapes search results, parses `__NEXT_DATA__` JSON | Best — human-curated, track-level |
| 2nd | **Last.fm** | `artist.getTopTags` API via curl, merged with ML predictions | Good — artist-level, supplemented by ML |
| 3rd | **Essentia ML** | Local TensorFlow model predictions only | Decent — broad Discogs taxonomy |

**Beatport** results are scored and the best match is selected (see [Remix-Aware Matching](#remix-aware-matching)). Results are cached in a bounded LRU cache (500 entries) to reduce repeat lookups.

**Last.fm** uses the `artist.getTopTags` endpoint. Only tags with a play count above a minimum threshold (default 20) are kept. The cleaned artist name is tried first, then the raw artist name. When Last.fm succeeds, its genres are merged with ML predictions (Last.fm genres first, then ML genres not already in the list).

**ML fallback** uses genres from the Discogs-EffNet model where the predicted probability exceeds the keep threshold (default 0.10). Genre labels follow the Discogs taxonomy (e.g., `Electronic---Drum n Bass`) and are cleaned to use only the sub-genre portion (e.g., `Drum n Bass`).

### Remix-Aware Matching

When a filename contains remix info (e.g., `"Teardrop (Friction & Subsonic Remix)"`), DJ Tagger extracts it and uses a scoring algorithm to find the correct version on Beatport:

| Score | Condition |
|-------|-----------|
| +25 | Exact remix match (after normalization) |
| +15 | Partial remix word overlap (plus +3 per overlapping word) |
| +10 | Track name match |
| +5 | Full artist match |
| +2 | Partial artist match (individual name from `Artist1 & Artist2`) |
| +3 | Generic mix when no specific remix requested |
| -10 | Wrong track name |
| -15 | Want specific remix but got generic (Original/Extended) |
| -20 | Want specific remix but got different remix |

If the best Beatport match scores below 10 for a specific remix, Beatport is skipped entirely — falling through to Last.fm/ML rather than tagging with the wrong version's genre.

**Normalization** strips filler words (`extended`, `original`, `radio`) and suffixes (`remix`, `mix`, `edit`, `dub`, `rework`), normalizes `&` to spaces, and lowercases everything before comparison.

**Recognized mix patterns:** remix, mix, edit, dub, rework, bootleg, version, VIP.

## ID3 Tags Written

| Tag | Frame | Content | Notes |
|-----|-------|---------|-------|
| Genre | `TCON` | Genre string | Only replaces generic genres (see below) |
| Energy | `TXXX:ENERGY` | Score 0–1 | Weighted blend: danceability + arousal + aggressive + relaxed |
| Valence | `TXXX:VALENCE` | Score 0–1 | From emomusic arousal/valence model |
| Danceability | `TXXX:DANCEABILITY` | Score 0–1 | Dedicated danceability model; still computed and tagged, but no longer shown in the visible comment (near-constant across this library) |
| Arousal | `TXXX:AROUSAL` | Score 0–1 | From emomusic model (normalized) |
| Peak energy | `TXXX:PEAK_ENERGY` | Score 0–1 | Highest energy across ~30s segments |
| Intro energy | `TXXX:INTRO_ENERGY` | Score 0–1 | Energy of the first ~30s |
| Energy variance | `TXXX:ENERGY_VARIANCE` | Small float | Low = steady groove, high = builds/drops |
| Happy | `TXXX:MOOD_HAPPY` | Score 0–1 | Raw ML prediction |
| Sad | `TXXX:MOOD_SAD` | Score 0–1 | Raw ML prediction |
| Aggressive | `TXXX:MOOD_AGGRESSIVE` | Score 0–1 | Raw ML prediction |
| Relaxed | `TXXX:MOOD_RELAXED` | Score 0–1 | Raw ML prediction |
| Genre source | `TXXX:GENRE_SOURCE` | `beatport`, `lastfm+ml`, or `ml` | Which tier provided the genre |
| Album | `TALB` | e.g. `Offender Remixes` | Filled from Beatport; replaces empty or URL/promo-junk existing values, preserves legit ones |
| Year | `TDRC` | e.g. `2022` | Filled from Beatport; replaces invalid-year existing values, preserves valid years |
| Genre detected | `TXXX:GENRE_DETECTED` | Full detected genre string | Stored even if TCON was preserved |
| Set role | `TXXX:SET_ROLE` | `Opener`, `Builder`, `Peak`, or `Closer` | See [Set Role](#set-role) |
| Arc level | `TXXX:ARC_LEVEL` | Score 0-1 | Overall intensity, derived from energy |
| Arc momentum | `TXXX:ARC_MOMENTUM` | Score -1 to +1 | Internal energy trajectory (rising/falling/flat) |
| Spectral centroid | `TXXX:SPECTRAL_CENTROID` | Raw value | Brightness, low-level DSP feature |
| Onset rate | `TXXX:ONSET_RATE` | Raw value | Percussive density/drive, low-level DSP feature |
| Dynamic range | `TXXX:DYNAMIC_RANGE` | Value in dB | Quiet-loud contrast, low-level DSP feature |
| Sub-bass ratio | `TXXX:SUB_BASS` | Score 0-1 | Fraction of energy below 120 Hz |
| Spectral flux | `TXXX:FLUX` | Score ~0.2-0.5 | How fast the spectrum changes (musical activity) |
| Pulse regularity | `TXXX:PULSE_REG` | Score 0-1 | Beat-pulse strength; captured for future use, not yet in the role decision |
| Vocal presence | `TXXX:VOCAL` | Score 0-1 | Voice/instrumental model, mean voice probability |
| Intro depth | `TXXX:INTRO_DB` | dB <= 0 | How far the first 20 s sit below the loudest moment |
| Outro depth | `TXXX:OUTRO_DB` | dB <= 0 | How far the last 20 s sit below the loudest moment |
| Loudness slope | `TXXX:ARC_SLOPE` | dB per track | Overall loudness trend (positive = gets louder) |
| Drop height | `TXXX:DROP_DB` | dB >= 0 | Largest breakdown-to-drop rise in the envelope |
| Peak position | `TXXX:PEAK_POS` | 0-1 | Where the loudest moment sits in the track |
| Drive index | `TXXX:DRIVE` | Score 0-1 | Builder character: flux + onsets + rising loudness |
| Emo index | `TXXX:EMO` | Score 0-1 | Closer character: valence + brightness + vocals + fade |
| Tagger version | `TXXX:TAGGER_VERSION` | e.g. `v7` | For tracking re-tag needs |
| Comment | `COMM::eng` | `Role: Builder \| Energy: Mid \| Mood: Neutral \| Edge: Soft \| Peak: 0.95 \| Intro: Mid` | Human-readable, visible in Serato/rekordbox |
| Comment (detail) | `COMM:djtagger:eng` | `E:0.81 \| V:0.34 \| Agg:0.55 \| Peak:0.92 \| Intro:0.78 \| D:0.89 \| Arousal:0.77` | Hidden reference with raw values |

All other existing ID3 frames (BPM, key, Serato cues, artwork, etc.) are left untouched.

### Genre Resolution Rules

When writing the `TCON` (Genre) frame, DJ Tagger compares the existing value against what Beatport / Last.fm / ML proposes, tokenizes both (case-insensitive, normalising `/`, `;`, `,`, `&`, `and`), and picks one of these actions:

| Existing vs proposed | Action |
|----------------------|--------|
| existing empty | **fill** with proposed |
| existing is junk (URL/promo spam) | **replace** with proposed |
| token-identical | **keep** existing (preserve original formatting) |
| existing ⊂ proposed (more specific) | **upgrade** (e.g. `House` → `Deep House`) |
| proposed ⊂ existing | **keep** existing (don't downgrade) |
| disjoint or partial overlap | **merge** — append proposed genres, deduped at token level, capped at 5 total |

The merge rule means a Coldplay rock/house crossover ending up with `Rock; Alternative; Britpop; House; Dance / Pop` — searchable under every tag, nothing lost.

Regardless of what ends up in `TCON`, the full detected genre string is always also stored in `TXXX:GENRE_DETECTED` for audit. Generic values like `Other`, `Unknown`, `Misc`, `Music`, and empty strings are treated as junk for replacement purposes.

### Comment Format

The human-readable comment (`COMM::eng`) leads with the set role (see [Set Role](#set-role)), followed by these labels:

| Metric | Range | Label |
|--------|-------|-------|
| Energy | < 0.4 | Low |
| Energy | 0.4 – 0.7 | Mid |
| Energy | > 0.7 | High |
| Valence (Mood) | < 0.58 | Dark |
| Valence (Mood) | 0.58 – 0.68 | Neutral |
| Valence (Mood) | ≥ 0.68 | Bright |
| Aggressive (Edge) | < 0.25 | Soft |
| Aggressive (Edge) | 0.25 – 0.5 | Mid |
| Aggressive (Edge) | ≥ 0.5 | Hard |
| Intro energy (Intro) | < 0.5 | Quiet |
| Intro energy (Intro) | 0.5 – 0.75 | Mid |
| Intro energy (Intro) | ≥ 0.75 | Hot |

Mood thresholds are tuned for electronic/dance music, where valence values cluster 0.45–0.80 — the 0–1 theoretical range is rarely exercised in practice.

Peak energy is shown as a raw number (e.g., `Peak: 0.98`) for precision.

Danceability is no longer part of the visible comment (it's near-constant across this library, so it carries little signal there), but it is still computed and written to the `DANCEABILITY` tag.

Example: `Role: Builder | Energy: Mid | Mood: Neutral | Edge: Soft | Peak: 0.95 | Intro: Mid`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DJTAGGER_MODEL_DIR` | `~/.local/essentia-models/` | Path to the directory containing Essentia TF model `.pb` files |
| `DJTAGGER_MUSIC_PATH` | `.` (current directory) | Default music path for `tag` and `stats` commands when no argument is given |
| `LASTFM_API_KEY` | *(none — disables Last.fm)* | Last.fm API key for genre lookups. Without this, tier 2 is skipped |

### ML Thresholds

These are compile-time constants in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `GENRE_MIN_PROB` | `0.05` | Minimum predicted probability for an ML genre to be considered at all |
| `GENRE_KEEP_PROB` | `0.10` | Minimum probability for an ML genre to appear in the final tag list |
| `LASTFM_MIN_COUNT` | `20` | Minimum Last.fm tag play count to consider a genre valid |

### Network Timeouts

| Constant | Default | Description |
|----------|---------|-------------|
| `BEATPORT_TIMEOUT` | 8 s | Curl timeout for Beatport page fetch |
| `LASTFM_TIMEOUT` | 5 s | Curl timeout for Last.fm API call |
| `SOCKET_TIMEOUT` | 10 s | Global Python socket timeout (set during `tag` command) |

## Monitoring

### Log Files

| File | Content |
|------|---------|
| `/tmp/dj-tagger.log` | Full processing log — one entry per track with genre, energy, valence |
| `/tmp/dj-tagger-errors.log` | Errors only — failed analyses and tag writes |

```bash
# Live log
tail -f /tmp/dj-tagger.log

# Errors only
cat /tmp/dj-tagger-errors.log
```

Log files are overwritten on each `tag` run (not appended).

### Status File

`/tmp/dj-tagger-status.json` is a JSON file updated after every track, designed for external tools to poll:

```json
{
  "state": "running",
  "mode": "TAGGING",
  "version": "v7",
  "total": 3312,
  "to_process": 150,
  "skipped": 3162,
  "processed": 42,
  "failed": 0,
  "current": "track.mp3",
  "current_folder": "Drum and Bass",
  "genre_sources": {
    "beatport": 28,
    "lastfm+ml": 10,
    "ml": 4
  },
  "started": "2025-06-15 14:30:00",
  "updated": "2025-06-15 14:39:12",
  "avg_seconds": 13.2,
  "eta_hours": 0.39,
  "last_track_seconds": 12.8
}
```

When processing completes, `state` changes to `"done"` and `finished` and `elapsed_hours` fields are added.

## Architecture

```
djtagger/
├── __init__.py       # Package version
├── cli.py            # Typer CLI: tag, info, stats, find, export, genre-stats, rerole, suggest, health; Rich UI
├── config.py         # Env vars, paths, constants, thresholds, set-role FEATURE_RANGE / ROLE_THRESHOLDS
├── analyzer.py       # Essentia model loading and ML inference (returns the analysis result dict)
├── dsp.py            # Pure-numpy audio features (centroid, flux, onset, sub-bass, dynamic range, loudness arc). NO Essentia, unit-tested
├── classify.py       # Set-role model: drive/emo indices, energy bands, genre-relative bands, decide_role. NO Essentia, unit-tested
├── genres.py         # Beatport scraping, Last.fm API, remix scoring, genre resolution
├── library.py        # Shared library scanning: reads all tracks into structured records
├── scanner.py        # Recursive MP3 discovery, resume filtering, version-aware upgrade filtering
└── tagger.py         # ID3 tag reading/writing via mutagen; role decision; filename parsing
```

**Data flow for the `tag` command:**

```
scanner.find_mp3s()          → list of MP3 paths (sorted alphabetically by directory/file)
  ↓
scanner.filter_untagged()    → remove already-tagged files (or filter_outdated for --upgrade)
  ↓
analyzer.load_models()       → load the TensorFlow models into memory once
  ↓
  For each MP3:
    tagger.parse_filename()  → extract artist + title from filename
    analyzer.analyze_track() → ML embeddings → genre/mood/danceability/vocal + energy
      └─ classify.compute_arc() → dsp features + drive/emo + provisional (global-band) role
    genres.resolve_metadata() → Beatport → Last.fm → ML fallback (skipped in --upgrade if genre exists)
    tagger.write_tags()      → SINGLE role decision point: classify.decide_role() using the
                               genre actually written (genre-relative when genre-stats exists),
                               then write ID3 tags, preserving existing data
```

**Set-role tuning is decoupled from analysis.** The role decision reads only
stored tags (`arc_level`, `drive`, `emo`, genre), so `djtagger rerole`
re-applies changed `ROLE_THRESHOLDS` / `FEATURE_RANGE` library-wide in
seconds without touching audio. See [Set Role](#set-role).

## Performance

- **~13 seconds per track** on Apple Silicon (first batch slower due to TF model warmup)
- ~3,300 tracks ≈ 12 hours
- Network lookups (Beatport/Last.fm) use `curl` with strict timeouts
- Beatport results are cached in memory (LRU, 500 entries) across a single run
- Resume is automatic — re-running skips tagged files based on `TXXX:GENRE_SOURCE` presence
- Files are processed in sorted order (alphabetically by directory, then filename)

## Troubleshooting

**"Cannot load genre labels"** — On first run, DJ Tagger fetches the 400-class genre label list from Essentia's servers and caches it locally in the model directory. If you're offline on first run, this will fail. Run once with an internet connection, or manually download the labels JSON to `~/.local/essentia-models/genre_discogs400_labels.json`.

**Beatport returns no results** — Beatport's page structure may change. If you see a "Beatport page structure changed" warning on stderr, scraping is broken for that run. Genres will fall through to Last.fm/ML. This does not affect already-tagged files.

**"LASTFM_API_KEY not set"** — This is a one-time warning printed to stderr. Last.fm lookups are disabled; genre resolution goes from Beatport directly to ML. Set the `LASTFM_API_KEY` environment variable to enable tier 2.

**Models not found** — Ensure all 6 `.pb` files are in `~/.local/essentia-models/` (or the directory set by `DJTAGGER_MODEL_DIR`). The expected filenames are listed in [Installation](#download-ml-models).

**Track analysis fails** — If a single track fails (corrupt file, unsupported encoding), it is logged to `/tmp/dj-tagger-errors.log` and skipped. The rest of the run continues.

**Re-tagging** — By default, already-tagged files are skipped. Use `--force` to re-analyze and re-tag everything. Use `--fix-comments` to only regenerate comments from existing energy/valence values (no ML re-analysis needed).

## License

[MIT](LICENSE) — Copyright (c) 2025 Dino Patti
