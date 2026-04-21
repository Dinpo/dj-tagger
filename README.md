# DJ Tagger

**Autonomous DJ music tagger powered by machine learning.**

Analyzes MP3 files using [Essentia](https://essentia.upf.edu/) TensorFlow models and enriches them with genre, energy, and mood metadata — written directly as ID3 tags. Genre detection combines three sources in priority order: Beatport (human-curated), Last.fm (artist tags), and local ML predictions.

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
  - [Suggest tracks for sets](#suggest-tracks-for-sets)
  - [Collection health check](#collection-health-check)
- [Filename Convention](#filename-convention)
- [How It Works](#how-it-works)
  - [Audio Analysis Pipeline](#audio-analysis-pipeline)
  - [Energy and Valence Formulas](#energy-and-valence-formulas)
  - [3-Tier Genre Resolution](#3-tier-genre-resolution)
  - [Remix-Aware Matching](#remix-aware-matching)
- [ID3 Tags Written](#id3-tags-written)
  - [Genre Preservation](#genre-preservation)
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
```

You should end up with 11 `.pb` files:

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
└── emomusic-msd-musicnn-2.pb         # Arousal/Valence regression
```

If any new model files are missing, DJ Tagger will still work — it falls back to v4-style heuristics for the affected features and logs a warning.

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
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Analyze files and resolve genres but don't write any tags |
| `--force` | Re-tag files even if they already have a `GENRE_SOURCE` tag |
| `--no-beatport` | Skip Beatport scraping; use Last.fm + ML only |
| `--fix-comments` | Regenerate `COMM` tags from existing energy/valence values (no ML re-analysis) |

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

 Comment              Energy: High | Mood: Bright | Edge: Hard | Peak: 0.98 | Intro: Hot | Dance: High
 Tagger version       v4
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
| Danceability | `TXXX:DANCEABILITY` | Score 0–1 | Dedicated danceability model |
| Arousal | `TXXX:AROUSAL` | Score 0–1 | From emomusic model (normalized) |
| Peak energy | `TXXX:PEAK_ENERGY` | Score 0–1 | Highest energy across ~30s segments |
| Intro energy | `TXXX:INTRO_ENERGY` | Score 0–1 | Energy of the first ~30s |
| Energy variance | `TXXX:ENERGY_VARIANCE` | Small float | Low = steady groove, high = builds/drops |
| Happy | `TXXX:MOOD_HAPPY` | Score 0–1 | Raw ML prediction |
| Sad | `TXXX:MOOD_SAD` | Score 0–1 | Raw ML prediction |
| Aggressive | `TXXX:MOOD_AGGRESSIVE` | Score 0–1 | Raw ML prediction |
| Relaxed | `TXXX:MOOD_RELAXED` | Score 0–1 | Raw ML prediction |
| Genre source | `TXXX:GENRE_SOURCE` | `beatport`, `lastfm+ml`, or `ml` | Which tier provided the genre |
| Genre detected | `TXXX:GENRE_DETECTED` | Full detected genre string | Stored even if TCON was preserved |
| Tagger version | `TXXX:TAGGER_VERSION` | e.g. `v5` | For tracking re-tag needs |
| Comment | `COMM::eng` | `Energy: High \| Mood: Bright \| Edge: Hard \| Peak: 0.98 \| Intro: Hot \| Dance: High` | Human-readable, visible in Serato/rekordbox |
| Comment (detail) | `COMM:djtagger:eng` | `E:0.81 \| V:0.34 \| Agg:0.55 \| Peak:0.92 \| Intro:0.78 \| D:0.89 \| Arousal:0.77` | Hidden reference with raw values |

All other existing ID3 frames (BPM, key, Serato cues, artwork, etc.) are left untouched.

### Genre Preservation

Existing non-generic genres are **never overwritten**. If a track already has `"Drum and Bass"` as its genre, DJ Tagger keeps it and stores its own detection in `TXXX:GENRE_DETECTED` for reference.

Only these generic/empty values get replaced: `Other`, `Unknown`, `Misc`, `Music`, `""` (empty).

### Comment Format

The human-readable comment (`COMM::eng`) uses these labels:

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
| Danceability (Dance) | < 0.4 | Low |
| Danceability (Dance) | 0.4 – 0.7 | Mid |
| Danceability (Dance) | > 0.7 | High |

Mood thresholds are tuned for electronic/dance music, where valence values cluster 0.45–0.80 — the 0–1 theoretical range is rarely exercised in practice.

Peak energy is shown as a raw number (e.g., `Peak: 0.98`) for precision.

Example: `Energy: High | Mood: Bright | Edge: Hard | Peak: 0.98 | Intro: Hot | Dance: High`

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
  "version": "v4",
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
├── cli.py            # Typer CLI — tag, info, stats, find, export, suggest, health; Rich UI
├── config.py         # Environment variables, paths, constants, thresholds
├── analyzer.py       # Essentia model loading and ML inference
├── genres.py         # Beatport scraping, Last.fm API, remix scoring, genre resolution
├── library.py        # Shared library scanning — reads all tracks into structured records
├── scanner.py        # Recursive MP3 discovery and resume filtering
└── tagger.py         # ID3 tag reading/writing via mutagen; filename parsing
```

**Data flow for the `tag` command:**

```
scanner.find_mp3s()          → list of MP3 paths (sorted alphabetically by directory/file)
  ↓
scanner.filter_untagged()    → remove already-tagged files (unless --force)
  ↓
analyzer.load_models()       → load 6 TensorFlow models into memory once
  ↓
  For each MP3:
    tagger.parse_filename()  → extract artist + title from filename
    analyzer.analyze_track() → ML embeddings → genre predictions + mood scores
    genres.resolve_genres()  → Beatport → Last.fm → ML fallback
    tagger.write_tags()      → write ID3 tags, preserving existing data
```

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
