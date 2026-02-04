# 🎛 DJ Tagger

**Autonomous DJ music tagger powered by machine learning.**

Analyzes audio files using [Essentia](https://essentia.upf.edu/) TensorFlow models and enriches them with genre, energy, and mood metadata — written directly as ID3 tags. Genre detection combines three sources in priority order: Beatport (human-curated), Last.fm (artist tags), and local ML predictions.

Designed to run unattended on large collections. No LLM calls, no cloud APIs for audio — everything runs locally except lightweight genre lookups.

## Features

- 🎵 **ML audio analysis** — Energy, valence, and 4 mood dimensions per track
- 🏷️ **3-tier genre resolution** — Beatport → Last.fm → Essentia ML fallback
- 🎯 **Remix-aware matching** — Scores Beatport results to find the right version
- 🛡️ **Non-destructive** — Preserves existing genres, Serato cues, rekordbox data, BPM, key
- 📊 **Rich CLI** — Live progress, stats dashboard, and beautiful output
- ⏸️ **Resume support** — Skips already-tagged files automatically
- 📡 **Status file** — JSON status at `/tmp/dj-tagger-status.json` for external monitoring

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

# Embedding model
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb

# Genre model
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb

# Mood models
for mood in happy sad aggressive relaxed; do
  curl -LO "https://essentia.upf.edu/models/classification-heads/mood_${mood}/mood_${mood}-discogs-effnet-1.pb"
done
```

### Last.fm API Key (Optional)

For Last.fm genre lookups, get a free API key at [last.fm/api](https://www.last.fm/api/account/create) and set it:

```bash
export LASTFM_API_KEY="your_api_key_here"
```

## Usage

### Tag files

```bash
# Tag a folder (recursive)
djtagger tag /path/to/music

# Dry run — analyze without writing tags
djtagger tag /path/to/music --dry-run

# Force re-tag already tagged files
djtagger tag /path/to/music --force

# Skip Beatport lookups (ML-only genres)
djtagger tag /path/to/music --no-beatport

# Fix comments on already-tagged files (no re-analysis)
djtagger tag /path/to/music --fix-comments
```

### Inspect a single track

```bash
djtagger info /path/to/track.mp3
```

Example output:

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
 
 Comment              Energy: High | Mood: Neutral
 Tagger version       v4
```

### Library statistics

```bash
djtagger stats /path/to/music
```

Example output:

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

## How It Works

### Audio Analysis

Each track is loaded at 16kHz mono and passed through an Essentia [Discogs-EffNet](https://essentia.upf.edu/models.html) embedding model. The embeddings feed into four classification heads:

- **Genre** — 400-class Discogs taxonomy, top predictions extracted
- **Mood** — Happy, Sad, Aggressive, Relaxed (independent scores 0–1)

From the mood scores, two composite metrics are derived:

- **Energy** = `min(1.0, (aggressive + (1 - relaxed)) / 2 × 1.5 + 0.3)` — biased toward DJ-useful range
- **Valence** = `(happy - sad + 1) / 2` — 0 = dark, 1 = bright

### 3-Tier Genre Resolution

Genre detection tries three sources in priority order, stopping at the first hit:

| Priority | Source | Method | Quality |
|----------|--------|--------|---------|
| 1st | **Beatport** | Scrapes search results, parses `__NEXT_DATA__` JSON | Best — human-curated, track-level |
| 2nd | **Last.fm** | `artist.getTopTags` API, merged with ML predictions | Good — artist-level, supplemented by ML |
| 3rd | **Essentia ML** | Local TensorFlow model predictions | Decent — broad Discogs taxonomy |

### Remix-Aware Matching

When a filename contains remix info (e.g., `"Teardrop (Friction & Subsonic Remix)"`), DJ Tagger extracts it and uses a scoring algorithm to find the correct version on Beatport:

```
+25  Exact remix match (normalized)
+15  Partial remix word overlap
+10  Track name match
 +5  Artist match
 +3  Generic mix when no specific remix requested
-10  Wrong track name
-15  Want specific remix but got generic (Original/Extended)
-20  Want specific remix but got different remix
```

If the best Beatport match scores below 10 for a specific remix, Beatport is skipped entirely — falling through to Last.fm/ML rather than tagging with the wrong version's genre.

## ID3 Tags Written

| Tag | Content | Notes |
|-----|---------|-------|
| `TCON` | Genre | Only replaces generic genres ("Other", "Unknown", empty) |
| `TXXX:ENERGY` | Energy score (0–1) | Composite of aggressive + relaxed |
| `TXXX:VALENCE` | Valence score (0–1) | Composite of happy + sad |
| `TXXX:MOOD_HAPPY` | Happy score (0–1) | Raw ML prediction |
| `TXXX:MOOD_SAD` | Sad score (0–1) | Raw ML prediction |
| `TXXX:MOOD_AGGRESSIVE` | Aggressive score (0–1) | Raw ML prediction |
| `TXXX:MOOD_RELAXED` | Relaxed score (0–1) | Raw ML prediction |
| `TXXX:GENRE_SOURCE` | `"beatport"`, `"lastfm+ml"`, or `"ml"` | Which tier provided the genre |
| `TXXX:GENRE_DETECTED` | Full detected genre string | Stored even if TCON was preserved |
| `TXXX:TAGGER_VERSION` | Version tag (e.g., `"v4"`) | For tracking re-tag needs |
| `COMM` | `"Energy: High \| Mood: Bright"` | Human-readable, visible in Serato |
| `COMM:djtagger` | Detailed energy/valence values | Hidden reference comment |

### Genre Preservation

Existing non-generic genres are **never overwritten**. If a track already has `"Drum and Bass"` as its genre, DJ Tagger keeps it and stores its own detection in `TXXX:GENRE_DETECTED` for reference.

Only these generic/empty values get replaced: `Other`, `Unknown`, `Misc`, `Music`, `""`.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DJTAGGER_MODEL_DIR` | `~/.local/essentia-models/` | Path to Essentia TF model files |
| `DJTAGGER_MUSIC_PATH` | `.` (current dir) | Default music path for `tag` and `stats` commands |
| `LASTFM_API_KEY` | *(none)* | Last.fm API key for genre lookups |

## Performance

- **~13 seconds per track** on Apple Silicon (first batch slower due to TF warmup)
- ~3,300 tracks ≈ 12 hours
- Network lookups (Beatport/Last.fm) use `curl` with strict timeouts
- Resume is automatic — re-running skips tagged files

## Dependencies

- [essentia-tensorflow](https://essentia.upf.edu/) — Audio analysis and ML inference
- [mutagen](https://mutagen.readthedocs.io/) — ID3 tag reading/writing
- [typer](https://typer.tiangolo.com/) + [rich](https://rich.readthedocs.io/) — CLI framework
- [numpy](https://numpy.org/) — Numerical operations
- Python ≥ 3.10

## Monitoring

```bash
# Poll status (JSON)
cat /tmp/dj-tagger-status.json

# Live log
tail -f /tmp/dj-tagger.log

# Errors only
cat /tmp/dj-tagger-errors.log
```

## License

[MIT](LICENSE)
