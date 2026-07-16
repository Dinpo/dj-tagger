"""Settings, paths, and constants."""

import os

# ─── Paths ──────────────────────────────────────────────────

MODEL_DIR = os.environ.get(
    "DJTAGGER_MODEL_DIR",
    os.path.expanduser("~/.local/essentia-models"),
)

DEFAULT_MUSIC_PATH = os.environ.get("DJTAGGER_MUSIC_PATH", ".")

STATUS_FILE = "/tmp/dj-tagger-status.json"
LOG_FILE = "/tmp/dj-tagger.log"
ERROR_FILE = "/tmp/dj-tagger-errors.log"

# ─── API Keys ───────────────────────────────────────────────

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"

# ─── Tagger ─────────────────────────────────────────────────

TAGGER_VERSION = "v7"

# Genres considered generic / empty — will be replaced
GENERIC_GENRES = {"other", "unknown", "misc", "music", ""}

# Patterns that indicate a junk/spam genre tag (URLs, foreign spam, nonsense)
JUNK_GENRE_PATTERNS = [
    "http", "www.", ".com", ".ru", ".net",
    "vk.com", "twitter", "mp3impulse",
    "muzdo", "muzpark", "realtones",
    "getliftedtonight", "stopbreathebump",
    "lmp music", "prime music", "[ ra",
]

# Exact junk genres (case-insensitive)
JUNK_GENRES_EXACT = {
    "genre",
    "танцевальная",
    "танцевальная музыка",
    "танцевальная/электронная музыка",
    "другое",
}


def is_junk_genre(genre: str) -> bool:
    """Check if a genre string is junk (URLs, spam, foreign nonsense)."""
    if not genre:
        return False
    g = genre.lower().strip()
    if g in GENERIC_GENRES:
        return True
    if g in JUNK_GENRES_EXACT:
        return True
    if any(pat in g for pat in JUNK_GENRE_PATTERNS):
        return True
    return False


def is_junk_album(album: str) -> bool:
    """Check if an album string looks like URL/promo spam rather than a real release.

    Reuses the same URL/domain patterns used for junk-genre detection, and also
    flags unreasonably long strings (typically promo copy rather than titles).
    """
    if not album:
        return False
    a = album.lower().strip()
    if any(pat in a for pat in JUNK_GENRE_PATTERNS):
        return True
    if len(album) > 120:
        return True
    return False


def is_valid_year(year: str) -> bool:
    """Check if a year string is a plausible release year (1950–2030).

    Accepts the first 4-digit year found in the string, so values like
    "2019-04-12" or "2022 / Ultra Records" still pass.
    """
    if not year:
        return False
    import re
    m = re.search(r"(19|20)\d{2}", year)
    if not m:
        return False
    try:
        y = int(m.group(0))
    except ValueError:
        return False
    return 1950 <= y <= 2030

# ─── Camelot Key Mapping ──────────────────────────────────────

CAMELOT_MAP = {
    # Minor keys → A wheel
    "Abm": "1A",  "G#m": "1A",  "Abmin": "1A",  "G#min": "1A",
    "Ebm": "2A",  "D#m": "2A",  "Ebmin": "2A",  "D#min": "2A",
    "Bbm": "3A",  "A#m": "3A",  "Bbmin": "3A",  "A#min": "3A",
    "Fm":  "4A",  "Fmin": "4A",
    "Cm":  "5A",  "Cmin": "5A",
    "Gm":  "6A",  "Gmin": "6A",
    "Dm":  "7A",  "Dmin": "7A",
    "Am":  "8A",  "Amin": "8A",
    "Em":  "9A",  "Emin": "9A",
    "Bm":  "10A", "Bmin": "10A",
    "F#m": "11A", "Gbm": "11A", "F#min": "11A", "Gbmin": "11A",
    "C#m": "12A", "Dbm": "12A", "C#min": "12A", "Dbmin": "12A",
    # Major keys → B wheel
    "B":   "1B",  "Bmaj": "1B",
    "F#":  "2B",  "Gb":  "2B",  "F#maj": "2B",  "Gbmaj": "2B",
    "C#":  "3B",  "Db":  "3B",  "C#maj": "3B",  "Dbmaj": "3B",
    "Ab":  "4B",  "G#":  "4B",  "Abmaj": "4B",  "G#maj": "4B",
    "Eb":  "5B",  "D#":  "5B",  "Ebmaj": "5B",  "D#maj": "5B",
    "Bb":  "6B",  "A#":  "6B",  "Bbmaj": "6B",  "A#maj": "6B",
    "F":   "7B",  "Fmaj": "7B",
    "C":   "8B",  "Cmaj": "8B",
    "G":   "9B",  "Gmaj": "9B",
    "D":   "10B", "Dmaj": "10B",
    "A":   "11B", "Amaj": "11B",
    "E":   "12B", "Emaj": "12B",
}


def camelot_distance(key1: str, key2: str) -> int | None:
    """Compute distance between two Camelot keys (0 = same, 1 = compatible, etc.).

    Returns None if either key is not valid Camelot notation.
    Accounts for the circular wheel (12 wraps to 1) and A/B mode switches.
    """
    import re
    pattern = re.compile(r"^(\d{1,2})([AB])$")
    m1 = pattern.match(key1)
    m2 = pattern.match(key2)
    if not m1 or not m2:
        return None
    num1, mode1 = int(m1.group(1)), m1.group(2)
    num2, mode2 = int(m2.group(1)), m2.group(2)
    # Circular distance on the 1-12 wheel
    circle_dist = min(abs(num1 - num2), 12 - abs(num1 - num2))
    # Mode switch costs 1 step (A↔B at same number is compatible)
    mode_dist = 0 if mode1 == mode2 else 1
    return circle_dist + mode_dist


# ─── Network ────────────────────────────────────────────────

BEATPORT_TIMEOUT = 8   # seconds (curl -m)
LASTFM_TIMEOUT = 5     # seconds (curl -m)
SOCKET_TIMEOUT = 10    # global socket default

# ─── ML Thresholds ──────────────────────────────────────────

GENRE_MIN_PROB = 0.05   # minimum probability for ML genre
GENRE_KEEP_PROB = 0.10  # threshold for inclusion in final list
LASTFM_MIN_COUNT = 20   # minimum tag count for Last.fm genres

# ─── Energy Formula Weights ────────────────────────────────

ENERGY_W_DANCEABILITY = 0.05
ENERGY_W_AROUSAL = 0.05
ENERGY_W_AGGRESSIVE = 0.70
ENERGY_W_RELAXED = 0.20

# Scale raw energy to use more of the 0-1 range
ENERGY_SCALE = 1.8
ENERGY_OFFSET = 0.05

# ─── Segment Analysis ──────────────────────────────────────

SEGMENT_LENGTH_SEC = 30   # embedding frames per segment (~1 frame/sec)
SEGMENT_HOP_SEC = 15      # hop between segments

# ─── Set-Role Classification (v7) ──────────────────────────
# Raw-feature normalization ranges (lo, hi) mapped to 0..1.
# First four calibrated 2026-07-14 on a 1500-track sample (p10..p90);
# flux/vocal/outro_fade/rise/valence calibrated 2026-07-15 on a 150-track
# sample run through the full v7 pipeline (p10..p90 of each feature).
FEATURE_RANGE = {
    "spectral_centroid": (1422.0, 2342.0),   # Hz on 16 kHz audio (p10..p90)
    "onset_rate":        (2.49, 3.90),        # onsets per second (p10..p90)
    "dynamic_range":     (9.2, 18.9),         # dB; metric is p90-p10 of frame RMS. Range below is that metric's p10..p90 across library
    "sub_bass":          (0.51, 0.81),        # fraction of energy below 120 Hz (p10..p90)
    "flux":              (0.29, 0.38),        # mean normalized spectral flux
    "vocal":             (0.07, 0.73),        # voice_instrumental model, mean voice prob
    "outro_fade":        (3.3, 27.3),         # dB the outro sits below the peak (-outro_db)
    "rise":              (0.0, 4.0),          # positive loudness slope, dB per track
    "valence":           (0.538, 0.716),      # emomusic valence (p10..p90)
}

# Per-segment energy slope is multiplied by this, then clipped to [-1, 1].
# arc_momentum is kept as an informational tag; the v7 role decision uses
# the loudness-arc slope instead (a far stronger structural signal).
MOMENTUM_SCALE = 8.0

# v7 role decision: energy bands set Opener/Peak; the mid band splits into
# Builder vs Closer by comparing a "drive" index (flux + onset + rising
# loudness) against an "emo" index (valence + brightness + vocals + fading
# outro). Genre-relative banding: when a genre-energy stats file exists
# (see GENRE_STATS_FILE) and the track's genre cohort is large enough, the
# energy band comes from the track's percentile WITHIN its genre, so a
# melodic-house peak is not measured against the whole library's bangers.
ROLE_THRESHOLDS = {
    "peak_level":        0.80,   # arc_level at or above this is Peak (global band)
    "opener_level":      0.55,   # arc_level at or below this is Opener (global band)
    "peak_genre_pctl":   0.78,   # within-genre energy percentile for Peak
    "opener_genre_pctl": 0.30,   # within-genre energy percentile for Opener
    "genre_min_n":       30,     # min cohort size to trust genre-relative bands
    "drive_bias":        0.0,    # positive favors Closer, negative favors Builder
}

# Per-genre energy percentile table, produced by `djtagger genre-stats`.
GENRE_STATS_FILE = os.path.join(MODEL_DIR, "genre_energy.json")

ROLE_OPENER = "Opener"
ROLE_BUILDER = "Builder"
ROLE_PEAK = "Peak"
ROLE_CLOSER = "Closer"

# Role names written by older tagger versions, mapped to their current
# equivalents. Applied when reading tags and when fixing comments, so
# files tagged by v6 display the v7 vocabulary without a full re-analysis.
LEGACY_ROLES = {"Warm-up": ROLE_OPENER}
