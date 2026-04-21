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

TAGGER_VERSION = "v5"

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
