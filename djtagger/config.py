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
