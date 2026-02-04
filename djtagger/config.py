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
LASTFM_URL = "http://ws.audioscrobbler.com/2.0/"

# ─── Tagger ─────────────────────────────────────────────────

TAGGER_VERSION = "v4"

# Genres considered generic / empty — will be replaced
GENERIC_GENRES = {"other", "unknown", "misc", "music", ""}

# ─── Network ────────────────────────────────────────────────

BEATPORT_TIMEOUT = 8   # seconds (curl -m)
LASTFM_TIMEOUT = 5     # seconds (curl -m)
SOCKET_TIMEOUT = 10    # global socket default

# ─── ML Thresholds ──────────────────────────────────────────

GENRE_MIN_PROB = 0.05   # minimum probability for ML genre
GENRE_KEEP_PROB = 0.10  # threshold for inclusion in final list
LASTFM_MIN_COUNT = 20   # minimum tag count for Last.fm genres
