"""DJ Tagger: DJ library tagger (genre, energy, moods, set role)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("djtagger")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"
