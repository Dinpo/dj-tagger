"""File discovery: recursive audio scan with resume support."""

import os

from .tagfile import TagFile, is_audio_file
from .tagger import is_already_tagged

# ─── Find audio files ───────────────────────────────────────


def find_audio_files(path: str) -> list[str]:
    """Find all supported audio files under *path* (recursively if directory).

    Supported extensions are `tagfile.AUDIO_EXTS`, matched case-insensitively
    (libraries contain `.MP3` / `.Mp3`). macOS AppleDouble files (`._name`)
    are skipped: they share the extension but hold no audio.
    """
    if os.path.isfile(path) and is_audio_file(path):
        return [path]

    found: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            if is_audio_file(f) and not f.startswith("._"):
                found.append(os.path.join(root, f))
    return found


def filter_untagged(mp3s: list[str]) -> tuple[list[str], int]:
    """Split a list of audio paths into untagged files and a skip count.

    Returns (files_to_process, skipped_count).
    """
    to_process: list[str] = []
    skipped = 0
    for f in mp3s:
        if is_already_tagged(f):
            skipped += 1
        else:
            to_process.append(f)
    return to_process, skipped


def filter_outdated(mp3s: list[str]) -> tuple[list[str], int]:
    """Split into files tagged by an older version (or untagged) and a skip count.

    Used by `tag --upgrade` to backfill fields added by newer tagger versions
    without re-processing files that are already current.
    Returns (files_to_process, skipped_count).
    """
    from .config import TAGGER_VERSION
    from .tagger import tagged_version

    to_process: list[str] = []
    skipped = 0
    for f in mp3s:
        if tagged_version(f) == TAGGER_VERSION:
            skipped += 1
        else:
            to_process.append(f)
    return to_process, skipped


def filter_by_source(mp3s: list[str], source: str) -> tuple[list[str], int]:
    """Keep only files whose existing GENRE_SOURCE tag matches *source*.

    Used by `tag --retag-source <source>` to surgically re-tag a subset
    (e.g. all ML-only tracks after a Beatport outage is fixed). Files
    with no GENRE_SOURCE, a different source, or unreadable tags are skipped.

    Returns (matching_files, skipped_count).
    """
    matching: list[str] = []
    skipped = 0
    for f in mp3s:
        try:
            existing = TagFile.open(f).get_custom("GENRE_SOURCE").strip()
        except Exception:
            existing = ""
        if existing and existing == source:
            matching.append(f)
        else:
            skipped += 1
    return matching, skipped
