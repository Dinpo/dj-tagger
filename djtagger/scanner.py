"""File discovery — recursive MP3 scan with resume support."""

import os

from mutagen.id3 import ID3

from .tagger import is_already_tagged

# ─── Find MP3s ──────────────────────────────────────────────


def find_mp3s(path: str) -> list[str]:
    """Find all MP3 files under *path* (recursively if directory)."""
    if os.path.isfile(path) and path.lower().endswith(".mp3"):
        return [path]

    mp3s: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(".mp3"):
                mp3s.append(os.path.join(root, f))
    return mp3s


def filter_untagged(mp3s: list[str]) -> tuple[list[str], int]:
    """Split a list of MP3 paths into untagged files and a skip count.

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


def filter_by_source(mp3s: list[str], source: str) -> tuple[list[str], int]:
    """Keep only MP3s whose existing TXXX:GENRE_SOURCE matches *source*.

    Used by `tag --retag-source <source>` to surgically re-tag a subset
    (e.g. all ML-only tracks after a Beatport outage is fixed). Files
    with no GENRE_SOURCE, a different source, or unreadable tags are skipped.

    Returns (matching_files, skipped_count).
    """
    matching: list[str] = []
    skipped = 0
    for f in mp3s:
        try:
            tags = ID3(f)
        except Exception:
            skipped += 1
            continue
        found = False
        for frame in tags.getall("TXXX"):
            if frame.desc == "GENRE_SOURCE" and frame.text:
                if frame.text[0].strip() == source:
                    matching.append(f)
                    found = True
                break
        if not found:
            skipped += 1
    return matching, skipped
