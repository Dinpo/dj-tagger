"""Shared library scanning — reads all tracks into structured records."""

import os
from typing import Callable

from .scanner import find_mp3s
from .tagger import parse_filename, read_tags


def scan_library(
    path: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Scan a directory and return a list of track records with all tag data.

    Each record is a dict with keys:
        path, folder, artist, artist_clean, title,
        genre, genre_source, genre_detected, tagger_version,
        energy, valence, mood_happy, mood_sad, mood_aggressive, mood_relaxed,
        comment, comment_detail, tagged (bool)

    *on_progress(current, total)* is called after each file if provided.
    """
    all_mp3s = find_mp3s(path)
    total = len(all_mp3s)
    tracks: list[dict] = []

    for i, mp3 in enumerate(all_mp3s, 1):
        artist, artist_clean, title = parse_filename(mp3)
        tags = read_tags(mp3)
        folder = os.path.basename(os.path.dirname(mp3))

        record = {
            "path": mp3,
            "folder": folder,
            "artist": artist,
            "artist_clean": artist_clean,
            "title": title,
            "tagged": bool(tags.get("tagger_version")),
            # Tag fields (strings, may be empty)
            "genre": tags.get("genre", ""),
            "genre_source": tags.get("genre_source", ""),
            "genre_detected": tags.get("genre_detected", ""),
            "tagger_version": tags.get("tagger_version", ""),
            "comment": tags.get("comment", ""),
            "comment_detail": tags.get("comment_detail", ""),
        }

        # String fields
        record["bpm"] = tags.get("bpm", "")
        record["key"] = tags.get("key", "")

        # Numeric fields — convert to float or None
        for key in ("energy", "valence", "mood_happy", "mood_sad",
                     "mood_aggressive", "mood_relaxed",
                     "danceability", "mood_party", "arousal",
                     "peak_energy", "intro_energy", "energy_variance"):
            raw = tags.get(key, "")
            record[key] = float(raw) if raw else None

        tracks.append(record)
        if on_progress:
            on_progress(i, total)

    return tracks
