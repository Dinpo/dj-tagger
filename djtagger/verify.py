"""Verify every track is *properly* tagged, not just marked as tagged.

`djtagger tag` skips a file as soon as it carries a GENRE_SOURCE frame, so a
"skipped (already tagged)" count does not prove a file is complete or current.
This reads every file and checks the full tag set. Run it after any large tag
run, or before claiming a library is done.

Run as `djtagger verify <folder>` (exit code 0 when every track passes, 1
when any track has issues).

Checks per track: current TAGGER_VERSION, a valid set role, PULSE_REG and
ENERGY present, GENRE_SOURCE present, a non-junk genre, no wide ML genre mix
(4+ genres from source "ml"), a Serato comment that leads with "Role:", and
a header that reports a real duration. Some exported FLACs carry a zero
sample count; the audio is fine, but after a tag write DJ software re-reads
the header and loads a 0-second track. Re-encode losslessly to fix it.
Files are found with the tagger's own scanner: every supported format
(MP3, FLAC, M4A, AIFF, WAV, Ogg), extensions matched case-insensitively.
"""

import os
from collections import Counter

import mutagen

from .config import TAGGER_VERSION, is_junk_genre
from .scanner import find_audio_files
from .tagger import read_tags

ROLES = {"Opener", "Builder", "Peak", "Closer"}


def track_issues(t: dict, path: str = "") -> list[str]:
    issues = []
    if path:
        try:
            audio = mutagen.File(path)
            if audio is not None and not audio.info.length:
                issues.append("header reports 0 length")
        except Exception:
            issues.append("unreadable audio header")
    if t["tagger_version"] != TAGGER_VERSION:
        issues.append(f"version={t['tagger_version'] or 'none'}")
    if t["set_role"] not in ROLES:
        issues.append("no/invalid role")
    if not str(t["pulse_reg"]).strip():
        issues.append("no PULSE_REG")
    if not t["energy"]:
        issues.append("no ENERGY")
    if not t["genre_source"]:
        issues.append("no GENRE_SOURCE")
    genre = t["genre"]
    if not genre:
        issues.append("no genre")
    elif is_junk_genre(genre):
        issues.append("junk genre")
    parts = [g.strip() for g in genre.split(";") if g.strip()]
    if t["genre_source"] == "ml" and len(parts) >= 4:
        issues.append("wide ml genre")
    if t["set_role"] and not t["comment"].startswith("Role:"):
        issues.append("comment missing role")
    return issues


def verify_library(root: str) -> tuple[int, int, Counter, dict[str, list[str]]]:
    """Check every audio file under *root*.

    Returns (total, ok, issue_counts, examples) where *examples* maps each
    issue to the relative paths that have it.
    """
    files = find_audio_files(root)
    problems: Counter = Counter()
    examples: dict[str, list[str]] = {}
    ok = 0
    for f in files:
        issues = track_issues(read_tags(f), f)
        if not issues:
            ok += 1
            continue
        for issue in issues:
            problems[issue] += 1
            examples.setdefault(issue, []).append(os.path.relpath(f, root))
    return len(files), ok, problems, examples
