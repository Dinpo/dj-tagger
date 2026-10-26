"""Verify every track is *properly* tagged, not just marked as tagged.

`djtagger tag` skips a file as soon as it carries a GENRE_SOURCE frame, so a
"skipped (already tagged)" count does not prove a file is complete or current.
This reads every file and checks the full tag set. Run it after any large tag
run, or before claiming a library is done.

Usage:
    .venv/bin/python scripts/verify_library.py "/Volumes/Multimedia/Music/_DJ Music"

Exit code 0 when every track passes, 1 when any track has issues.

Checks per track: current TAGGER_VERSION, a valid set role, PULSE_REG and
ENERGY present, GENRE_SOURCE present, a non-junk genre, no wide ML genre mix
(4+ genres from source "ml"), and a Serato comment that leads with "Role:".
Files are found with the tagger's own scanner, so `.MP3` / `.Mp3` count too.
"""

import os
import sys
from collections import Counter

from djtagger.config import TAGGER_VERSION, is_junk_genre
from djtagger.scanner import find_mp3s
from djtagger.tagger import read_tags

ROLES = {"Opener", "Builder", "Peak", "Closer"}


def track_issues(t: dict) -> list[str]:
    issues = []
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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    files = find_mp3s(root)
    problems: Counter = Counter()
    examples: dict[str, list[str]] = {}
    ok = 0
    for f in files:
        issues = track_issues(read_tags(f))
        if not issues:
            ok += 1
            continue
        for issue in issues:
            problems[issue] += 1
            examples.setdefault(issue, []).append(os.path.relpath(f, root))

    n = len(files)
    print(f"Total tracks: {n}")
    print(f"Fully OK:     {ok} ({100 * ok / max(n, 1):.1f}%)")
    print(f"With issues:  {n - ok}")
    for issue, count in problems.most_common():
        print(f"\n  {count:5d}  {issue}")
        for e in examples[issue][:5]:
            print(f"           {e[:80]}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
