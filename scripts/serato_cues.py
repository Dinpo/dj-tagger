"""Read Serato cue points (name + position) from an MP3's ID3 tags.

Serato stores cues in a base64-encoded `GEOB:Serato Markers2` frame. This
parses that frame and yields each cue's index, position, and label, which is
how the user's own ENTRY / EXIT / phrase markers (and Mixed In Key's
"Energy N" cues) can be read back as ground-truth data.

Usage:
    .venv/bin/python scripts/serato_cues.py "/path/to/track.mp3" [more.mp3 ...]

    # Find every track carrying ENTRY/EXIT markers:
    .venv/bin/python scripts/serato_cues.py --scan "/Volumes/.../_DJ Music"

Kept for the parked ENTRY/EXIT detection work; see
docs/superpowers/notes/2026-09-03-entry-exit-cue-detection.md
"""

import base64
import glob
import os
import struct
import sys

from mutagen.id3 import ID3


def markers2_payload(tags) -> bytes:
    """Decode the Serato Markers2 GEOB frame, or return b'' if absent."""
    for key in tags.keys():
        if key.startswith("GEOB:Serato Markers2"):
            data = tags[key].data
            b64 = data[2:].replace(b"\n", b"").replace(b"\r", b"")
            b64 += b"=" * (-len(b64) % 4)
            try:
                return base64.b64decode(b64)
            except Exception:
                return b""
    return b""


def parse_cues(payload: bytes) -> list[tuple[int, float, str]]:
    """Return [(index, seconds, name)] for the CUE entries in a payload."""
    cues: list[tuple[int, float, str]] = []
    i = 2 if payload[:2] == b"\x01\x01" else 0
    while i < len(payload) - 5:
        j = payload.find(b"\x00", i)
        if j < 0:
            break
        tag = payload[i:j].decode("latin1", "ignore")
        if not tag:
            break
        length = struct.unpack(">I", payload[j + 1:j + 5])[0]
        body = payload[j + 5:j + 5 + length]
        i = j + 5 + length
        # CUE body: \x00 idx(1) pos_ms(4 BE) \x00 color(3) \x00\x00 name...
        if tag == "CUE" and len(body) >= 13:
            idx = body[1]
            pos_ms = struct.unpack(">I", body[2:6])[0]
            name = body[12:].split(b"\x00")[0].decode("utf-8", "ignore")
            cues.append((idx, pos_ms / 1000.0, name))
    return cues


def read_cues(path: str) -> list[tuple[int, float, str]]:
    """Convenience: cues for one file ([] if unreadable or none)."""
    try:
        payload = markers2_payload(ID3(path))
    except Exception:
        return []
    return parse_cues(payload) if payload else []


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "--scan":
        root = args[1]
        wanted = ("ENTRY", "EXIT")
        found = 0
        for f in glob.glob(os.path.join(root, "**", "*.mp3"), recursive=True):
            cues = read_cues(f)
            marks = [(n, t) for _i, t, n in
                     [(i, t, n) for i, t, n in cues]
                     if any(w in n.strip().upper() for w in wanted)]
            if marks:
                found += 1
                print(f"\n{os.path.basename(f)}")
                for n, t in marks:
                    print(f"   {t:8.2f}s  {n}")
        print(f"\n{found} tracks with ENTRY/EXIT markers")
        return
    for f in args:
        print(f"\n{os.path.basename(f)}")
        for idx, secs, name in read_cues(f):
            print(f"   cue[{idx}] {secs:8.2f}s  '{name}'")


if __name__ == "__main__":
    main()
