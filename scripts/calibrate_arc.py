"""Sample the library, run analysis, and print feature distributions.

Used once to turn the provisional FEATURE_RANGE / ROLE_THRESHOLDS in
config.py into values grounded in the real collection. Samples across
eras: the root folder holds the oldest tracks, so newer dated subfolders
must be represented too.

Usage:
    .venv/bin/python scripts/calibrate_arc.py "/Volumes/Multimedia/Music/_DJ Music" 300
"""

import os
import sys

import numpy as np

from djtagger.analyzer import load_models, analyze_track


def sample_paths(root: str, n: int) -> list[str]:
    """Pick up to n mp3s spread across top-level subfolders and the root."""
    buckets: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        top = "." if rel == "." else rel.split(os.sep)[0]
        for f in files:
            if f.lower().endswith(".mp3"):
                buckets.setdefault(top, []).append(os.path.join(dirpath, f))
    if not buckets:
        return []
    per = max(1, n // len(buckets))
    picked: list[str] = []
    for _top, paths in sorted(buckets.items()):
        step = max(1, len(paths) // per)
        picked.extend(paths[::step][:per])
    return picked[:n]


def main() -> None:
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    paths = sample_paths(root, n)
    print(f"Sampling {len(paths)} tracks from {root}", file=sys.stderr)

    models = load_models()
    keys = ["energy", "spectral_centroid", "onset_rate", "dynamic_range",
            "sub_bass", "arc_momentum"]
    collected: dict[str, list[float]] = {k: [] for k in keys}

    for i, p in enumerate(paths, 1):
        try:
            r = analyze_track(p, models, detect_bpm_key=False)
        except Exception as ex:
            print(f"  skip {os.path.basename(p)}: {ex}", file=sys.stderr)
            continue
        for k in keys:
            collected[k].append(float(r[k]))
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", file=sys.stderr)

    print("\nfeature                p5      p10     p25     p50     p75     p90     p95")
    for k in keys:
        vals = np.array(collected[k])
        if len(vals) == 0:
            continue
        ps = [np.percentile(vals, q) for q in (5, 10, 25, 50, 75, 90, 95)]
        print(f"{k:20s} " + " ".join(f"{v:7.3f}" for v in ps))


if __name__ == "__main__":
    main()
