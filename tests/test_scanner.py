from mutagen.id3 import ID3, TXXX

from djtagger.config import TAGGER_VERSION
from djtagger.scanner import filter_outdated


def _mk(tmp_path, name, version=None):
    p = str(tmp_path / name)
    tags = ID3()
    if version is not None:
        tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=[version]))
        tags.add(TXXX(encoding=3, desc="GENRE_SOURCE", text=["ml"]))
    tags.save(p)
    return p


def test_filter_outdated_splits_by_version(tmp_path):
    old = _mk(tmp_path, "old.mp3", "v6")
    cur = _mk(tmp_path, "cur.mp3", TAGGER_VERSION)
    raw = _mk(tmp_path, "raw.mp3", None)

    process, skipped = filter_outdated([old, cur, raw])
    assert old in process
    assert raw in process
    assert cur not in process
    assert skipped == 1
