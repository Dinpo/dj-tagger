"""Multi-format tag support: every format round-trips through the tagger API."""

import os
import shutil

import mutagen
import pytest
from mutagen.mp4 import MP4

from djtagger import tagger
from djtagger.scanner import filter_by_source, find_audio_files
from djtagger.tagfile import AUDIO_EXTS, TagFile

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
FORMATS = ["flac", "m4a", "aif", "wav", "ogg"]


def _copy(tmp_path, ext, name="Artist - Title"):
    p = str(tmp_path / f"{name}.{ext}")
    shutil.copy(os.path.join(FIXTURES, f"silent.{ext}"), p)
    return p


def _result(**over):
    r = {
        "energy": 0.9, "valence": 0.6,
        "moods": {"happy": 0.5, "sad": 0.2, "aggressive": 0.7, "relaxed": 0.1},
        "danceability": 0.9, "arousal": 0.5,
        "peak_energy": 0.95, "intro_energy": 0.4, "energy_variance": 0.001,
        "set_role": "Opener", "arc_level": 0.9, "arc_momentum": 0.01,
        "spectral_centroid": 1800.0, "onset_rate": 3.2, "dynamic_range": 12.0,
        "sub_bass": 0.7, "flux": 0.33, "pulse_reg": 0.2, "vocal": 0.4,
        "intro_db": -5.0, "outro_db": -10.0, "arc_slope": 0.5, "drop_db": 12.0,
        "peak_pos": 0.6, "drive": 0.5, "emo": 0.5, "arc_ok": True,
        "bpm": 0, "key": "",
    }
    r.update(over)
    return r


@pytest.mark.parametrize("ext", FORMATS)
def test_write_then_read_round_trip(tmp_path, ext):
    p = _copy(tmp_path, ext)
    assert not tagger.is_already_tagged(p)

    ok, _ = tagger.write_tags(p, _result(), "ml", ["Techno", "Peak Time"],
                              album="Some Album", year="2021")
    assert ok

    info = tagger.read_tags(p)
    assert info["genre"] == "Techno; Peak Time"
    assert info["set_role"] == "Peak"  # arc_level 0.9, unknown genre cohort
    assert info["energy"] == "0.9"
    assert info["pulse_reg"] == "0.2"
    assert info["genre_source"] == "ml"
    assert info["album"] == "Some Album"
    assert info["year"] == "2021"
    assert info["comment"].startswith("Role: Peak | ")
    assert info["comment_detail"].startswith("E:0.9 | ")
    assert tagger.is_already_tagged(p)
    assert tagger.tagged_version(p) == info["tagger_version"] != ""


@pytest.mark.parametrize("ext", FORMATS)
def test_rerole_and_fix_comments(tmp_path, ext):
    p = _copy(tmp_path, ext)
    tagger.write_tags(p, _result(), "ml", ["Techno"])
    t = TagFile.open(p)
    t.set_custom("SET_ROLE", "Warm-up")  # legacy v6 name
    t.save()

    assert tagger.fix_comments(p) == "fixed"
    assert tagger.read_tags(p)["set_role"] == "Opener"
    # Stored Opener but arc_level 0.9 decides Peak.
    assert tagger.rerole_file(p) == ("reroled", "Peak")
    assert tagger.read_tags(p)["comment"].startswith("Role: Peak | ")
    assert tagger.rerole_file(p) == ("unchanged", "Peak")


@pytest.mark.parametrize("ext", FORMATS)
def test_retag_source_filter(tmp_path, ext):
    ml = _copy(tmp_path, ext, "A - ml")
    bp = _copy(tmp_path, ext, "B - bp")
    raw = _copy(tmp_path, ext, "C - raw")
    tagger.write_tags(ml, _result(), "ml", ["House"])
    tagger.write_tags(bp, _result(), "beatport", ["House"])
    matching, skipped = filter_by_source([ml, bp, raw], "ml")
    assert matching == [ml]
    assert skipped == 2


def test_find_audio_files_covers_formats_case_insensitively(tmp_path):
    for name in ["a.mp3", "b.FLAC", "c.m4a", "d.aif", "e.aiff", "f.Wav", "g.ogg",
                 "h.mp4", "i.txt", "._j.flac"]:
        (tmp_path / name).write_bytes(b"")
    found = [os.path.basename(f) for f in find_audio_files(str(tmp_path))]
    assert found == ["a.mp3", "b.FLAC", "c.m4a", "d.aif", "e.aiff", "f.Wav", "g.ogg"]
    assert ".mp4" not in AUDIO_EXTS  # usually video in a music folder


def test_flac_custom_tags_do_not_clobber_mixed_in_key(tmp_path):
    # Mixed In Key stores base64 blobs under "energy" and "key" in FLAC, and
    # Vorbis field names are case-insensitive, so ours must be namespaced.
    p = _copy(tmp_path, "flac")
    f = mutagen.File(p)
    f.tags["energy"] = ["eyJlbmVyZ3lMZXZlbCI6N30="]
    f.tags["key"] = ["eyJrZXkiOiIxMEEifQ=="]
    f.tags["initialkey"] = ["10A"]
    f.save()

    tagger.write_tags(p, _result(), "ml", ["House"])

    f = mutagen.File(p)
    assert f.tags["energy"] == ["eyJlbmVyZ3lMZXZlbCI6N30="]
    assert f.tags["key"] == ["eyJrZXkiOiIxMEEifQ=="]
    assert f.tags["DJTAGGER_ENERGY"] == ["0.9"]
    assert tagger.read_tags(p)["key"] == "10A"  # INITIALKEY is the real key


def test_flac_description_mirror_follows_comment_only_when_mirrored(tmp_path):
    mirrored = _copy(tmp_path, "flac", "A - mirrored")
    f = mutagen.File(mirrored)
    f.tags["comment"] = ["10A - Energy 7"]
    f.tags["description"] = ["10A - Energy 7"]
    f.save()
    own = _copy(tmp_path, "flac", "B - own")
    f = mutagen.File(own)
    f.tags["comment"] = ["10A - Energy 7"]
    f.tags["description"] = ["my own notes"]
    f.save()

    for p in (mirrored, own):
        tagger.write_tags(p, _result(), "ml", ["House"])

    assert mutagen.File(mirrored).tags["description"][0].startswith("Role: Peak")
    assert mutagen.File(own).tags["description"] == ["my own notes"]


def test_m4a_uses_standard_atoms(tmp_path):
    p = _copy(tmp_path, "m4a")
    tagger.write_tags(p, _result(bpm=128.4, key="8A"), "ml", ["House"])

    m = MP4(p)
    assert m.tags["\xa9gen"] == ["House"]
    assert m.tags["\xa9cmt"][0].startswith("Role: Peak")
    assert m.tags["tmpo"] == [128]
    assert bytes(m.tags["----:com.apple.iTunes:initialkey"][0]) == b"8A"
    assert bytes(m.tags["----:com.apple.iTunes:DJTAGGER_SET_ROLE"][0]) == b"Peak"


def test_m4a_reads_key_case_insensitively(tmp_path):
    p = _copy(tmp_path, "m4a")
    t = TagFile.open(p)
    t._set_ff("INITIALKEY", "5A")
    t.save()
    assert tagger.read_tags(p)["key"] == "5A"


def test_m4a_existing_bpm_and_key_are_kept(tmp_path):
    p = _copy(tmp_path, "m4a")
    t = TagFile.open(p)
    t.set("bpm", "124")
    t.set("key", "3B")
    t.save()
    tagger.write_tags(p, _result(bpm=128.0, key="8A"), "ml", ["House"])
    info = tagger.read_tags(p)
    assert (info["bpm"], info["key"]) == ("124", "3B")


@pytest.mark.parametrize("ext", ["aif", "wav"])
def test_container_id3_keeps_v23(tmp_path, ext):
    # Serato writes ID3v2.3 into some AIFF/WAV files; do not silently upgrade.
    p = _copy(tmp_path, ext)
    f = mutagen.File(p)
    f.add_tags()
    f.save(v2_version=3)
    tagger.write_tags(p, _result(), "ml", ["House"])
    f = mutagen.File(p)
    assert f.tags.version[:2] == (2, 3)
    assert f.tags.get("TXXX:SET_ROLE").text == ["Peak"]


def test_unsupported_extension_is_rejected(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        TagFile.open(str(p))
    assert tagger.read_tags(str(p))["genre"] == ""
    assert tagger.write_tags(str(p), _result(), "ml", ["House"])[0] is False


def test_zero_length_header_is_refused_and_file_untouched(tmp_path):
    # FFmpeg-style export with total_samples=0: tagging it would make DJ
    # software re-read the header and load a 0-second track, so refuse.
    p = _copy(tmp_path, "flac")
    f = mutagen.File(p)
    f.info.total_samples = 0
    f.save()
    before = open(p, "rb").read()

    ok, msg = tagger.write_tags(p, _result(), "ml", ["House"])
    assert ok is False
    assert "0 length" in msg
    assert tagger.fix_comments(p) == "skipped"  # never tagged by us
    assert open(p, "rb").read() == before
