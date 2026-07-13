import os

from mutagen.id3 import ID3, TXXX

from djtagger import tagger


def test_build_comment_role_first_and_no_dance():
    comment, detail = tagger._build_comment(
        energy=0.85, valence=0.6, set_role="Builder", danceability=0.9,
        peak_energy=0.92, arousal=0.5, aggressive=0.7, intro_energy=0.3,
        arc_level=0.85, arc_momentum=0.4,
    )
    assert comment.startswith("Role: Builder | ")
    assert "Dance:" not in comment
    # Danceability is retained in the hidden detail string.
    assert "D:0.9" in detail
    assert "Lvl:0.85" in detail
    assert "Mom:0.4" in detail


def test_build_comment_without_role_omits_role_prefix():
    comment, _ = tagger._build_comment(energy=0.5, valence=0.6)
    assert not comment.startswith("Role:")


def test_read_tags_picks_up_new_frames(tmp_path):
    p = str(tmp_path / "t.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=["v6"]))
    tags.add(TXXX(encoding=3, desc="SET_ROLE", text=["Peak"]))
    tags.add(TXXX(encoding=3, desc="ARC_LEVEL", text=["0.88"]))
    tags.add(TXXX(encoding=3, desc="ARC_MOMENTUM", text=["0.12"]))
    tags.add(TXXX(encoding=3, desc="SUB_BASS", text=["0.31"]))
    tags.save(p)

    info = tagger.read_tags(p)
    assert info["set_role"] == "Peak"
    assert info["arc_level"] == "0.88"
    assert info["arc_momentum"] == "0.12"
    assert info["sub_bass"] == "0.31"
