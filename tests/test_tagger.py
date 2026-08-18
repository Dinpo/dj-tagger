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


def _full_result(**over):
    r = {
        "energy": 0.9, "valence": 0.6,
        "moods": {"happy": 0.5, "sad": 0.2, "aggressive": 0.7, "relaxed": 0.1},
        "danceability": 0.9, "arousal": 0.5,
        "peak_energy": 0.95, "intro_energy": 0.4, "energy_variance": 0.001,
        "set_role": "Opener", "arc_level": 0.9, "arc_momentum": 0.01,
        "spectral_centroid": 1800.0, "onset_rate": 3.2, "dynamic_range": 12.0,
        "sub_bass": 0.7, "flux": 0.33, "vocal": 0.4, "intro_db": -5.0,
        "outro_db": -10.0, "arc_slope": 0.5, "drop_db": 12.0, "peak_pos": 0.6,
        "drive": 0.5, "emo": 0.5, "arc_ok": True,
        "bpm": 0, "key": "",
    }
    r.update(over)
    return r


def test_read_tags_normalizes_legacy_warmup(tmp_path):
    p = str(tmp_path / "legacy.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=["v6"]))
    tags.add(TXXX(encoding=3, desc="SET_ROLE", text=["Warm-up"]))
    tags.save(p)
    assert tagger.read_tags(p)["set_role"] == "Opener"


def test_fix_comments_translates_and_persists_legacy_role(tmp_path):
    p = str(tmp_path / "legacy2.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=["v6"]))
    tags.add(TXXX(encoding=3, desc="ENERGY", text=["0.5"]))
    tags.add(TXXX(encoding=3, desc="VALENCE", text=["0.6"]))
    tags.add(TXXX(encoding=3, desc="SET_ROLE", text=["Warm-up"]))
    tags.save(p)

    assert tagger.fix_comments(p) == "fixed"
    reread = ID3(p)
    assert str(reread["TXXX:SET_ROLE"]) == "Opener"
    assert str(reread["COMM::eng"]).startswith("Role: Opener | ")


def test_write_tags_decides_role_from_written_genre(tmp_path):
    # arc_level 0.9 with an unknown genre (no cohort) -> global bands -> Peak,
    # overriding the provisional "Opener" the analyzer supplied.
    p = str(tmp_path / "role.mp3")
    ID3().save(p)
    ok, _ = tagger.write_tags(p, _full_result(), "ml", ["Zzz Unknown Genre"])
    assert ok
    assert tagger.read_tags(p)["set_role"] == "Peak"


def test_write_tags_arc_failure_writes_empty_role(tmp_path):
    p = str(tmp_path / "failed.mp3")
    ID3().save(p)
    ok, _ = tagger.write_tags(
        p, _full_result(set_role="", arc_ok=False), "ml", ["House"])
    assert ok
    info = tagger.read_tags(p)
    assert info["set_role"] == ""
    assert "Role:" not in info["comment"]


def test_rerole_changes_role_from_stored_tags(tmp_path):
    # Stored as Opener but arc_level 0.9 with unknown genre -> global bands -> Peak.
    p = str(tmp_path / "rr.mp3")
    tags = ID3()
    for desc, val in [("TAGGER_VERSION", "v7"), ("SET_ROLE", "Opener"),
                      ("ARC_LEVEL", "0.9"), ("DRIVE", "0.5"), ("EMO", "0.5"),
                      ("ENERGY", "0.9"), ("VALENCE", "0.6")]:
        tags.add(TXXX(encoding=3, desc=desc, text=[val]))
    tags.save(p)

    status, role = tagger.rerole_file(p)
    assert status == "reroled"
    assert role == "Peak"
    reread = tagger.read_tags(p)
    assert reread["set_role"] == "Peak"
    assert reread["comment"].startswith("Role: Peak | ")


def test_rerole_skips_non_v7_file(tmp_path):
    p = str(tmp_path / "old.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="TAGGER_VERSION", text=["v5"]))
    tags.add(TXXX(encoding=3, desc="ENERGY", text=["0.7"]))
    tags.save(p)
    assert tagger.rerole_file(p)[0] == "skipped"


def test_rerole_unchanged_when_role_stable(tmp_path):
    p = str(tmp_path / "stable.mp3")
    tags = ID3()
    for desc, val in [("SET_ROLE", "Peak"), ("ARC_LEVEL", "0.95"),
                      ("DRIVE", "0.5"), ("EMO", "0.5"), ("ENERGY", "0.95"),
                      ("VALENCE", "0.6")]:
        tags.add(TXXX(encoding=3, desc=desc, text=[val]))
    tags.save(p)
    assert tagger.rerole_file(p) == ("unchanged", "Peak")


def test_write_tags_keeps_existing_role_on_arc_failure(tmp_path):
    # A previously-tagged Peak; this run's arc analysis failed (arc_ok False,
    # empty role). The existing role must be preserved, not erased.
    p = str(tmp_path / "arcfail.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="SET_ROLE", text=["Peak"]))
    tags.save(p)
    ok, _ = tagger.write_tags(
        p, _full_result(set_role="", arc_ok=False), "ml", ["House"])
    assert ok
    info = tagger.read_tags(p)
    assert info["set_role"] == "Peak"
    assert "Role: Peak" in info["comment"]


def test_write_tags_preserves_genre_detected_when_no_new_genres(tmp_path):
    # --upgrade keeps the existing genre and passes genre_list=[]; the
    # GENRE_DETECTED audit trail must not be blanked.
    p = str(tmp_path / "gd.mp3")
    tags = ID3()
    tags.add(TXXX(encoding=3, desc="GENRE_DETECTED", text=["House; Techno"]))
    tags.save(p)
    ok, _ = tagger.write_tags(p, _full_result(), "beatport", [])
    assert ok
    assert tagger.read_tags(p)["genre_detected"] == "House; Techno"


def test_rerole_persists_legacy_name_when_role_matches(tmp_path):
    # v6 "Warm-up" that recomputes to its translation "Opener": the raw frame
    # must be rewritten to the current vocabulary, not left as "Warm-up".
    p = str(tmp_path / "legacy_rr.mp3")
    tags = ID3()
    for d, v in [("SET_ROLE", "Warm-up"), ("ARC_LEVEL", "0.45"),
                 ("DRIVE", "0.3"), ("EMO", "0.7"), ("ENERGY", "0.45"),
                 ("VALENCE", "0.6")]:
        tags.add(TXXX(encoding=3, desc=d, text=[v]))
    tags.save(p)
    status, role = tagger.rerole_file(p)
    assert role == "Opener"
    assert str(ID3(p)["TXXX:SET_ROLE"]) == "Opener"   # raw frame normalized


def test_merge_genres_same_source_refresh_replaces_not_merges():
    # Re-resolving an ml genre must replace the stale wide mix, not merge it.
    new, action = tagger._merge_genres(
        "ambient; house; dnb; Electro House",
        ["House", "Electro House", "Tropical House"],
        existing_source="ml", new_source="ml")
    assert new == "House; Electro House; Tropical House"
    assert "ambient" not in new.lower()


def test_merge_genres_cross_source_still_merges():
    # A different existing source (or user genre) is still merged, not clobbered.
    new, _ = tagger._merge_genres(
        "Techno", ["House"], existing_source="", new_source="ml")
    assert "techno" in new.lower() and "house" in new.lower()


def test_merge_genres_beatport_still_upgrades_over_ml():
    new, _ = tagger._merge_genres(
        "ambient; house", ["Melodic Techno"], existing_source="ml", new_source="beatport")
    assert new == "Melodic Techno"
