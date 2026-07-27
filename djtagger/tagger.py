"""ID3 tag reading / writing with mutagen."""

import os
import re

from mutagen.id3 import ID3, TXXX, TCON, TBPM, TKEY, COMM, TALB, TDRC, ID3NoHeaderError

from .config import (
    GENERIC_GENRES,
    LEGACY_ROLES,
    TAGGER_VERSION,
    is_junk_album,
    is_junk_genre,
    is_valid_year,
)

# ─── Read helpers ───────────────────────────────────────────


def is_already_tagged(filepath: str) -> bool:
    """Check if file already has our GENRE_SOURCE tag."""
    try:
        tags = ID3(filepath)
        for frame in tags.getall("TXXX"):
            if frame.desc == "GENRE_SOURCE" and frame.text:
                return True
    except Exception:
        pass
    return False


def tagged_version(filepath: str) -> str:
    """Return the file's TAGGER_VERSION tag value, or "" if absent/unreadable."""
    try:
        tags = ID3(filepath)
    except Exception:
        return ""
    for frame in tags.getall("TXXX"):
        if frame.desc == "TAGGER_VERSION" and frame.text:
            return str(frame.text[0]).strip()
    return ""


def parse_filename(filepath: str) -> tuple[str, str, str]:
    """Parse artist, clean-artist, and title from filename.

    Expects format: 'Artist - Title.mp3'
    Returns (artist, artist_clean, title).
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in basename:
        parts = basename.split(" - ", 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        artist, title = "", basename.strip()
    artist_clean = re.sub(r"\s*\([A-Z]{2}\)\s*$", "", artist).strip()
    return artist, artist_clean, title


def read_tags(filepath: str) -> dict:
    """Read all DJ-tagger related tags from a file.

    Returns dict with tag values (or empty strings for missing tags).
    """
    info: dict = {
        "genre": "",
        "energy": "",
        "valence": "",
        "mood_happy": "",
        "mood_sad": "",
        "mood_aggressive": "",
        "mood_relaxed": "",
        "genre_source": "",
        "genre_detected": "",
        "tagger_version": "",
        "comment": "",
        "comment_detail": "",
        "bpm": "",
        "key": "",
        "album": "",
        "year": "",
        # v5 fields
        "danceability": "",
        "arousal": "",
        "peak_energy": "",
        "intro_energy": "",
        "energy_variance": "",
        # v6 fields
        "set_role": "",
        "arc_level": "",
        "arc_momentum": "",
        "spectral_centroid": "",
        "onset_rate": "",
        "dynamic_range": "",
        "sub_bass": "",
        # v7 fields
        "flux": "",
        "vocal": "",
        "intro_db": "",
        "outro_db": "",
        "arc_slope": "",
        "drop_db": "",
        "peak_pos": "",
        "drive": "",
        "emo": "",
    }
    try:
        tags = ID3(filepath)
    except Exception:
        return info

    # TCON genre
    tcon = tags.getall("TCON")
    if tcon and tcon[0].text:
        info["genre"] = tcon[0].text[0].strip()

    # TBPM / TKEY (standard frames)
    tbpm = tags.getall("TBPM")
    if tbpm and tbpm[0].text:
        info["bpm"] = tbpm[0].text[0].strip()
    tkey = tags.getall("TKEY")
    if tkey and tkey[0].text:
        info["key"] = tkey[0].text[0].strip()
    talb = tags.getall("TALB")
    if talb and talb[0].text:
        info["album"] = str(talb[0].text[0]).strip()
    tdrc = tags.getall("TDRC")
    if tdrc and tdrc[0].text:
        info["year"] = str(tdrc[0].text[0]).strip()

    # TXXX custom tags
    tag_map = {
        "ENERGY": "energy",
        "VALENCE": "valence",
        "MOOD_HAPPY": "mood_happy",
        "MOOD_SAD": "mood_sad",
        "MOOD_AGGRESSIVE": "mood_aggressive",
        "MOOD_RELAXED": "mood_relaxed",
        "GENRE_SOURCE": "genre_source",
        "GENRE_DETECTED": "genre_detected",
        "TAGGER_VERSION": "tagger_version",
        # v5 tags
        "DANCEABILITY": "danceability",
        "AROUSAL": "arousal",
        "PEAK_ENERGY": "peak_energy",
        "INTRO_ENERGY": "intro_energy",
        "ENERGY_VARIANCE": "energy_variance",
        # v6 tags
        "SET_ROLE": "set_role",
        "ARC_LEVEL": "arc_level",
        "ARC_MOMENTUM": "arc_momentum",
        "SPECTRAL_CENTROID": "spectral_centroid",
        "ONSET_RATE": "onset_rate",
        "DYNAMIC_RANGE": "dynamic_range",
        "SUB_BASS": "sub_bass",
        # v7 tags
        "FLUX": "flux",
        "VOCAL": "vocal",
        "INTRO_DB": "intro_db",
        "OUTRO_DB": "outro_db",
        "ARC_SLOPE": "arc_slope",
        "DROP_DB": "drop_db",
        "PEAK_POS": "peak_pos",
        "DRIVE": "drive",
        "EMO": "emo",
    }
    for frame in tags.getall("TXXX"):
        if frame.desc in tag_map:
            info[tag_map[frame.desc]] = frame.text[0] if frame.text else ""

    # Normalize role names written by older tagger versions (v6 "Warm-up").
    info["set_role"] = LEGACY_ROLES.get(info["set_role"], info["set_role"])

    # Comments
    comm = tags.get("COMM::eng")
    if comm:
        info["comment"] = str(comm)
    comm_dj = tags.get("COMM:djtagger:eng")
    if comm_dj:
        info["comment_detail"] = str(comm_dj)

    return info


# ─── Write tags ─────────────────────────────────────────────


def _genre_tokens(s: str) -> set[str]:
    """Tokenize a genre string, normalizing separators/punctuation/casing.

    "Dance / Pop", "Dance; Pop", and "Dance & Pop" all produce {"dance", "pop"}.
    The word "and" is dropped so "Drum and Bass" ≡ "Drum & Bass".
    """
    parts = re.split(r"[\s;/&,\-]+", s.lower())
    return {p.strip("()[]{}") for p in parts if p and p != "and"}


_SOURCE_TIER = {"ml": 1, "lastfm+ml": 2, "beatport": 3}


def _merge_genres(
    existing: str,
    proposed: list[str],
    max_total: int = 5,
    existing_source: str = "",
    new_source: str = "",
) -> tuple[str, str]:
    """Decide the TCON value to write based on existing vs proposed genres.

    Rules (checked top to bottom):
        - proposed empty → keep existing
        - existing empty or junk → replace with proposed
        - existing was set by djtagger from a *weaker* source than the new one
          (ml → lastfm+ml / beatport, or lastfm+ml → beatport) → replace
          (don't merge old ML guesses into authoritative new data)
        - token-identical → keep existing (preserve original formatting)
        - existing ⊂ proposed (strictly more specific) → upgrade to proposed
        - proposed ⊂ existing → keep (don't downgrade specificity)
        - disjoint or partial overlap → merge deduped, capped at max_total

    Returns (new_tcon, action_label). If new_tcon == existing, caller should
    skip the write. action_label describes what happened for the log.
    """
    if not proposed:
        return existing, "no-proposed"

    proposed_str = "; ".join(proposed[:4])

    if not existing:
        return proposed_str, "filled"

    if is_junk_genre(existing):
        return proposed_str, f"replaced junk '{existing}'"

    # Tier-upgrade replace: if a previous djtagger run wrote this tag from a
    # weaker source than what we have now, drop the old guess instead of
    # merging stale ML tokens into authoritative Beatport/Last.fm data.
    e_tier = _SOURCE_TIER.get(existing_source, 0)
    n_tier = _SOURCE_TIER.get(new_source, 0)
    if existing_source and n_tier > e_tier:
        return (
            proposed_str,
            f"upgraded source {existing_source}→{new_source}: "
            f"'{existing}' → '{proposed_str}'",
        )

    e_tokens = _genre_tokens(existing)
    p_tokens = _genre_tokens(proposed_str)

    if e_tokens == p_tokens:
        return existing, "matches"
    if e_tokens < p_tokens:
        return proposed_str, f"upgraded '{existing}' → '{proposed_str}'"
    if p_tokens < e_tokens:
        return existing, f"kept more-specific '{existing}'"

    # Disjoint or partial overlap — merge, deduped at token level
    existing_items = [g.strip() for g in re.split(r";", existing) if g.strip()]
    merged = existing_items[:]
    for p in proposed:
        pt = _genre_tokens(p)
        if any(_genre_tokens(m) == pt for m in merged):
            continue
        merged.append(p)
    merged_str = "; ".join(merged[:max_total])
    if merged_str == existing:
        return existing, "matches"
    return merged_str, f"merged '{existing}' + '{proposed_str}'"


def _build_comment(
    energy: float,
    valence: float,
    set_role: str = "",
    danceability: float = 0.0,
    peak_energy: float = 0.0,
    arousal: float = 0.0,
    aggressive: float = 0.0,
    intro_energy: float = 0.0,
    arc_level: float = 0.0,
    arc_momentum: float = 0.0,
) -> tuple[str, str]:
    """Build human-readable comment and detail string."""
    e_lbl = "Low" if energy < 0.4 else "Mid" if energy < 0.7 else "High"
    # Valence thresholds tuned for electronic/dance music distribution
    # (median ~0.63, range ~0.45-0.80); the 0-1 model range is rarely exercised.
    v_lbl = "Dark" if valence < 0.58 else "Neutral" if valence < 0.68 else "Bright"
    agg_lbl = "Soft" if aggressive < 0.25 else "Mid" if aggressive < 0.5 else "Hard"
    intro_lbl = "Quiet" if intro_energy < 0.5 else "Mid" if intro_energy < 0.75 else "Hot"

    # Role goes first so it is the glanceable thing in Serato. Danceability is
    # dropped from the visible comment (near-constant across this library) but
    # kept in the hidden detail below.
    role_part = f"Role: {set_role} | " if set_role else ""
    comment = (
        f"{role_part}Energy: {e_lbl} | Mood: {v_lbl} | Edge: {agg_lbl} | "
        f"Peak: {peak_energy:.2f} | Intro: {intro_lbl}"
    )

    detail = (
        f"E:{energy} | V:{valence} | Agg:{aggressive} | "
        f"Peak:{peak_energy} | Intro:{intro_energy} | D:{danceability} | "
        f"Arousal:{arousal} | Lvl:{arc_level} | Mom:{arc_momentum}"
    )

    return comment, detail


def write_tags(
    filepath: str,
    result: dict,
    genre_source: str,
    genre_list: list[str],
    album: str = "",
    year: str = "",
) -> tuple[bool, str]:
    """Write analysis results as ID3 tags.

    Returns (success, genre_action_description).

    *album* and *year* are optional — when passed, they are written to TALB /
    TDRC respectively, but only if the existing frame is empty or looks like
    junk (URL/promo spam for album; non-year for year). Legitimate existing
    values are preserved.
    """
    try:
        try:
            tags = ID3(filepath)
        except ID3NoHeaderError:
            tags = ID3()

        # Genre: only overwrite if existing is generic/empty/junk
        existing_genre = ""
        tcon = tags.getall("TCON")
        if tcon and tcon[0].text:
            existing_genre = tcon[0].text[0].strip()

        # Existing GENRE_SOURCE — used by the merge logic to decide whether
        # this is a tier upgrade (weaker prior source → stronger new source).
        existing_source = ""
        for frame in tags.getall("TXXX"):
            if frame.desc == "GENRE_SOURCE" and frame.text:
                existing_source = frame.text[0].strip()
                break

        if genre_list:
            new_genre, genre_action = _merge_genres(
                existing_genre,
                genre_list,
                existing_source=existing_source,
                new_source=genre_source,
            )
            if new_genre != existing_genre:
                tags.delall("TCON")
                tags.add(TCON(encoding=3, text=[new_genre]))
        else:
            new_genre = existing_genre
            genre_action = "no genre"

        # Set role: decided HERE, once, for every write path (tag, fix-audit,
        # future callers), using the genre that actually ends up on the file
        # after the merge above. compute_arc's role is only a global-band
        # provisional; this applies genre-relative bands when stats exist.
        # Skipped when arc analysis failed (arc_ok False): an empty role is
        # written rather than one fabricated from neutral values.
        if result.get("arc_ok", True) and result.get("set_role", "") != "":
            from . import classify
            try:
                result["set_role"] = classify.decide_role(result, new_genre)
            except Exception:
                # Stats lookup failed for this track; the analyzer's
                # global-band role is still a valid decision, keep it.
                pass

        # TXXX custom tags (only our namespaced keys)
        for key, val in [
            ("ENERGY", result["energy"]),
            ("VALENCE", result["valence"]),
            ("MOOD_HAPPY", result["moods"]["happy"]),
            ("MOOD_SAD", result["moods"]["sad"]),
            ("MOOD_AGGRESSIVE", result["moods"]["aggressive"]),
            ("MOOD_RELAXED", result["moods"]["relaxed"]),
            ("GENRE_SOURCE", genre_source),
            ("GENRE_DETECTED", "; ".join(genre_list[:4])),
            ("TAGGER_VERSION", TAGGER_VERSION),
            # v5 tags
            ("DANCEABILITY", result["danceability"]),
            ("AROUSAL", result["arousal"]),
            ("PEAK_ENERGY", result["peak_energy"]),
            ("INTRO_ENERGY", result["intro_energy"]),
            ("ENERGY_VARIANCE", result["energy_variance"]),
            # v6 tags
            ("SET_ROLE", result["set_role"]),
            ("ARC_LEVEL", result["arc_level"]),
            ("ARC_MOMENTUM", result["arc_momentum"]),
            ("SPECTRAL_CENTROID", result["spectral_centroid"]),
            ("ONSET_RATE", result["onset_rate"]),
            ("DYNAMIC_RANGE", result["dynamic_range"]),
            ("SUB_BASS", result["sub_bass"]),
            # v7 tags
            ("FLUX", result.get("flux", 0.0)),
            ("VOCAL", result.get("vocal", 0.0)),
            ("INTRO_DB", result.get("intro_db", 0.0)),
            ("OUTRO_DB", result.get("outro_db", 0.0)),
            ("ARC_SLOPE", result.get("arc_slope", 0.0)),
            ("DROP_DB", result.get("drop_db", 0.0)),
            ("PEAK_POS", result.get("peak_pos", 0.5)),
            ("DRIVE", result.get("drive", 0.0)),
            ("EMO", result.get("emo", 0.0)),
        ]:
            tags.delall(f"TXXX:{key}")
            tags.add(TXXX(encoding=3, desc=key, text=[str(val)]))

        # Remove retired tags from older versions
        tags.delall("TXXX:MOOD_PARTY")

        # Comments
        comment, detail = _build_comment(
            energy=result["energy"],
            valence=result["valence"],
            set_role=result.get("set_role", ""),
            danceability=result.get("danceability", 0.0),
            peak_energy=result.get("peak_energy", 0.0),
            arousal=result.get("arousal", 0.0),
            aggressive=result["moods"]["aggressive"],
            intro_energy=result.get("intro_energy", 0.0),
            arc_level=result.get("arc_level", 0.0),
            arc_momentum=result.get("arc_momentum", 0.0),
        )
        tags.delall("COMM::eng")
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.delall("COMM:djtagger:eng")
        tags.add(COMM(encoding=3, lang="eng", desc="djtagger", text=detail))

        # BPM — only write if not already set
        if result.get("bpm"):
            existing_bpm = tags.getall("TBPM")
            if not existing_bpm or not existing_bpm[0].text or not existing_bpm[0].text[0].strip():
                tags.delall("TBPM")
                tags.add(TBPM(encoding=3, text=[str(result["bpm"])]))

        # Key — only write if not already set
        if result.get("key"):
            existing_key = tags.getall("TKEY")
            if not existing_key or not existing_key[0].text or not existing_key[0].text[0].strip():
                tags.delall("TKEY")
                tags.add(TKEY(encoding=3, text=[str(result["key"])]))

        # Album — fill if empty or replace if junk; never overwrite a legit value
        if album:
            existing_album = ""
            talb = tags.getall("TALB")
            if talb and talb[0].text:
                existing_album = str(talb[0].text[0]).strip()
            if not existing_album or is_junk_album(existing_album):
                tags.delall("TALB")
                tags.add(TALB(encoding=3, text=[album]))

        # Year — fill if empty or replace if not a plausible year
        if year:
            existing_year = ""
            tdrc = tags.getall("TDRC")
            if tdrc and tdrc[0].text:
                existing_year = str(tdrc[0].text[0]).strip()
            if not existing_year or not is_valid_year(existing_year):
                tags.delall("TDRC")
                tags.add(TDRC(encoding=3, text=[year]))

        tags.save(filepath)
        return True, genre_action
    except Exception as ex:
        return False, f"error: {ex}"


def _txxx_float(tags, desc, default=None):
    """Read a TXXX float value, returning default if absent/blank/non-numeric."""
    fr = tags.get(f"TXXX:{desc}")
    if fr and fr.text and str(fr.text[0]).strip() != "":
        try:
            return float(fr.text[0])
        except ValueError:
            return default
    return default


def rerole_file(filepath: str) -> tuple[str, str]:
    """Re-decide SET_ROLE from already-stored tags, no audio or ML.

    Applies the current thresholds and genre-relative bands to a track that
    was already analyzed by v7 (has ARC_LEVEL / DRIVE / EMO), rewriting the
    SET_ROLE frame and the comment only when the role actually changes. Lets
    role tuning iterate in seconds without re-running the analysis pipeline.

    Returns (status, new_role): status is "reroled" (role changed and was
    written), "unchanged", "skipped" (not a v7-analyzed file), or "error".
    """
    try:
        tags = ID3(filepath)
    except Exception:
        return "skipped", ""

    lvl = _txxx_float(tags, "ARC_LEVEL")
    drive = _txxx_float(tags, "DRIVE")
    emo = _txxx_float(tags, "EMO")
    if lvl is None or drive is None or emo is None:
        return "skipped", ""  # not analyzed by v7, nothing to re-decide from

    genre = ""
    tcon = tags.getall("TCON")
    if tcon and tcon[0].text:
        genre = str(tcon[0].text[0]).strip()

    from . import classify
    try:
        new_role = classify.decide_role(
            {"arc_level": lvl, "drive": drive, "emo": emo,
             "sub_bass": _txxx_float(tags, "SUB_BASS", 0.0),
             "drop_db": _txxx_float(tags, "DROP_DB", 0.0)},
            genre,
        )
    except Exception:
        return "error", ""

    role_fr = tags.get("TXXX:SET_ROLE")
    stored = str(role_fr.text[0]).strip() if role_fr and role_fr.text else ""
    old_role = LEGACY_ROLES.get(stored, stored)
    if new_role == old_role:
        return "unchanged", new_role  # no write, keep it cheap

    try:
        tags.delall("TXXX:SET_ROLE")
        tags.add(TXXX(encoding=3, desc="SET_ROLE", text=[new_role]))
        comment, detail = _build_comment(
            energy=_txxx_float(tags, "ENERGY", 0.0),
            valence=_txxx_float(tags, "VALENCE", 0.0),
            set_role=new_role,
            danceability=_txxx_float(tags, "DANCEABILITY", 0.0),
            peak_energy=_txxx_float(tags, "PEAK_ENERGY", 0.0),
            arousal=_txxx_float(tags, "AROUSAL", 0.0),
            aggressive=_txxx_float(tags, "MOOD_AGGRESSIVE", 0.0),
            intro_energy=_txxx_float(tags, "INTRO_ENERGY", 0.0),
            arc_level=lvl,
            arc_momentum=_txxx_float(tags, "ARC_MOMENTUM", 0.0),
        )
        tags.delall("COMM::eng")
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.delall("COMM:djtagger:eng")
        tags.add(COMM(encoding=3, lang="eng", desc="djtagger", text=detail))
        tags.save(filepath)
        return "reroled", new_role
    except Exception:
        return "error", ""


def fix_comments(filepath: str) -> str:
    """Re-write comments from existing TXXX energy/valence/danceability tags.

    Returns "fixed" on success, "skipped" if file lacks djtagger tags,
    or "error" on read/write failure.
    """
    try:
        tags = ID3(filepath)
    except Exception:
        return "skipped"

    # Only fix files that have our tagger version tag
    tv = tags.get("TXXX:TAGGER_VERSION")
    if not tv:
        return "skipped"
    e_tag = tags.get("TXXX:ENERGY")
    v_tag = tags.get("TXXX:VALENCE")
    if not e_tag or not e_tag.text or not v_tag or not v_tag.text:
        return "skipped"

    try:
        e = float(e_tag.text[0])
        v = float(v_tag.text[0])

        # Read v5 fields if present (graceful for v4 files)
        d_tag = tags.get("TXXX:DANCEABILITY")
        d = float(d_tag.text[0]) if d_tag and d_tag.text else 0.0
        p_tag = tags.get("TXXX:PEAK_ENERGY")
        p = float(p_tag.text[0]) if p_tag and p_tag.text else 0.0
        a_tag = tags.get("TXXX:AROUSAL")
        a = float(a_tag.text[0]) if a_tag and a_tag.text else 0.0
        agg_tag = tags.get("TXXX:MOOD_AGGRESSIVE")
        agg = float(agg_tag.text[0]) if agg_tag and agg_tag.text else 0.0
        intro_tag = tags.get("TXXX:INTRO_ENERGY")
        intro = float(intro_tag.text[0]) if intro_tag and intro_tag.text else 0.0

        role_tag = tags.get("TXXX:SET_ROLE")
        stored_role = role_tag.text[0] if role_tag and role_tag.text else ""
        # Translate role names from older versions (v6 "Warm-up" -> "Opener")
        # and persist the rename so the frame matches the current vocabulary.
        role = LEGACY_ROLES.get(stored_role, stored_role)
        if role != stored_role:
            tags.delall("TXXX:SET_ROLE")
            tags.add(TXXX(encoding=3, desc="SET_ROLE", text=[role]))
        lvl_tag = tags.get("TXXX:ARC_LEVEL")
        lvl = float(lvl_tag.text[0]) if lvl_tag and lvl_tag.text else 0.0
        mom_tag = tags.get("TXXX:ARC_MOMENTUM")
        mom = float(mom_tag.text[0]) if mom_tag and mom_tag.text else 0.0

        comment, detail = _build_comment(
            energy=e, valence=v, set_role=role, danceability=d, peak_energy=p,
            arousal=a, aggressive=agg, intro_energy=intro, arc_level=lvl,
            arc_momentum=mom,
        )

        tags.delall("COMM::eng")
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.delall("COMM:djtagger:eng")
        tags.add(COMM(encoding=3, lang="eng", desc="djtagger", text=detail))
        tags.save(filepath)
        return "fixed"
    except Exception:
        return "error"


def clean_junk_genre(filepath: str) -> tuple[bool, str]:
    """Remove junk genre tags, replacing with GENRE_DETECTED if available.

    Returns (changed, description). changed=True if genre was cleaned.
    """
    try:
        tags = ID3(filepath)
    except Exception:
        return False, "no tags"

    tcon = tags.getall("TCON")
    if not tcon or not tcon[0].text:
        return False, "no genre"

    existing = tcon[0].text[0].strip()
    if not is_junk_genre(existing):
        return False, "ok"

    # Try to use GENRE_DETECTED as replacement
    detected = ""
    for frame in tags.getall("TXXX"):
        if frame.desc == "GENRE_DETECTED" and frame.text:
            detected = frame.text[0].strip()
            break

    if detected and not is_junk_genre(detected):
        tags.delall("TCON")
        tags.add(TCON(encoding=3, text=[detected]))
        tags.save(filepath)
        return True, f"replaced '{existing}' → '{detected}'"
    else:
        # No good replacement — just clear the junk
        tags.delall("TCON")
        tags.save(filepath)
        return True, f"cleared '{existing}'"
