"""Tag reading / writing for every supported format (via `tagfile.TagFile`)."""

import os
import re

from .config import (
    GENERIC_GENRES,
    LEGACY_ROLES,
    TAGGER_VERSION,
    is_junk_album,
    is_junk_genre,
    is_valid_year,
)
from .tagfile import TagFile

# ─── Read helpers ───────────────────────────────────────────


def is_already_tagged(filepath: str) -> bool:
    """Check if file already has our GENRE_SOURCE tag."""
    try:
        return bool(TagFile.open(filepath).get_custom("GENRE_SOURCE"))
    except Exception:
        return False


def tagged_version(filepath: str) -> str:
    """Return the file's TAGGER_VERSION tag value, or "" if absent/unreadable."""
    try:
        return TagFile.open(filepath).get_custom("TAGGER_VERSION").strip()
    except Exception:
        return ""


def parse_filename(filepath: str) -> tuple[str, str, str]:
    """Parse artist, clean-artist, and title from filename.

    Expects format: 'Artist - Title.<ext>'
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
        "pulse_reg": "",
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
        tags = TagFile.open(filepath)
    except Exception:
        return info

    # Standard fields (genre, BPM, key, album, year)
    for field in ("genre", "bpm", "key", "album", "year"):
        info[field] = tags.get(field)

    # Custom tags
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
        "PULSE_REG": "pulse_reg",
        "VOCAL": "vocal",
        "INTRO_DB": "intro_db",
        "OUTRO_DB": "outro_db",
        "ARC_SLOPE": "arc_slope",
        "DROP_DB": "drop_db",
        "PEAK_POS": "peak_pos",
        "DRIVE": "drive",
        "EMO": "emo",
    }
    for name, key in tag_map.items():
        info[key] = tags.get_custom(name)

    # Normalize role names written by older tagger versions (v6 "Warm-up").
    info["set_role"] = LEGACY_ROLES.get(info["set_role"], info["set_role"])

    # Comments
    info["comment"] = tags.get("comment")
    info["comment_detail"] = tags.get("detail")

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
            f"upgraded source {existing_source}->{new_source}: "
            f"'{existing}' -> '{proposed_str}'",
        )

    # Same-source refresh: a fresh resolution from the same djtagger source
    # supersedes our own prior output for it. Replace rather than merge, so a
    # re-tag (e.g. tag --retag-source ml after a resolver fix) does not merge
    # the stale wide genre back into the newly cleaned one.
    if existing_source and existing_source == new_source:
        if _genre_tokens(existing) == _genre_tokens(proposed_str):
            return existing, "matches"
        return proposed_str, f"refreshed {new_source}: '{existing}' -> '{proposed_str}'"

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
    """Write analysis results as tags (any supported format).

    Returns (success, genre_action_description).

    *album* and *year* are optional. When passed, they are written to the
    album / year fields respectively, but only if the existing frame is empty or looks like
    junk (URL/promo spam for album; non-year for year). Legitimate existing
    values are preserved.
    """
    try:
        tags = TagFile.open(filepath)

        # Genre: only overwrite if existing is generic/empty/junk
        existing_genre = tags.get("genre")

        # Existing GENRE_SOURCE — used by the merge logic to decide whether
        # this is a tier upgrade (weaker prior source → stronger new source).
        existing_source = tags.get_custom("GENRE_SOURCE").strip()

        if genre_list:
            new_genre, genre_action = _merge_genres(
                existing_genre,
                genre_list,
                existing_source=existing_source,
                new_source=genre_source,
            )
            if new_genre != existing_genre:
                tags.set("genre", new_genre)
        else:
            new_genre = existing_genre
            genre_action = "no genre"

        # Set role: decided HERE, once, for every write path (tag, fix-audit,
        # future callers), using the genre that actually ends up on the file
        # after the merge above. compute_arc's role is only a global-band
        # provisional; this applies genre-relative bands when stats exist.
        # Skipped when arc analysis failed (arc_ok False): an empty role is
        # written rather than one fabricated from neutral values.
        # Existing role/detected genre, used to avoid erasing good data on a
        # run where analysis did not (re)produce them.
        existing_role = tags.get_custom("SET_ROLE").strip()
        existing_detected = tags.get_custom("GENRE_DETECTED")

        if result.get("arc_ok", True) and result.get("set_role", "") != "":
            from . import classify
            try:
                result["set_role"] = classify.decide_role(result, new_genre)
            except Exception:
                # Stats lookup failed for this track; the analyzer's
                # global-band role is still a valid decision, keep it.
                pass
        elif not result.get("set_role", ""):
            # Arc analysis failed this run (arc_ok False): keep the previously
            # written role rather than erasing it with an empty string.
            result["set_role"] = LEGACY_ROLES.get(existing_role, existing_role)

        # GENRE_DETECTED: keep the existing value when this run resolved no
        # genres by design (e.g. --upgrade keeping the file's existing genre),
        # so the ML-detected audit trail is not blanked.
        detected_val = "; ".join(genre_list[:4]) if genre_list else existing_detected

        # Custom tags (only our namespaced keys)
        for key, val in [
            ("ENERGY", result["energy"]),
            ("VALENCE", result["valence"]),
            ("MOOD_HAPPY", result["moods"]["happy"]),
            ("MOOD_SAD", result["moods"]["sad"]),
            ("MOOD_AGGRESSIVE", result["moods"]["aggressive"]),
            ("MOOD_RELAXED", result["moods"]["relaxed"]),
            ("GENRE_SOURCE", genre_source),
            ("GENRE_DETECTED", detected_val),
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
            ("PULSE_REG", result.get("pulse_reg", 0.0)),
            ("VOCAL", result.get("vocal", 0.0)),
            ("INTRO_DB", result.get("intro_db", 0.0)),
            ("OUTRO_DB", result.get("outro_db", 0.0)),
            ("ARC_SLOPE", result.get("arc_slope", 0.0)),
            ("DROP_DB", result.get("drop_db", 0.0)),
            ("PEAK_POS", result.get("peak_pos", 0.5)),
            ("DRIVE", result.get("drive", 0.0)),
            ("EMO", result.get("emo", 0.0)),
        ]:
            tags.set_custom(key, str(val))

        # Remove retired tags from older versions
        tags.delete_custom("MOOD_PARTY")

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
        tags.set("comment", comment)
        tags.set("detail", detail)

        # BPM — only write if not already set
        if result.get("bpm") and not tags.get("bpm"):
            tags.set("bpm", str(result["bpm"]))

        # Key — only write if not already set
        if result.get("key") and not tags.get("key"):
            tags.set("key", str(result["key"]))

        # Album — fill if empty or replace if junk; never overwrite a legit value
        if album:
            existing_album = tags.get("album")
            if not existing_album or is_junk_album(existing_album):
                tags.set("album", album)

        # Year — fill if empty or replace if not a plausible year
        if year:
            existing_year = tags.get("year")
            if not existing_year or not is_valid_year(existing_year):
                tags.set("year", year)

        tags.save()
        return True, genre_action
    except Exception as ex:
        return False, f"error: {ex}"


def _custom_float(tags: TagFile, name: str, default=None):
    """Read a custom float tag, returning default if absent/blank/non-numeric."""
    raw = tags.get_custom(name).strip()
    if raw != "":
        try:
            return float(raw)
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
        tags = TagFile.open(filepath)
    except Exception:
        return "skipped", ""

    lvl = _custom_float(tags, "ARC_LEVEL")
    drive = _custom_float(tags, "DRIVE")
    emo = _custom_float(tags, "EMO")
    if lvl is None or drive is None or emo is None:
        return "skipped", ""  # not analyzed by v7, nothing to re-decide from

    genre = tags.get("genre")

    from . import classify
    try:
        new_role = classify.decide_role(
            {"arc_level": lvl, "drive": drive, "emo": emo,
             "sub_bass": _custom_float(tags, "SUB_BASS", 0.0),
             "drop_db": _custom_float(tags, "DROP_DB", 0.0)},
            genre,
        )
    except Exception:
        return "error", ""

    stored = tags.get_custom("SET_ROLE").strip()
    if new_role == stored:
        return "unchanged", new_role  # raw frame already correct, keep it cheap
    # Note: when only the label changed (a legacy v6 name like "Warm-up" whose
    # translation equals new_role), we still fall through and rewrite the frame
    # so the on-disk value matches the current vocabulary.

    try:
        tags.set_custom("SET_ROLE", new_role)
        comment, detail = _build_comment(
            energy=_custom_float(tags, "ENERGY", 0.0),
            valence=_custom_float(tags, "VALENCE", 0.0),
            set_role=new_role,
            danceability=_custom_float(tags, "DANCEABILITY", 0.0),
            peak_energy=_custom_float(tags, "PEAK_ENERGY", 0.0),
            arousal=_custom_float(tags, "AROUSAL", 0.0),
            aggressive=_custom_float(tags, "MOOD_AGGRESSIVE", 0.0),
            intro_energy=_custom_float(tags, "INTRO_ENERGY", 0.0),
            arc_level=lvl,
            arc_momentum=_custom_float(tags, "ARC_MOMENTUM", 0.0),
        )
        tags.set("comment", comment)
        tags.set("detail", detail)
        tags.save()
        return "reroled", new_role
    except Exception:
        return "error", ""


def fix_comments(filepath: str) -> str:
    """Re-write comments from existing energy/valence/danceability tags.

    Returns "fixed" on success, "skipped" if file lacks djtagger tags,
    or "error" on read/write failure.
    """
    try:
        tags = TagFile.open(filepath)
    except Exception:
        return "skipped"

    # Only fix files that have our tagger version tag
    if not tags.get_custom("TAGGER_VERSION"):
        return "skipped"
    e_raw = tags.get_custom("ENERGY")
    v_raw = tags.get_custom("VALENCE")
    if not e_raw or not v_raw:
        return "skipped"

    try:
        e = float(e_raw)
        v = float(v_raw)

        # v5+ fields default to 0.0 when absent (graceful for v4 files)
        stored_role = tags.get_custom("SET_ROLE")
        # Translate role names from older versions (v6 "Warm-up" -> "Opener")
        # and persist the rename so the frame matches the current vocabulary.
        role = LEGACY_ROLES.get(stored_role, stored_role)
        if role != stored_role:
            tags.set_custom("SET_ROLE", role)

        comment, detail = _build_comment(
            energy=e, valence=v, set_role=role,
            danceability=_custom_float(tags, "DANCEABILITY", 0.0),
            peak_energy=_custom_float(tags, "PEAK_ENERGY", 0.0),
            arousal=_custom_float(tags, "AROUSAL", 0.0),
            aggressive=_custom_float(tags, "MOOD_AGGRESSIVE", 0.0),
            intro_energy=_custom_float(tags, "INTRO_ENERGY", 0.0),
            arc_level=_custom_float(tags, "ARC_LEVEL", 0.0),
            arc_momentum=_custom_float(tags, "ARC_MOMENTUM", 0.0),
        )

        tags.set("comment", comment)
        tags.set("detail", detail)
        tags.save()
        return "fixed"
    except Exception:
        return "error"


def clean_junk_genre(filepath: str) -> tuple[bool, str]:
    """Remove junk genre tags, replacing with GENRE_DETECTED if available.

    Returns (changed, description). changed=True if genre was cleaned.
    """
    try:
        tags = TagFile.open(filepath)
    except Exception:
        return False, "no tags"

    existing = tags.get("genre")
    if not existing:
        return False, "no genre"
    if not is_junk_genre(existing):
        return False, "ok"

    # Try to use GENRE_DETECTED as replacement
    detected = tags.get_custom("GENRE_DETECTED").strip()

    if detected and not is_junk_genre(detected):
        tags.set("genre", detected)
        tags.save()
        return True, f"replaced '{existing}' → '{detected}'"
    else:
        # No good replacement — just clear the junk
        tags.delete("genre")
        tags.save()
        return True, f"cleared '{existing}'"
