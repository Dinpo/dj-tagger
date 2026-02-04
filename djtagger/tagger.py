"""ID3 tag reading / writing with mutagen."""

import os
import re

from mutagen.id3 import ID3, TXXX, TCON, COMM, ID3NoHeaderError

from .config import GENERIC_GENRES, TAGGER_VERSION

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
    }
    try:
        tags = ID3(filepath)
    except Exception:
        return info

    # TCON genre
    tcon = tags.getall("TCON")
    if tcon and tcon[0].text:
        info["genre"] = tcon[0].text[0].strip()

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
    }
    for frame in tags.getall("TXXX"):
        if frame.desc in tag_map:
            info[tag_map[frame.desc]] = frame.text[0] if frame.text else ""

    # Comments
    comm = tags.get("COMM::eng")
    if comm:
        info["comment"] = str(comm)
    comm_dj = tags.get("COMM:djtagger:eng")
    if comm_dj:
        info["comment_detail"] = str(comm_dj)

    return info


# ─── Write tags ─────────────────────────────────────────────


def _build_comment(energy: float, valence: float) -> tuple[str, str]:
    """Build human-readable comment and detail string."""
    e_lbl = "Low" if energy < 0.4 else "Mid" if energy < 0.7 else "High"
    v_lbl = "Dark" if valence < 0.33 else "Neutral" if valence < 0.66 else "Bright"
    comment = f"Energy: {e_lbl} | Mood: {v_lbl}"
    detail = f"Energy:{e_lbl} Mood:{v_lbl} | E:{energy} V:{valence}"
    return comment, detail


def write_tags(
    filepath: str,
    result: dict,
    genre_source: str,
    genre_list: list[str],
) -> tuple[bool, str]:
    """Write analysis results as ID3 tags.

    Returns (success, genre_action_description).
    """
    try:
        try:
            tags = ID3(filepath)
        except ID3NoHeaderError:
            tags = ID3()

        # Genre: only overwrite if existing is generic/empty
        existing_genre = ""
        tcon = tags.getall("TCON")
        if tcon and tcon[0].text:
            existing_genre = tcon[0].text[0].strip()

        if genre_list:
            genre_str = "; ".join(genre_list[:4])
            if existing_genre.lower() in GENERIC_GENRES or not existing_genre:
                tags.delall("TCON")
                tags.add(TCON(encoding=3, text=[genre_str]))
                genre_action = "replaced"
            elif existing_genre.lower() != genre_str.lower():
                genre_action = f"kept '{existing_genre}'"
            else:
                genre_action = "matches"
        else:
            genre_action = "no genre"

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
        ]:
            tags.delall(f"TXXX:{key}")
            tags.add(TXXX(encoding=3, desc=key, text=[str(val)]))

        # Comments
        comment, detail = _build_comment(result["energy"], result["valence"])
        tags.delall("COMM::eng")
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.delall("COMM:djtagger:eng")
        tags.add(COMM(encoding=3, lang="eng", desc="djtagger", text=detail))

        tags.save(filepath)
        return True, genre_action
    except Exception as ex:
        return False, f"error: {ex}"


def fix_comments(filepath: str) -> bool:
    """Re-write comments from existing TXXX energy/valence tags.

    Returns True if comment was updated, False if skipped.
    """
    try:
        tags = ID3(filepath)

        # Only fix files that have our tagger version tag
        tv = tags.get("TXXX:TAGGER_VERSION")
        if not tv:
            return False
        e_tag = tags.get("TXXX:ENERGY")
        v_tag = tags.get("TXXX:VALENCE")
        if not e_tag or not v_tag:
            return False

        e = float(e_tag.text[0])
        v = float(v_tag.text[0])
        comment, detail = _build_comment(e, v)

        tags.delall("COMM::eng")
        tags.add(COMM(encoding=3, lang="eng", desc="", text=comment))
        tags.delall("COMM:djtagger:eng")
        tags.add(COMM(encoding=3, lang="eng", desc="djtagger", text=detail))
        tags.save(filepath)
        return True
    except Exception:
        return False
