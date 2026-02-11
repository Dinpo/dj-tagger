"""Beatport scraping + Last.fm API + remix matching / scoring."""

import json
import re
import subprocess
import sys
from collections import OrderedDict
from urllib.parse import quote as _url_quote

from .config import (
    BEATPORT_TIMEOUT,
    LASTFM_API_KEY,
    LASTFM_TIMEOUT,
    LASTFM_URL,
    LASTFM_MIN_COUNT,
)

# ─── Beatport cache (bounded LRU) ───────────────────────────

_BEATPORT_CACHE_MAX = 500


class _BoundedCache(OrderedDict):
    """Simple bounded LRU cache using OrderedDict."""

    def __setitem__(self, key: str, value: list[str]) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > _BEATPORT_CACHE_MAX:
            self.popitem(last=False)


_beatport_cache: _BoundedCache = _BoundedCache()

# ─── Last.fm API key warning (shown once) ────────────────────

_lastfm_warned = False

# ─── Mix / Remix helpers ────────────────────────────────────


def _extract_mix_info(title: str) -> tuple[str, str]:
    """Extract remix/mix info and base title from track title."""
    mix_match = re.search(
        r"\(([^)]*(?:remix|mix|edit|dub|rework|bootleg|version|vip)[^)]*)\)",
        title,
        re.IGNORECASE,
    )
    mix_info = mix_match.group(1).strip() if mix_match else ""
    base_title = re.sub(r"\s*\(.*?\)\s*", " ", title).strip()
    return base_title, mix_info


def _normalize_mix(mix_str: str) -> str:
    """Normalize a mix name for comparison: lowercase, strip filler words."""
    s = mix_str.lower()
    s = re.sub(r"\b(extended|original|radio)\b", "", s).strip()
    s = re.sub(r"\s*(remix|mix|edit|dub|rework)\s*$", "", s).strip()
    s = re.sub(r"\s*&\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_generic_mix(mix_str: str) -> bool:
    """Check if mix name is generic (Original Mix, Extended Mix, etc.)."""
    s = mix_str.lower().strip()
    return s in (
        "",
        "original mix",
        "extended mix",
        "radio edit",
        "radio mix",
        "extended",
        "original",
        "club mix",
        "extended club mix",
    )


def _remix_words(mix_str: str) -> set[str]:
    """Extract meaningful words from a remix name."""
    s = _normalize_mix(mix_str)
    s = re.sub(
        r"\b(feat\.?|ft\.?|featuring|the|a|an|of|in|on|at|to|for|and|vs\.?)\b",
        "",
        s,
    )
    return set(w for w in s.split() if len(w) > 1)


# ─── Beatport scoring ──────────────────────────────────────


def _score_beatport_result(
    item: dict,
    artist_lower: str,
    base_title_lower: str,
    file_mix_info: str,
) -> int:
    """Score a Beatport result for match quality. Higher = better."""
    track_name = (item.get("track_name") or "").lower()
    mix_name = item.get("mix_name", "") or ""
    item_artists = [a.get("artist_name", "").lower() for a in item.get("artists", [])]

    score = 0

    # Track name match (required for a good match)
    if base_title_lower in track_name or track_name in base_title_lower:
        score += 10
    elif any(w in track_name for w in base_title_lower.split() if len(w) > 2):
        score += 3  # Partial title match
    else:
        score -= 10  # Wrong track entirely

    # Artist match
    if any(a in artist_lower or artist_lower in a for a in item_artists if a):
        score += 5
    # Check individual artist names (for "Artist1 & Artist2" cases)
    for part in re.split(r"\s*[&,]\s*", artist_lower):
        part = part.strip()
        if part and any(part in a or a in part for a in item_artists if a):
            score += 2

    # Mix / remix matching (the critical part)
    if file_mix_info:
        file_mix_norm = _normalize_mix(file_mix_info)
        bp_mix_norm = _normalize_mix(mix_name)

        if file_mix_norm and bp_mix_norm and file_mix_norm == bp_mix_norm:
            score += 25  # Exact remix match
        elif file_mix_norm and bp_mix_norm:
            # Fuzzy: check if key remix words overlap
            file_words = _remix_words(file_mix_info)
            bp_words = _remix_words(mix_name)
            if file_words and bp_words:
                overlap = file_words & bp_words
                if overlap:
                    score += 15 + len(overlap) * 3
                else:
                    score -= 20  # Different remix

        if not _is_generic_mix(file_mix_info) and _is_generic_mix(mix_name):
            score -= 15  # We want a specific remix, this is original/extended
    else:
        # No specific remix in filename — prefer original/extended
        if _is_generic_mix(mix_name):
            score += 3

    return score


# ─── Beatport lookup ────────────────────────────────────────


def get_beatport_genre(artist: str, title: str) -> list[str]:
    """Look up genre on Beatport via search page scraping.

    Returns up to 3 genre names, or empty list on miss.
    """
    cache_key = f"{artist}|{title}".lower()
    if cache_key in _beatport_cache:
        return _beatport_cache[cache_key]

    base_title, file_mix_info = _extract_mix_info(title)
    artist_search = re.sub(r"\s*&\s*", " ", artist).strip()

    # Include remix info in search query for better results
    if file_mix_info and not _is_generic_mix(file_mix_info):
        remix_terms = _normalize_mix(file_mix_info)
        query = f"{artist_search} {base_title} {remix_terms}".strip()
    else:
        query = f"{artist_search} {base_title}".strip()

    genres: list[str] = []
    try:
        url = f"https://www.beatport.com/search/tracks?q={_url_quote(query)}"
        result = subprocess.run(
            ["curl", "-s", "-m", str(BEATPORT_TIMEOUT), "-A", "Mozilla/5.0", url],
            capture_output=True,
            text=True,
            timeout=BEATPORT_TIMEOUT + 2,
        )
        if result.returncode != 0 or not result.stdout:
            _beatport_cache[cache_key] = []
            return []

        match = re.search(
            r'__NEXT_DATA__.*?type="application/json">(.*?)</script>',
            result.stdout,
        )
        if not match:
            _beatport_cache[cache_key] = []
            return []

        data = json.loads(match.group(1))
        # Guard against Beatport page structure changes
        try:
            items = (
                data["props"]["pageProps"]["dehydratedState"]
                ["queries"][0]["state"]["data"]["data"]
            )
        except (KeyError, IndexError, TypeError):
            print(
                "[djtagger] Warning: Beatport page structure changed — "
                "scraping may be broken. Falling back to other sources.",
                file=sys.stderr,
            )
            _beatport_cache[cache_key] = []
            return []
        if not items:
            _beatport_cache[cache_key] = []
            return []

        artist_lower = artist.lower()
        base_title_lower = base_title.lower()

        # Score all results and pick the best
        scored = []
        for item in items[:10]:
            s = _score_beatport_result(
                item, artist_lower, base_title_lower, file_mix_info
            )
            scored.append((s, item))
        scored.sort(key=lambda x: -x[0])

        best_score, best = scored[0]

        # If we wanted a specific remix but best match is poor, skip Beatport
        if file_mix_info and not _is_generic_mix(file_mix_info) and best_score < 10:
            _beatport_cache[cache_key] = []
            return []

        for g in best.get("genre", []):
            gname = g.get("genre_name", "")
            if gname:
                genres.append(gname)

    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    _beatport_cache[cache_key] = genres[:3]
    return genres[:3]


# ─── Last.fm lookup ─────────────────────────────────────────


def get_lastfm_genre(artist: str, artist_clean: str, title: str) -> list[str]:
    """Look up genre tags on Last.fm via artist.getTopTags API.

    Tries cleaned artist name first, then raw.
    Returns up to 3 genre names.
    """
    global _lastfm_warned
    if not LASTFM_API_KEY:
        if not _lastfm_warned:
            print(
                "[djtagger] Warning: LASTFM_API_KEY not set — "
                "Last.fm lookups disabled. Set the env var for better genre results.",
                file=sys.stderr,
            )
            _lastfm_warned = True
        return []

    genres: list[str] = []
    for art in [artist_clean, artist]:
        if not art:
            continue
        try:
            url = (
                f"{LASTFM_URL}?method=artist.getTopTags"
                f"&artist={_url_quote(art)}"
                f"&api_key={LASTFM_API_KEY}&format=json"
            )
            result = subprocess.run(
                ["curl", "-s", "-m", str(LASTFM_TIMEOUT), url],
                capture_output=True,
                text=True,
                timeout=LASTFM_TIMEOUT + 3,
            )
            if result.returncode != 0 or not result.stdout:
                continue
            data = json.loads(result.stdout)
            if "toptags" in data and "tag" in data["toptags"]:
                tags = data["toptags"]["tag"]
                if isinstance(tags, list):
                    genres = [
                        t["name"].title()
                        for t in tags[:3]
                        if int(t.get("count", 0)) > LASTFM_MIN_COUNT
                    ]
                    if genres:
                        break
        except Exception:
            pass
    return genres


# ─── Resolve genre from all sources ─────────────────────────


def resolve_genres(
    artist: str,
    artist_clean: str,
    title: str,
    ml_genres: list[tuple[str, float]],
    use_beatport: bool = True,
    genre_keep_prob: float = 0.10,
) -> tuple[list[str], str]:
    """Resolve genre from Beatport > Last.fm > ML.

    Returns (genre_list, source_name).
    """
    bp_genres = (
        get_beatport_genre(artist_clean or artist, title) if use_beatport else []
    )
    fm_genres = get_lastfm_genre(artist, artist_clean, title)
    ml_list = [g[0] for g in ml_genres[:3] if g[1] >= genre_keep_prob]

    if bp_genres:
        return bp_genres, "beatport"

    if fm_genres:
        final = fm_genres[:]
        for g in ml_list:
            if g.lower() not in [x.lower() for x in final]:
                final.append(g)
        return final, "lastfm+ml"

    return ml_list, "ml"
