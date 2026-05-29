"""Beatport scraping + Last.fm API + remix matching / scoring."""

import json
import re
import subprocess
import sys
from collections import OrderedDict
from typing import Any
from urllib.parse import quote as _url_quote

from curl_cffi import requests as _cffi_requests
from curl_cffi.requests.errors import RequestsError as _CffiRequestsError

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

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > _BEATPORT_CACHE_MAX:
            self.popitem(last=False)


_beatport_cache: _BoundedCache = _BoundedCache()

# ─── Last.fm API key warning (shown once) ────────────────────

_lastfm_warned = False

# ─── Mix / Remix helpers ────────────────────────────────────


_FEAT_RE = re.compile(r"\b(?:feat\.?|ft\.?|featuring)\b", re.IGNORECASE)
_TOP_LEVEL_FEAT_RE = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring)\s+[^()\[\]]+?(?=\s*[\(\[]|$)",
    re.IGNORECASE,
)
_MIX_KEYWORDS_RE = re.compile(
    r"\b(?:remix|mix|edit|dub|rework|bootleg|version|vip)\b",
    re.IGNORECASE,
)
_PAREN_OR_BRACKET_RE = re.compile(r"[\(\[]([^)\]]*)[\)\]]")


def _extract_mix_info(title: str) -> tuple[str, str, str]:
    """Parse a track title into (base_title, subtitle_extras, mix_info).

    The three buckets:
      - mix_info: paren/bracket content with a remix-marker keyword (the first
        one wins, e.g. "Friction Remix", "Extended Mix").
      - subtitle_extras: other paren/bracket content that disambiguates the
        track but isn't a remix marker (e.g. "ASOT 950 Anthem",
        "Love Lesson"). Joined with spaces if multiple.
      - base_title: the main title with top-level "feat. X" segments AND all
        paren/bracket sections stripped.

    Featured-artist parens like "(feat. Someone)" are dropped entirely — they
    pollute both the search query and the title-token comparison.
    """
    mix_parts: list[str] = []
    subtitle_parts: list[str] = []
    for m in _PAREN_OR_BRACKET_RE.finditer(title):
        content = m.group(1).strip()
        if not content or _FEAT_RE.search(content):
            continue
        if _MIX_KEYWORDS_RE.search(content):
            mix_parts.append(content)
        else:
            subtitle_parts.append(content)
    mix_info = mix_parts[0] if mix_parts else ""
    subtitle_extras = " ".join(subtitle_parts).strip()

    # Strip top-level "feat. X" then all parens/brackets
    base = _TOP_LEVEL_FEAT_RE.sub("", title)
    base = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*", " ", base)
    base_title = re.sub(r"\s+", " ", base).strip()
    return base_title, subtitle_extras, mix_info


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


_TITLE_STOPWORDS = {"the", "a", "an", "feat", "ft", "and"}


def _title_tokens(s: str) -> set[str]:
    """Extract meaningful title words from a string, stopword-stripped."""
    return {t for t in re.findall(r"\w+", s.lower()) if t not in _TITLE_STOPWORDS}


def _score_beatport_result(
    item: dict,
    artist_lower: str,
    base_title_lower: str,
    file_mix_info: str,
) -> int:
    """Score a Beatport result for match quality. Higher = better.

    Returns -100 as a hard reject when the track's title shares no meaningful
    tokens with the file — no amount of artist or mix-name bonus can rescue
    a completely different track.
    """
    track_name = (item.get("track_name") or "").lower()
    mix_name = item.get("mix_name", "") or ""
    item_artists = [a.get("artist_name", "").lower() for a in item.get("artists", [])]

    # Strip parenthesised / bracketed sections from track_name for title
    # comparison — that content (e.g. "(VIP)", "[Radio Edit]") belongs to the
    # mix comparison instead.
    track_base = re.sub(r"[\(\[][^)\]]*[\)\]]", " ", track_name)
    file_toks = _title_tokens(base_title_lower)
    tn_toks = _title_tokens(track_base)

    overlap = file_toks & tn_toks if (file_toks and tn_toks) else set()
    if file_toks and not overlap:
        # Different track entirely — hard reject
        return -100

    # Title match — scaled by how many file-title tokens appear in the track
    score = 0
    if file_toks:
        frac = len(overlap) / len(file_toks)
        if frac >= 1.0:
            score += 15  # All file tokens present
        elif frac >= 0.5:
            score += 8   # Majority match
        else:
            score += 2   # Weak — some overlap but mostly different
    else:
        # Degenerate: empty file tokens (stopwords only?). Fall back to substring.
        if base_title_lower and base_title_lower in track_name:
            score += 10

    # Artist match
    if any(a in artist_lower or artist_lower in a for a in item_artists if a):
        score += 5
    for part in re.split(r"\s*[&,]\s*", artist_lower):
        part = part.strip()
        if part and any(part in a or a in part for a in item_artists if a):
            score += 2

    # Beatport sometimes puts remix info inside the track_name parens/brackets
    # (e.g. "More Baby (VIP)" + mix_name="Extended Mix"). Combine both for the
    # mix comparison so we don't miss these split-encoding cases.
    paren_parts = " ".join(re.findall(r"[\(\[]([^)\]]+)[\)\]]", track_name))
    bp_mix_effective = f"{paren_parts} {mix_name}".strip()

    # Mix / remix matching
    if file_mix_info:
        file_mix_norm = _normalize_mix(file_mix_info)
        bp_mix_norm = _normalize_mix(bp_mix_effective)

        if file_mix_norm and bp_mix_norm and file_mix_norm == bp_mix_norm:
            score += 25  # Exact remix match
        elif file_mix_norm and bp_mix_norm:
            file_words = _remix_words(file_mix_info)
            bp_words = _remix_words(bp_mix_effective)
            if file_words and bp_words:
                word_overlap = file_words & bp_words
                if word_overlap:
                    score += 15 + len(word_overlap) * 3
                else:
                    score -= 20  # Different remix

        if not _is_generic_mix(file_mix_info) and _is_generic_mix(bp_mix_effective):
            score -= 15  # We want a specific remix, this is original/extended
    else:
        if _is_generic_mix(mix_name):
            score += 3

    return score


# ─── Beatport lookup ────────────────────────────────────────


_EMPTY_BP: dict = {"genres": [], "album": "", "year": ""}

_beatport_warned = False


def _warn_beatport_once(message: str) -> None:
    """Print a Beatport-broken warning at most once per process."""
    global _beatport_warned
    if _beatport_warned:
        return
    _beatport_warned = True
    print(f"[djtagger] Warning: {message}", file=sys.stderr)


def get_beatport_metadata(artist: str, title: str) -> dict:
    """Look up a track on Beatport via search page scraping.

    Returns {"genres": [...up to 3], "album": str, "year": str} — any field
    empty if the corresponding data wasn't found. Returns an all-empty dict on
    transient network failure (not cached, so the next run can retry).
    """
    cache_key = f"{artist}|{title}".lower()
    if cache_key in _beatport_cache:
        return _beatport_cache[cache_key]

    base_title, subtitle_extras, file_mix_info = _extract_mix_info(title)
    artist_search = re.sub(r"\s*&\s*", " ", artist).strip()

    # Build the search query from artist + base title + any subtitle (e.g.
    # "ASOT 950 Anthem"). Subtitle is high-signal for Beatport's relevance
    # ranking — without it, common-word titles get drowned out by other
    # tracks containing the artist's name.
    query_parts = [artist_search, base_title]
    if subtitle_extras:
        query_parts.append(subtitle_extras)
    if file_mix_info and not _is_generic_mix(file_mix_info):
        query_parts.append(_normalize_mix(file_mix_info))
    query = " ".join(p for p in query_parts if p).strip()

    try:
        url = f"https://www.beatport.com/search/tracks?q={_url_quote(query)}"
        # curl_cffi impersonates Chrome's TLS fingerprint — Beatport sits behind
        # Cloudflare which 403s plain curl/python-requests even with browser
        # User-Agent. impersonate="chrome" is what gets past the challenge.
        try:
            resp = _cffi_requests.get(
                url, impersonate="chrome", timeout=BEATPORT_TIMEOUT
            )
        except _CffiRequestsError:
            # Transient network/TLS error — don't cache, allow retry next run
            return dict(_EMPTY_BP)

        body = resp.text
        if resp.status_code != 200 or not body:
            blocked = (
                resp.status_code == 403
                or "Just a moment" in body[:2000]
                or "challenge-platform" in body[:2000]
            )
            if blocked:
                _warn_beatport_once(
                    f"Beatport returning Cloudflare challenge (HTTP "
                    f"{resp.status_code}). All Beatport lookups will fall "
                    "through to Last.fm/ML for the rest of this run."
                )
            else:
                _warn_beatport_once(
                    f"Beatport returned HTTP {resp.status_code} — lookups "
                    "will fall through to Last.fm/ML."
                )
            _beatport_cache[cache_key] = dict(_EMPTY_BP)
            return dict(_EMPTY_BP)

        match = re.search(
            r'__NEXT_DATA__.*?type="application/json">(.*?)</script>',
            body,
        )
        if not match:
            _warn_beatport_once(
                "Beatport response missing __NEXT_DATA__ (page format may have "
                "changed). Lookups will fall through to Last.fm/ML."
            )
            _beatport_cache[cache_key] = dict(_EMPTY_BP)
            return dict(_EMPTY_BP)

        data = json.loads(match.group(1))
        # Guard against Beatport page structure changes
        try:
            items = (
                data["props"]["pageProps"]["dehydratedState"]
                ["queries"][0]["state"]["data"]["data"]
            )
        except (KeyError, IndexError, TypeError):
            _warn_beatport_once(
                "Beatport __NEXT_DATA__ shape changed — scraping path broken. "
                "Lookups will fall through to Last.fm/ML."
            )
            _beatport_cache[cache_key] = dict(_EMPTY_BP)
            return dict(_EMPTY_BP)
        if not items:
            _beatport_cache[cache_key] = dict(_EMPTY_BP)
            return dict(_EMPTY_BP)

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

        # Universal acceptance floor — any match needs title+artist confidence
        # beyond chance. The title gate already rejects unrelated tracks;
        # this catches cases where the best remaining match is still weak.
        min_score = 15 if (file_mix_info and not _is_generic_mix(file_mix_info)) else 10
        if best_score < min_score:
            _beatport_cache[cache_key] = dict(_EMPTY_BP)
            return dict(_EMPTY_BP)

        genres: list[str] = []
        for g in best.get("genre", []):
            gname = g.get("genre_name", "")
            if gname:
                genres.append(gname)

        release = best.get("release") or {}
        # Beatport uses `release_name` (not `name`) inside the release object
        album = str(release.get("release_name") or "").strip()
        # Year lives at the track level as `publish_date` (e.g. "2022-08-22T00:00:00")
        # with `release_date` as an occasional fallback
        date_str = str(best.get("publish_date") or best.get("release_date") or "").strip()
        year_match = re.search(r"(19|20)\d{2}", date_str)
        year = year_match.group(0) if year_match else ""

    except Exception:
        return dict(_EMPTY_BP)

    result_dict = {"genres": genres[:3], "album": album, "year": year}
    _beatport_cache[cache_key] = result_dict
    return result_dict


def get_beatport_genre(artist: str, title: str) -> list[str]:
    """Backwards-compatible wrapper: return just the genre list."""
    return get_beatport_metadata(artist, title)["genres"]


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


def resolve_metadata(
    artist: str,
    artist_clean: str,
    title: str,
    ml_genres: list[tuple[str, float]],
    ml_electronic_genres: list[tuple[str, float]] | None = None,
    use_beatport: bool = True,
    genre_keep_prob: float = 0.10,
) -> dict:
    """Resolve genre + album/year from Beatport > Last.fm > ML.

    Album and year only come from Beatport (the other sources don't carry
    reliable release info). Source is one of:
    "beatport", "lastfm+ml", "ml".
    """
    # Beatport first (primary source for electronic dance music)
    bp = (
        get_beatport_metadata(artist_clean or artist, title)
        if use_beatport else dict(_EMPTY_BP)
    )
    album = bp.get("album", "")
    year = bp.get("year", "")
    bp_genres = bp.get("genres", [])
    if bp_genres:
        return {"genres": bp_genres, "source": "beatport", "album": album, "year": year}

    # Last.fm (artist-level) + ML fallback
    fm_genres = get_lastfm_genre(artist, artist_clean, title)
    ml_list = [g[0] for g in ml_genres[:3] if g[1] >= genre_keep_prob]

    # Prefer electronic sub-genre labels when confident
    if ml_electronic_genres:
        elec_list = [g[0] for g in ml_electronic_genres[:3] if g[1] >= genre_keep_prob]
        if elec_list and ml_electronic_genres[0][1] > 0.20:
            ml_list = elec_list + [g for g in ml_list if g.lower() not in
                                   [e.lower() for e in elec_list]]

    if fm_genres:
        final = fm_genres[:]
        for g in ml_list:
            if g.lower() not in [x.lower() for x in final]:
                final.append(g)
        return {"genres": final, "source": "lastfm+ml", "album": album, "year": year}

    return {"genres": ml_list, "source": "ml", "album": album, "year": year}


def resolve_genres(
    artist: str,
    artist_clean: str,
    title: str,
    ml_genres: list[tuple[str, float]],
    ml_electronic_genres: list[tuple[str, float]] | None = None,
    use_beatport: bool = True,
    genre_keep_prob: float = 0.10,
) -> tuple[list[str], str]:
    """Backwards-compatible wrapper returning just (genre_list, source_name)."""
    m = resolve_metadata(
        artist, artist_clean, title, ml_genres,
        ml_electronic_genres=ml_electronic_genres,
        use_beatport=use_beatport,
        genre_keep_prob=genre_keep_prob,
    )
    return m["genres"], m["source"]
