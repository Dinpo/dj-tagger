from djtagger import genres


def _no_network(monkeypatch):
    monkeypatch.setattr(genres, "get_beatport_metadata",
                        lambda *a, **k: {"genres": [], "album": "", "year": ""})
    monkeypatch.setattr(genres, "get_lastfm_genre", lambda *a, **k: [])


def test_flat_electronic_head_ignored_uses_discogs(monkeypatch):
    # Flat electronic head (top 0.212 barely above 0.207) is uncertain: it must
    # not be prepended; the coherent Discogs-400 labels lead instead.
    _no_network(monkeypatch)
    disc = [("House", 0.552), ("Electro House", 0.390), ("Tropical House", 0.340)]
    elec = [("ambient", 0.212), ("house", 0.207), ("dnb", 0.199)]
    meta = genres.resolve_metadata("a", "a", "t", disc, ml_electronic_genres=elec)
    assert meta["source"] == "ml"
    assert meta["genres"][:3] == ["House", "Electro House", "Tropical House"]
    assert "ambient" not in [g.lower() for g in meta["genres"]]


def test_dominant_electronic_head_leads(monkeypatch):
    # A clearly dominant electronic top (0.55 vs 0.10) is trusted and leads.
    _no_network(monkeypatch)
    disc = [("House", 0.40), ("Deep House", 0.20)]
    elec = [("Techno", 0.55), ("House", 0.10)]
    meta = genres.resolve_metadata("a", "a", "t", disc, ml_electronic_genres=elec)
    assert meta["genres"][0] == "Techno"


def test_near_chance_electronic_head_ignored(monkeypatch):
    # Real case (Faithless - We Come 1, MANNA Remix): the 5-class electronic
    # head says ambient 0.28 vs dnb 0.20, barely above the 0.20 chance level.
    # It must not prepend "ambient; dnb" ahead of the coherent trance labels.
    _no_network(monkeypatch)
    disc = [("Tech Trance", 0.31), ("Psy-Trance", 0.19), ("Techno", 0.16)]
    elec = [("ambient", 0.28), ("dnb", 0.20), ("house", 0.19)]
    meta = genres.resolve_metadata("a", "a", "t", disc, ml_electronic_genres=elec)
    assert meta["genres"][0] == "Tech Trance"
    low = [g.lower() for g in meta["genres"]]
    assert "ambient" not in low and "dnb" not in low


def _lastfm(monkeypatch, tags):
    monkeypatch.setattr(genres, "get_beatport_metadata",
                        lambda *a, **k: {"genres": [], "album": "", "year": ""})
    monkeypatch.setattr(genres, "get_lastfm_genre", lambda *a, **k: tags)


HOUSE = [("House", 0.5), ("Electro House", 0.3)]


def test_is_genre_tag_drops_non_genres():
    for tag in ("Minimal Techno", "Hip-Hop", "Drum and Bass", "Synth-pop", "rnb", "Tech House"):
        assert genres._is_genre_tag(tag), tag
    for tag in ("Australia", "female vocalists", "80s", "british", "seen live", "merge",
                "funk_add_to_lidarr_batch_10", "under 2000 listeners", "lady gaga"):
        assert not genres._is_genre_tag(tag), tag


def test_is_remix():
    assert genres._is_remix("Enter Sandman (Joel Fletcher Remix)")
    assert genres._is_remix("Get Lucky (Daft Punk Cover)")
    assert genres._is_remix("Song - Bootleg")
    assert not genres._is_remix("Night & Day")
    assert not genres._is_remix("Analog Love (Original Mix)")
    assert not genres._is_remix("Wonderwall (Extended Mix)")


def test_remix_ignores_lastfm(monkeypatch):
    _lastfm(monkeypatch, ["Thrash Metal", "Heavy Metal"])
    meta = genres.resolve_metadata("Metallica", "Metallica", "Enter Sandman (X Remix)", HOUSE)
    assert meta["source"] == "ml"
    assert meta["genres"] == ["House", "Electro House"]


def test_lastfm_specific_genre_leads_on_an_original(monkeypatch):
    _lastfm(monkeypatch, ["Minimal Techno", "Techno", "Electronic"])
    meta = genres.resolve_metadata("Boris Brejcha", "Boris Brejcha", "Night & Day",
                                   [("Tech Trance", 0.5), ("Techno", 0.3)])
    assert meta["source"] == "lastfm+ml"
    assert meta["genres"] == ["Minimal Techno", "Techno", "Tech Trance"]


def test_lastfm_tag_contradicting_electronic_audio_is_dropped(monkeypatch):
    _lastfm(monkeypatch, ["Metal", "Hip-Hop"])
    meta = genres.resolve_metadata("DJ X", "DJ X", "Track", HOUSE)
    assert meta["source"] == "ml"
    assert meta["genres"] == ["House", "Electro House"]


def test_lastfm_only_generic_tags_add_nothing(monkeypatch):
    _lastfm(monkeypatch, ["Electronic", "Dance"])
    assert genres.resolve_metadata("A", "A", "Track", HOUSE)["source"] == "ml"


def test_lastfm_rock_kept_when_audio_is_not_electronic(monkeypatch):
    # Not every library is electronic: rock tags must still work for rock.
    _lastfm(monkeypatch, ["Thrash Metal", "Heavy Metal"])
    meta = genres.resolve_metadata("Metallica", "Metallica", "Battery", [("Thrash", 0.6)])
    assert meta["source"] == "lastfm+ml"
    assert meta["genres"][0] == "Thrash Metal"
