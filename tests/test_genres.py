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
