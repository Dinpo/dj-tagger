"""Format-neutral tag access: MP3 / AIFF / WAV (ID3), FLAC / Ogg (Vorbis), M4A (MP4).

The rest of DJ Tagger reads and writes tags through `TagFile` and never
touches a format's native frames. Two kinds of field:

- **standard** fields that DJ software displays: genre, bpm, key, album,
  year, artist, title (read only, as a filename fallback), comment (the
  visible Serato comment) and detail (our hidden detail comment).
- **custom** fields holding our analysis (ENERGY, SET_ROLE, ...).

Native mapping per format:

| field   | ID3 (mp3/aif/wav) | Vorbis (flac/ogg)  | MP4 (m4a)                     |
|---------|-------------------|--------------------|-------------------------------|
| genre   | TCON              | GENRE              | ©gen                          |
| bpm     | TBPM              | BPM                | tmpo                          |
| key     | TKEY              | INITIALKEY         | ----:com.apple.iTunes:initialkey |
| album   | TALB              | ALBUM              | ©alb                          |
| year    | TDRC              | DATE               | ©day                          |
| artist  | TPE1              | ARTIST             | ©ART                          |
| title   | TIT2              | TITLE              | ©nam                          |
| comment | COMM::eng         | COMMENT            | ©cmt                          |
| detail  | COMM:djtagger:eng | DJTAGGER_DETAIL    | ----:com.apple.iTunes:DJTAGGER_DETAIL |
| custom  | TXXX:<NAME>       | DJTAGGER_<NAME>    | ----:com.apple.iTunes:DJTAGGER_<NAME> |

Custom fields are namespaced with `DJTAGGER_` outside ID3 because Vorbis
field names are case-insensitive and Mixed In Key already stores its own
base64 blobs under `energy` and `key` in FLAC files: a plain `ENERGY` field
would overwrite them. ID3 keeps the bare `TXXX` names it has always used, so
existing MP3 libraries read back unchanged.
"""

import os

import mutagen
from mutagen.id3 import COMM, ID3, TALB, TBPM, TCON, TDRC, TIT2, TKEY, TPE1, TXXX, ID3NoHeaderError
from mutagen.mp4 import MP4, AtomDataType, MP4FreeForm

# Extensions DJ Tagger analyzes and tags. .mp4 is left out on purpose: in a
# music folder it is usually a video.
AUDIO_EXTS = (".mp3", ".flac", ".m4a", ".aif", ".aiff", ".wav", ".ogg")

_ID3_EXTS = (".mp3", ".aif", ".aiff", ".wav")
_VORBIS_EXTS = (".flac", ".ogg")
_MP4_EXTS = (".m4a",)

CUSTOM_PREFIX = "DJTAGGER_"


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXTS)


class BrokenHeaderError(Exception):
    """The file's header reports zero length, so writing tags is refused.

    Some exports (typically FFmpeg-based) leave the FLAC sample count at
    zero. The audio is intact and players with a cached length keep working,
    but once a tag write changes the file, DJ software re-reads the header
    and loads a 0-second track. Fix by re-encoding losslessly
    (`ffmpeg -i in.flac -c:a flac out.flac`) and copying the tags back.
    """


def _check_header(audio, path: str) -> None:
    if not audio.info.length:
        raise BrokenHeaderError(
            f"header reports 0 length, not writing tags (re-encode losslessly "
            f"to repair, see README Troubleshooting): {os.path.basename(path)}"
        )


class TagFile:
    """One file's tags. Open with `TagFile.open(path)`; call `save()` to write."""

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def open(cls, path: str) -> "TagFile":
        """Open *path* for tag access. Raises on an unreadable or unsupported file.

        A file without any tags opens as an empty tag set (saving creates one).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mp3":
            return _ID3File(path)
        if ext in _ID3_EXTS:
            return _ContainerID3File(path)
        if ext in _VORBIS_EXTS:
            return _VorbisFile(path)
        if ext in _MP4_EXTS:
            return _MP4File(path)
        raise ValueError(f"unsupported audio format: {ext or path}")

    # Subclasses implement these.
    def get(self, field: str) -> str:
        raise NotImplementedError

    def set(self, field: str, value: str) -> None:
        raise NotImplementedError

    def delete(self, field: str) -> None:
        raise NotImplementedError

    def get_custom(self, name: str) -> str:
        raise NotImplementedError

    def set_custom(self, name: str, value: str) -> None:
        raise NotImplementedError

    def delete_custom(self, name: str) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError


# ─── ID3 (MP3, AIFF, WAV) ───────────────────────────────────

_ID3_TEXT = {"genre": TCON, "bpm": TBPM, "key": TKEY, "album": TALB, "year": TDRC,
             "artist": TPE1, "title": TIT2}
_ID3_COMM_DESC = {"comment": "", "detail": "djtagger"}


class _ID3File(TagFile):
    """Bare ID3 file (MP3). Behaves exactly like the pre-TagFile code."""

    def __init__(self, path: str):
        super().__init__(path)
        self.tags = self._load()

    def _load(self):
        try:
            return ID3(self.path)
        except ID3NoHeaderError:
            return ID3()

    def get(self, field: str) -> str:
        if field in _ID3_COMM_DESC:
            fr = self.tags.get(f"COMM:{_ID3_COMM_DESC[field]}:eng")
            return str(fr) if fr else ""
        fr = self.tags.get(_ID3_TEXT[field].__name__)
        if fr and fr.text:
            return str(fr.text[0]).strip()
        return ""

    def set(self, field: str, value: str) -> None:
        if field in _ID3_COMM_DESC:
            desc = _ID3_COMM_DESC[field]
            self.tags.delall(f"COMM:{desc}:eng")
            self.tags.add(COMM(encoding=3, lang="eng", desc=desc, text=value))
            return
        frame = _ID3_TEXT[field]
        self.tags.delall(frame.__name__)
        self.tags.add(frame(encoding=3, text=[value]))

    def delete(self, field: str) -> None:
        if field in _ID3_COMM_DESC:
            self.tags.delall(f"COMM:{_ID3_COMM_DESC[field]}:eng")
        else:
            self.tags.delall(_ID3_TEXT[field].__name__)

    def get_custom(self, name: str) -> str:
        fr = self.tags.get(f"TXXX:{name}")
        return str(fr.text[0]) if fr and fr.text else ""

    def set_custom(self, name: str, value: str) -> None:
        self.tags.delall(f"TXXX:{name}")
        self.tags.add(TXXX(encoding=3, desc=name, text=[value]))

    def delete_custom(self, name: str) -> None:
        self.tags.delall(f"TXXX:{name}")

    def save(self) -> None:
        self.tags.save(self.path)


class _ContainerID3File(_ID3File):
    """ID3 inside an AIFF or WAV container."""

    def _load(self):
        self.audio = mutagen.File(self.path)
        if self.audio is None:
            raise ValueError(f"unreadable audio file: {self.path}")
        if self.audio.tags is None:
            self.audio.add_tags()
        # Keep the ID3 version the file already uses (Serato writes v2.3 into
        # some AIFF/WAV files); a fresh tag gets mutagen's default v2.4.
        self._v2 = 3 if self.audio.tags.version[:2] == (2, 3) else 4
        return self.audio.tags

    def save(self) -> None:
        _check_header(self.audio, self.path)
        self.audio.save(v2_version=self._v2)


# ─── Vorbis comments (FLAC, Ogg) ────────────────────────────

_VORBIS_STD = {
    "genre": "GENRE", "bpm": "BPM", "key": "INITIALKEY", "album": "ALBUM",
    "year": "DATE", "comment": "COMMENT", "detail": CUSTOM_PREFIX + "DETAIL",
    "artist": "ARTIST", "title": "TITLE",
}


class _VorbisFile(TagFile):
    def __init__(self, path: str):
        super().__init__(path)
        self.audio = mutagen.File(path)
        if self.audio is None:
            raise ValueError(f"unreadable audio file: {path}")
        if self.audio.tags is None:
            self.audio.add_tags()
        self.tags = self.audio.tags

    def _get(self, key: str) -> str:
        vals = self.tags.get(key)
        return str(vals[0]) if vals else ""

    def _set(self, key: str, value: str) -> None:
        self.tags[key] = [value]

    def _delete(self, key: str) -> None:
        if key in self.tags:
            del self.tags[key]

    def get(self, field: str) -> str:
        val = self._get(_VORBIS_STD[field])
        return val if field in ("comment", "detail") else val.strip()

    def set(self, field: str, value: str) -> None:
        if field == "comment":
            # Many taggers (Mixed In Key among them) mirror the comment into
            # DESCRIPTION because players disagree on which one to show. Keep
            # the mirror in step, but never touch a DESCRIPTION that holds
            # something else.
            old = self._get("COMMENT")
            if "DESCRIPTION" in self.tags and self._get("DESCRIPTION") == old:
                self._set("DESCRIPTION", value)
        self._set(_VORBIS_STD[field], value)

    def delete(self, field: str) -> None:
        self._delete(_VORBIS_STD[field])

    def get_custom(self, name: str) -> str:
        return self._get(CUSTOM_PREFIX + name)

    def set_custom(self, name: str, value: str) -> None:
        self._set(CUSTOM_PREFIX + name, value)

    def delete_custom(self, name: str) -> None:
        self._delete(CUSTOM_PREFIX + name)

    def save(self) -> None:
        _check_header(self.audio, self.path)
        self.audio.save()


# ─── MP4 atoms (M4A) ────────────────────────────────────────

_MP4_ATOM = {"genre": "\xa9gen", "album": "\xa9alb", "year": "\xa9day", "comment": "\xa9cmt",
             "artist": "\xa9ART", "title": "\xa9nam"}
_MP4_FREEFORM = {"key": "initialkey", "detail": CUSTOM_PREFIX + "DETAIL"}
_MP4_MEAN = "----:com.apple.iTunes:"


class _MP4File(TagFile):
    def __init__(self, path: str):
        super().__init__(path)
        self.audio = MP4(path)
        if self.audio.tags is None:
            self.audio.add_tags()
        self.tags = self.audio.tags

    def _ff_key(self, name: str) -> str:
        """The stored freeform key for *name*, matching case-insensitively
        (writers disagree on `initialkey` vs `INITIALKEY`)."""
        want = (_MP4_MEAN + name).lower()
        for k in self.tags.keys():
            if k.lower() == want:
                return k
        return _MP4_MEAN + name

    def _get_ff(self, name: str) -> str:
        vals = self.tags.get(self._ff_key(name))
        return bytes(vals[0]).decode("utf-8", "replace") if vals else ""

    def _set_ff(self, name: str, value: str) -> None:
        self._delete_ff(name)
        self.tags[_MP4_MEAN + name] = [
            MP4FreeForm(value.encode("utf-8"), dataformat=AtomDataType.UTF8)
        ]

    def _delete_ff(self, name: str) -> None:
        key = self._ff_key(name)
        if key in self.tags:
            del self.tags[key]

    def get(self, field: str) -> str:
        if field == "bpm":
            vals = self.tags.get("tmpo")
            return str(vals[0]) if vals else ""
        if field in _MP4_FREEFORM:
            val = self._get_ff(_MP4_FREEFORM[field])
            return val if field == "detail" else val.strip()
        vals = self.tags.get(_MP4_ATOM[field])
        val = str(vals[0]) if vals else ""
        return val if field == "comment" else val.strip()

    def set(self, field: str, value: str) -> None:
        if field == "bpm":
            self.tags["tmpo"] = [int(round(float(value)))]
        elif field in _MP4_FREEFORM:
            self._set_ff(_MP4_FREEFORM[field], value)
        else:
            # (mutagen already reads a legacy numeric `gnre` atom as \xa9gen
            # and writes it back as \xa9gen, so there is no duplicate to clear.)
            self.tags[_MP4_ATOM[field]] = [value]

    def delete(self, field: str) -> None:
        if field == "bpm":
            self.tags.pop("tmpo", None)
        elif field in _MP4_FREEFORM:
            self._delete_ff(_MP4_FREEFORM[field])
        else:
            self.tags.pop(_MP4_ATOM[field], None)

    def get_custom(self, name: str) -> str:
        return self._get_ff(CUSTOM_PREFIX + name)

    def set_custom(self, name: str, value: str) -> None:
        self._set_ff(CUSTOM_PREFIX + name, value)

    def delete_custom(self, name: str) -> None:
        self._delete_ff(CUSTOM_PREFIX + name)

    def save(self) -> None:
        _check_header(self.audio, self.path)
        self.audio.save()
