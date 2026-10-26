# ML genre quality: wide incoherent mixes (fixed 2026-10-26)

Status: **fixed and applied to the whole library.** Code in
`genres.resolve_metadata` and `tagger._merge_genres`; regression test
`tests/test_genres.py::test_near_chance_electronic_head_ignored`.

## Symptom

About 400 tracks have no Beatport or Last.fm match, so their genre comes from
the ML models alone (`GENRE_SOURCE=ml`). Many of them got a wide, incoherent
genre list such as `ambient; house; dnb; Electro House`. Before the fix,
essentially every ML-source track carried 4+ genres, and `ambient` was the
primary genre on 72 of them, on dance tracks that are nothing like ambient.

## Root causes (three, found in sequence)

1. **A flat electronic head was trusted.** Besides the Discogs-400 head, the
   resolver uses a 5-class electronic head (ambient / dnb / house / techno /
   trance). Its labels were prepended ahead of the Discogs labels whenever the
   top probability passed `> 0.20`. With 5 classes, 0.20 is exactly chance,
   so the gate fired even when the head was uncertain, and its noise led the
   list ahead of the coherent Discogs labels.

2. **Same-source re-tags merged instead of replacing.** The first fix
   (August: lead only with top >= 0.30, or >= 0.20 with a 0.08 lead) was
   applied with `tag --retag-source ml`, but the genres stayed wide.
   `_merge_genres` treated the old ML genre as an existing value and merged
   the new, clean ML result into it, so the old pollution survived. Fix
   (commit 541ed0f): when the existing `GENRE_SOURCE` equals the new source,
   the new value replaces the old one. Cross-source merges (a user or other
   tool's genre, or a tier upgrade ml → beatport) are unchanged.

3. **The August gate was still too lenient for 5 classes.** A full-library
   verification found 6 ML tracks still with `ambient; dnb` in the lead, all
   just past the 0.08 lead (Faithless - We Come 1 (MANNA Remix): ambient
   0.28 vs dnb 0.20, a 0.084 lead; Eric Prydz - Generate (Kydus Remix):
   0.080). A top of ~0.27 is barely above a coin flip for a 5-class head.
   Fix (commit c5bc8b6): the head may lead only when **top >= 0.40** (twice
   chance) **and** it beats the runner-up by **>= 0.10**.

## Applying it

`djtagger tag "<lib>" --retag-source ml` re-resolved all 407 ML tracks
(about 10 minutes). Side benefit: 26 of them now matched on Beatport and were
upgraded to curated genres. Result: 0 ML tracks with a wide genre mix, and
`ambient` is gone as a false primary. `scripts/verify_library.py` reported
3,690/3,690 tracks OK on 2026-10-26.

## Do not use `--force` for this

`tag --force` re-runs the full Beatport → Last.fm → ML chain on every track.
If a Beatport lookup fails transiently (timeout, rate limit), that track falls
through to ML, and the ML genres are then **merged into its curated Beatport
genre** (different source, so the merge rule applies). Prefer:

- `--retag-source ml` to re-resolve only ML-source tracks, or
- `--upgrade` to backfill new analysis fields while keeping resolved genres
  (no network at all).

## Lessons

- **Verify the tags, not the run summary.** The first retag reported 407
  tagged / 0 failed and still left every genre wide. Only reading the actual
  tags back caught it (root cause 2), and a later full-library check caught
  root cause 3. That check is now `scripts/verify_library.py`.
- **Calibrate a confidence gate against the head's class count.** A
  threshold that is reasonable for a 400-class head is chance level for a
  5-class head.
