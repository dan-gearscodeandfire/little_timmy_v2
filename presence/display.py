"""Canonical -> spoken/UI display names (auto-suffix aware).

The expo duplicate-name mechanism (2026-07-16) mints canonical forks like
``mike_2`` for a second, biometrically-different "Mike". The fork is an
internal disambiguator (facts subjects, .npy files, Postgres rows stay
distinct) and must NEVER be spoken or shown as the person's name. The
id-map's ``_meta`` section records which names are auto-suffixed forks and
of what base — display strips the suffix ONLY on a meta hit, so a genuine
underscored name (``dan_the_barbarian``) or the ``{name}#{id}.retired``
rows minted by ``db/speakers.py`` are never mangled by string parsing.

Import-light on purpose (no encoders — unlike ``presence.face_identifier``):
reads the shared id-map JSON directly with an mtime-keyed cache; falls back
to plain de-slugging when the map is unavailable (hermetic tests, offline
scripts).
"""

import logging
import os
from pathlib import Path

from presence import creators
from presence.prototype_base import IdMap

log = logging.getLogger(__name__)

# Same shared id-map as speaker/identifier.py and presence/face_identifier.py
# (one speaker_id space across both biometrics).
_SHARED_ID_MAP = Path(os.path.expanduser(
    "~/little_timmy/models/speaker/_id_map.json"))
_RESERVED_IDS = {"dan": 1, "timmy": 2}
_FIRST_FREE_ID = 3

# (id-map mtime, decoded meta) — refreshed whenever the file changes. A stat()
# per lookup is microseconds; no TTL games needed.
_cache: tuple[float, dict] | None = None


def _meta() -> dict:
    """Auto-suffix meta (``name -> {"base", "at"}``) from the shared id-map."""
    global _cache
    try:
        mtime = _SHARED_ID_MAP.stat().st_mtime
    except OSError:
        return {}
    if _cache is not None and _cache[0] == mtime:
        return _cache[1]
    try:
        meta = IdMap(_SHARED_ID_MAP, reserved_ids=_RESERVED_IDS,
                     first_free_id=_FIRST_FREE_ID).meta()
    except Exception:
        log.exception("Failed to read id-map meta; display falls back to de-slug")
        return {}
    _cache = (mtime, meta)
    return meta


def display_base(canonical: str) -> str:
    """Canonical lowercase base for comparisons: the auto-suffix marker's base
    when one exists (``mike_2`` -> ``mike``), else the canonical itself."""
    clean = (canonical or "").strip().lower()
    info = _meta().get(clean)
    return info["base"] if info else clean


def display_name(canonical: str) -> str:
    """Spoken/UI form: auto-suffix stripped (meta-driven only), creators
    registry wins, else de-slugged + title-cased (``mike_2`` -> ``Mike``,
    ``william_osman`` -> ``William Osman``)."""
    return creators.display_name(display_base(canonical))
