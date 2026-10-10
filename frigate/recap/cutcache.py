"""Per-event cutout cache, so a refresh only cuts out new events.

A rolling recap is rebuilt every half hour over a window that mostly
overlaps the last build. Cutting an object out of its recording is the
slow part, so the finished ghost frames for each event are kept on disk
and reused. The key covers everything that changes the pixels: the
event id and end time, the output size, the sample rate, the per-object
time cap, the category, the camera's time shift, and ``CACHE_VERSION``.
Entries not used for ``MAX_AGE_HOURS`` are deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from frigate.const import RECAP_DIR

logger = logging.getLogger(__name__)

# Bump when cutout, sampling, or the stored layout changes.
# 2: path boxes aligned to the recording, clip window around the best view,
# window ghosts for infrared and dusk.
# 3: no empty windows inside a moving track, vehicle masks without the road
# halo, clips cut at retention holes.
CACHE_VERSION = 3
# Longest window a cached cutout is useful for (72 hours) plus slack.
MAX_AGE_HOURS = 74.0
# A failed cutout is only cached once the recording is surely written.
NEGATIVE_SETTLE_SECONDS = 600.0


def cache_root() -> Path:
    return Path(RECAP_DIR) / ".cutcache"


def cache_key(
    event_id: str,
    end_time: float | None,
    width: int,
    height: int,
    sample_fps: float,
    max_object_seconds: float,
    category: str,
    time_shift: float = 0.0,
) -> str:
    """Stable digest for one event's cutouts.

    ``end_time`` is part of the key so an event that was still open at the
    last build is cut again once it ends. Vehicles and other objects use
    different sample rates and cutout rules, so ``category`` is included.
    """
    payload = json.dumps(
        [
            CACHE_VERSION,
            str(event_id),
            None if end_time is None else round(float(end_time), 2),
            int(width),
            int(height),
            round(float(sample_fps), 3),
            round(float(max_object_seconds), 3),
            str(category),
            round(float(time_shift), 3),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedCutout:
    """What the cutout loop needs to skip decoding an event."""

    status: str  # "ok", "no_frames", or "no_cutout"
    frames: list[dict[str, Any]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    context_first: bytes | None = None
    context_last: bytes | None = None
    still: bool = False


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stored: int = 0


def _path(key: str, root: Path | None = None) -> Path:
    base = root or cache_root()
    return base / key[:2] / f"{key}.pkl"


def load(key: str, root: Path | None = None) -> CachedCutout | None:
    path = _path(key, root)
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        data = pickle.loads(zlib.decompress(raw))
    except Exception:
        logger.debug("Dropping unreadable recap cache entry %s", path.name)
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if not isinstance(data, dict) or data.get("v") != CACHE_VERSION:
        return None
    try:
        os.utime(path, None)
    except OSError:
        pass
    return CachedCutout(
        status=str(data.get("status") or ""),
        frames=list(data.get("frames") or []),
        boxes=[tuple(box) for box in data.get("boxes") or []],
        times=[float(item) for item in data.get("times") or []],
        context_first=data.get("context_first"),
        context_last=data.get("context_last"),
        still=bool(data.get("still")),
    )


def store(key: str, entry: CachedCutout, root: Path | None = None) -> bool:
    path = _path(key, root)
    payload = {
        "v": CACHE_VERSION,
        "status": entry.status,
        "frames": entry.frames,
        "boxes": [tuple(float(v) for v in box) for box in entry.boxes],
        "times": [float(item) for item in entry.times],
        "context_first": entry.context_first,
        "context_last": entry.context_last,
        "still": bool(entry.still),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(zlib.compress(pickle.dumps(payload, protocol=4), 3))
        os.replace(temporary, path)
    except OSError:
        logger.debug("Could not write recap cache entry %s", path.name)
        return False
    return True


def should_cache_failure(end_time: float | None, now: float | None = None) -> bool:
    """A missing recording may still be on its way for a recent event."""
    if end_time is None:
        return False
    moment = time.time() if now is None else now
    return moment - float(end_time) >= NEGATIVE_SETTLE_SECONDS


def pack_frames(ghost_frames: list[Any]) -> list[dict[str, Any]]:
    """Packed GhostFrame fields. The alpha mask is compressed again on store."""
    packed: list[dict[str, Any]] = []
    for ghost in ghost_frames:
        if not ghost.jpeg or not ghost.alpha_bytes or not ghost.alpha_shape:
            continue
        packed.append(
            {
                "x": int(ghost.x),
                "y": int(ghost.y),
                "box": tuple(float(v) for v in ghost.box),
                "jpeg": ghost.jpeg,
                "alpha_shape": tuple(int(v) for v in ghost.alpha_shape),
                "alpha_bytes": ghost.alpha_bytes,
                "window": bool(getattr(ghost, "window", False)),
            }
        )
    return packed


def purge(
    max_age_hours: float = MAX_AGE_HOURS,
    root: Path | None = None,
    now: float | None = None,
) -> tuple[int, int, int]:
    """Delete entries unused for ``max_age_hours``.

    Returns ``(removed, kept, kept_bytes)`` so the caller can log the size.
    """
    base = root or cache_root()
    moment = time.time() if now is None else now
    cutoff = moment - max_age_hours * 3600
    removed = kept = kept_bytes = 0
    if not base.is_dir():
        return 0, 0, 0
    for path in base.glob("*/*"):
        try:
            info = path.stat()
        except OSError:
            continue
        if path.suffix == ".tmp" and info.st_mtime < moment - 3600:
            path.unlink(missing_ok=True)
            continue
        if path.suffix != ".pkl":
            continue
        if info.st_mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
            continue
        kept += 1
        kept_bytes += info.st_size
    return removed, kept, kept_bytes
