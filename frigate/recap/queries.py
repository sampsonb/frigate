"""Load Frigate events and recordings for a recap window.

Real 0.17.2 rows leave several columns NULL. ``false_positive`` is never
written, so it is unset rather than false. An event that is still in
progress has ``end_time`` NULL. SQLite ``=`` does not match NULL, which
used to drop every object. Review segments are not queried here.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from frigate.models import Event, Recordings

logger = logging.getLogger(__name__)


def load_events(
    camera: str, after: float, before: float, labels: list[str]
) -> list[dict[str, Any]]:
    """Events for one camera that overlap ``[after, before]``.

    NULL ``false_positive`` counts as not a false positive. NULL
    ``has_clip`` counts as unknown and is kept; an explicit false is
    dropped. An in-progress event (NULL ``end_time``) runs through
    ``before``. ``sub_label``, ``top_score``, ``zones``, and
    ``has_snapshot`` are not filters. ``data`` is always a dict.
    """
    if not labels:
        return []
    query = (
        Event.select()
        .where(Event.camera == camera)
        .where(Event.label.in_(labels))
        .where(
            (Event.has_clip.is_null(True)) | (Event.has_clip == True)  # noqa: E712
        )
        .where(
            (Event.false_positive.is_null(True)) | (Event.false_positive == False)  # noqa: E712
        )
        .where(Event.start_time < before)
        .where((Event.end_time.is_null(True)) | (Event.end_time > after))
        .order_by(Event.start_time.asc())
    )
    events: list[dict[str, Any]] = []
    for event in query:
        row = _event_row(event, before)
        if row is not None:
            events.append(row)
    return events


def load_recordings(camera: str, after: float, before: float) -> list[dict[str, Any]]:
    """Recording files overlapping the window.

    A segment with NULL ``end_time`` is treated as still open and is
    closed at ``before`` so later math always sees a number.
    """
    query = (
        Recordings.select(Recordings.path, Recordings.start_time, Recordings.end_time)
        .where(Recordings.camera == camera)
        .where(Recordings.start_time < before)
        .where((Recordings.end_time.is_null(True)) | (Recordings.end_time > after))
        .order_by(Recordings.start_time.asc())
    )
    rows: list[dict[str, Any]] = []
    for row in query:
        if row.start_time is None or not row.path:
            continue
        try:
            start = float(row.start_time)
            end = float(before if row.end_time is None else row.end_time)
        except (TypeError, ValueError):
            logger.debug("Skipped recording %s with unreadable times", row.path)
            continue
        rows.append({"path": row.path, "start": start, "end": end})
    return rows


def recordings_overlap(
    camera: str, start: float, end: float, now: float | None = None
) -> bool:
    """True when a recording file overlaps ``[start, end]``.

    ``end`` may be None for an event that is still in progress. The
    search then runs through ``now``. A recording with NULL ``end_time``
    still counts.
    """
    if not camera:
        return False
    bound = now if now is not None else time.time()
    try:
        start_epoch = float(start)
        end_epoch = bound if end is None else float(end)
    except (TypeError, ValueError):
        return False
    return (
        Recordings.select()
        .where(Recordings.camera == camera)
        .where(Recordings.start_time < end_epoch)
        .where(
            (Recordings.end_time.is_null(True)) | (Recordings.end_time > start_epoch)
        )
        .exists()
    )


def live_recordings_exist(event: Any, now: float | None = None) -> bool:
    """True when this event should play from Frigate's recordings.

    Explicit ``has_clip`` false means the clip was removed. NULL means
    the flag was never written. NULL ``end_time`` means the event is
    still open.
    """
    if getattr(event, "has_clip", None) is False:
        return False
    camera = getattr(event, "camera", None)
    start = getattr(event, "start_time", None)
    if not camera or start is None:
        return False
    return recordings_overlap(str(camera), start, getattr(event, "end_time", None), now)


def _event_row(event: Any, before: float) -> dict[str, Any] | None:
    if event.start_time is None:
        return None
    try:
        start = float(event.start_time)
        end = float(before if event.end_time is None else event.end_time)
    except (TypeError, ValueError):
        logger.debug("Skipped event %s with unreadable times", event.id)
        return None
    if end <= start:
        end = start + 0.5
    return {
        "id": event.id,
        "label": event.label,
        "sub_label": event.sub_label,
        "start_time": start,
        "end_time": end,
        "data": _event_data(event.data),
    }


def _event_data(value: Any) -> dict[str, Any]:
    """JSON object from the event row, or an empty dict.

    The column is NULL when nothing was stored. A driver may also hand
    back the raw text. Lists and other shapes are ignored.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}
