"""Optional archive of Frigate media kept outside the normal retention.

The owner's nightly job copies ``/media/frigate`` onto another disk and
keeps that layout: ``recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4`` (UTC),
``clips/<camera>-<event id>.jpg``, and ``recap/<camera>/<id>/`` for
Frigate-built recaps. It also writes ``index/YYYY-MM-DD.json`` (rows with
id, camera, label, start, end, ``paths`` such as ``events/<review id>.mp4``,
and a snapshot) plus a dated copy of the Frigate database.

Recap uses the archive only when ``recap.archive.path`` or
``recap.archive.url`` is set. Live files win when they are still present.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Shown when an index row exists but the file is gone, including when it
# disappears between lookup and the moment we open it.
CLIP_GONE = "This clip is no longer archived"

# A recording file is included when it starts before the event ends and
# its start is within this many seconds of the event. Frigate segments
# are about 10 seconds. The extra room covers a segment that began earlier.
SEGMENT_SPAN = 60.0
# Events stay in Frigate for about two weeks, so a dated DB from the
# night of the event through this many later nights can still name it.
DB_LOOKAHEAD_DAYS = 16
INDEX_NAMES = (
    "index/{day}.json",
    "index/{day}.jsonl",
    "index/{day}.ndjson",
    "index/{day}.csv",
    "{day}/index.json",
    "{day}/index.jsonl",
    "{day}/index.csv",
)
DB_NAMES = (
    "db/frigate-{day}.db",
    "db/{day}.db",
    "db/{day}/frigate.db",
    "databases/frigate-{day}.db",
    "frigate-{day}.db",
)
UNDATED_DBS = ("db/frigate.db", "frigate.db")

_cache: dict[str, tuple[float, Any]] = {}


def clear_archive_cache() -> None:
    """Drop cached index parses. Tests call this between cases."""
    _cache.clear()


@dataclass
class ArchiveLocation:
    """One configured archive, as a mount, a URL, or both."""

    path: Path | None = None
    url: str | None = None


@dataclass
class ArchiveSegment:
    """One recording file covering part of an event."""

    path: Path | None
    url: str | None
    start: float | None
    end: float | None


@dataclass
class ArchivePlayback:
    """Where to play an event that is no longer in Frigate."""

    camera: str | None
    label: str | None
    start: float | None
    end: float | None
    segments: list[ArchiveSegment] = field(default_factory=list)
    snapshot: Path | None = None
    snapshot_url: str | None = None
    message: str = ""


def archive_locations(config: Any) -> list[ArchiveLocation]:
    """Unique archives from the global recap config and camera overrides."""
    found: list[ArchiveLocation] = []
    seen: set[tuple[str | None, str | None]] = set()
    blocks = [config.recap.archive]
    blocks.extend(camera.recap.archive for camera in config.cameras.values())
    for block in blocks:
        path = (block.path or "").strip() or None
        url = (block.url or "").strip() or None
        key = (path, url)
        if key == (None, None) or key in seen:
            continue
        seen.add(key)
        found.append(ArchiveLocation(path=Path(path) if path else None, url=url))
    return found


def archive_recap_dirs(locations: list[ArchiveLocation]) -> list[Path]:
    """``recap/`` directories that exist on mounted archives."""
    dirs: list[Path] = []
    for location in locations:
        if location.path is None:
            continue
        candidate = location.path / "recap"
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def event_start_from_id(event_id: str) -> float | None:
    """Frigate event ids start with the unix time the object was first seen."""
    head, separator, _tail = event_id.rpartition("-")
    if not separator or not head:
        return None
    try:
        return float(head)
    except ValueError:
        return None


def candidate_days(moment: float) -> list[str]:
    """UTC dates that can hold the index row for a timestamp."""
    base = datetime.fromtimestamp(moment, timezone.utc)
    days: list[str] = []
    for delta in (0, 1, -1):
        day = (base + timedelta(days=delta)).date().isoformat()
        if day not in days:
            days.append(day)
    return days


def _clean_relative(value: str) -> str | None:
    """Keep a path that stays inside the archive, including live media paths."""
    text = value.strip()
    if not text:
        return None
    rewritten = rewrite_media_path(text)
    if rewritten and _relative_ok(rewritten):
        return rewritten
    if _relative_ok(text):
        return text
    return None


def _relative_ok(relative: str) -> bool:
    text = relative.strip().replace("\\", "/")
    if not text or text.startswith("/"):
        return False
    return ".." not in text.split("/")


def safe_archive_path(root: Path, relative: str) -> Path | None:
    """Join a relative archive path and reject anything that escapes root."""
    text = relative.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not _relative_ok(text):
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / text).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def rewrite_media_path(stored: str) -> str | None:
    """Turn a live Frigate path into a path relative to the archive root.

    Dated database copies still contain paths such as
    ``/media/frigate/recordings/2026-10-07/15/front/50.00.mp4``.
    """
    text = stored.replace("\\", "/")
    for marker in ("/recordings/", "/clips/", "/recap/", "/snapshots/", "/events/"):
        index = text.find(marker)
        if index >= 0:
            return text[index + 1 :]
    if text.startswith(("recordings/", "clips/", "recap/", "snapshots/", "events/")):
        return text
    return None


def parse_index_text(text: str) -> list[dict[str, Any]]:
    """Read a per-day index as JSON, JSON lines, or CSV."""
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return _rows_from_json(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    rows: list[dict[str, Any]] = []
    jsonl_failed = False
    for line in stripped.splitlines():
        piece = line.strip()
        if not piece:
            continue
        try:
            parsed = json.loads(piece)
        except json.JSONDecodeError:
            jsonl_failed = True
            break
        if isinstance(parsed, dict):
            rows.append(parsed)
    if rows and not jsonl_failed:
        return rows
    header = stripped.splitlines()[0].lower()
    if "id" in header and ("," in header or ";" in header):
        return _rows_from_csv(stripped)
    return []


def lookup_playback(
    locations: list[ArchiveLocation], event_id: str
) -> ArchivePlayback | None:
    """Find an archived event and the files that can still play it."""
    if not locations or not event_id:
        return None
    for location in locations:
        hit = _lookup_one(location, event_id)
        if hit is not None:
            return hit
    return None


def present_local_segments(
    segments: list[ArchiveSegment],
) -> list[ArchiveSegment] | None:
    """Local segments whose files still exist, checked at call time.

    Returns None when any local file is missing, including a file deleted
    after lookup. Callers should answer 410 with ``CLIP_GONE`` and must
    not open the path. URL-only segments are omitted. An empty list means
    there is nothing on disk to stream.
    """
    local = [segment for segment in segments if segment.path is not None]
    for segment in local:
        path = segment.path
        try:
            ready = path is not None and path.is_file()
        except OSError:
            ready = False
        if not ready:
            return None
    return local


def remote_recap_summaries(locations: list[ArchiveLocation]) -> list[dict[str, Any]]:
    """Summaries published at ``<url>/recap/index.json`` when no disk is mounted."""
    summaries: list[dict[str, Any]] = []
    for location in locations:
        if not location.url or location.path is not None:
            continue
        payload = _fetch_text(f"{location.url}/recap/index.json")
        if not payload:
            continue
        for row in parse_index_text(payload):
            recap_id = str(row.get("id") or "")
            camera = str(row.get("camera") or "")
            if not recap_id or not camera:
                continue
            summaries.append(
                {
                    "id": recap_id,
                    "camera": camera,
                    "status": row.get("status") or "complete",
                    "progress": row.get("progress", 100),
                    "message": row.get("message") or "",
                    "created": row.get("created"),
                    "after": row.get("after"),
                    "before": row.get("before"),
                    "seconds": row.get("seconds"),
                    "event_count": row.get("event_count", 0),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "source": "archive",
                }
            )
    return summaries


def remote_manifest(
    locations: list[ArchiveLocation], recap_id: str
) -> tuple[str, dict[str, Any]] | None:
    """Fetch a recap manifest from an archive URL. Returns ``(base url, manifest)``."""
    for location in locations:
        if not location.url:
            continue
        if location.path is not None:
            continue
        payload = _fetch_text(f"{location.url}/recap/index.json")
        camera = ""
        embedded: dict[str, Any] | None = None
        if payload:
            for row in parse_index_text(payload):
                if str(row.get("id") or "") == recap_id:
                    camera = str(row.get("camera") or "")
                    if row.get("tracks"):
                        embedded = row
                    break
        if embedded is not None:
            return location.url, embedded
        if not camera:
            continue
        body = _fetch_text(f"{location.url}/recap/{camera}/{recap_id}/manifest.json")
        if not body:
            continue
        try:
            manifest = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(manifest, dict):
            return location.url, manifest
    return None


def recordings_covering(
    root: Path, camera: str, start: float, end: float
) -> list[ArchiveSegment]:
    """Recording segments in the archive whose UTC names overlap ``[start, end]``."""
    if end <= start or not camera:
        return []
    earliest = start - SEGMENT_SPAN
    begin = datetime.fromtimestamp(earliest, timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    finish = datetime.fromtimestamp(end, timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    found: list[ArchiveSegment] = []
    cursor = begin
    while cursor <= finish:
        day = cursor.strftime("%Y-%m-%d")
        hour = cursor.strftime("%H")
        directory = root / "recordings" / day / hour / camera
        if directory.is_dir():
            for file in directory.glob("*.mp4"):
                parsed = _recording_start(day, hour, file.name)
                if parsed is None:
                    continue
                if parsed < end and parsed + SEGMENT_SPAN > start:
                    found.append(
                        ArchiveSegment(
                            path=file, url=None, start=parsed, end=parsed + 10.0
                        )
                    )
        cursor += timedelta(hours=1)
    found.sort(key=lambda segment: segment.start or 0)
    return found


def _lookup_one(location: ArchiveLocation, event_id: str) -> ArchivePlayback | None:
    moment = event_start_from_id(event_id)
    days = candidate_days(moment) if moment is not None else []
    row = _index_row(location, event_id, days)
    start = moment
    end: float | None = None
    camera = ""
    label = ""
    relative_paths: list[str] = []
    snapshot_rel = ""
    if row is not None:
        camera = str(row.get("camera") or "")
        label = str(row.get("label") or "")
        start = row.get("start") if row.get("start") is not None else start
        end = row.get("end")
        relative_paths = list(row.get("paths") or [])
        snapshot_rel = str(row.get("snapshot") or "")
    if location.path is not None and (not relative_paths or not camera or end is None):
        db_row = _db_row(location.path, event_id, days)
        if db_row is not None:
            camera = camera or str(db_row.get("camera") or "")
            label = label or str(db_row.get("label") or "")
            if start is None:
                start = db_row.get("start")
            if end is None:
                end = db_row.get("end")
            if not relative_paths:
                relative_paths = list(db_row.get("paths") or [])
            if not snapshot_rel:
                snapshot_rel = str(db_row.get("snapshot") or "")
    if not camera and not relative_paths and location.path is None:
        return None
    segments = _segments_for(location, relative_paths, camera, start, end)
    snapshot, snapshot_url = _snapshot_for(location, snapshot_rel, camera, event_id)
    if not segments and snapshot is None and snapshot_url is None:
        if row is None and not camera:
            return None
        return ArchivePlayback(
            camera=camera or None,
            label=label or None,
            start=start,
            end=end,
            message=CLIP_GONE,
        )
    message = ""
    if not segments:
        message = CLIP_GONE
    return ArchivePlayback(
        camera=camera or None,
        label=label or None,
        start=start,
        end=end,
        segments=segments,
        snapshot=snapshot,
        snapshot_url=snapshot_url,
        message=message,
    )


def _segments_for(
    location: ArchiveLocation,
    relative_paths: list[str],
    camera: str,
    start: float | None,
    end: float | None,
) -> list[ArchiveSegment]:
    segments: list[ArchiveSegment] = []
    if location.path is not None and relative_paths:
        for relative in relative_paths:
            file = safe_archive_path(location.path, relative)
            if file is not None and file.is_file():
                parsed = _start_from_recording_path(file)
                segments.append(
                    ArchiveSegment(
                        path=file,
                        url=None,
                        start=parsed,
                        end=None if parsed is None else parsed + 10.0,
                    )
                )
    if segments:
        return segments
    if location.path is not None and camera and start is not None and end is not None:
        covered = recordings_covering(location.path, camera, float(start), float(end))
        if covered:
            return covered
    if location.url and len(relative_paths) == 1 and _relative_ok(relative_paths[0]):
        return [
            ArchiveSegment(
                path=None,
                url=_join_url(location.url, relative_paths[0]),
                start=start,
                end=end,
            )
        ]
    return []


def _snapshot_for(
    location: ArchiveLocation,
    relative: str,
    camera: str,
    event_id: str,
) -> tuple[Path | None, str | None]:
    names = []
    if relative:
        names.append(relative)
    if camera and event_id:
        names.append(f"clips/{camera}-{event_id}.jpg")
        names.append(f"clips/{camera}-{event_id}-clean.webp")
    if location.path is not None:
        for name in names:
            file = safe_archive_path(location.path, name)
            if file is not None and file.is_file():
                return file, None
    if location.url and names:
        name = names[0]
        if _relative_ok(name):
            return None, _join_url(location.url, name)
    return None, None


def _index_row(
    location: ArchiveLocation, event_id: str, days: list[str]
) -> dict[str, Any] | None:
    for day in days:
        for row in _day_rows(location, day):
            if str(row.get("id") or "") == event_id:
                return row
    return None


def _day_rows(location: ArchiveLocation, day: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if location.path is not None:
        for pattern in INDEX_NAMES:
            file = location.path / pattern.format(day=day)
            if not file.is_file():
                continue
            cached = _cached_file(file)
            if cached is not None:
                rows.extend(cached)
                break
    elif location.url:
        for pattern in INDEX_NAMES:
            if not pattern.endswith((".json", ".jsonl", ".csv")):
                continue
            text = _fetch_text(_join_url(location.url, pattern.format(day=day)))
            if text:
                rows.extend(_normalize_rows(parse_index_text(text)))
                break
    return rows


def _cached_file(file: Path) -> list[dict[str, Any]] | None:
    try:
        modified = file.stat().st_mtime
    except OSError:
        return None
    key = str(file)
    cached = _cache.get(key)
    if cached is not None and cached[0] == modified:
        return cached[1]
    try:
        text = file.read_text(encoding="utf-8")
        rows = _normalize_rows(parse_index_text(text))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        # Remember the failure until the file changes so a bad index
        # does not raise, and does not get re-read on every clip.
        logger.warning("Could not read archive index %s", file)
        _cache[key] = (modified, [])
        return []
    _cache[key] = (modified, rows)
    return rows


def _db_row(root: Path, event_id: str, days: list[str]) -> dict[str, Any] | None:
    moment = event_start_from_id(event_id)
    search_days = list(days)
    if moment is not None:
        base = datetime.fromtimestamp(moment, timezone.utc).date()
        for offset in range(DB_LOOKAHEAD_DAYS + 1):
            day = (base + timedelta(days=offset)).isoformat()
            if day not in search_days:
                search_days.append(day)
    paths = [root / name.format(day=day) for day in search_days for name in DB_NAMES]
    paths.extend(root / name for name in UNDATED_DBS)
    best: dict[str, Any] | None = None
    for path in paths:
        if not path.is_file():
            continue
        row = _read_db(path, event_id)
        if row is None:
            continue
        # Prefer a copy that still names recording files.
        if row.get("paths"):
            return row
        best = row
    return best


def _read_db(path: Path, event_id: str) -> dict[str, Any] | None:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        cursor = connection.execute(
            "SELECT camera, label, start_time, end_time FROM event WHERE id = ?",
            (event_id,),
        )
        event = cursor.fetchone()
        if event is None:
            return None
        camera, label, start, end = event
        start_epoch = _as_epoch(start)
        end_epoch = _as_epoch(end)
        paths: list[str] = []
        if camera and start_epoch is not None and end_epoch is not None:
            recordings = connection.execute(
                """
                SELECT path FROM recordings
                WHERE camera = ? AND start_time < ? AND end_time > ?
                ORDER BY start_time ASC
                """,
                (camera, end_epoch, start_epoch),
            )
            for (stored,) in recordings:
                relative = rewrite_media_path(str(stored))
                if relative:
                    paths.append(relative)
        return {
            "id": event_id,
            "camera": camera,
            "label": label,
            "start": start_epoch,
            "end": end_epoch,
            "paths": paths,
            "snapshot": "",
        }
    except sqlite3.Error:
        logger.debug("Could not read archived database %s", path)
        return None
    finally:
        connection.close()


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_row(row)
        if item is not None:
            normalized.append(item)
    return normalized


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    event_id = row.get("id") or row.get("event_id") or row.get("event")
    if not event_id:
        return None
    paths: list[str] = []
    for key in ("paths", "files", "recordings", "clip", "clip_path"):
        for item in _path_values(row.get(key)):
            relative = _clean_relative(item)
            if relative:
                paths.append(relative)
    snapshot = row.get("snapshot") or row.get("snapshot_path") or ""
    if isinstance(snapshot, list):
        snapshot = snapshot[0] if snapshot else ""
    snapshot_rel = _clean_relative(str(snapshot or "")) or ""
    start = row.get("start")
    if start is None:
        start = row.get("start_time")
    end = row.get("end")
    if end is None:
        end = row.get("end_time")
    return {
        "id": str(event_id),
        "camera": str(row.get("camera") or row.get("camera_name") or ""),
        "label": str(row.get("label") or row.get("object") or ""),
        "start": _as_epoch(start),
        "end": _as_epoch(end),
        "paths": paths,
        "snapshot": snapshot_rel,
    }


def _path_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                return _path_values(json.loads(text))
            except json.JSONDecodeError:
                return []
        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]
        return [text]
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            if isinstance(item, str):
                found.extend(_path_values(item))
            elif isinstance(item, dict):
                found.extend(_path_values(item.get("path") or item.get("file")))
        return found
    return []


def _rows_from_json(data: object) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("events", "items", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    if data.get("id") or data.get("event_id") or data.get("event"):
        return [data]
    rows: list[dict[str, Any]] = []
    markers = {"camera", "label", "paths", "files", "start", "start_time"}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if not markers.intersection(value):
            continue
        copied = dict(value)
        copied.setdefault("id", key)
        rows.append(copied)
    return rows


def _rows_from_csv(text: str) -> list[dict[str, Any]]:
    sample = text.splitlines()[0]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def _as_epoch(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text.replace("Z", ""), pattern)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    return None


def _recording_start(day: str, hour: str, name: str) -> float | None:
    stem = name[:-4] if name.endswith(".mp4") else name
    parts = stem.split(".")
    if len(parts) != 2:
        return None
    try:
        minute = int(parts[0])
        second = int(parts[1])
    except ValueError:
        return None
    try:
        moment = datetime.strptime(
            f"{day} {hour}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return None
    return moment.replace(tzinfo=timezone.utc).timestamp()


def _start_from_recording_path(path: Path) -> float | None:
    parts = path.parts
    # .../recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
    if len(parts) < 4:
        return None
    name = parts[-1]
    hour = parts[-3]
    day = parts[-4]
    if len(hour) != 2 or len(day) != 10:
        return None
    return _recording_start(day, hour, name)


def _join_url(base: str, relative: str) -> str:
    return base.rstrip("/") + "/" + relative.lstrip("/")


def _fetch_text(url: str) -> str | None:
    cached = _cache.get(url)
    now = time.time()
    if cached is not None and now - cached[0] < 600:
        return cached[1]
    request = urllib.request.Request(
        url, headers={"Accept": "application/json,text/plain"}
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = response.read(2_000_000)
    except Exception:
        logger.debug("Archive request failed for %s", url)
        _cache[url] = (now, None)
        return None
    text = payload.decode("utf-8", "replace")
    _cache[url] = (now, text)
    return text
