"""Background recap jobs, the nightly schedule, and retention.

One job runs at a time. The worker thread lowers its priority so live
object detection keeps the CPU on a small machine. A second request
waits in a queue instead of starting a second encode.
"""

from __future__ import annotations

import json
import logging
import os
import random
import string
import threading
import time
from datetime import datetime
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from frigate.config import FrigateConfig
from frigate.config.recap import RecapConfig
from frigate.const import CACHE_DIR, RECAP_DIR
from frigate.models import Event, Recordings
from frigate.recap.categories import (
    DELIVERY_QUERIES_PERSON,
    DELIVERY_QUERIES_VEHICLE,
    DOG_WALKER_QUERY,
)
from frigate.recap.frames import (
    concat_sample,
    grab_frame,
    probe_size,
    scaled_size,
)
from frigate.recap.generate import (
    RecapCancelled,
    generate_recap,
    plate_timestamps,
    sample_times,
)
from frigate.recap.storage import (
    ensure_tree,
    purge_expired,
    read_manifest,
    update_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)


def resolve_zone(config: FrigateConfig):
    """UI timezone when one is set, otherwise the container's local zone."""
    name = config.ui.timezone
    if name:
        try:
            return ZoneInfo(name)
        except Exception:
            logger.warning(
                "Recap timezone %s is not recognized, using local time", name
            )
    return datetime.now().astimezone().tzinfo


class RecapManager:
    """Owns the queue. The API and the scheduler both call ``start``."""

    def __init__(self, config: FrigateConfig, embeddings: Any = None) -> None:
        self.config = config
        self.embeddings = embeddings
        self._lock = threading.Lock()
        self._running: RecapJob | None = None
        self._queue: list[RecapJob] = []
        self._jobs: dict[str, RecapJob] = {}
        self._stop = threading.Event()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel()
        for job in jobs:
            job.join(timeout=5)

    def start(
        self,
        camera: str,
        after: float,
        before: float,
        *,
        reason: str = "manual",
    ) -> dict[str, Any]:
        """Queue a recap. Raises ValueError when the camera or range is unusable."""
        camera_config = self.config.cameras.get(camera)
        if camera_config is None:
            raise ValueError(f"{camera} is not a camera")
        settings: RecapConfig = camera_config.recap
        if not settings.enabled:
            raise ValueError(f"Recap is not enabled for {camera}")
        if before <= after:
            raise ValueError("The end of the range has to be after the start")
        hours = (before - after) / 3600
        if hours > settings.max_window_hours:
            raise ValueError(
                f"That range is {hours:.1f} hours. The longest recap for {camera} is {settings.max_window_hours:g} hours"
            )
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        recap_id = f"{camera}_{stamp}_{suffix}"
        directory = ensure_tree(camera, recap_id)
        manifest = {
            "id": recap_id,
            "camera": camera,
            "status": "queued",
            "progress": 0,
            "message": "Waiting",
            "created": time.time(),
            "after": after,
            "before": before,
            "reason": reason,
            "tracks": [],
            "event_count": 0,
        }
        write_manifest(directory, manifest)
        job = RecapJob(self, recap_id, camera, after, before, settings, directory)
        with self._lock:
            self._jobs[recap_id] = job
            self._queue.append(job)
            self._pump_locked()
        return manifest

    def cancel(self, recap_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(recap_id)
            if job is None:
                return False
            if job in self._queue:
                self._queue.remove(job)
                self._jobs.pop(recap_id, None)
                update_manifest(
                    job.directory, status="cancelled", message="Cancelled", progress=0
                )
                return True
        job.cancel()
        return True

    def forget(self, recap_id: str) -> None:
        with self._lock:
            self._jobs.pop(recap_id, None)
            if self._running and self._running.recap_id == recap_id:
                self._running = None
            self._pump_locked()

    def _pump_locked(self) -> None:
        if self._running is not None or not self._queue or self._stop.is_set():
            return
        job = self._queue.pop(0)
        self._running = job
        job.start()

    def search_text(self, query: str) -> list[tuple[str, float]]:
        """Thumbnail semantic search, or nothing when embeddings are off."""
        if self.embeddings is None or not self.config.semantic_search.enabled:
            return []
        try:
            hits = self.embeddings.search_thumbnail(query)
        except Exception:
            logger.warning("Semantic search failed while building a recap")
            return []
        cleaned: list[tuple[str, float]] = []
        for item in hits or []:
            if len(item) < 2:
                continue
            cleaned.append((str(item[0]), float(item[1])))
        return cleaned


class RecapJob(threading.Thread):
    """One synopsis, from event query through ffmpeg."""

    def __init__(
        self,
        manager: RecapManager,
        recap_id: str,
        camera: str,
        after: float,
        before: float,
        settings: RecapConfig,
        directory: Path,
    ) -> None:
        super().__init__(name=f"recap-{recap_id}", daemon=True)
        self.manager = manager
        self.recap_id = recap_id
        self.camera = camera
        self.after = after
        self.before = before
        self.settings = settings
        self.directory = directory
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            os.nice(10)
        except OSError:
            pass
        try:
            self._run()
        except RecapCancelled:
            update_manifest(self.directory, status="cancelled", message="Cancelled")
            logger.info("Recap %s cancelled", self.recap_id)
        except Exception:
            logger.exception("Recap %s failed", self.recap_id)
            update_manifest(
                self.directory,
                status="failed",
                message="Recap failed. Check the Frigate logs",
            )
        finally:
            self.manager.forget(self.recap_id)

    def _progress(self, percent: float, message: str) -> None:
        update_manifest(
            self.directory,
            status="running",
            progress=int(percent),
            message=message,
        )

    def _run(self) -> None:
        update_manifest(
            self.directory, status="running", progress=1, message="Looking up events"
        )
        config = self.manager.config
        ffmpeg = config.ffmpeg.ffmpeg_path
        ffprobe = config.ffmpeg.ffprobe_path
        events = _load_events(
            self.camera, self.after, self.before, self.settings.labels
        )
        if self._cancel.is_set():
            raise RecapCancelled()
        self._progress(4, f"Found {len(events)} events")
        delivery_ids, dog_ids = self._semantic(events)
        rows = _recordings(self.camera, self.after, self.before)
        if not rows:
            raise RuntimeError("No recordings in that time range")
        size = None
        for row in rows:
            size = probe_size(ffprobe, row["path"])
            if size:
                break
        if size is None:
            raise RuntimeError("Could not read the recording size")
        width, height = scaled_size(size[0], size[1], self.settings.max_width)
        self._progress(6, "Building a background")
        plate_frames = self._plate(ffmpeg, rows, width, height)
        if self._cancel.is_set():
            raise RecapCancelled()

        def load_clip(
            event: dict[str, Any], sample_fps: float, out_w: int, out_h: int
        ) -> tuple[list, list[float]] | None:
            if self._cancel.is_set():
                raise RecapCancelled()
            segments = _segments(
                rows, float(event["start_time"]), float(event["end_time"])
            )
            if not segments:
                return None
            work = Path(CACHE_DIR)
            work.mkdir(parents=True, exist_ok=True)
            max_frames = max(4, int(self.settings.max_object_seconds * sample_fps) + 2)
            frames = concat_sample(
                ffmpeg,
                segments,
                out_w,
                out_h,
                sample_fps,
                max_frames,
                work,
            )
            if len(frames) < 3:
                return None
            times = sample_times(
                float(event["start_time"]), float(event["end_time"]), len(frames)
            )
            return frames, times

        body = generate_recap(
            camera=self.camera,
            after=self.after,
            before=self.before,
            settings=self.settings,
            zone=resolve_zone(config),
            events=events,
            load_clip=load_clip,
            plate_frames=plate_frames,
            out_dir=self.directory,
            ffmpeg=ffmpeg,
            cancel=self._cancel.is_set,
            progress=self._progress,
            delivery_ids=delivery_ids,
            dog_walker_ids=dog_ids,
            width=width,
            height=height,
        )
        current = read_manifest(self.directory) or {}
        current.update(body)
        current["id"] = self.recap_id
        current["status"] = "complete"
        current["progress"] = 100
        current["message"] = "Ready"
        write_manifest(self.directory, current)
        logger.info(
            "Recap %s ready, %s objects, %ss",
            self.recap_id,
            current.get("event_count"),
            current.get("seconds"),
        )

    def _semantic(self, events: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        if not self.settings.delivery_search:
            return set(), set()
        wanted = {event["id"] for event in events}
        delivery: set[str] = set()
        dogs: set[str] = set()
        cutoff = self.settings.delivery_distance
        for query in DELIVERY_QUERIES_VEHICLE + DELIVERY_QUERIES_PERSON:
            if self._cancel.is_set():
                raise RecapCancelled()
            for event_id, distance in self.manager.search_text(query):
                if event_id in wanted and distance <= cutoff:
                    delivery.add(event_id)
        for event_id, distance in self.manager.search_text(DOG_WALKER_QUERY):
            if event_id in wanted and distance <= cutoff:
                dogs.add(event_id)
        return delivery, dogs

    def _plate(
        self,
        ffmpeg: str,
        rows: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> list:
        frames = []
        for moment in plate_timestamps(self.after, self.before, 12):
            if self._cancel.is_set():
                raise RecapCancelled()
            segment = _segment_at(rows, moment)
            if segment is None:
                continue
            path, offset = segment
            frame = grab_frame(ffmpeg, path, offset, width, height)
            if frame is not None:
                frames.append(frame)
        return frames


class RecapMaintainer(threading.Thread):
    """Fires the daily schedule and deletes recaps past their retention."""

    def __init__(
        self,
        config: FrigateConfig,
        stop_event: threading.Event | MpEvent,
        manager: RecapManager,
    ) -> None:
        super().__init__(name="recap-maintainer", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.manager = manager
        self._ticks = 0

    def run(self) -> None:
        while not self.stop_event.wait(20):
            try:
                self._schedule()
                self._ticks += 1
                if self._ticks % 180 == 1:
                    self._retain()
            except Exception:
                logger.exception("Recap maintainer failed")

    def _retain(self) -> None:
        days = {
            name: camera.recap.retain_days
            for name, camera in self.config.cameras.items()
            if camera.recap.enabled
        }
        removed = purge_expired(days)
        if removed:
            logger.info("Removed %d expired recaps", removed)

    def _schedule(self) -> None:
        zone = resolve_zone(self.config)
        now = datetime.now(zone)
        state_file = Path(RECAP_DIR) / ".schedule.json"
        try:
            state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        changed = False
        for name, camera in self.config.cameras.items():
            settings = camera.recap
            if not settings.enabled or not settings.schedule or not camera.enabled:
                continue
            hour, minute = (int(part) for part in settings.schedule.split(":"))
            scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < scheduled or (now - scheduled).total_seconds() > 90:
                continue
            key = f"{name}:{now.date().isoformat()}:{settings.schedule}"
            if state.get(key):
                continue
            before = now.timestamp()
            after = before - settings.window_hours * 3600
            try:
                self.manager.start(name, after, before, reason="schedule")
            except ValueError as err:
                logger.info("Skipped scheduled recap for %s: %s", name, err)
                continue
            state[key] = now.timestamp()
            changed = True
            logger.info("Scheduled recap started for %s", name)
        if changed:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            today = now.date().isoformat()
            fresh = {
                key: value
                for key, value in state.items()
                if len(str(key).split(":")) >= 2 and str(key).split(":")[1] == today
            }
            state_file.write_text(json.dumps(fresh))


def _load_events(
    camera: str, after: float, before: float, labels: list[str]
) -> list[dict[str, Any]]:
    if not labels:
        return []
    query = (
        Event.select()
        .where(Event.camera == camera)
        .where(Event.label.in_(labels))
        .where(Event.has_clip == True)  # noqa: E712
        .where(Event.false_positive == False)  # noqa: E712
        .where(Event.end_time.is_null(False))
        .where(Event.start_time < before)
        .where(Event.end_time > after)
        .order_by(Event.start_time.asc())
    )
    events: list[dict[str, Any]] = []
    for event in query:
        events.append(
            {
                "id": event.id,
                "label": event.label,
                "sub_label": event.sub_label,
                "start_time": float(event.start_time),
                "end_time": float(event.end_time),
                "data": event.data or {},
            }
        )
    return events


def _recordings(camera: str, after: float, before: float) -> list[dict[str, Any]]:
    query = (
        Recordings.select(Recordings.path, Recordings.start_time, Recordings.end_time)
        .where(Recordings.camera == camera)
        .where(Recordings.start_time < before)
        .where(Recordings.end_time > after)
        .order_by(Recordings.start_time.asc())
    )
    return [
        {
            "path": row.path,
            "start": float(row.start_time),
            "end": float(row.end_time),
        }
        for row in query
    ]


def _segments(
    rows: list[dict[str, Any]], start: float, end: float
) -> list[tuple[str, float, float]]:
    segments: list[tuple[str, float, float]] = []
    for row in rows:
        if row["end"] <= start or row["start"] >= end:
            continue
        offset = max(0.0, start - row["start"])
        duration = min(row["end"], end) - max(row["start"], start)
        if duration > 0.05:
            segments.append((row["path"], offset, duration))
    return segments


def _segment_at(rows: list[dict[str, Any]], moment: float) -> tuple[str, float] | None:
    for row in rows:
        if row["start"] <= moment < row["end"]:
            return row["path"], max(0.0, moment - row["start"])
    return None
