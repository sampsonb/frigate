"""Background recap jobs, the nightly schedule, and retention.

One job runs at a time. The worker thread lowers its priority so live
object detection keeps the CPU on a small machine. A second request
waits in a queue instead of starting a second encode.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import shutil
import string
import threading
import time
from datetime import datetime
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2

from frigate.config import FrigateConfig
from frigate.config.recap import RecapConfig
from frigate.const import CACHE_DIR, CLIPS_DIR, RECAP_DIR
from frigate.recap import cutcache
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
    frame_times,
    generate_recap,
)
from frigate.recap.layout import rolling_decision, rolling_slot_key, schedule_due
from frigate.recap.plates import gap_sample_times, plate_sample_times
from frigate.recap.queries import (
    attach_timeline,
    event_fingerprint,
    fingerprint_changed,
    load_events,
    load_recordings,
)
from frigate.recap.storage import (
    ROLLING_REASON,
    ensure_tree,
    purge_expired,
    purge_rolling_archives,
    read_manifest,
    recap_dir,
    recap_kind,
    remove_superseded,
    rolling_id,
    staging_dir,
    swap_into_place,
    update_manifest,
    write_manifest,
)

# Lower runs first. A person waiting on a recap beats a background refresh.
PRIORITY_MANUAL = 0
PRIORITY_SCHEDULED = 1

# Progress bands the job reports, used for per-stage timing in the manifest.
STAGES = (
    (6.0, "events"),
    (12.0, "background"),
    (78.0, "cutouts"),
    (88.0, "labels"),
    (101.0, "encode"),
)


def stage_for(percent: float) -> str:
    for limit, name in STAGES:
        if percent < limit:
            return name
    return STAGES[-1][1]


def dedupe_key(
    camera: str,
    reason: str,
    after: float,
    before: float,
    *,
    explicit_range: bool,
    kind: str = "",
) -> str:
    """Jobs with the same key are the same request and are not queued twice."""
    if reason == ROLLING_REASON:
        return f"{camera}:rolling"
    if reason == "interval":
        return f"{camera}:interval:{kind}"
    if explicit_range:
        return f"{camera}:range:{int(round(after))}:{int(round(before))}"
    if reason in ("schedule", "backfill"):
        return f"{camera}:{reason}:{kind}"
    minutes = int(round((before - after) / 60))
    return f"{camera}:last:{minutes}m"


logger = logging.getLogger(__name__)


def resolve_zone(config: FrigateConfig):
    """Timezone for label clocks and the nightly ``HH:MM`` schedule.

    This is ``ui.timezone``. The container clock is often UTC, so a
    schedule of 02:00 is 02:00 in the UI zone, not 02:00 UTC. An empty
    or unknown name falls back to the container's local zone.
    """
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
        self._seq = itertools.count()
        # Last rolling outcome per camera, for the UI ("skipped", errors).
        self.rolling_state: dict[str, dict[str, Any]] = {}

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
        explicit_range: bool = False,
        priority: int | None = None,
    ) -> dict[str, Any]:
        """Queue a recap. Raises ValueError when the camera or range is unusable.

        ``explicit_range`` is set when the caller passed after and before,
        rather than a last-N-hours button. Those windows are not grouped
        as last-Nh.
        """
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
        if priority is None:
            priority = (
                PRIORITY_SCHEDULED
                if reason in ("schedule", "backfill", "interval")
                else PRIORITY_MANUAL
            )
        zone = resolve_zone(self.config)
        kind = recap_kind(reason, after, before, zone, explicit=explicit_range)
        key = dedupe_key(
            camera, reason, after, before, explicit_range=explicit_range, kind=kind
        )
        with self._lock:
            existing = self._find_locked(key)
            if existing is not None:
                if priority < existing.priority and existing in self._queue:
                    existing.priority = priority
                    self._sort_locked()
                current = read_manifest(existing.directory)
                if current is not None:
                    current["deduped"] = True
                    return current
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
            "kind": kind,
            "explicit_range": explicit_range,
            "hours": round(hours, 3),
            "tracks": [],
            "event_count": 0,
        }
        write_manifest(directory, manifest)
        job = RecapJob(
            self,
            recap_id,
            camera,
            after,
            before,
            settings,
            directory,
            priority=priority,
            key=key,
            reason=reason,
        )
        with self._lock:
            self._enqueue_locked(job)
        return manifest

    def start_rolling(
        self, camera: str, *, manual: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """Check the rolling recap and queue a refresh when it is out of date.

        Returns ``{"status": "queued"|"running"|"skipped", ...}``. A refresh
        never waits behind another refresh of the same camera. A manual
        request moves a waiting scheduled refresh to the front.
        """
        camera_config = self.config.cameras.get(camera)
        if camera_config is None:
            raise ValueError(f"{camera} is not a camera")
        settings: RecapConfig = camera_config.recap
        if not settings.enabled:
            raise ValueError(f"Recap is not enabled for {camera}")
        recap_id = rolling_id(camera)
        key = f"{camera}:rolling"
        priority = PRIORITY_MANUAL if manual else PRIORITY_SCHEDULED
        with self._lock:
            existing = self._find_locked(key)
            if existing is not None:
                if priority < existing.priority and existing in self._queue:
                    existing.priority = priority
                    self._sort_locked()
                running = self._running is existing
                return {
                    "status": "running" if running else "queued",
                    "id": recap_id,
                    "deduped": True,
                }
        now = time.time()
        hours = float(settings.rolling_hours)
        live = read_manifest(recap_dir(camera, recap_id))
        reason = self.rolling_reason(camera, settings, live, now, force=force)
        if reason is None:
            if live is not None:
                update_manifest(recap_dir(camera, recap_id), checked=now)
            self.rolling_state[camera] = {"last": "skipped", "at": now}
            logger.debug("Rolling recap for %s is up to date", camera)
            return {"status": "skipped", "id": recap_id, "checked": now}
        directory = staging_dir(recap_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": recap_id,
            "camera": camera,
            "status": "queued",
            "progress": 0,
            "message": "Waiting",
            "created": now,
            "after": now - hours * 3600,
            "before": now,
            "reason": ROLLING_REASON,
            "kind": ROLLING_REASON,
            "explicit_range": False,
            "hours": hours,
            "trigger": "manual" if manual else "schedule",
            "rebuild_reason": reason,
            "tracks": [],
            "event_count": 0,
        }
        write_manifest(directory, manifest)
        job = RecapJob(
            self,
            recap_id,
            camera,
            now - hours * 3600,
            now,
            settings,
            directory,
            priority=priority,
            key=key,
            reason=ROLLING_REASON,
            rolling_hours=hours,
        )
        with self._lock:
            self._enqueue_locked(job)
        self.rolling_state[camera] = {"last": "queued", "at": now, "why": reason}
        return {"status": "queued", "id": recap_id, "why": reason}

    def rolling_reason(
        self,
        camera: str,
        settings: RecapConfig,
        live: dict[str, Any] | None,
        now: float,
        *,
        force: bool = False,
    ) -> str | None:
        """Why the rolling recap needs a rebuild, or None to skip."""
        if force:
            return "requested"
        if live is None or live.get("status") != "complete":
            return "missing"
        hours = float(settings.rolling_hours)
        if abs(float(live.get("hours") or 0) - hours) > 1e-6:
            return "window changed"
        fingerprint = event_fingerprint(
            camera, now - hours * 3600, now, settings.labels
        )
        previous = live.get("fingerprint")
        if not isinstance(previous, dict):
            previous = None
        if fingerprint_changed(previous, fingerprint):
            return "new events"
        finished = float(live.get("finished") or live.get("created") or 0)
        max_age = settings.rolling_max_age_minutes
        if max_age and now - finished >= max_age * 60:
            return "aged"
        return None

    def rolling_status(self, camera: str) -> dict[str, Any]:
        """Live rolling recap, any refresh in progress, and the last outcome."""
        recap_id = rolling_id(camera)
        live = read_manifest(recap_dir(camera, recap_id))
        building = None
        with self._lock:
            job = self._find_locked(f"{camera}:rolling")
            if job is not None:
                building = read_manifest(job.directory) or {}
                building["status"] = "running" if self._running is job else "queued"
        if live is not None:
            live = dict(live)
            live.pop("fingerprint", None)
            live.pop("tracks", None)
            live.pop("excluded", None)
        if building is not None:
            building.pop("fingerprint", None)
            building.pop("tracks", None)
        return {
            "camera": camera,
            "id": recap_id,
            "current": live,
            "building": building,
            "state": self.rolling_state.get(camera, {}),
        }

    def has_pending(self, camera: str, reason: str) -> bool:
        """True when this camera already has a queued or running job of ``reason``."""
        with self._lock:
            jobs = ([self._running] if self._running else []) + list(self._queue)
            return any(job.camera == camera and job.reason == reason for job in jobs)

    def _find_locked(self, key: str) -> RecapJob | None:
        if self._running is not None and self._running.key == key:
            return self._running
        for job in self._queue:
            if job.key == key:
                return job
        return None

    def _sort_locked(self) -> None:
        self._queue.sort(key=lambda job: (job.priority, job.seq))

    def _enqueue_locked(self, job: RecapJob) -> None:
        job.seq = next(self._seq)
        self._jobs[job.recap_id] = job
        self._queue.append(job)
        self._sort_locked()
        self._pump_locked()

    def queue_snapshot(self) -> list[dict[str, Any]]:
        """Running job first, then waiting jobs in the order they will run."""
        with self._lock:
            jobs = ([self._running] if self._running else []) + list(self._queue)
            return [
                {
                    "id": job.recap_id,
                    "camera": job.camera,
                    "priority": job.priority,
                    "key": job.key,
                    "running": job is self._running,
                }
                for job in jobs
            ]

    def cancel(self, recap_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(recap_id)
            if job is None:
                return False
            if job in self._queue:
                self._queue.remove(job)
                self._jobs.pop(recap_id, None)
                if job.rolling_hours is not None:
                    shutil.rmtree(job.directory, ignore_errors=True)
                else:
                    update_manifest(
                        job.directory,
                        status="cancelled",
                        message="Cancelled",
                        progress=0,
                    )
                return True
        job.cancel()
        return True

    def forget(self, job: RecapJob) -> None:
        with self._lock:
            if self._jobs.get(job.recap_id) is job:
                self._jobs.pop(job.recap_id, None)
            if self._running is job:
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
        *,
        priority: int = PRIORITY_MANUAL,
        key: str = "",
        reason: str = "",
        rolling_hours: float | None = None,
    ) -> None:
        super().__init__(name=f"recap-{recap_id}", daemon=True)
        self.manager = manager
        self.recap_id = recap_id
        self.camera = camera
        self.after = after
        self.before = before
        self.settings = settings
        self.directory = directory
        self.priority = priority
        self.key = key or recap_id
        self.reason = reason
        self.seq = 0
        # Set for a rolling refresh. The window is fixed when the job starts.
        self.rolling_hours = rolling_hours
        self._cancel = threading.Event()
        self._t0 = 0.0
        self._stage = ""
        self._stage_times: dict[str, float] = {}
        self._cache_stats = cutcache.CacheStats()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            os.nice(10)
        except OSError:
            pass
        self._t0 = time.time()
        try:
            self._run()
        except RecapCancelled:
            self._finish_failed("cancelled", "Cancelled")
            logger.info("Recap %s cancelled", self.recap_id)
        except Exception as err:
            logger.exception("Recap %s failed", self.recap_id)
            self._finish_failed(
                "failed", "Recap failed. Check the Frigate logs", str(err)
            )
        finally:
            self.manager.forget(self)
            self._purge_cache()

    def _finish_failed(self, status: str, message: str, error: str = "") -> None:
        finished = time.time()
        if self.rolling_hours is not None:
            # The live rolling recap stays as it was. Drop the half-built copy.
            self.manager.rolling_state[self.camera] = {
                "last": status,
                "at": finished,
                "error": error[:200],
                "took_s": round(finished - self._t0, 1),
            }
            shutil.rmtree(self.directory, ignore_errors=True)
            return
        if not self.directory.is_dir():
            return
        try:
            update_manifest(
                self.directory,
                status=status,
                message=message,
                finished=finished,
                took_s=round(finished - self._t0, 1),
            )
        except OSError:
            logger.debug("Recap %s folder is gone", self.recap_id)

    def _purge_cache(self) -> None:
        try:
            removed, kept, size = cutcache.purge()
            logger.info(
                "Recap cutout cache: %d entries, %.1f MB (%d expired removed)",
                kept,
                size / 1e6,
                removed,
            )
        except Exception:
            logger.debug("Recap cache cleanup failed", exc_info=True)

    def _progress(self, percent: float, message: str) -> None:
        elapsed = round(time.time() - self._t0, 1)
        stage = stage_for(percent)
        if stage != self._stage:
            self._stage = stage
            self._stage_times.setdefault(stage, elapsed)
        update_manifest(
            self.directory,
            status="running",
            progress=int(percent),
            message=message,
            stage=stage,
            elapsed_s=elapsed,
            stage_started=dict(self._stage_times),
        )

    def _run(self) -> None:
        if self.rolling_hours is not None:
            self.before = time.time()
            self.after = self.before - self.rolling_hours * 3600
        update_manifest(
            self.directory,
            status="running",
            progress=1,
            message="Looking up events",
            started=self._t0,
            after=self.after,
            before=self.before,
        )
        self._progress(1, "Looking up events")
        fingerprint = event_fingerprint(
            self.camera, self.after, self.before, self.settings.labels
        )
        config = self.manager.config
        ffmpeg = config.ffmpeg.ffmpeg_path
        ffprobe = config.ffmpeg.ffprobe_path
        events = load_events(self.camera, self.after, self.before, self.settings.labels)
        try:
            attach_timeline(events)
        except Exception:
            # Boxes fall back to the path and the snapshot alone.
            logger.exception("Could not read the timeline for recap %s", self.recap_id)
        if self._cancel.is_set():
            raise RecapCancelled()
        self._progress(4, f"Found {len(events)} events")
        delivery_ids, dog_ids = self._semantic(events)
        rows = load_recordings(self.camera, self.after, self.before)
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
            span = float(event["end_time"]) - float(event["start_time"])
            max_frames = max(4, int(span * sample_fps) + 2)
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
            # ffmpeg's fps filter emits frame k at k / fps from the seek
            # point. Spreading the frames over the event span instead put
            # every box of a long event on the wrong frame.
            times = frame_times(float(event["start_time"]), sample_fps, len(frames))
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
            use_cache=True,
            cache_stats=self._cache_stats,
            time_shift=camera_time_shift(config, self.camera),
            load_snapshot=lambda event: load_snapshot(self.camera, str(event["id"])),
        )
        finished = time.time()
        current = read_manifest(self.directory) or {}
        current.update(body)
        current["id"] = self.recap_id
        current["status"] = "complete"
        current["progress"] = 100
        current["message"] = "Ready"
        current["started"] = self._t0
        current["finished"] = finished
        current["took_s"] = round(finished - self._t0, 1)
        current["stage_started"] = dict(self._stage_times)
        current["stage"] = "done"
        current["cache"] = {
            "hits": self._cache_stats.hits,
            "misses": self._cache_stats.misses,
            "stored": self._cache_stats.stored,
        }
        if self.rolling_hours is not None:
            current["created"] = finished
            current["checked"] = finished
            current["fingerprint"] = fingerprint
            current["kind"] = ROLLING_REASON
            write_manifest(self.directory, current)
            _live, saved = swap_into_place(
                self.directory,
                self.camera,
                self.recap_id,
                archive=float(getattr(self.settings, "rolling_keep_hours", 0) or 0) > 0,
            )
            if saved:
                logger.info("Saved replaced rolling recap as %s", saved)
            self.manager.rolling_state[self.camera] = {
                "last": "built",
                "at": finished,
                "took_s": current["took_s"],
                "cache": current["cache"],
            }
            logger.info(
                "Rolling recap %s ready in %ss, %s objects, cache %s",
                self.recap_id,
                current["took_s"],
                current.get("event_count"),
                current["cache"],
            )
            return
        # Set again after generate_recap's body is merged so the kind
        # used for replacement is the finished window, not a stale value.
        current["kind"] = recap_kind(
            str(current.get("reason") or "manual"),
            float(current["after"] if current.get("after") is not None else self.after),
            float(
                current["before"] if current.get("before") is not None else self.before
            ),
            resolve_zone(config),
            explicit=bool(current.get("explicit_range")),
        )
        write_manifest(self.directory, current)
        logger.info(
            "Recap %s ready, %s objects, %ss video, rendered in %ss",
            self.recap_id,
            current.get("event_count"),
            current.get("seconds"),
            current.get("took_s"),
        )
        # A deletion error must not flip a Ready recap to failed.
        try:
            remove_superseded(
                self.camera,
                self.recap_id,
                resolve_zone(config),
                enabled=self.settings.replace_superseded,
            )
        except Exception:
            logger.exception("Could not remove superseded recaps for %s", self.recap_id)

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

    def _grab_plate(
        self,
        ffmpeg: str,
        rows: list[dict[str, Any]],
        moment: float,
        width: int,
        height: int,
    ):
        segment = _segment_at(rows, moment)
        if segment is None:
            return None
        path, offset = segment
        return grab_frame(ffmpeg, path, offset, width, height)

    def _plate(
        self,
        ffmpeg: str,
        rows: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> list[tuple[float, Any]]:
        """Frames about every 10 minutes, plus extra grabs at a day/night switch.

        Each sample keeps its timestamp so the synopsis can use a plate from
        the same lighting period as the objects on screen.
        """
        samples: list[tuple[float, Any]] = []
        for moment in plate_sample_times(self.after, self.before):
            if self._cancel.is_set():
                raise RecapCancelled()
            frame = self._grab_plate(ffmpeg, rows, moment, width, height)
            if frame is not None:
                samples.append((moment, frame))
        # A 10 minute step can jump over the infrared switch. Bisect the
        # gaps whose neighbors disagree, a few times, so the plate changes
        # near the real switch instead of half an hour later.
        for _pass in range(3):
            extras = gap_sample_times(samples)
            if not extras:
                break
            added = False
            for moment in extras:
                if self._cancel.is_set():
                    raise RecapCancelled()
                frame = self._grab_plate(ffmpeg, rows, moment, width, height)
                if frame is None:
                    continue
                samples.append((moment, frame))
                added = True
            if not added:
                break
            samples.sort(key=lambda item: item[0])
        return samples


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
        # Last rolling check per camera (epoch seconds).
        self._rolling_checked: dict[str, float] = {}

    def run(self) -> None:
        # A refresh that was building when Frigate stopped is not resumed.
        shutil.rmtree(Path(RECAP_DIR) / ".staging", ignore_errors=True)
        while not self.stop_event.wait(20):
            try:
                self._schedule()
                self._rolling()
                self._interval()
                self._ticks += 1
                if self._ticks % 180 == 1:
                    self._retain()
            except Exception:
                logger.exception("Recap maintainer failed")

    def _rolling(self, now: float | None = None) -> list[str]:
        """Check each camera's rolling recap on its interval. Returns cameras checked."""
        moment = time.time() if now is None else now
        checked: list[str] = []
        for name, camera in self.config.cameras.items():
            settings = camera.recap
            if (
                not settings.enabled
                or not camera.enabled
                or settings.rolling_interval_minutes <= 0
            ):
                continue
            last = self._rolling_checked.get(name)
            if last is None:
                live = read_manifest(recap_dir(name, rolling_id(name))) or {}
                last = float(live.get("checked") or live.get("finished") or 0)
            if moment - last < settings.rolling_interval_minutes * 60:
                self._rolling_checked[name] = last
                continue
            self._rolling_checked[name] = moment
            checked.append(name)
            try:
                result = self.manager.start_rolling(name)
                logger.info(
                    "Rolling recap check for %s: %s", name, result.get("status")
                )
            except ValueError as err:
                logger.info("Skipped rolling recap for %s: %s", name, err)
        return checked

    def _interval(self) -> None:
        """Start one last-N style recap per camera per ``interval_minutes`` slot.

        The window is the last ``interval_minutes`` of footage. The kind
        is ``rolling-<minutes>m``, so a Ready copy replaces older ones of
        that kind only. A slot that arrives while that recap is still
        queued or rendering is skipped, so missed runs do not pile up.
        This is separate from the in-place ``rolling_hours`` recap.
        """
        zone = resolve_zone(self.config)
        now = datetime.now(zone)
        state_file = Path(RECAP_DIR) / ".interval.json"
        try:
            state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        changed = False
        for name, camera in self.config.cameras.items():
            settings = camera.recap
            minutes = settings.interval_minutes
            if not camera.enabled or not settings.enabled or not minutes:
                continue
            slot = rolling_slot_key(now, minutes)
            key = f"{name}:{slot}"
            action = rolling_decision(
                bool(state.get(key)),
                self.manager.has_pending(name, "interval"),
            )
            if action == "wait":
                continue
            if action == "start":
                before = now.timestamp()
                after = before - minutes * 60
                try:
                    self.manager.start(name, after, before, reason="interval")
                except ValueError as err:
                    logger.info("Skipped interval recap for %s: %s", name, err)
                else:
                    logger.info("Interval recap started for %s (%sm)", name, minutes)
            else:
                logger.info(
                    "Skipped interval recap for %s because one is still running",
                    name,
                )
            state[key] = now.timestamp()
            changed = True
        if changed:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            today = now.date().isoformat()
            fresh = {key: value for key, value in state.items() if today in str(key)}
            state_file.write_text(json.dumps(fresh))

    def _retain(self) -> None:
        days = {
            name: camera.recap.retain_days
            for name, camera in self.config.cameras.items()
            if camera.recap.enabled
        }
        removed = purge_expired(days)
        if removed:
            logger.info("Removed %d expired recaps", removed)
        hours = {
            name: camera.recap.rolling_keep_hours
            for name, camera in self.config.cameras.items()
            if camera.recap.enabled
        }
        archived = purge_rolling_archives(hours)
        if archived:
            logger.info("Removed %d saved rolling recaps past their limit", archived)

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
            # settings.schedule is wall time in ui.timezone (``now``).
            if not schedule_due(now, settings.schedule):
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


def load_snapshot(camera: str, event_id: str) -> Any:
    """Frigate's clean snapshot for an event, or the boxed one, as BGR."""
    for suffix in ("-clean.webp", "-clean.png", ".jpg"):
        path = os.path.join(CLIPS_DIR, f"{camera}-{event_id}{suffix}")
        if not os.path.isfile(path):
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return None


def camera_time_shift(config: FrigateConfig, camera: str) -> float:
    """Starting guess, in seconds, from recording time to detector time.

    Frigate's ``detect.annotation_offset`` lines the review overlay up
    with the recording: the frame at recording time ``t`` shows what the
    detector saw at ``t - offset``. Each event refines this guess.
    """
    camera_config = config.cameras.get(camera)
    if camera_config is None:
        return 0.0
    try:
        offset_ms = float(camera_config.detect.annotation_offset or 0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return -offset_ms / 1000.0


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
