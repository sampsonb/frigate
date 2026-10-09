"""Rolling recap: event cap order, cutout cache, skip-if-unchanged, queue order."""

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _stub_config_package():
    """Import recap modules without loading the whole Frigate config package."""
    package = "frigate.config"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path("frigate/config").resolve())]
        module.__package__ = package
        sys.modules[package] = module
    module = sys.modules[package]
    if not hasattr(module, "FrigateConfig"):
        module.FrigateConfig = object  # type: ignore[attr-defined]
    from frigate.config.recap import RecapConfig

    return RecapConfig


RecapConfig = _stub_config_package()

from frigate.recap import cutcache
from frigate.recap.generate import _cutouts_for, cap_newest
from frigate.recap.layout import MotionTrack
from frigate.recap.queries import fingerprint_changed


def _active(starts):
    items = []
    for index, start in enumerate(starts):
        track = MotionTrack(
            id=f"e{index}",
            label="person",
            category="person",
            start=float(start),
            end=float(start) + 5,
            boxes=[(0, 0, 10, 10)],
            times=[float(start)],
        )
        items.append(({"id": f"e{index}"}, "person", track))
    return items


class TestEventCap(unittest.TestCase):
    def test_keeps_newest_events_in_time_order(self):
        active = _active([50, 10, 40, 20, 30])
        kept, dropped = cap_newest(active, 3)
        self.assertEqual([item[2].start for item in kept], [30.0, 40.0, 50.0])
        self.assertEqual(sorted(item[2].start for item in dropped), [10.0, 20.0])

    def test_under_cap_keeps_everything(self):
        kept, dropped = cap_newest(_active([3, 1, 2]), 10)
        self.assertEqual([item[2].start for item in kept], [1.0, 2.0, 3.0])
        self.assertEqual(dropped, [])


class TestCacheKey(unittest.TestCase):
    base = dict(
        event_id="1.0-abc",
        end_time=100.0,
        width=1280,
        height=720,
        sample_fps=4.0,
        max_object_seconds=12.0,
        category="person",
    )

    def key(self, **changes):
        values = dict(self.base)
        values.update(changes)
        return cutcache.cache_key(**values)

    def test_same_inputs_same_key(self):
        self.assertEqual(self.key(), self.key())
        # Float noise below the rounding does not change the key.
        self.assertEqual(self.key(end_time=100.001), self.key())

    def test_every_input_changes_the_key(self):
        base = self.key()
        for change in (
            {"event_id": "1.0-xyz"},
            {"end_time": 101.0},
            {"end_time": None},
            {"width": 960},
            {"height": 540},
            {"sample_fps": 8.0},
            {"max_object_seconds": 10.0},
            {"category": "vehicle"},
        ):
            self.assertNotEqual(self.key(**change), base, change)

    def test_cache_version_changes_the_key(self):
        base = self.key()
        with patch.object(cutcache, "CACHE_VERSION", cutcache.CACHE_VERSION + 1):
            self.assertNotEqual(self.key(), base)

    def test_round_trip_and_purge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = cutcache.CachedCutout(
                status="ok",
                frames=[
                    {
                        "x": 1,
                        "y": 2,
                        "box": (1.0, 2.0, 3.0, 4.0),
                        "jpeg": b"jpg",
                        "alpha_shape": (2, 2),
                        "alpha_bytes": b"\x00\xff\xff\x00",
                    }
                ],
                boxes=[(1.0, 2.0, 3.0, 4.0)],
                times=[5.0],
            )
            key = self.key()
            self.assertTrue(cutcache.store(key, entry, root))
            loaded = cutcache.load(key, root)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.frames[0]["alpha_bytes"], b"\x00\xff\xff\x00")
            self.assertEqual(loaded.boxes, [(1.0, 2.0, 3.0, 4.0)])
            removed, kept, size = cutcache.purge(root=root)
            self.assertEqual((removed, kept), (0, 1))
            self.assertGreater(size, 0)
            removed, kept, _ = cutcache.purge(root=root, now=time.time() + 75 * 3600)
            self.assertEqual((removed, kept), (1, 0))
            self.assertIsNone(cutcache.load(key, root))

    def test_recent_failures_are_not_cached(self):
        now = 10_000.0
        self.assertFalse(cutcache.should_cache_failure(None, now))
        self.assertFalse(cutcache.should_cache_failure(now - 60, now))
        self.assertTrue(cutcache.should_cache_failure(now - 900, now))


class TestCutoutReuse(unittest.TestCase):
    def test_second_build_reads_the_cache(self):
        settings = RecapConfig(enabled=True)
        event = {
            "id": "1.0-abc",
            "label": "person",
            "start_time": 1.0,
            "end_time": 4.0,
            "data": {"box": [0.1, 0.25, 0.25, 0.45]},
        }
        calls = []

        def load_clip(evt, fps, w, h):
            calls.append(evt["id"])
            frames = [np.full((72, 128, 3), 40, np.uint8) for _ in range(6)]
            for index, frame in enumerate(frames):
                frame[20:50, 10 + index * 8 : 30 + index * 8] = 220
            return frames, [1.0 + 0.5 * i for i in range(6)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = cutcache.CacheStats()
            first = _cutouts_for(
                event,
                "person",
                4.0,
                settings,
                128,
                72,
                load_clip,
                use_cache=True,
                cache_root=root,
                stats=stats,
            )
            second = _cutouts_for(
                event,
                "person",
                4.0,
                settings,
                128,
                72,
                load_clip,
                use_cache=True,
                cache_root=root,
                stats=stats,
            )
        self.assertEqual(first.status, "ok")
        self.assertEqual(len(calls), 1)
        self.assertTrue(second.from_cache)
        self.assertEqual((stats.hits, stats.misses, stats.stored), (1, 1, 1))
        self.assertEqual(len(second.frames), len(first.frames))
        self.assertEqual(second.boxes, first.boxes)


class TestFingerprint(unittest.TestCase):
    def test_new_or_changed_events_trigger_a_rebuild(self):
        previous = {"a": 10.0, "b": None}
        self.assertTrue(fingerprint_changed(None, {}))
        self.assertFalse(fingerprint_changed(previous, {"a": 10.0, "b": None}))
        # Only aged out: no rebuild.
        self.assertFalse(fingerprint_changed(previous, {"b": None}))
        self.assertTrue(fingerprint_changed(previous, {"a": 10.0, "b": 20.0}))
        self.assertTrue(fingerprint_changed(previous, {"a": 10.0, "c": 5.0}))


def _fake_config(tmp: str, **recap):
    settings = RecapConfig(enabled=True, **recap)
    return SimpleNamespace(
        cameras={
            "front": SimpleNamespace(recap=settings, enabled=True),
            "door": SimpleNamespace(recap=settings, enabled=True),
        },
        ui=SimpleNamespace(timezone="UTC"),
        semantic_search=SimpleNamespace(enabled=False),
    )


class TestRollingQueue(unittest.TestCase):
    def setUp(self):
        from frigate.recap import manager as manager_module

        self.module = manager_module
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.patches = [
            patch("frigate.recap.storage.RECAP_DIR", str(self.root)),
            patch("frigate.recap.manager.RECAP_DIR", str(self.root)),
            # Jobs are inspected in the queue, never run.
            patch.object(manager_module.RecapJob, "start", lambda self: None),
        ]
        for item in self.patches:
            item.start()
        self.manager = manager_module.RecapManager(_fake_config(self._tmp.name))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self._tmp.cleanup()

    def _order(self):
        return [
            (job["camera"], job["key"], job["priority"])
            for job in self.manager.queue_snapshot()
        ]

    def test_manual_jobs_run_before_scheduled_refreshes(self):
        with patch.object(self.module, "event_fingerprint", return_value={}):
            # First job takes the worker. The rest wait.
            self.manager.start("door", 0, 3600, reason="schedule")
            self.manager.start_rolling("front")
            now = time.time()
            self.manager.start("front", now - 12 * 3600, now)
        order = self._order()
        self.assertTrue(self.manager.queue_snapshot()[0]["running"])
        waiting = order[1:]
        self.assertEqual(waiting[0][1], "front:last:720m")
        self.assertEqual(waiting[0][2], self.module.PRIORITY_MANUAL)
        self.assertEqual(waiting[1][1], "front:rolling")

    def test_identical_requests_are_not_queued_twice(self):
        now = time.time()
        first = self.manager.start("front", now - 12 * 3600, now)
        again = self.manager.start("front", now - 12 * 3600 + 5, now + 5)
        self.assertEqual(first["id"], again["id"])
        self.assertTrue(again.get("deduped"))
        explicit = self.manager.start("front", 1000, 5000, explicit_range=True)
        explicit_again = self.manager.start("front", 1000, 5000, explicit_range=True)
        self.assertEqual(explicit["id"], explicit_again["id"])
        self.assertEqual(len(self.manager.queue_snapshot()), 2)

    def test_rolling_never_queues_behind_itself_and_manual_moves_it_up(self):
        now = time.time()
        self.manager.start("door", now - 3600, now)  # occupies the worker
        self.manager.start("door", now - 7200, now, explicit_range=True)
        with patch.object(self.module, "event_fingerprint", return_value={}):
            first = self.manager.start_rolling("front")
            second = self.manager.start_rolling("front")
            self.assertEqual(first["status"], "queued")
            self.assertTrue(second.get("deduped"))
            rolling = [
                job
                for job in self.manager.queue_snapshot()
                if job["key"] == "front:rolling"
            ]
            self.assertEqual(len(rolling), 1)
            self.assertEqual(rolling[0]["priority"], self.module.PRIORITY_SCHEDULED)
            self.manager.start_rolling("front", manual=True)
        rolling = [
            job
            for job in self.manager.queue_snapshot()
            if job["key"] == "front:rolling"
        ]
        self.assertEqual(len(rolling), 1)
        self.assertEqual(rolling[0]["priority"], self.module.PRIORITY_MANUAL)
        # Manual refresh now waits ahead of the earlier explicit door range? Same
        # priority, so it keeps arrival order behind that manual job.
        keys = [job["key"] for job in self.manager.queue_snapshot()]
        self.assertLess(
            keys.index(f"door:range:{int(round(now - 7200))}:{int(round(now))}"),
            keys.index("front:rolling"),
        )

    def _live(self, **fields):
        from frigate.recap.storage import recap_dir, rolling_id, write_manifest

        directory = recap_dir("front", rolling_id("front"))
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": rolling_id("front"),
            "camera": "front",
            "status": "complete",
            "message": "Ready",
            "reason": "rolling",
            "hours": 6.0,
            "finished": time.time() - 600,
            "fingerprint": {"a": 10.0},
        }
        manifest.update(fields)
        write_manifest(directory, manifest)
        return directory

    def test_refresh_is_skipped_when_nothing_changed(self):
        directory = self._live()
        with patch.object(self.module, "event_fingerprint", return_value={"a": 10.0}):
            result = self.manager.start_rolling("front")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.manager.queue_snapshot(), [])
        from frigate.recap.storage import read_manifest

        self.assertIn("checked", read_manifest(directory))

    def test_refresh_runs_for_new_events_or_old_builds(self):
        self._live()
        settings = self.manager.config.cameras["front"].recap
        now = time.time()
        live = {
            "status": "complete",
            "hours": 6.0,
            "finished": now - 600,
            "fingerprint": {"a": 10.0},
        }
        with patch.object(
            self.module, "event_fingerprint", return_value={"a": 10.0, "b": None}
        ):
            self.assertEqual(
                self.manager.rolling_reason("front", settings, live, now), "new events"
            )
        with patch.object(self.module, "event_fingerprint", return_value={"a": 10.0}):
            self.assertIsNone(self.manager.rolling_reason("front", settings, live, now))
            old = dict(live, finished=now - 4 * 3600)
            self.assertEqual(
                self.manager.rolling_reason("front", settings, old, now), "aged"
            )
            self.assertEqual(
                self.manager.rolling_reason("front", settings, None, now), "missing"
            )
            self.assertEqual(
                self.manager.rolling_reason("front", settings, live, now, force=True),
                "requested",
            )

    def test_maintainer_checks_each_camera_once_per_interval(self):
        maintainer = self.module.RecapMaintainer(
            self.manager.config, SimpleNamespace(wait=lambda _s: True), self.manager
        )
        with patch.object(
            self.manager, "start_rolling", return_value={"status": "skipped"}
        ) as call:
            now = time.time()
            self.assertEqual(sorted(maintainer._rolling(now)), ["door", "front"])
            self.assertEqual(maintainer._rolling(now + 60), [])
            self.assertEqual(
                sorted(maintainer._rolling(now + 31 * 60)), ["door", "front"]
            )
        self.assertEqual(call.call_count, 4)


class TestRecapJobThread(unittest.TestCase):
    def test_job_starts_as_a_thread_and_records_timing(self):
        from frigate.recap import manager as manager_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("frigate.recap.storage.RECAP_DIR", str(root)),
                patch("frigate.recap.manager.RECAP_DIR", str(root)),
                patch.object(manager_module.cutcache, "purge", return_value=(0, 0, 0)),
            ):
                manager = manager_module.RecapManager(_fake_config(tmp))
                ran = []

                def fake_run(job):
                    ran.append(job.recap_id)
                    job._progress(50, "Cutting out person 1 of 2")

                with patch.object(manager_module.RecapJob, "_run", fake_run):
                    manifest = manager.start("front", 0, 3600)
                    job = manager._jobs.get(manifest["id"])
                    if job is not None:
                        job.join(timeout=5)
                self.assertEqual(ran, [manifest["id"]])
                from frigate.recap.storage import read_manifest, recap_dir

                saved = read_manifest(recap_dir("front", manifest["id"]))
                self.assertEqual(saved["stage"], "cutouts")
                self.assertIn("cutouts", saved["stage_started"])
                self.assertEqual(manager.queue_snapshot(), [])


class TestRollingSwapAndKeep(unittest.TestCase):
    """The live rolling recap always plays; replaced ones move to the saved list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch("frigate.recap.storage.RECAP_DIR", str(self.root))
        self.patch.start()
        from frigate.recap import storage

        self.storage = storage

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _build(self, directory, before, fingerprint, video=b"v"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "video.mp4").write_bytes(video)
        self.storage.write_manifest(
            directory,
            {
                "id": "front_rolling",
                "camera": "front",
                "status": "complete",
                "message": "Ready",
                "reason": "rolling",
                "kind": "rolling",
                "after": before - 6 * 3600,
                "before": before,
                "created": before,
                "finished": before,
                "hours": 6,
                "fingerprint": fingerprint,
            },
        )

    def _staging(self, before, fingerprint, video=b"new"):
        staging = self.storage.staging_dir("front_rolling")
        self._build(staging, before, fingerprint, video)
        return staging

    def test_replaced_recap_is_saved_and_chains(self):
        live = self.storage.recap_dir("front", "front_rolling")
        self._build(live, 1_000_000.0, {"a": 1.0}, b"first")
        _, saved1 = self.storage.swap_into_place(
            self._staging(1_001_800.0, {"a": 1.0, "b": 2.0}, b"second"),
            "front",
            "front_rolling",
            archive=True,
        )
        _, saved2 = self.storage.swap_into_place(
            self._staging(1_003_600.0, {"a": 1.0, "b": 2.0, "c": 3.0}, b"third"),
            "front",
            "front_rolling",
            archive=True,
        )
        self.assertTrue(saved1 and saved2 and saved1 != saved2)
        self.assertEqual((live / "video.mp4").read_bytes(), b"third")
        first = self.storage.recap_dir("front", saved1)
        self.assertEqual((first / "video.mp4").read_bytes(), b"first")
        manifest = self.storage.read_manifest(first)
        self.assertEqual(manifest["id"], saved1)
        self.assertEqual(manifest["reason"], self.storage.ARCHIVE_REASON)
        self.assertNotIn("fingerprint", manifest)
        self.assertFalse(self.storage.is_rolling(manifest))
        ids = sorted(item["id"] for item in self.storage.list_recaps())
        self.assertEqual(ids, sorted(["front_rolling", saved1, saved2]))
        self.assertFalse((self.root / ".staging" / "front_rolling.old").exists())

    def test_same_events_or_same_window_is_not_saved_twice(self):
        live = self.storage.recap_dir("front", "front_rolling")
        self._build(live, 1_000_000.0, {"a": 1.0})
        _, saved = self.storage.swap_into_place(
            self._staging(1_000_100.0, {"a": 1.0}),
            "front",
            "front_rolling",
            archive=True,
        )
        self.assertIsNone(saved)
        self.assertEqual(len(self.storage.list_recaps()), 1)

    def test_archive_off_deletes_the_old_copy(self):
        live = self.storage.recap_dir("front", "front_rolling")
        self._build(live, 1_000_000.0, {"a": 1.0})
        _, saved = self.storage.swap_into_place(
            self._staging(1_001_800.0, {"b": 1.0}), "front", "front_rolling"
        )
        self.assertIsNone(saved)
        self.assertEqual(len(self.storage.list_recaps()), 1)

    def test_purge_only_removes_old_saved_rolling(self):
        now = 2_000_000.0
        old_saved = self.storage.recap_dir("front", "front_old_r6h")
        new_saved = self.storage.recap_dir("front", "front_new_r6h")
        nightly = self.storage.recap_dir("front", "front_nightly")
        manual = self.storage.recap_dir("front", "front_manual")
        for directory, reason, before in [
            (old_saved, self.storage.ARCHIVE_REASON, now - 73 * 3600),
            (new_saved, self.storage.ARCHIVE_REASON, now - 71 * 3600),
            (nightly, "schedule", now - 100 * 3600),
            (manual, "manual", now - 100 * 3600),
        ]:
            directory.mkdir(parents=True)
            self.storage.write_manifest(
                directory,
                {
                    "id": directory.name,
                    "camera": "front",
                    "status": "complete",
                    "reason": reason,
                    "before": before,
                    "created": before,
                },
            )
        removed = self.storage.purge_rolling_archives({"front": 72}, now=now)
        self.assertEqual(removed, 1)
        self.assertFalse(old_saved.exists())
        for directory in (new_saved, nightly, manual):
            self.assertTrue(directory.exists())
        self.assertEqual(self.storage.purge_rolling_archives({"front": 0}, now=now), 0)


class TestRollingFailureKeepsLive(unittest.TestCase):
    def test_failed_or_cancelled_refresh_keeps_the_live_copy(self):
        from frigate.recap import manager as manager_module
        from frigate.recap import storage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("frigate.recap.storage.RECAP_DIR", str(root)),
                patch("frigate.recap.manager.RECAP_DIR", str(root)),
                patch.object(manager_module.cutcache, "purge", return_value=(0, 0, 0)),
                patch.object(
                    manager_module, "event_fingerprint", return_value={"x": 1.0}
                ),
            ):
                live = storage.recap_dir("front", "front_rolling")
                live.mkdir(parents=True)
                (live / "video.mp4").write_bytes(b"good")
                storage.write_manifest(
                    live,
                    {
                        "id": "front_rolling",
                        "camera": "front",
                        "status": "complete",
                        "reason": "rolling",
                        "finished": 1.0,
                        "checked": 1.0,
                        "fingerprint": {"old": 1.0},
                    },
                )
                manager = manager_module.RecapManager(_fake_config(tmp))

                def boom(job):
                    raise RuntimeError("ffmpeg failed")

                with patch.object(manager_module.RecapJob, "_run", boom):
                    result = manager.start_rolling("front", manual=True, force=True)
                    job = manager._jobs.get(result["id"])
                    if job is not None:
                        job.join(timeout=5)
                self.assertEqual((live / "video.mp4").read_bytes(), b"good")
                self.assertEqual(storage.read_manifest(live)["finished"], 1.0)
                self.assertFalse(storage.staging_dir("front_rolling").exists())
                self.assertEqual(manager.rolling_state["front"]["last"], "failed")


if __name__ == "__main__":
    unittest.main()
