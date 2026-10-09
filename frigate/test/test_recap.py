"""Recap layout, time labels, parked cars, cutouts, and archive lookup."""

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from frigate.recap.archive import (
    CLIP_GONE,
    ArchiveLocation,
    ArchiveSegment,
    clear_archive_cache,
    event_start_from_id,
    lookup_playback,
    parse_index_text,
    present_local_segments,
    recordings_covering,
    rewrite_media_path,
    safe_archive_path,
)
from frigate.recap.clips import (
    live_clip_urls,
    playable_mp4_response,
    render_faststart_mp4,
)
from frigate.recap.cutout import attach_motion, build_cutouts
from frigate.recap.layout import (
    MotionTrack,
    ScheduledUnit,
    _rects_hit,
    assign_label_texts,
    dedupe_tracks,
    format_clock,
    is_stationary,
    link_parked_cars,
    plan_label_rects,
    repeat_for_min_show,
    rolling_decision,
    rolling_slot_key,
    schedule_due,
    schedule_units,
)
from frigate.recap.render import (
    GhostFrame,
    Tube,
    compose_frame,
    leader_without_ghost,
)
from frigate.recap.storage import list_visible_recaps, recap_kind, remove_superseded


def _recap_config():
    """Load RecapConfig without importing the full Frigate config package."""
    package = "frigate.config"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path("frigate/config").resolve())]
        module.__package__ = package
        sys.modules[package] = module
    from frigate.config.recap import RecapConfig

    return RecapConfig


def _track(
    track_id: str,
    label: str,
    category: str,
    start: float,
    boxes: list[tuple[float, float, float, float]],
    step: float = 1.0,
) -> MotionTrack:
    times = [start + index * step for index in range(len(boxes))]
    return MotionTrack(
        id=track_id,
        label=label,
        category=category,
        start=start,
        end=times[-1],
        boxes=boxes,
        times=times,
    )


class TestRecapLabels(unittest.TestCase):
    def test_seconds_only_when_a_minute_is_shared(self):
        zone = ZoneInfo("America/Chicago")
        first = datetime(2025, 10, 7, 17, 50, 12, tzinfo=zone).timestamp()
        second = datetime(2025, 10, 7, 17, 50, 47, tzinfo=zone).timestamp()
        alone = datetime(2025, 10, 7, 15, 16, 4, tzinfo=zone).timestamp()
        texts = assign_label_texts(
            [first, second, alone],
            [None, "got out", None],
            zone,
        )
        self.assertEqual(
            texts,
            ["5:50:12 PM", "5:50:47 PM got out", "3:16 PM"],
        )
        self.assertTrue(all("(2)" not in text for text in texts))

    def test_a_single_label_omits_seconds(self):
        zone = ZoneInfo("America/Chicago")
        moment = datetime(2025, 10, 7, 15, 16, 4, tzinfo=zone).timestamp()
        self.assertEqual(
            assign_label_texts([moment], [None], zone),
            ["3:16 PM"],
        )

    def test_config_is_off_and_archive_is_optional(self):
        RecapConfig = _recap_config()
        config = RecapConfig()
        self.assertFalse(config.enabled)
        self.assertIsNone(config.interval_minutes)
        self.assertIsNone(RecapConfig(interval_minutes=0).interval_minutes)
        self.assertIsNone(RecapConfig(interval_minutes="").interval_minutes)
        self.assertEqual(RecapConfig(interval_minutes=30).interval_minutes, 30)
        self.assertTrue(config.replace_superseded)
        self.assertFalse(RecapConfig(replace_superseded=False).replace_superseded)
        self.assertIsNone(config.archive.path)
        self.assertIsNone(config.archive.url)
        self.assertIsNone(RecapConfig(schedule="").schedule)
        self.assertEqual(RecapConfig(schedule="2:5").schedule, "02:05")
        self.assertEqual(RecapConfig(labels="person, dog").labels, ["person", "dog"])
        configured = RecapConfig(
            archive={"path": "/mnt/archive", "url": "https://archive.example/frigate/"}
        )
        self.assertEqual(configured.archive.path, "/mnt/archive")
        self.assertEqual(configured.archive.url, "https://archive.example/frigate")
        with self.assertRaises(ValueError):
            RecapConfig(archive={"url": "ftp://nope"})


class TestParkedCars(unittest.TestCase):
    def test_got_out_links_a_stationary_car(self):
        car = _track(
            "car-1",
            "car",
            "vehicle",
            900,
            [(400, 320, 760, 500)] * 4,
            step=30,
        )
        person = _track(
            "person-1",
            "person",
            "person",
            1000,
            [
                (520, 360, 580, 450),
                (500, 300, 570, 470),
                (300, 280, 370, 450),
                (180, 280, 250, 450),
            ],
        )
        links = link_parked_cars([person], [car], 1280, 720)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].kind, "got out")
        self.assertEqual(links[0].vehicle_id, "car-1")
        self.assertEqual(links[0].person_id, "person-1")

    def test_walked_past_is_not_a_link(self):
        car = _track(
            "car-1",
            "car",
            "vehicle",
            900,
            [(400, 320, 760, 500)] * 3,
            step=30,
        )
        person = _track(
            "person-1",
            "person",
            "person",
            1000,
            [
                (520, 280, 590, 450),
                (300, 280, 370, 450),
                (80, 280, 150, 450),
            ],
        )
        self.assertEqual(link_parked_cars([person], [car], 1280, 720), [])

    def test_edge_and_moving_cars_are_skipped(self):
        parked = _track(
            "car-1",
            "car",
            "vehicle",
            900,
            [(400, 320, 760, 500)] * 3,
            step=30,
        )
        moving = _track(
            "car-2",
            "car",
            "vehicle",
            900,
            [
                (100, 320, 460, 500),
                (500, 320, 860, 500),
                (800, 320, 1160, 500),
            ],
            step=5,
        )
        self.assertTrue(is_stationary(parked.boxes, 1280, 720))
        self.assertFalse(is_stationary(moving.boxes, 1280, 720))
        leaving = _track(
            "person-1",
            "person",
            "person",
            1000,
            [
                (0, 360, 40, 450),
                (20, 300, 80, 470),
                (300, 280, 370, 450),
            ],
        )
        self.assertEqual(link_parked_cars([leaving], [parked], 1280, 720), [])
        beside = _track(
            "person-2",
            "person",
            "person",
            1000,
            [
                (520, 360, 580, 450),
                (500, 300, 570, 470),
                (180, 280, 250, 450),
            ],
        )
        self.assertEqual(link_parked_cars([beside], [moving], 1280, 720), [])


class TestLayout(unittest.TestCase):
    def test_duplicate_tracks_merge_and_side_by_side_tracks_stay(self):
        left = _track(
            "a",
            "person",
            "person",
            10,
            [(100, 100, 180, 260), (110, 100, 190, 260)],
        )
        copy = _track(
            "b",
            "person",
            "person",
            10.1,
            [(102, 102, 170, 250), (112, 102, 180, 250)],
        )
        other = _track(
            "c",
            "person",
            "person",
            10,
            [(400, 100, 480, 260), (410, 100, 490, 260)],
        )
        dropped = dedupe_tracks([left, copy, other])
        self.assertIn(1, dropped)
        self.assertNotIn(2, dropped)

    def test_schedule_caps_simultaneous_labels_and_links_start_together(self):
        units = [
            ScheduledUnit(
                event_id=str(index),
                clip_event_id=str(index),
                label="person",
                category="person",
                start_time=float(index),
                boxes=[(10, 10, 80, 160)] * 6,
            )
            for index in range(4)
        ]
        starts, length = schedule_units(units, 320, 240, 1000, max_active=2)

        def concurrent(frame: int) -> int:
            return sum(
                1
                for start, unit in zip(starts, units, strict=True)
                if start <= frame < start + len(unit.boxes)
            )

        self.assertTrue(all(concurrent(frame) <= 2 for frame in range(length)))
        person = ScheduledUnit(
            event_id="person",
            clip_event_id="person",
            label="person",
            category="person",
            start_time=1,
            boxes=[(10, 10, 40, 80)] * 5,
        )
        parked = ScheduledUnit(
            event_id="parked",
            clip_event_id="car",
            label="car",
            category="parked",
            start_time=1,
            boxes=[(100, 100, 200, 160)] * 5,
            suffix="got out",
            link_index=0,
        )
        linked, _total = schedule_units([person, parked], 320, 240, 100)
        self.assertEqual(linked[0], linked[1])

    def test_labels_that_share_the_screen_keep_a_gap(self):
        box = [(400.0, 400.0, 480.0, 560.0)]
        units = [
            ScheduledUnit(
                event_id="a",
                clip_event_id="a",
                label="person",
                category="person",
                start_time=1,
                boxes=box * 4,
            ),
            ScheduledUnit(
                event_id="b",
                clip_event_id="b",
                label="person",
                category="person",
                start_time=1,
                boxes=[(420.0, 420.0, 500.0, 580.0)] * 4,
            ),
        ]
        rects = plan_label_rects(
            units,
            [0, 0],
            1280,
            720,
            text_size={0: (80, 16), 1: (80, 16)},
            header_h=28,
            min_gap=12,
        )
        self.assertFalse(_rects_hit(rects[0], rects[1], 12))

    def test_short_tracks_are_repeated_until_readable(self):
        self.assertGreaterEqual(4 * repeat_for_min_show(4, 12, 2.5), 30)


class TestCutout(unittest.TestCase):
    def test_connected_cart_is_included_and_a_separate_blob_is_not(self):
        height, width = 180, 280
        background = np.full((height, width, 3), 90, np.uint8)
        frame = background.copy()
        frame[8:24, 8:24] = (30, 30, 220)
        frame[78:130, 70:190] = (0, 210, 230)
        frame[48:88, 110:150] = (220, 60, 40)
        frame[32:46, 68:84] = (40, 40, 220)
        _x, _y, mask = attach_motion(
            frame, background, (108.0, 46.0, 152.0, 92.0), "person"
        )
        full = np.zeros((height, width), np.uint8)
        full[_y : _y + mask.shape[0], _x : _x + mask.shape[1]] = mask
        self.assertGreater(float(full[90:120, 80:170].mean()), 0.4)
        self.assertGreater(float(full[50:80, 115:145].mean()), 0.4)
        self.assertEqual(int(full[8:24, 8:24].sum()), 0)
        self.assertEqual(int(full[32:46, 68:84].sum()), 0)

    def test_moving_cart_survives_the_background_plate(self):
        height, width = 160, 320
        plate = np.full((height, width, 3), 90, np.uint8)
        frames = []
        boxes = []
        for index in range(6):
            frame = plate.copy()
            x = 20 + index * 40
            frame[70:120, x : x + 70] = (0, 210, 230)
            frame[40:80, x + 20 : x + 45] = (220, 60, 40)
            frames.append(frame)
            boxes.append((float(x + 18), 36.0, float(x + 48), 82.0))
        ghosts = build_cutouts(frames, boxes, "person")
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        self.assertEqual(len(ghosts), len(frames))
        sample = ghosts[len(ghosts) // 2]
        alpha = sample["alpha"]
        self.assertGreater(int(alpha.sum()), 200)

    def test_stopped_car_still_gets_a_feathered_ghost(self):
        height, width = 180, 320
        frames = [np.full((height, width, 3), (40, 40, 40), np.uint8) for _ in range(4)]
        box = (100.0, 40.0, 180.0, 110.0)
        ghosts = build_cutouts(frames, [box] * 4, "vehicle")
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        self.assertEqual(len(ghosts), 4)
        for ghost in ghosts:
            self.assertGreater(int(np.asarray(ghost["alpha"]).sum()), 0)

    def test_vehicle_gap_is_repaired_from_a_neighbor(self):
        height, width = 180, 320
        plate = np.full((height, width, 3), (90, 90, 90), np.uint8)
        car = np.array((0, 0, 220), np.uint8)
        frames = []
        boxes = []
        for index in range(6):
            frame = plate.copy()
            x = 30 + index * 28
            if index not in (2, 3):
                frame[50:110, x : x + 50] = car
            frames.append(frame)
            boxes.append((float(x), 50.0, float(x + 50), 110.0))
        ghosts = build_cutouts(frames, boxes, "vehicle")
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        self.assertEqual(len(ghosts), 6)
        for ghost in ghosts:
            alpha = np.asarray(ghost["alpha"])
            crop = np.asarray(ghost["crop"])
            strong = alpha > 40
            self.assertGreater(int(strong.sum()), 10)
            self.assertGreater(float(crop[strong][:, 2].mean()), 150)

    def test_low_contrast_vehicle_is_not_dropped(self):
        plate = np.full((120, 200, 3), 90, np.uint8)
        frame = plate.copy()
        frame[40:90, 60:140] = 80
        _x, _y, mask = attach_motion(frame, plate, (60.0, 40.0, 140.0, 90.0), "vehicle")
        self.assertGreater(int(mask.sum()), 20)

    def test_stable_vehicle_measures_motion_once(self):
        height, width = 120, 200
        plate = np.full((height, width, 3), 90, np.uint8)
        box = (40.0, 30.0, 120.0, 90.0)
        frames = []
        for _index in range(8):
            frame = plate.copy()
            frame[32:88, 42:118] = (0, 0, 210)
            frames.append(frame)
        with patch("frigate.recap.cutout.attach_motion", wraps=attach_motion) as spy:
            ghosts = build_cutouts(frames, [box] * 8, "vehicle")
        self.assertEqual(spy.call_count, 1)
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        self.assertEqual(len(ghosts), 8)
        for ghost in ghosts:
            alpha = np.asarray(ghost["alpha"])
            crop = np.asarray(ghost["crop"])
            strong = alpha > 40
            self.assertGreater(int(strong.sum()), 10)
            self.assertGreater(float(crop[strong][:, 2].mean()), 150)

    def test_moving_vehicle_is_measured_on_each_frame(self):
        height, width = 160, 320
        plate = np.full((height, width, 3), 90, np.uint8)
        frames = []
        boxes = []
        for index in range(4):
            frame = plate.copy()
            x = 20 + index * 40
            frame[40:100, x : x + 50] = (0, 0, 210)
            frames.append(frame)
            boxes.append((float(x), 40.0, float(x + 50), 100.0))
        with patch("frigate.recap.cutout.attach_motion", wraps=attach_motion) as spy:
            ghosts = build_cutouts(frames, boxes, "vehicle")
        self.assertEqual(spy.call_count, 4)
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        for ghost in ghosts:
            self.assertGreater(int(np.asarray(ghost["alpha"]).sum()), 0)

    def test_large_and_tiny_vehicles_stay_visible(self):
        plate = np.full((360, 640, 3), (80, 80, 80), np.uint8)
        frames = []
        boxes = []
        for index in range(4):
            frame = plate.copy()
            x = 40 + index * 20
            frame[70:270, x : x + 360] = (20, 20, 200)
            frame[20:36, 20:38] = (0, 0, 200)
            frames.append(frame)
            boxes.append((float(x), 70.0, float(x + 360), 270.0))
        ghosts = build_cutouts(frames, boxes, "vehicle")
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        sample = ghosts[1]
        alpha = np.asarray(sample["alpha"])
        crop = np.asarray(sample["crop"])
        strong = alpha > 40
        self.assertGreater(int(strong.sum()), 1000)
        self.assertGreater(float(crop[strong][:, 2].mean()), 150)

        tiny_plate = np.full((180, 320, 3), 90, np.uint8)
        tiny_frames = []
        tiny_box = (40.0, 40.0, 62.0, 58.0)
        for _index in range(4):
            frame = tiny_plate.copy()
            frame[40:58, 40:62] = (0, 0, 220)
            tiny_frames.append(frame)
        tiny = build_cutouts(tiny_frames, [tiny_box] * 4, "vehicle")
        self.assertIsNotNone(tiny)
        assert tiny is not None
        for ghost in tiny:
            alpha = np.asarray(ghost["alpha"])
            crop = np.asarray(ghost["crop"])
            strong = alpha > 40
            self.assertGreater(int(strong.sum()), 10)
            self.assertGreater(float(crop[strong][:, 2].mean()), 150)

    def test_leader_is_not_drawn_without_a_ghost(self):
        plate = np.zeros((180, 320, 3), np.uint8)
        box = (100.0, 40.0, 180.0, 100.0)
        unit = ScheduledUnit(
            event_id="car",
            clip_event_id="car",
            label="car",
            category="vehicle",
            start_time=0.0,
            boxes=[box],
            text="5:50 PM",
        )
        empty = GhostFrame(
            x=100,
            y=40,
            box=box,
            crop=np.zeros((60, 80, 3), np.uint8),
            alpha=np.zeros((60, 80), np.uint8),
        )
        tube = Tube(
            event_id="car",
            clip_event_id="car",
            label="car",
            category="vehicle",
            start_time=0.0,
            frames=[empty],
        )
        rect = (10, 10, 80, 36)
        blank = compose_frame(
            plate,
            [unit],
            [tube],
            [0],
            [rect],
            0,
            header="",
            header_h=0,
            fade_frames=1,
            label_opacity=0.5,
            font_scale=1.0,
            repeats=[1],
        )
        vehicle = np.array((40, 165, 255), np.uint8)
        near_tip = blank[30:48, 130:150]
        self.assertEqual(int((near_tip == vehicle).all(axis=2).sum()), 0)
        strong = GhostFrame(
            x=100,
            y=40,
            box=box,
            crop=np.full((60, 80, 3), (0, 0, 200), np.uint8),
            alpha=np.full((60, 80), 255, np.uint8),
        )
        tube.frames = [strong]
        drawn = compose_frame(
            plate,
            [unit],
            [tube],
            [0],
            [rect],
            0,
            header="",
            header_h=0,
            fade_frames=1,
            label_opacity=0.5,
            font_scale=1.0,
            repeats=[1],
        )
        near_tip = drawn[30:48, 130:150]
        self.assertGreater(int((near_tip == vehicle).all(axis=2).sum()), 0)

    def test_leader_without_ghost_counts_mismatches(self):
        self.assertEqual(leader_without_ghost([True, False], [True, False]), 0)
        self.assertEqual(leader_without_ghost([False], [True]), 1)


class TestArchive(unittest.TestCase):
    def setUp(self):
        clear_archive_cache()

    def test_index_formats_and_path_rewrite(self):
        rows = parse_index_text(
            json.dumps(
                {
                    "events": [
                        {
                            "event_id": "1.5-aaaaaa",
                            "camera_name": "front",
                            "object": "person",
                            "start_time": 1.5,
                            "end_time": 4,
                            "files": [
                                "/media/frigate/recordings/2026-10-07/15/front/50.00.mp4"
                            ],
                        }
                    ]
                }
            )
        )
        self.assertEqual(rows[0]["event_id"], "1.5-aaaaaa")
        self.assertEqual(
            rewrite_media_path(
                "/media/frigate/recordings/2026-10-07/15/front/50.00.mp4"
            ),
            "recordings/2026-10-07/15/front/50.00.mp4",
        )
        csv_rows = parse_index_text(
            "id,camera,label,start,end,paths\n"
            "1.5-aaaaaa,front,person,1.5,4,recordings/a.mp4|recordings/b.mp4\n"
        )
        self.assertEqual(len(csv_rows), 1)
        normalized = parse_index_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "1.5-aaaaaa",
                            "camera": "front",
                            "paths": ["clips/a.jpg"],
                        }
                    ),
                ]
            )
        )
        self.assertEqual(normalized[0]["camera"], "front")

    def test_lookup_uses_the_recording_tree_and_rejects_escape(self):
        moment = datetime(2026, 10, 7, 15, 50, 5, tzinfo=timezone.utc).timestamp()
        event_id = f"{moment}-abc123"
        self.assertEqual(event_start_from_id(event_id), moment)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "secret.txt"
            outside.write_text("nope")
            folder = root / "recordings" / "2026-10-07" / "15" / "front"
            folder.mkdir(parents=True)
            (folder / "50.00.mp4").write_bytes(b"a")
            (folder / "50.10.mp4").write_bytes(b"b")
            (folder / "49.00.mp4").write_bytes(b"c")
            end = moment + 15
            covered = recordings_covering(root, "front", moment, end)
            names = [segment.path.name for segment in covered if segment.path]
            self.assertEqual(names, ["50.00.mp4", "50.10.mp4"])
            index = root / "index"
            index.mkdir()
            (index / "2026-10-07.json").write_text(
                json.dumps(
                    [
                        {
                            "id": event_id,
                            "camera": "front",
                            "label": "person",
                            "start": moment,
                            "end": end,
                            "paths": [
                                "../../secret.txt",
                                "/media/frigate/recordings/2026-10-07/15/front/50.00.mp4",
                            ],
                            "snapshot": "clips/front-" + event_id + ".jpg",
                        }
                    ]
                )
            )
            snap = root / "clips"
            snap.mkdir()
            (snap / f"front-{event_id}.jpg").write_bytes(b"jpg")
            self.assertIsNone(safe_archive_path(root, "../../secret.txt"))
            playback = lookup_playback([ArchiveLocation(path=root)], event_id)
            self.assertIsNotNone(playback)
            assert playback is not None
            files = [segment.path.name for segment in playback.segments if segment.path]
            self.assertEqual(files, ["50.00.mp4"])
            self.assertTrue(
                all("secret" not in str(segment.path) for segment in playback.segments)
            )
            self.assertIsNotNone(playback.snapshot)

    def test_dated_database_rewrites_live_paths(self):
        moment = datetime(2026, 10, 7, 15, 50, 5, tzinfo=timezone.utc).timestamp()
        event_id = f"{moment}-db1234"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = "2026-10-07"
            database = root / "db"
            database.mkdir()
            db_path = database / f"frigate-{day}.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE event (id TEXT, camera TEXT, label TEXT, start_time REAL, end_time REAL)"
            )
            connection.execute(
                "CREATE TABLE recordings (path TEXT, camera TEXT, start_time REAL, end_time REAL)"
            )
            connection.execute(
                "INSERT INTO event VALUES (?, ?, ?, ?, ?)",
                (event_id, "front", "car", moment, moment + 20),
            )
            connection.execute(
                "INSERT INTO recordings VALUES (?, ?, ?, ?)",
                (
                    "/media/frigate/recordings/2026-10-07/15/front/50.00.mp4",
                    "front",
                    moment - 5,
                    moment + 5,
                ),
            )
            connection.commit()
            connection.close()
            folder = root / "recordings" / day / "15" / "front"
            folder.mkdir(parents=True)
            (folder / "50.00.mp4").write_bytes(b"mp4")
            playback = lookup_playback([ArchiveLocation(path=root)], event_id)
            self.assertIsNotNone(playback)
            assert playback is not None
            self.assertEqual(playback.camera, "front")
            self.assertEqual(
                [segment.path.name for segment in playback.segments if segment.path],
                ["50.00.mp4"],
            )

    def test_archive_recaps_are_listed_without_hiding_local_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_root_holder = root / "unused"
            local_root_holder.mkdir()
            archive = root / "archive" / "recap" / "front" / "front_20261007_abcd"
            archive.mkdir(parents=True)
            manifest = {
                "id": "front_20261007_abcd",
                "camera": "front",
                "status": "complete",
                "created": 10,
            }
            (archive / "manifest.json").write_text(json.dumps(manifest))
            listed = list_visible_recaps([root / "archive" / "recap"])
            match = [item for item in listed if item["id"] == "front_20261007_abcd"]
            self.assertEqual(len(match), 1)
            self.assertEqual(match[0]["source"], "archive")
            self.assertEqual(match[0]["categories"], [])
            self.assertEqual(match[0]["reason"], "")

    def test_summary_keeps_schedule_day_and_category_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "front" / "front_20261007_abcd"
            folder.mkdir(parents=True)
            manifest = {
                "id": "front_20261007_abcd",
                "camera": "front",
                "status": "complete",
                "created": 20,
                "reason": "schedule",
                "event_count": 4,
                "before": 1_700_000_000,
                "categories": [
                    {
                        "key": "person",
                        "name": "People",
                        "color": "#3cb9ff",
                        "count": 3,
                    },
                    {
                        "key": "vehicle",
                        "name": "Vehicles",
                        "color": "#ffa528",
                        "count": "1",
                    },
                    "skip",
                ],
            }
            (folder / "manifest.json").write_text(json.dumps(manifest))
            listed = list_visible_recaps([root])
            match = [item for item in listed if item["id"] == "front_20261007_abcd"]
            self.assertEqual(len(match), 1)
            self.assertEqual(match[0]["reason"], "schedule")
            self.assertEqual(match[0]["event_count"], 4)
            self.assertEqual(
                match[0]["categories"],
                [
                    {
                        "key": "person",
                        "name": "People",
                        "color": "#3cb9ff",
                        "count": 3,
                    },
                    {
                        "key": "vehicle",
                        "name": "Vehicles",
                        "color": "#ffa528",
                        "count": 1,
                    },
                ],
            )

    def test_events_index_and_missing_files(self):
        moment = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc).timestamp()
        event_id = f"{moment}-abc123"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events"
            events.mkdir()
            first = events / "review-1.mp4"
            second = events / "review-1b.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            (events / "review-1.jpg").write_bytes(b"jpg")
            index = root / "index"
            index.mkdir()
            (index / "2026-10-07.json").write_text(
                json.dumps(
                    [
                        {
                            "id": event_id,
                            "camera": "front",
                            "label": "person",
                            "start": moment,
                            "end": moment + 20,
                            "paths": ["events/review-1.mp4", "events/review-1b.mp4"],
                            "snapshot": "events/review-1.jpg",
                        }
                    ]
                )
            )
            location = ArchiveLocation(path=root, url=None)
            playback = lookup_playback([location], event_id)
            self.assertIsNotNone(playback)
            assert playback is not None
            names = [segment.path.name for segment in playback.segments if segment.path]
            self.assertEqual(names, ["review-1.mp4", "review-1b.mp4"])
            ready = present_local_segments(playback.segments)
            self.assertIsNotNone(ready)
            assert ready is not None
            self.assertEqual(len(ready), 2)
            second.unlink()
            self.assertIsNone(present_local_segments(playback.segments))
            first.unlink()
            gone = lookup_playback([location], event_id)
            self.assertIsNotNone(gone)
            assert gone is not None
            self.assertEqual(gone.segments, [])
            self.assertEqual(gone.message, CLIP_GONE)
            self.assertEqual(
                rewrite_media_path("/media/frigate/events/review-1.mp4"),
                "events/review-1.mp4",
            )

    def test_unreadable_index_does_not_raise(self):
        moment = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc).timestamp()
        event_id = f"{moment}-abc123"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "index"
            index.mkdir()
            (index / "2026-10-07.json").write_bytes(b"\xff\xfe not utf-8")
            playback = lookup_playback([ArchiveLocation(path=root, url=None)], event_id)
            self.assertIsNone(playback)
            (index / "2026-10-07.json").write_text("{not json", encoding="utf-8")
            clear_archive_cache()
            playback = lookup_playback([ArchiveLocation(path=root, url=None)], event_id)
            self.assertIsNone(playback)
            rows = parse_index_text("{not json")
            self.assertEqual(rows, [])


class TestTimezone(unittest.TestCase):
    def test_labels_and_schedule_use_ui_timezone_not_utc(self):
        zone = ZoneInfo("America/New_York")
        instant = datetime(2026, 7, 15, 6, 0, 30, tzinfo=timezone.utc)
        local = instant.astimezone(zone)
        self.assertEqual(local.hour, 2)
        self.assertEqual(
            format_clock(instant.timestamp(), zone, seconds=False), "2:00 AM"
        )
        self.assertEqual(
            format_clock(instant.timestamp(), timezone.utc, seconds=False), "6:00 AM"
        )
        self.assertTrue(schedule_due(local, "02:00"))
        self.assertFalse(schedule_due(instant, "02:00"))
        later = local.replace(minute=1, second=31)
        self.assertFalse(schedule_due(later, "02:00"))
        self.assertTrue(schedule_due(local.replace(minute=1, second=30), "02:00"))


class TestNullableEventRows(unittest.TestCase):
    """Rows shaped like a real 0.17.2 database.

    ``false_positive`` is NULL because the maintainer never writes it.
    An event that has not ended has NULL ``end_time``.
    """

    def setUp(self):
        from playhouse.sqlite_ext import SqliteExtDatabase

        from frigate.models import Event, Recordings

        self.Event = Event
        self.Recordings = Recordings
        self._event_db = Event._meta.database
        self._recording_db = Recordings._meta.database
        self._tmp = tempfile.TemporaryDirectory()
        self.db = SqliteExtDatabase(str(Path(self._tmp.name) / "frigate.db"))
        self.db.bind([Event, Recordings])
        self.db.connect()
        _create_nullable_table(self.db, Event)
        _create_nullable_table(self.db, Recordings)
        self.after = 1_000.0
        self.before = 4_600.0
        self.labels = ["person", "car"]

    def tearDown(self):
        self.db.close()
        self.Event._meta.database = self._event_db
        self.Recordings._meta.database = self._recording_db
        self._tmp.cleanup()

    def test_null_false_positive_and_open_events_are_included(self):
        from frigate.recap.queries import live_recordings_exist, load_events

        data = {
            "box": [0.4, 0.3, 0.1, 0.2],
            "region": [0.2, 0.1, 0.5, 0.6],
            "score": 0.82,
            "top_score": 0.91,
            "attributes": None,
            "path_data": [[[0.45, 0.5], 1050.0], [[0.5, 0.55], 1080.0]],
            "type": "object",
        }
        self._event(
            "1050.0-person",
            label="person",
            start=1050.0,
            end=1200.0,
            has_clip=1,
            false_positive=None,
            sub_label=None,
            top_score=None,
            zones=None,
            data=data,
        )
        # Explicit false still counts as a real object.
        self._event(
            "1300.0-car00",
            label="car",
            start=1300.0,
            end=1400.0,
            has_clip=1,
            false_positive=0,
            data={"box": [0.1, 0.1, 0.2, 0.2], "top_score": None},
        )
        # Still in progress: end_time NULL, same as a live track.
        self._event(
            "2000.0-open01",
            label="person",
            start=2000.0,
            end=None,
            has_clip=1,
            false_positive=None,
            has_snapshot=None,
            data=None,
        )
        # has_clip was never written. Do not treat that as "no clip".
        self._event(
            "2100.0-nullcl",
            label="person",
            start=2100.0,
            end=2200.0,
            has_clip=None,
            false_positive=None,
        )
        self._event(
            "1500.0-fp0001",
            label="person",
            start=1500.0,
            end=1600.0,
            has_clip=1,
            false_positive=1,
        )
        self._event(
            "1600.0-noclip",
            label="person",
            start=1600.0,
            end=1700.0,
            has_clip=0,
            false_positive=None,
        )
        self._event(
            "0100.0-ended",
            label="person",
            start=100.0,
            end=500.0,
            has_clip=1,
            false_positive=None,
        )
        self._event(
            "1800.0-back1",
            label="person",
            camera="back",
            start=1800.0,
            end=1900.0,
            has_clip=1,
            false_positive=None,
        )

        found = load_events("front", self.after, self.before, self.labels)
        by_id = {row["id"]: row for row in found}
        self.assertEqual(
            set(by_id),
            {"1050.0-person", "1300.0-car00", "2000.0-open01", "2100.0-nullcl"},
        )
        real = by_id["1050.0-person"]
        self.assertIsNone(real["sub_label"])
        self.assertEqual(real["data"]["box"], [0.4, 0.3, 0.1, 0.2])
        self.assertIsNone(real["data"]["attributes"])
        self.assertEqual(real["data"]["top_score"], 0.91)
        open_event = by_id["2000.0-open01"]
        self.assertEqual(open_event["end_time"], self.before)
        self.assertEqual(open_event["data"], {})
        self.assertIsInstance(by_id["1300.0-car00"]["end_time"], float)

        stored = self.Event.get(self.Event.id == "2000.0-open01")
        self.assertIsNone(stored.false_positive)
        self.assertIsNone(stored.end_time)
        self.assertIsNone(stored.top_score)
        self.assertIsNone(stored.zones)
        self._recording("rec-open", start=1900.0, end=None)
        self.assertTrue(live_recordings_exist(stored, now=self.before))
        no_clip = self.Event.get(self.Event.id == "1600.0-noclip")
        self.assertFalse(live_recordings_exist(no_clip, now=self.before))

    def test_open_recordings_overlap_the_window(self):
        from frigate.recap.queries import load_recordings

        self._recording("rec-closed", start=1000.0, end=1060.0)
        self._recording("rec-open", start=4000.0, end=None)
        self._recording("rec-old", start=100.0, end=200.0)
        self._recording("rec-back", camera="back", start=1000.0, end=1100.0)
        rows = load_recordings("front", self.after, self.before)
        by_path = {row["path"]: row for row in rows}
        self.assertEqual(set(by_path), {"rec-closed", "rec-open"})
        self.assertEqual(by_path["rec-open"]["end"], self.before)
        self.assertEqual(by_path["rec-closed"]["start"], 1000.0)

    def test_archived_open_event_still_names_recordings(self):
        moment = datetime(2026, 10, 7, 15, 50, 5, tzinfo=timezone.utc).timestamp()
        event_id = f"{moment:.0f}-open01"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "db"
            database.mkdir()
            db_path = database / "frigate-2026-10-07.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE event (id TEXT, camera TEXT, label TEXT, "
                "start_time REAL, end_time REAL, false_positive INTEGER)"
            )
            connection.execute(
                "CREATE TABLE recordings (path TEXT, camera TEXT, "
                "start_time REAL, end_time REAL)"
            )
            connection.execute(
                "INSERT INTO event VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, "front", "person", moment, None, None),
            )
            connection.execute(
                "INSERT INTO recordings VALUES (?, ?, ?, ?)",
                (
                    "/media/frigate/recordings/2026-10-07/15/front/50.00.mp4",
                    "front",
                    moment - 5,
                    None,
                ),
            )
            connection.commit()
            connection.close()
            folder = root / "recordings" / "2026-10-07" / "15" / "front"
            folder.mkdir(parents=True)
            (folder / "50.00.mp4").write_bytes(b"mp4")
            playback = lookup_playback([ArchiveLocation(path=root)], event_id)
            self.assertIsNotNone(playback)
            assert playback is not None
            self.assertEqual(
                [segment.path.name for segment in playback.segments if segment.path],
                ["50.00.mp4"],
            )

    def _event(
        self,
        event_id: str,
        label: str,
        start: float,
        end: float | None,
        has_clip: int | None,
        false_positive: int | None,
        camera: str = "front",
        sub_label: str | None = None,
        top_score: float | None = None,
        zones: str | None = None,
        has_snapshot: int | None = 1,
        data: dict | None = None,
    ) -> None:
        _insert(
            self.db,
            "event",
            {
                "id": event_id,
                "label": label,
                "sub_label": sub_label,
                "camera": camera,
                "start_time": start,
                "end_time": end,
                "top_score": top_score,
                "false_positive": false_positive,
                "zones": zones,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "data": None if data is None else json.dumps(data),
            },
        )

    def _recording(
        self,
        path: str,
        start: float,
        end: float | None,
        camera: str = "front",
    ) -> None:
        _insert(
            self.db,
            "recordings",
            {
                "id": path,
                "camera": camera,
                "path": path,
                "start_time": start,
                "end_time": end,
            },
        )


def _create_nullable_table(db, model) -> None:
    columns = []
    for field in model._meta.sorted_fields:
        name = f'"{field.column_name}"'
        if field.primary_key:
            columns.append(f"{name} TEXT PRIMARY KEY")
        else:
            columns.append(name)
    db.execute_sql(f'CREATE TABLE "{model._meta.table_name}" ({", ".join(columns)})')


def _insert(db, table: str, values: dict) -> None:
    names = ", ".join(f'"{name}"' for name in values)
    placeholders = ", ".join("?" for _ in values)
    db.execute_sql(
        f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
        list(values.values()),
    )


class TestClipPlayback(unittest.TestCase):
    def test_live_clip_uses_the_vod_playlist(self):
        urls = live_clip_urls("front", 1728330612.5, 1728330640.0, "evt-1")
        self.assertEqual(
            urls["clip"],
            "vod/front/start/1728330612.5/end/1728330640/index.m3u8",
        )
        self.assertEqual(urls["download"], "events/evt-1/clip.mp4")

    def test_archive_clip_is_faststart_h264_and_ranged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "camera.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:s=160x120:d=1:r=12",
                    "-c:v",
                    "mpeg4",
                    str(source),
                ],
                check=True,
            )
            rendered = render_faststart_mp4(
                "ffmpeg",
                [ArchiveSegment(path=source, url=None, start=0, end=1)],
                None,
                None,
            )
            self.addCleanup(rendered.unlink, missing_ok=True)
            data = rendered.read_bytes()
            self.assertLess(data.find(b"moov"), data.find(b"mdat"))
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    str(rendered),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "h264")
            response = playable_mp4_response(rendered, "clip.mp4")
            self.assertEqual(response.media_type, "video/mp4")
            self.assertIn("inline", response.headers["content-disposition"])
            messages: list[dict] = []

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                messages.append(message)

            asyncio.run(
                response(
                    {
                        "type": "http",
                        "http_version": "1.1",
                        "method": "GET",
                        "scheme": "http",
                        "path": "/clip.mp4",
                        "raw_path": b"/clip.mp4",
                        "query_string": b"",
                        "headers": [(b"range", b"bytes=0-7")],
                        "client": ("127.0.0.1", 123),
                        "server": ("127.0.0.1", 80),
                    },
                    receive,
                    send,
                )
            )
            self.assertEqual(messages[0]["status"], 206)
            headers = {
                key.decode().lower(): value.decode()
                for key, value in messages[0]["headers"]
            }
            self.assertTrue(headers["content-type"].startswith("video/mp4"))
            self.assertTrue(headers["content-range"].startswith("bytes 0-7/"))


def _write_recap(root: Path, camera: str, recap_id: str, **fields) -> None:
    folder = root / camera / recap_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": recap_id,
        "camera": camera,
        "status": "complete",
        "message": "Ready",
        "created": 1,
        "after": 0.0,
        "before": 3600.0,
        "reason": "manual",
    }
    payload.update(fields)
    (folder / "manifest.json").write_text(json.dumps(payload))


class TestSupersededRecaps(unittest.TestCase):
    def test_on_demand_kind_is_window_length(self):
        zone = ZoneInfo("America/New_York")
        self.assertEqual(recap_kind("manual", 0, 3600, zone), "last-1h")
        self.assertEqual(recap_kind("manual", 0, 6 * 3600, zone), "last-6h")
        self.assertEqual(recap_kind("manual", 0, 12 * 3600, zone), "last-12h")
        self.assertEqual(recap_kind("manual", 0, 24 * 3600, zone), "last-24h")
        self.assertEqual(recap_kind("", 0, 24 * 3600, zone), "last-24h")
        self.assertEqual(recap_kind("manual", 0, 3600 + 30, zone), "last-1h")
        self.assertEqual(recap_kind("manual", 0, 3600 + 90, zone), "last-1h")
        self.assertEqual(recap_kind("manual", 0, 3600 + 91, zone), "last-62m")
        self.assertEqual(recap_kind("manual", 0, 90 * 60, zone), "last-90m")

    def test_nightly_kind_uses_the_ui_timezone_date(self):
        zone = ZoneInfo("America/New_York")
        # 03:00 UTC is still the previous evening in New York.
        before = datetime(2026, 10, 8, 3, 0, tzinfo=timezone.utc).timestamp()
        after = before - 24 * 3600
        self.assertEqual(recap_kind("schedule", after, before, zone), "day:2026-10-07")
        self.assertEqual(recap_kind("backfill", after, before, zone), "day:2026-10-07")
        self.assertEqual(recap_kind("manual", after, before, zone), "last-24h")

    def test_ready_recap_replaces_older_completed_duplicates_only(self):
        zone = ZoneInfo("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "local"
            archive = Path(tmp) / "archive"
            root.mkdir()
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(root, "front", "front_new", created=20, kind="last-1h")
                _write_recap(root, "front", "front_old", created=10)
                _write_recap(root, "front", "front_same_time", created=20)
                _write_recap(root, "front", "front_newer", created=30, kind="last-1h")
                _write_recap(root, "front", "front_six", created=5, before=6 * 3600)
                _write_recap(root, "door", "door_old", created=5, kind="last-1h")
                _write_recap(
                    root,
                    "front",
                    "front_failed",
                    created=5,
                    status="failed",
                    message="Recap failed. Check the Frigate logs",
                )
                _write_recap(
                    root,
                    "front",
                    "front_running",
                    created=5,
                    status="running",
                    message="Building a background",
                )
                _write_recap(
                    root,
                    "front",
                    "front_archived",
                    created=5,
                    source="archive",
                    kind="last-1h",
                )
                archive_copy = archive / "recap" / "front" / "front_archive_copy"
                _write_recap(
                    archive / "recap",
                    "front",
                    "front_archive_copy",
                    created=5,
                    kind="last-1h",
                )
                with self.assertLogs("frigate.recap.storage", level="INFO") as logs:
                    removed = remove_superseded("front", "front_new", zone)
                self.assertCountEqual(removed, ["front_old", "front_same_time"])
                kept = {
                    "front_new",
                    "front_newer",
                    "front_six",
                    "door_old",
                    "front_failed",
                    "front_running",
                    "front_archived",
                }
                for recap_id in kept:
                    camera = "door" if recap_id.startswith("door") else "front"
                    self.assertTrue(
                        (root / camera / recap_id / "manifest.json").is_file(),
                        recap_id,
                    )
                self.assertFalse((root / "front" / "front_old").exists())
                self.assertFalse((root / "front" / "front_same_time").exists())
                self.assertTrue(
                    (archive_copy / "manifest.json").is_file(),
                )
                text = "\n".join(logs.output)
                self.assertIn(
                    "Removed superseded recap front_old for front (last-1h)", text
                )
                self.assertIn(
                    "Removed superseded recap front_same_time for front (last-1h)",
                    text,
                )

    def test_nothing_is_deleted_before_the_new_recap_is_ready(self):
        zone = ZoneInfo("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(
                    root,
                    "front",
                    "front_new",
                    created=20,
                    status="running",
                    message="Looking up events",
                )
                _write_recap(root, "front", "front_old", created=10)
                self.assertEqual(remove_superseded("front", "front_new", zone), [])
                self.assertTrue(
                    (root / "front" / "front_old" / "manifest.json").is_file()
                )
                _write_recap(
                    root,
                    "front",
                    "front_new",
                    created=20,
                    status="complete",
                    message="Finishing",
                )
                self.assertEqual(remove_superseded("front", "front_new", zone), [])
                self.assertTrue(
                    (root / "front" / "front_old" / "manifest.json").is_file()
                )

    def test_flag_off_keeps_duplicates(self):
        zone = ZoneInfo("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(root, "front", "front_new", created=20, kind="last-1h")
                _write_recap(root, "front", "front_old", created=10, kind="last-1h")
                removed = remove_superseded("front", "front_new", zone, enabled=False)
                self.assertEqual(removed, [])
                self.assertTrue(
                    (root / "front" / "front_old" / "manifest.json").is_file()
                )

    def test_nightly_replaces_the_same_local_day_only(self):
        zone = ZoneInfo("America/New_York")
        same = datetime(2026, 10, 8, 3, 0, tzinfo=timezone.utc).timestamp()
        previous = datetime(2026, 10, 7, 3, 0, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(
                    root,
                    "front",
                    "front_new",
                    created=50,
                    reason="schedule",
                    after=same - 24 * 3600,
                    before=same,
                )
                _write_recap(
                    root,
                    "front",
                    "front_same_day",
                    created=10,
                    reason="backfill",
                    after=same - 24 * 3600,
                    before=same,
                )
                _write_recap(
                    root,
                    "front",
                    "front_previous_day",
                    created=5,
                    reason="schedule",
                    after=previous - 24 * 3600,
                    before=previous,
                )
                _write_recap(
                    root,
                    "front",
                    "front_manual_day",
                    created=8,
                    reason="manual",
                    after=same - 24 * 3600,
                    before=same,
                )
                with self.assertLogs("frigate.recap.storage", level="INFO") as logs:
                    removed = remove_superseded("front", "front_new", zone)
                self.assertEqual(removed, ["front_same_day"])
                self.assertFalse((root / "front" / "front_same_day").exists())
                self.assertTrue(
                    (root / "front" / "front_previous_day" / "manifest.json").is_file()
                )
                self.assertTrue(
                    (root / "front" / "front_manual_day" / "manifest.json").is_file()
                )
                self.assertTrue(
                    (root / "front" / "front_new" / "manifest.json").is_file()
                )
                self.assertIn("day:2026-10-07", "\n".join(logs.output))

    def test_explicit_range_is_never_a_last_n_kind(self):
        zone = ZoneInfo("America/New_York")
        day_start = datetime(2026, 10, 7, 0, 0, tzinfo=zone).timestamp()
        day_end = datetime(2026, 10, 8, 0, 0, tzinfo=zone).timestamp()
        self.assertEqual(
            recap_kind("manual", day_start, day_end, zone, explicit=True),
            "day:2026-10-07",
        )
        self.assertEqual(recap_kind("manual", day_start, day_end, zone), "last-24h")
        spring_start = datetime(2026, 3, 8, 0, 0, tzinfo=zone).timestamp()
        spring_end = datetime(2026, 3, 9, 0, 0, tzinfo=zone).timestamp()
        self.assertEqual(spring_end - spring_start, 23 * 3600)
        self.assertEqual(
            recap_kind("manual", spring_start, spring_end, zone, explicit=True),
            "day:2026-03-08",
        )
        fall_start = datetime(2026, 11, 1, 0, 0, tzinfo=zone).timestamp()
        fall_end = datetime(2026, 11, 2, 0, 0, tzinfo=zone).timestamp()
        self.assertEqual(fall_end - fall_start, 25 * 3600)
        self.assertEqual(
            recap_kind("manual", fall_start, fall_end, zone, explicit=True),
            "day:2026-11-01",
        )
        before = datetime(2026, 10, 8, 3, 0, tzinfo=timezone.utc).timestamp()
        after = before - 24 * 3600
        shifted = recap_kind("manual", after, before, zone, explicit=True)
        self.assertTrue(shifted.startswith("range:"))
        self.assertNotIn("last-", shifted)
        self.assertEqual(
            recap_kind("manual", 0, 6 * 3600, zone, explicit=True),
            "range:0-21600",
        )
        two_days = recap_kind(
            "manual",
            day_start,
            datetime(2026, 10, 9, 0, 0, tzinfo=zone).timestamp(),
            zone,
            explicit=True,
        )
        self.assertTrue(two_days.startswith("range:"))
        utc_start = datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc).timestamp()
        utc_end = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertTrue(
            recap_kind("manual", utc_start, utc_end, zone, explicit=True).startswith(
                "range:"
            )
        )
        self.assertEqual(
            recap_kind("manual", utc_start, utc_end, timezone.utc, explicit=True),
            "day:2026-10-08",
        )
        self.assertTrue(
            recap_kind(
                "manual", day_start + 1, day_end, zone, explicit=True
            ).startswith("range:")
        )

    def test_backfill_day_groups_with_the_calendar_day(self):
        zone = ZoneInfo("America/New_York")
        start = datetime(2026, 10, 7, 0, 0, tzinfo=zone).timestamp()
        end = datetime(2026, 10, 8, 0, 0, tzinfo=zone).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(
                    root,
                    "front",
                    "front_day",
                    created=40,
                    reason="manual",
                    explicit_range=True,
                    after=start,
                    before=end,
                )
                _write_recap(
                    root,
                    "front",
                    "front_hand",
                    created=10,
                    kind="backfill-day:2026-10-07",
                    after=start,
                    before=end,
                )
                _write_recap(
                    root,
                    "front",
                    "front_other_day",
                    created=9,
                    kind="backfill-day:2026-10-06",
                )
                _write_recap(
                    root,
                    "front",
                    "front_last",
                    created=8,
                    kind="last-24h",
                    after=end - 24 * 3600,
                    before=end,
                )
                _write_recap(
                    root,
                    "front",
                    "front_partial",
                    created=7,
                    kind="range:1-21601",
                    after=1,
                    before=21601,
                    explicit_range=True,
                )
                with self.assertLogs("frigate.recap.storage", level="INFO") as logs:
                    removed = remove_superseded("front", "front_day", zone)
                self.assertEqual(removed, ["front_hand"])
                self.assertFalse((root / "front" / "front_hand").exists())
                for recap_id in (
                    "front_day",
                    "front_other_day",
                    "front_last",
                    "front_partial",
                ):
                    self.assertTrue(
                        (root / "front" / recap_id / "manifest.json").is_file(),
                        recap_id,
                    )
                self.assertIn(
                    "Removed superseded recap front_hand for front (day:2026-10-07)",
                    "\n".join(logs.output),
                )
                _write_recap(
                    root,
                    "front",
                    "front_night",
                    created=60,
                    reason="schedule",
                    after=end - 24 * 3600,
                    before=datetime(2026, 10, 7, 23, 0, tzinfo=zone).timestamp(),
                )
                removed = remove_superseded("front", "front_night", zone)
                self.assertEqual(removed, ["front_day"])
                self.assertTrue(
                    (root / "front" / "front_last" / "manifest.json").is_file()
                )

    def test_explicit_range_does_not_replace_last_n_hours(self):
        zone = ZoneInfo("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(
                    root,
                    "front",
                    "front_range",
                    created=20,
                    reason="manual",
                    explicit_range=True,
                    after=100,
                    before=100 + 6 * 3600,
                )
                _write_recap(
                    root,
                    "front",
                    "front_last",
                    created=5,
                    kind="last-6h",
                    after=0,
                    before=6 * 3600,
                )
                self.assertEqual(
                    remove_superseded("front", "front_range", zone),
                    [],
                )
                self.assertTrue(
                    (root / "front" / "front_last" / "manifest.json").is_file()
                )
                _write_recap(
                    root,
                    "front",
                    "front_again",
                    created=30,
                    kind="range:100-21700",
                    after=100,
                    before=100 + 6 * 3600,
                    explicit_range=True,
                )
                removed = remove_superseded("front", "front_again", zone)
                self.assertEqual(removed, ["front_range"])
                self.assertTrue(
                    (root / "front" / "front_last" / "manifest.json").is_file()
                )


class TestRecapOrder(unittest.TestCase):
    def test_window_end_then_creation_time_across_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "local"
            archive = Path(tmp) / "archive"
            root.mkdir()
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(root, "door", "door_same_end", before=300, created=1)
                _write_recap(root, "front", "front_same_end", before=300, created=9)
                _write_recap(
                    root, "front", "front_older_window", before=100, created=50
                )
                _write_recap(root, "yard", "yard_newest", before=400, created=2)
                _write_recap(
                    archive,
                    "front",
                    "front_archive_later",
                    before=500,
                    created=1,
                )
                listed = list_visible_recaps([archive])
                self.assertEqual(
                    [item["id"] for item in listed],
                    [
                        "front_archive_later",
                        "yard_newest",
                        "front_same_end",
                        "door_same_end",
                        "front_older_window",
                    ],
                )


class TestIntervalRecap(unittest.TestCase):
    def test_kind_is_the_interval_not_last_n_hours(self):
        zone = ZoneInfo("America/New_York")
        self.assertEqual(recap_kind("interval", 0, 30 * 60, zone), "rolling-30m")
        self.assertEqual(recap_kind("interval", 0, 90 * 60, zone), "rolling-90m")
        self.assertEqual(recap_kind("manual", 0, 30 * 60, zone), "last-30m")

    def test_slot_follows_the_local_clock_and_does_not_queue(self):
        zone = ZoneInfo("America/New_York")
        ten = datetime(2026, 10, 9, 10, 0, tzinfo=zone)
        self.assertEqual(rolling_slot_key(ten, 30), "2026-10-09:20")
        self.assertEqual(
            rolling_slot_key(ten.replace(minute=29, second=59), 30),
            "2026-10-09:20",
        )
        self.assertEqual(rolling_slot_key(ten.replace(minute=30), 30), "2026-10-09:21")
        self.assertEqual(
            rolling_slot_key(ten.replace(hour=0, minute=0), 30),
            "2026-10-09:0",
        )
        self.assertEqual(rolling_decision(False, False), "start")
        self.assertEqual(rolling_decision(False, True), "skip")
        self.assertEqual(rolling_decision(True, False), "wait")
        self.assertEqual(rolling_decision(True, True), "wait")

    def test_a_ready_interval_recap_replaces_only_that_kind(self):
        zone = ZoneInfo("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                _write_recap(
                    root,
                    "front",
                    "front_new",
                    created=20,
                    reason="interval",
                    after=0,
                    before=30 * 60,
                )
                _write_recap(
                    root,
                    "front",
                    "front_old",
                    created=5,
                    kind="rolling-30m",
                    after=0,
                    before=30 * 60,
                )
                _write_recap(
                    root,
                    "front",
                    "front_last",
                    created=4,
                    kind="last-30m",
                    after=0,
                    before=30 * 60,
                )
                _write_recap(
                    root,
                    "front",
                    "front_hour",
                    created=3,
                    kind="rolling-60m",
                )
                _write_recap(
                    root,
                    "front",
                    "front_standing",
                    created=6,
                    reason="rolling",
                    kind="rolling",
                    after=0,
                    before=6 * 3600,
                )
                _write_recap(
                    root,
                    "door",
                    "door_rolling",
                    created=2,
                    kind="rolling-30m",
                )
                removed = remove_superseded("front", "front_new", zone)
                self.assertEqual(removed, ["front_old"])
                for recap_id in (
                    "front_new",
                    "front_last",
                    "front_hour",
                    "front_standing",
                    "door_rolling",
                ):
                    camera = "door" if recap_id.startswith("door") else "front"
                    self.assertTrue(
                        (root / camera / recap_id / "manifest.json").is_file(),
                        recap_id,
                    )


if __name__ == "__main__":
    unittest.main()
