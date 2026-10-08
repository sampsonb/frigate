"""Recap layout, time labels, parked cars, cutouts, and archive lookup."""

import json
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from frigate.recap.archive import (
    CLIP_GONE,
    ArchiveLocation,
    clear_archive_cache,
    event_start_from_id,
    lookup_playback,
    parse_index_text,
    present_local_segments,
    recordings_covering,
    rewrite_media_path,
    safe_archive_path,
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
    schedule_due,
    schedule_units,
)
from frigate.recap.render import (
    GhostFrame,
    Tube,
    compose_frame,
    leader_without_ghost,
)
from frigate.recap.storage import list_visible_recaps


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


if __name__ == "__main__":
    unittest.main()
