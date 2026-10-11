"""Recap objects stay recognizable: alignment, windows, snapshots, pacing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from frigate.config.recap import RecapConfig
from frigate.recap import cutcache
from frigate.recap.align import estimate_shift
from frigate.recap.clips import download_name
from frigate.recap.cutout import (
    build_cutouts,
    needs_window,
    rounded_alpha,
    window_ghost,
)
from frigate.recap.frames import concat_sample
from frigate.recap.generate import (
    _cutouts_for,
    _merge_dog_walkers,
    boxes_at,
    clip_window,
    fit_snapshot,
    frame_times,
    last_seen,
    path_track,
    still_cut,
    too_small,
)
from frigate.recap.layout import MotionTrack, ScheduledUnit, schedule_units
from frigate.recap.manager import clip_segments
from frigate.recap.plates import label_should_draw

WIDTH, HEIGHT = 320, 180
DAYLIGHT = (80, 140, 70)
RED = (0, 0, 220)
CAR_W, CAR_H, CAR_TOP = 40, 24, 80
SPEED = 100.0  # pixels per second of recording
LAG = 0.6  # detector time minus recording time for the same moment


def _car_left(recording_time: float) -> float | None:
    """Left edge of the car at a recording time, or None when it is gone."""
    if recording_time < 0 or recording_time > 2:
        return None
    return 40 + SPEED * recording_time


def _scene(recording_time: float, tone=DAYLIGHT) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), tone, np.uint8)
    left = _car_left(recording_time)
    if left is not None:
        x = int(round(left))
        frame[CAR_TOP : CAR_TOP + CAR_H, x : x + CAR_W] = RED
    return frame


def _box_xywh(left: float) -> list[float]:
    return [left / WIDTH, CAR_TOP / HEIGHT, CAR_W / WIDTH, CAR_H / HEIGHT]


def _event(lag: float = LAG, path_moves: bool = True) -> dict:
    """A car seen by the detector ``lag`` seconds after the recording shows it."""
    start = 0.0 + lag
    seen = 2.0 + lag
    path = []
    for step in range(6):
        recording = step * 0.4
        left = _car_left(recording) if path_moves else 40.0
        foot = ((left + CAR_W / 2) / WIDTH, (CAR_TOP + CAR_H) / HEIGHT)
        path.append([list(foot), recording + lag])
    return {
        "id": f"{start:.1f}-car",
        "label": "car",
        "start_time": start,
        # Frigate keeps the event open a few seconds after the car is gone.
        "end_time": seen + 3.0,
        "data": {"box": _box_xywh(_car_left(1.0)), "path_data": path},
        "timeline": [
            [start, "visible", _box_xywh(_car_left(0.0) if path_moves else 40.0)],
            [seen, "gone", _box_xywh(_car_left(2.0) if path_moves else 40.0)],
        ],
    }


def _load_clip(calls: list, tone=DAYLIGHT):
    def load_clip(event, fps, width, height):
        calls.append((event["start_time"], event["end_time"]))
        count = int((event["end_time"] - event["start_time"]) * fps) + 1
        times = frame_times(event["start_time"], fps, count)
        return [_scene(moment, tone) for moment in times], times

    return load_clip


class TestAlignment(unittest.TestCase):
    def test_a_lagging_path_is_lined_up_with_the_recording(self):
        event = _event()
        times = frame_times(-1.0, 8, 33)
        frames = [_scene(moment) for moment in times]
        found = estimate_shift(
            frames,
            times,
            lambda moments, w, h: boxes_at(event, moments, w, h),
            default=0.0,
            valid=(event["start_time"], last_seen(event)),
        )
        self.assertTrue(found.found)
        self.assertAlmostEqual(found.seconds, LAG, delta=0.11)
        self.assertGreater(found.fill, 0.5)

    def test_an_object_that_does_not_move_keeps_the_starting_guess(self):
        event = _event(path_moves=False)
        times = frame_times(0.0, 8, 16)
        frames = []
        for _moment in times:
            frame = np.full((HEIGHT, WIDTH, 3), DAYLIGHT, np.uint8)
            frame[CAR_TOP : CAR_TOP + CAR_H, 40 : 40 + CAR_W] = RED
            frames.append(frame)
        found = estimate_shift(
            frames,
            times,
            lambda moments, w, h: boxes_at(event, moments, w, h),
            default=0.3,
        )
        self.assertFalse(found.found)
        self.assertEqual(found.seconds, 0.3)


class TestBoxes(unittest.TestCase):
    def test_box_size_follows_the_timeline(self):
        event = {
            "id": "1.0-x",
            "label": "car",
            "start_time": 1.0,
            "end_time": 9.0,
            "data": {"box": [0.4, 0.4, 0.2, 0.2], "path_data": []},
            "timeline": [
                [1.0, "visible", [0.1, 0.4, 0.1, 0.1]],
                [5.0, "gone", [0.5, 0.3, 0.3, 0.3]],
            ],
        }
        early, late = boxes_at(event, [1.0, 5.0], 100, 100)
        self.assertAlmostEqual(early[2] - early[0], 10, delta=0.6)
        self.assertAlmostEqual(late[2] - late[0], 30, delta=0.6)
        # Halfway, the box is between the two, not the snapshot size forever.
        (middle,) = boxes_at(event, [3.0], 100, 100)
        self.assertGreater(middle[2] - middle[0], 12)
        self.assertLess(middle[2] - middle[0], 28)

    def test_shift_moves_the_box_along_the_path(self):
        event = _event()
        (raw,) = boxes_at(event, [1.0], WIDTH, HEIGHT)
        (lined_up,) = boxes_at(event, [1.0], WIDTH, HEIGHT, LAG)
        self.assertAlmostEqual(lined_up[0], _car_left(1.0), delta=2)
        self.assertLess(raw[0], lined_up[0] - 40)

    def test_window_stops_at_the_last_sighting(self):
        event = _event()
        start, end = clip_window(event, 12)
        self.assertAlmostEqual(start, LAG)
        self.assertAlmostEqual(end, 2.0 + LAG)
        self.assertLess(end, event["end_time"] - 2.5)

    def test_long_event_is_trimmed_to_where_it_moves_most(self):
        # Faster before 40 s than after: the stretch ends there, the latest
        # of the equally busy ones.
        event = {
            "id": "0.0-p",
            "label": "person",
            "start_time": 0.0,
            "end_time": 60.0,
            "data": {
                "box": [0.70, 0.40, 0.05, 0.20],
                "path_data": [
                    [[0.1, 0.6], 2.0],
                    [[0.725, 0.6], 40.0],
                    [[0.9, 0.6], 55.0],
                ],
            },
            "timeline": [],
        }
        start, end = clip_window(event, 12)
        self.assertAlmostEqual(end - start, 12)
        self.assertLessEqual(start, 40.0)
        self.assertGreaterEqual(end, 40.0)

    def test_frame_times_match_the_fps_filter(self):
        self.assertEqual(frame_times(10.0, 4, 3), [10.0, 10.25, 10.5])
        self.assertEqual(frame_times(10.0, 4, 0), [])


class TestSmallObjects(unittest.TestCase):
    def test_far_cars_and_specks_are_left_out(self):
        far_car = {"data": {"box": [0.4, 0.3, 0.03, 0.02]}}
        near_car = {"data": {"box": [0.4, 0.3, 0.15, 0.10]}}
        far_person = {"data": {"box": [0.4, 0.3, 0.01, 0.03]}}
        walker = {"data": {"box": [0.4, 0.3, 0.02, 0.05]}}
        self.assertTrue(too_small(far_car, "vehicle", 0.0012))
        self.assertFalse(too_small(near_car, "vehicle", 0.0012))
        self.assertTrue(too_small(far_person, "person", 0.0012))
        self.assertFalse(too_small(walker, "person", 0.0012))
        self.assertFalse(too_small(far_car, "vehicle", 0.0))

    def test_every_drawn_ghost_keeps_its_label(self):
        self.assertTrue(label_should_draw(np.full((10, 10), 255, np.uint8)))
        self.assertFalse(label_should_draw(np.full((5, 5), 255, np.uint8)))


class TestWindows(unittest.TestCase):
    def test_infrared_and_dusk_clips_use_windows(self):
        infrared = np.full((HEIGHT, WIDTH, 3), 70, np.uint8)
        dusk = np.full((HEIGHT, WIDTH, 3), (40, 70, 60), np.uint8)
        day = np.full((HEIGHT, WIDTH, 3), DAYLIGHT, np.uint8)
        self.assertTrue(needs_window(infrared))
        self.assertTrue(needs_window(dusk))
        self.assertFalse(needs_window(day))

    def test_a_window_holds_the_whole_object_with_room_for_a_head(self):
        frame = _scene(1.0)
        box = (140.0, 80.0, 180.0, 104.0)
        ghost = window_ghost(frame, box, "person")
        self.assertIsNotNone(ghost)
        assert ghost is not None
        self.assertTrue(ghost["window"])
        self.assertLess(ghost["y"], 80 - 2)
        self.assertLessEqual(ghost["x"], 140 - 4)
        alpha = ghost["alpha"]
        crop = ghost["crop"]
        # Opaque inside, transparent in the rounded corner.
        self.assertEqual(int(alpha[alpha.shape[0] // 2, alpha.shape[1] // 2]), 255)
        self.assertLess(int(alpha[0, 0]), 8)
        self.assertTrue(
            np.array_equal(crop[alpha.shape[0] // 2, alpha.shape[1] // 2], RED)
        )

    def test_infrared_frames_become_windows_not_motion_masks(self):
        frames = [_scene(moment, (70, 70, 70)) for moment in (0.5, 0.75, 1.0, 1.25)]
        boxes = [
            (_car_left(moment), CAR_TOP, _car_left(moment) + CAR_W, CAR_TOP + CAR_H)
            for moment in (0.5, 0.75, 1.0, 1.25)
        ]
        ghosts = build_cutouts(frames, boxes, "vehicle")
        self.assertIsNotNone(ghosts)
        assert ghosts is not None
        self.assertTrue(all(ghost.get("window") for ghost in ghosts))

    def test_rounded_alpha_is_opaque_inside(self):
        alpha = rounded_alpha(40, 60)
        self.assertEqual(int(alpha[20, 30]), 255)
        self.assertLess(int(alpha[0, 0]), 8)


class TestDaylightCutouts(unittest.TestCase):
    def _moving(self, present, box_pad=0, step=8):
        frames, boxes = [], []
        for index in range(len(present)):
            frame = np.full((HEIGHT, WIDTH, 3), DAYLIGHT, np.uint8)
            x = 20 + index * step
            if present[index]:
                frame[CAR_TOP : CAR_TOP + CAR_H, x : x + CAR_W] = RED
            frames.append(frame)
            boxes.append(
                (
                    float(x - box_pad),
                    float(CAR_TOP - box_pad),
                    float(x + CAR_W + box_pad),
                    float(CAR_TOP + CAR_H + box_pad),
                )
            )
        return frames, boxes

    def test_a_missing_frame_holds_the_car_instead_of_an_empty_window(self):
        present = [index < 5 or index >= 19 for index in range(24)]
        frames, boxes = self._moving(present)
        ghosts = build_cutouts(frames, boxes, "vehicle")
        assert ghosts is not None
        for index, ghost in enumerate(ghosts):
            self.assertFalse(ghost.get("window"), index)
            crop = np.asarray(ghost["crop"])
            alpha = np.asarray(ghost["alpha"])
            strong = alpha > 200
            self.assertGreater(int(strong.sum()), 50, index)
            # Red car pixels, not road, wherever the ghost is drawn.
            self.assertGreater(float(crop[strong][:, 2].mean()), 150, index)
        self.assertTrue(
            all(ghost.get("held") or ghost.get("repaired") for ghost in ghosts[5:19])
        )
        self.assertFalse(any(ghost.get("held") for ghost in ghosts[:5]))

    def test_a_car_that_is_gone_is_not_frozen_on_the_road(self):
        present = [index < 5 for index in range(24)]
        frames, boxes = self._moving(present)
        ghosts = build_cutouts(frames, boxes, "vehicle")
        assert ghosts is not None
        self.assertTrue(
            all(int(np.asarray(g["alpha"]).max()) > 200 for g in ghosts[:5])
        )
        # Past the few frames a neighbor can fill, the frames are left out.
        tail = ghosts[13:]
        self.assertTrue(all(int(np.asarray(g["alpha"]).max()) < 8 for g in tail))
        self.assertFalse(any(g.get("held") or g.get("window") for g in tail))

    def test_a_box_that_ran_ahead_snaps_back_onto_the_car(self):
        frames, boxes = self._moving([True] * 12, step=16)
        # Speeding up between two sparse path points, the box runs a car
        # length ahead for the last half of the clip.
        ahead = boxes[:6] + [
            (b[0] + CAR_W, b[1], b[2] + CAR_W, b[3]) for b in boxes[6:]
        ]
        ghosts = build_cutouts(frames, ahead, "vehicle")
        assert ghosts is not None
        for index, ghost in enumerate(ghosts):
            crop = np.asarray(ghost["crop"])
            alpha = np.asarray(ghost["alpha"])
            strong = alpha > 200
            self.assertGreater(int(strong.sum()), 50, index)
            self.assertGreater(float(crop[strong][:, 2].mean()), 150, index)

    def test_a_loose_box_does_not_bring_road_along(self):
        # A street car crosses most of its own length every frame.
        frames, boxes = self._moving([True] * 10, box_pad=14, step=24)
        ghosts = build_cutouts(frames, boxes, "vehicle")
        assert ghosts is not None
        ghost = ghosts[4]
        alpha = np.asarray(ghost["alpha"])
        x0, y0 = int(ghost["x"]), int(ghost["y"])
        box = boxes[4]
        # A corner of the loose box is road. It must not be painted.
        corner_x = int(box[0]) + 3 - x0
        corner_y = int(box[1]) + 3 - y0
        if 0 <= corner_y < alpha.shape[0] and 0 <= corner_x < alpha.shape[1]:
            self.assertLess(int(alpha[corner_y, corner_x]), 64)
        car_x = int(box[0]) + 14 + CAR_W // 2 - x0
        car_y = int(box[1]) + 14 + CAR_H // 2 - y0
        self.assertEqual(int(alpha[car_y, car_x]), 255)


class TestSnapshots(unittest.TestCase):
    def test_a_cropped_snapshot_is_not_used(self):
        self.assertIsNone(
            fit_snapshot(np.zeros((300, 300, 3), np.uint8), WIDTH, HEIGHT)
        )
        fitted = fit_snapshot(np.zeros((360, 640, 3), np.uint8), WIDTH, HEIGHT)
        self.assertIsNotNone(fitted)
        assert fitted is not None
        self.assertEqual(fitted.shape[:2], (HEIGHT, WIDTH))

    def test_still_is_the_snapshot_box(self):
        event = _event()
        cut = still_cut(event, "vehicle", _scene(1.0), WIDTH, HEIGHT)
        self.assertEqual(cut.status, "ok")
        self.assertTrue(cut.still)
        self.assertEqual(len(cut.frames), 1)
        self.assertTrue(cut.frames[0]["window"])
        box = cut.boxes[0]
        self.assertAlmostEqual(box[0], _car_left(1.0), delta=1)

    def _cut(self, event, snapshot_tone, clip_tone=DAYLIGHT):
        calls: list = []
        settings = RecapConfig(enabled=True)
        snapshot = _scene(1.0, snapshot_tone)
        with tempfile.TemporaryDirectory() as tmp:
            cut = _cutouts_for(
                event,
                "vehicle",
                8.0,
                settings,
                WIDTH,
                HEIGHT,
                _load_clip(calls, clip_tone),
                use_cache=True,
                cache_root=Path(tmp),
                stats=cutcache.CacheStats(),
                load_snapshot=lambda _event: snapshot,
            )
        return cut, calls

    def test_night_object_is_its_snapshot_and_no_clip_is_decoded(self):
        cut, calls = self._cut(_event(), (70, 70, 70), (70, 70, 70))
        self.assertEqual(cut.status, "ok")
        self.assertTrue(cut.still)
        self.assertEqual(calls, [])

    def test_lined_up_daylight_object_moves(self):
        cut, calls = self._cut(_event(), DAYLIGHT)
        self.assertEqual(cut.status, "ok")
        self.assertFalse(cut.still)
        self.assertEqual(len(calls), 1)
        self.assertGreater(len(cut.frames), 8)

    def test_object_lost_by_the_recording_moves_while_it_is_seen(self):
        event = _event()
        # The recording loses the car after half a second (a hole the path
        # does not know about), so most frames would only repeat it.
        calls: list = []

        def load_clip(evt, fps, width, height):
            calls.append(evt["id"])
            count = int((evt["end_time"] - evt["start_time"]) * fps) + 1
            times = frame_times(evt["start_time"], fps, count)
            frames = []
            for moment in times:
                frame = _scene(moment)
                if moment > 0.5:
                    frame[:] = DAYLIGHT
                frames.append(frame)
            return frames, times

        settings = RecapConfig(enabled=True)
        with tempfile.TemporaryDirectory() as tmp:
            cut = _cutouts_for(
                event,
                "vehicle",
                8.0,
                settings,
                WIDTH,
                HEIGHT,
                load_clip,
                use_cache=True,
                cache_root=Path(tmp),
                stats=cutcache.CacheStats(),
                load_snapshot=lambda _event: _scene(1.0),
            )
        self.assertEqual(cut.status, "ok")
        # Seeing it move for a moment beats a snapshot frozen in the road.
        self.assertFalse(cut.still)
        self.assertLess(max(cut.times) - min(cut.times), 1.6)

    def test_a_path_that_stands_still_is_followed_to_the_car(self):
        # The path says the car never moved. The cutout follows the car it
        # finds next to the box instead of freezing it at the box.
        cut, calls = self._cut(_event(path_moves=False), DAYLIGHT)
        self.assertEqual(cut.status, "ok")
        self.assertFalse(cut.still)
        self.assertEqual(len(calls), 1)
        lefts = [box[0] for box in cut.boxes]
        self.assertGreater(lefts[-1] - lefts[0], 100)

    def test_cache_keeps_the_still_flag_and_the_shift(self):
        a = cutcache.cache_key("e", 1.0, 320, 180, 8.0, 12.0, "vehicle", 0.0)
        b = cutcache.cache_key("e", 1.0, 320, 180, 8.0, 12.0, "vehicle", 0.5)
        self.assertNotEqual(a, b)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = cutcache.CachedCutout(status="ok", still=True)
            self.assertTrue(cutcache.store(a, entry, root))
            loaded = cutcache.load(a, root)
        assert loaded is not None
        self.assertTrue(loaded.still)


class TestDogWalkers(unittest.TestCase):
    def test_a_walker_and_their_dog_are_one_label(self):
        person = MotionTrack(
            id="p",
            label="person",
            category="person",
            start=0.0,
            end=3.0,
            boxes=[(100.0, 40.0, 140.0, 140.0)] * 4,
            times=[0.0, 1.0, 2.0, 3.0],
        )
        # The dog walks beside the person: under half of it overlaps.
        dog = MotionTrack(
            id="d",
            label="dog",
            category="animal",
            start=0.0,
            end=3.0,
            boxes=[(120.0, 110.0, 170.0, 140.0)] * 4,
            times=[0.2, 1.2, 2.2, 3.2],
        )
        self.assertEqual(_merge_dog_walkers([person, dog]), {"d"})
        self.assertEqual(person.category, "animal")

    def test_a_held_walker_is_judged_on_its_path(self):
        # The person is a snapshot (one box, one moment); the dog is a clip
        # from other seconds. Their paths still overlap the whole time.
        def event(event_id, label, box):
            return {
                "id": event_id,
                "label": label,
                "start_time": 10.0,
                "end_time": 20.0,
                "data": {"box": box, "path_data": []},
                "timeline": [
                    [10.0, "visible", box],
                    [16.0, "gone", box],
                ],
            }

        person_event = event("p", "person", [0.40, 0.20, 0.10, 0.50])
        dog_event = event("d", "dog", [0.45, 0.55, 0.12, 0.15])
        held = MotionTrack(
            id="p",
            label="person",
            category="person",
            start=10.0,
            end=20.0,
            boxes=[(128.0, 36.0, 160.0, 126.0)],
            times=[19.5],
        )
        clip = MotionTrack(
            id="d",
            label="dog",
            category="animal",
            start=10.0,
            end=20.0,
            boxes=[(144.0, 99.0, 182.0, 126.0)] * 3,
            times=[10.0, 10.5, 11.0],
        )
        self.assertEqual(_merge_dog_walkers([held, clip]), set())
        judged = [
            path_track(person_event, held, WIDTH, HEIGHT),
            path_track(dog_event, clip, WIDTH, HEIGHT),
        ]
        self.assertEqual(_merge_dog_walkers(judged), {"d"})


class TestPacing(unittest.TestCase):
    def _units(self) -> list[ScheduledUnit]:
        box = (100.0, 60.0, 160.0, 100.0)
        return [
            ScheduledUnit(
                event_id=name,
                clip_event_id=name,
                label="car",
                category="vehicle",
                start_time=float(index),
                boxes=[box] * 10,
            )
            for index, name in enumerate(("first", "second"))
        ]

    def test_objects_on_the_same_spot_wait_instead_of_stacking(self):
        starts, length = schedule_units(
            self._units(), WIDTH, HEIGHT, 10, max_delay=30, max_overlap=0.0
        )
        self.assertGreaterEqual(starts[1], 10)
        self.assertGreaterEqual(length, 20)
        # The cap allows a short handoff, never most of the object.
        starts, length = schedule_units(
            self._units(), WIDTH, HEIGHT, 10, max_delay=30, max_overlap=0.3
        )
        self.assertGreaterEqual(starts[1], 7)
        self.assertGreaterEqual(length, 17)
        # The old behavior stacked them to hit the target length.
        _starts, stacked = schedule_units(
            self._units(), WIDTH, HEIGHT, 10, max_delay=30, max_overlap=64
        )
        self.assertLessEqual(stacked, 12)


class TestRetentionHoles(unittest.TestCase):
    def _rows(self) -> list[dict]:
        # 100 to 110 kept, 110 to 120 deleted (no motion), then 120 to 140.
        return [
            {"path": "/a.mp4", "start": 100.0, "end": 110.0},
            {"path": "/c.mp4", "start": 120.0, "end": 130.0},
            {"path": "/d.mp4", "start": 130.0, "end": 140.0},
        ]

    def test_a_hole_is_not_joined_over(self):
        segments, first = clip_segments(self._rows(), 105.0, 135.0)
        self.assertEqual([item[0] for item in segments], ["/c.mp4", "/d.mp4"])
        self.assertEqual(first, 120.0)
        self.assertEqual(segments[0][1], 0.0)

    def test_first_frame_time_skips_a_missing_start(self):
        segments, first = clip_segments(self._rows(), 115.0, 125.0)
        self.assertEqual([item[0] for item in segments], ["/c.mp4"])
        self.assertEqual(first, 120.0)

    def test_rounding_between_segments_is_not_a_hole(self):
        rows = [
            {"path": "/a.mp4", "start": 100.0, "end": 109.6},
            {"path": "/b.mp4", "start": 111.0, "end": 121.0},
        ]
        segments, first = clip_segments(rows, 104.0, 115.0)
        self.assertEqual([item[0] for item in segments], ["/a.mp4", "/b.mp4"])
        self.assertEqual(first, 104.0)


class TestDownloadName(unittest.TestCase):
    def test_saved_recaps_get_a_safe_readable_name(self):
        self.assertEqual(
            download_name("Street recap Sat Oct 10, 12:35 PM to 6:35 PM"),
            "Street recap Sat Oct 10, 12.35 PM to 6.35 PM.mp4",
        )
        self.assertEqual(download_name("../../etc/passwd"), "etcpasswd.mp4")
        self.assertEqual(download_name("Door.mp4"), "Door.mp4")
        self.assertEqual(download_name(None), "recap.mp4")
        self.assertEqual(download_name("  ...  "), "recap.mp4")


class TestSegmentJoin(unittest.TestCase):
    def test_joined_segments_are_seeked_not_cut_at_a_keyframe(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frigate.recap.frames.read_raw", return_value=[]) as read,
        ):
            concat_sample(
                "ffmpeg",
                [("/a.mp4", 4.0, 6.0), ("/b.mp4", 0.0, 3.0)],
                WIDTH,
                HEIGHT,
                8,
                100,
                Path(tmp),
            )
            playlist = (Path(tmp) / "recap-concat.txt").read_text()
        self.assertIn("file '/a.mp4'", playlist)
        self.assertIn("file '/b.mp4'", playlist)
        self.assertNotIn("inpoint", playlist)
        args = read.call_args.args
        self.assertEqual(args[2], 4.0)
        self.assertEqual(args[3], 9.0)
        self.assertEqual(read.call_args.kwargs.get("input_format"), "concat")


if __name__ == "__main__":
    unittest.main()
