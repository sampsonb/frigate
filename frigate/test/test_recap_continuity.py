"""Recap keeps cause and effect: arrivals, stays, departures, and who waits."""

import shutil
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from frigate.config.recap import RecapConfig
from frigate.recap import cutcache
from frigate.recap.generate import (
    LEAVES,
    PIECE_SETTLE,
    STATIONARY_LAG,
    _vehicle_stays,
    clip_window,
    frame_times,
    generate_recap,
    last_seen,
    parked_spans,
    vehicle_pieces,
)
from frigate.recap.layout import (
    MotionTrack,
    ScheduledUnit,
    follows,
    schedule_units,
)
from frigate.recap.plates import (
    ParkedSpan,
    build_timed_plates,
    clean_plates,
    parked_cutout,
    parked_region,
)
from frigate.recap.render import ParkedPatch

WIDTH, HEIGHT = 320, 180
ROAD = (80, 140, 70)
RED = (0, 0, 220)
CAR_W, CAR_H, CAR_TOP = 40, 24, 100
PARK_LEFT = 170.0
T0 = 1_000_000.0


def _car_left(moment: float) -> float | None:
    """Pulls in over 2 s, sits for a minute, drives off right over 2 s."""
    seconds = moment - T0
    if seconds < 0 or seconds > 64:
        return None
    if seconds <= 2:
        return 10 + 80 * seconds
    if seconds <= 62:
        return PARK_LEFT
    return PARK_LEFT + 70 * (seconds - 62)


def _road() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), ROAD, np.uint8)


def _scene(moment: float) -> np.ndarray:
    frame = _road()
    left = _car_left(moment)
    if left is not None:
        x = int(round(left))
        frame[CAR_TOP : CAR_TOP + CAR_H, max(0, x) : min(WIDTH, x + CAR_W)] = RED
    return frame


def _xywh(left: float) -> list[float]:
    return [left / WIDTH, CAR_TOP / HEIGHT, CAR_W / WIDTH, CAR_H / HEIGHT]


def _foot(left: float) -> list[float]:
    return [(left + CAR_W / 2) / WIDTH, (CAR_TOP + CAR_H) / HEIGHT]


def _parked_event(arrives: bool = True, leaves: bool = True) -> dict:
    """Frigate's event for the car: stationary 10 s after it stops."""
    begin = 0.0 if arrives else 2.0
    path = (
        [
            [_foot(_car_left(T0 + step)), T0 + step]
            for step in (0, 0.4, 0.8, 1.2, 1.6, 2)
        ]
        if arrives
        else [[_foot(PARK_LEFT), T0 + 2.0]]
    )
    timeline = [
        [T0 + begin, "visible", _xywh(_car_left(T0 + begin))],
        [T0 + 12.0, "stationary", _xywh(PARK_LEFT)],
    ]
    if leaves:
        path += [
            [_foot(_car_left(T0 + step)), T0 + step]
            for step in (62.4, 62.8, 63.2, 63.6, 64.0)
        ]
        # Frigate marks it active once it has already moved a little.
        timeline += [
            [T0 + 62.6, "active", _xywh(_car_left(T0 + 62.6))],
            [T0 + 64.0, "gone", _xywh(_car_left(T0 + 64.0))],
        ]
    return {
        "id": f"{T0 + begin:.6f}-truck",
        "label": "car",
        "start_time": T0 + begin,
        "end_time": T0 + (67.0 if leaves else 100.0),
        "data": {"box": _xywh(PARK_LEFT), "path_data": path},
        "timeline": timeline,
    }


def _track(
    track_id: str,
    times: list[float],
    boxes: list[tuple[float, float, float, float]],
    label: str = "person",
) -> MotionTrack:
    return MotionTrack(
        id=track_id,
        label=label,
        category="vehicle" if label == "car" else label,
        start=times[0],
        end=times[-1],
        boxes=boxes,
        times=times,
    )


def _unit(
    unit_id: str,
    frames: int,
    box: tuple[float, float, float, float],
    category: str = "person",
    clip_id: str | None = None,
) -> ScheduledUnit:
    return ScheduledUnit(
        event_id=unit_id,
        clip_event_id=clip_id or unit_id,
        label="car" if category == "vehicle" else category,
        category=category,
        start_time=0.0,
        boxes=[box] * frames,
    )


class TestVehiclePieces(unittest.TestCase):
    def test_pulling_in_and_driving_off_are_two_clips(self):
        event = _parked_event()
        arrival, departure = vehicle_pieces(event, 12)
        self.assertEqual(arrival[0], T0)
        self.assertGreaterEqual(arrival[1], T0 + 2)
        self.assertLessEqual(arrival[1], T0 + 12 + PIECE_SETTLE)
        # It starts before Frigate's late active mark, while still parked.
        self.assertLess(departure[0], T0 + 62.0)
        self.assertGreater(departure[0], arrival[1])
        self.assertAlmostEqual(departure[1], last_seen(event))

    def test_a_car_that_stays_shows_only_pulling_in(self):
        pieces = vehicle_pieces(_parked_event(leaves=False), 12)
        self.assertEqual(len(pieces), 1)
        self.assertLessEqual(pieces[0][1], T0 + 12 + PIECE_SETTLE)

    def test_a_car_found_parked_shows_only_driving_off(self):
        pieces = vehicle_pieces(_parked_event(arrives=False), 12)
        self.assertEqual(len(pieces), 1)
        self.assertGreater(pieces[0][0], T0 + 50)

    def test_a_wobbling_box_on_a_parked_car_is_not_a_drive(self):
        event = _parked_event(arrives=False, leaves=False)
        wobble = [_foot(PARK_LEFT + 2), T0 + 30.0]
        event["data"]["path_data"].append(wobble)
        event["timeline"] += [
            [T0 + 40.0, "active", _xywh(PARK_LEFT + 2)],
            [T0 + 41.0, "gone", _xywh(PARK_LEFT)],
        ]
        self.assertEqual(vehicle_pieces(event, 12), [])

    def test_a_short_stop_is_one_clip(self):
        event = _parked_event()
        event["timeline"] = [
            row if row[1] != "stationary" else [T0 + 3.0, "stationary", row[2]]
            for row in event["timeline"]
        ]
        event["timeline"] = [
            row if row[1] != "active" else [T0 + 4.0, "active", row[2]]
            for row in event["timeline"]
        ]
        event["data"]["path_data"] = [
            point for point in event["data"]["path_data"] if point[1] <= T0 + 2
        ] + [[_foot(PARK_LEFT + 30), T0 + 4.5], [_foot(PARK_LEFT + 80), T0 + 5.0]]
        pieces = vehicle_pieces(event, 12)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0][0], T0)

    def test_without_a_stationary_mark_it_is_the_busiest_stretch(self):
        event = _parked_event()
        event["timeline"] = [row for row in event["timeline"] if row[1] == "visible"]
        self.assertEqual(vehicle_pieces(event, 12), [clip_window(event, 12)])

    def test_a_piece_window_is_what_the_clip_cuts(self):
        event = dict(_parked_event(), window=(T0 + 1.0, T0 + 5.0))
        self.assertEqual(clip_window(event, 12), (T0 + 1.0, T0 + 5.0))


class TestParkedSpans(unittest.TestCase):
    def test_span_runs_from_when_it_stopped_until_it_moved(self):
        (begin, until, box), *rest = parked_spans(_parked_event(), WIDTH, HEIGHT)
        self.assertEqual(rest, [])
        self.assertAlmostEqual(begin, T0 + 12.0 - STATIONARY_LAG)
        self.assertAlmostEqual(until, T0 + 62.6)
        self.assertEqual([round(value) for value in box], [170, 100, 210, 124])

    def test_a_car_still_parked_sits_until_it_was_last_seen(self):
        event = _parked_event(leaves=False)
        (_begin, until, _box), *_rest = parked_spans(event, WIDTH, HEIGHT)
        self.assertAlmostEqual(until, last_seen(event))


class TestFollows(unittest.TestCase):
    TRUCK = (150.0, 60.0, 250.0, 120.0)

    def _truck(self) -> MotionTrack:
        times = [float(second) for second in range(11)]
        boxes = [
            (10.0 + 14 * step, 60.0, 110.0 + 14 * step, 120.0) for step in range(10)
        ] + [self.TRUCK]
        return _track("truck", times, boxes, label="car")

    def test_people_who_got_out_wait_for_the_truck_to_pull_in(self):
        truck = self._truck()
        person = _track("person", [16.0, 20.0], [(240.0, 70.0, 260.0, 120.0)] * 2)
        waits = follows([truck, person], [truck, person], WIDTH, HEIGHT)
        self.assertEqual(waits, [[], [(0, 1.0)]])

    def test_someone_appearing_mid_clip_waits_for_that_moment(self):
        truck = self._truck()
        near = (80.0, 70.0, 100.0, 120.0)
        person = _track("person", [5.0, 9.0], [near, near])
        waits = follows([truck, person], [truck, person], WIDTH, HEIGHT)
        self.assertEqual(len(waits[1]), 1)
        self.assertEqual(waits[1][0][0], 0)
        self.assertAlmostEqual(waits[1][0][1], 5 / 11)

    def test_cars_passing_at_the_edge_keep_their_own_pace(self):
        out = _track("a", [0.0, 3.0], [(200.0, 60.0, 320.0, 100.0)] * 2, "car")
        back = _track("b", [5.0, 8.0], [(210.0, 60.0, 320.0, 100.0)] * 2, "car")
        self.assertEqual(follows([out, back], [out, back], WIDTH, HEIGHT), [[], []])

    def test_minutes_later_is_not_a_follow_up(self):
        truck = self._truck()
        person = _track("person", [300.0, 304.0], [(240.0, 70.0, 260.0, 120.0)] * 2)
        waits = follows([truck, person], [truck, person], WIDTH, HEIGHT)
        self.assertEqual(waits, [[], []])

    def test_a_held_person_is_matched_on_where_their_path_starts(self):
        truck = self._truck()
        door = (20.0, 20.0, 40.0, 70.0)
        shown = _track("person", [18.0], [door])
        path = _track("person", [16.0, 18.0], [(240.0, 70.0, 260.0, 120.0), door])
        waits = follows([truck, shown], [truck, path], WIDTH, HEIGHT)
        self.assertEqual(waits[1], [(0, 1.0)])

    def test_the_schedule_honors_the_wait(self):
        truck = _unit("truck", 24, (10.0, 60.0, 110.0, 120.0), "vehicle")
        whole = _unit("person", 12, (280.0, 10.0, 300.0, 50.0))
        whole.after = [(0, 1.0)]
        half = _unit("dog", 12, (280.0, 120.0, 300.0, 150.0), "animal")
        half.after = [(0, 0.5)]
        starts, _length = schedule_units([truck, whole, half], WIDTH, HEIGHT, 200)
        self.assertGreaterEqual(starts[1], starts[0] + 24)
        self.assertGreaterEqual(starts[2], starts[0] + 12)

    def test_objects_that_were_together_play_together(self):
        pickup = _unit("pickup", 24, (100.0, 60.0, 160.0, 120.0), "vehicle")
        trailer = _unit("trailer", 24, (160.0, 60.0, 260.0, 120.0), "vehicle")
        # The trailer came into view 6 frames into the pickup's clip, right
        # beside it. Without this the two overlap, so one would wait.
        trailer.together = (0, 6)
        starts, _length = schedule_units([pickup, trailer], WIDTH, HEIGHT, 200)
        self.assertEqual(starts[1] - starts[0], 6)

    def test_without_a_wait_far_apart_objects_share_the_screen(self):
        truck = _unit("truck", 24, (10.0, 60.0, 110.0, 120.0), "vehicle")
        person = _unit("person", 12, (280.0, 10.0, 300.0, 50.0))
        starts, _length = schedule_units([truck, person], WIDTH, HEIGHT, 200)
        self.assertEqual(starts, [0, 0])


class TestCleanPlates(unittest.TestCase):
    SPAN = ParkedSpan(T0 + 2.0, T0 + 62.6, (170.0, 100.0, 210.0, 124.0))

    def test_a_parked_car_is_taken_out_of_the_plate(self):
        samples = [
            (T0 - 30, _road()),
            (T0 + 30, _scene(T0 + 30)),
            (T0 + 50, _scene(T0 + 50)),
            (T0 + 90, _road()),
        ]
        plates = build_timed_plates(samples)
        self.assertEqual(len(plates), 1)
        # Half the samples have the car, so the plain median is half car.
        self.assertGreater(int(plates[0].image[112, 190, 2]), 120)
        cleaned = clean_plates(plates, samples, [self.SPAN], WIDTH, HEIGHT)
        np.testing.assert_allclose(cleaned[0].image[112, 190], ROAD, atol=2)
        # Away from the car the plate is untouched.
        np.testing.assert_array_equal(cleaned[0].image[20:60], plates[0].image[20:60])

    def test_a_plate_that_never_had_it_is_left_alone(self):
        samples = [(T0 - 30, _road()), (T0 + 90, _road())]
        plates = build_timed_plates(samples)
        cleaned = clean_plates(plates, samples, [self.SPAN], WIDTH, HEIGHT)
        self.assertIs(cleaned[0], plates[0])

    def test_a_plate_made_only_while_it_sat_borrows_the_nearest_road(self):
        samples = [
            (T0 - 3000, _road()),
            (T0 + 30, _scene(T0 + 30)),
            (T0 + 50, _scene(T0 + 50)),
        ]
        plates = build_timed_plates(samples)
        self.assertEqual(len(plates), 2)
        cleaned = clean_plates(plates, samples, [self.SPAN], WIDTH, HEIGHT)
        np.testing.assert_allclose(cleaned[1].image[112, 190], ROAD, atol=2)


class TestParkedCutout(unittest.TestCase):
    def test_the_car_is_cut_against_the_road(self):
        region = parked_region((170.0, 100.0, 210.0, 124.0), WIDTH, HEIGHT)
        crop, alpha = parked_cutout([_scene(T0 + 30)], _road(), region)
        x, y = 190 - region[0], 112 - region[1]
        self.assertGreater(alpha[y, x], 0.95)
        self.assertLess(alpha[0, 0], 0.05)
        np.testing.assert_array_equal(crop[y, x], RED)

    def test_nothing_to_draw_when_the_frames_match_the_plate(self):
        region = parked_region((170.0, 100.0, 210.0, 124.0), WIDTH, HEIGHT)
        self.assertIsNone(parked_cutout([_road()], _road(), region))


class TestParkedPatch(unittest.TestCase):
    def test_drawn_from_the_arrival_end_to_the_departure_start(self):
        patch = ParkedPatch(
            x=0, y=0, begin=100.0, until=200.0, on_frame=10, off_frame=50
        )
        self.assertEqual(patch.wanted(5, 150.0), (False, True))
        self.assertEqual(patch.wanted(10, 90.0), (True, True))
        self.assertEqual(patch.wanted(49, 300.0), (True, True))
        self.assertEqual(patch.wanted(50, 150.0), (False, True))

    def test_without_its_clips_the_story_time_fades_it(self):
        patch = ParkedPatch(x=0, y=0, begin=100.0, until=200.0)
        self.assertEqual(patch.wanted(0, 90.0), (False, False))
        self.assertEqual(patch.wanted(0, 150.0), (True, False))
        self.assertEqual(patch.wanted(0, 250.0), (False, False))


class TestVehicleStays(unittest.TestCase):
    def test_one_event_pulls_in_sits_and_drives_off(self):
        event = _parked_event()
        arrival, departure = vehicle_pieces(event, 12)
        leaving_id = event["id"] + LEAVES
        events = {
            event["id"]: dict(event, window=arrival),
            leaving_id: dict(
                event, id=leaving_id, source_id=event["id"], window=departure
            ),
        }
        units = [
            _unit(event["id"], 24, (10.0, 100.0, 210.0, 124.0), "vehicle"),
            _unit("walker", 12, (20.0, 10.0, 40.0, 60.0)),
            _unit(leaving_id, 24, (170.0, 100.0, 320.0, 124.0), "vehicle", event["id"]),
        ]
        tracks = {
            event["id"]: _track(
                event["id"],
                [arrival[0], arrival[1]],
                [(10.0, 100.0, 50.0, 124.0)] * 2,
                "car",
            ),
            "walker": _track(
                "walker", [T0 + 30, T0 + 34], [(20.0, 10.0, 40.0, 60.0)] * 2
            ),
            leaving_id: _track(
                leaving_id,
                [departure[0], departure[1]],
                [(170.0, 100.0, 210.0, 124.0)] * 2,
                "car",
            ),
        }
        (stay,) = _vehicle_stays(
            units, tracks, events, WIDTH, HEIGHT, T0 - 60, T0 + 200
        )
        self.assertEqual((stay.arrival, stay.departure), (0, 2))
        self.assertEqual(stay.sources, {event["id"]})

    def test_a_car_lost_and_found_again_is_one_stay(self):
        first = _parked_event(leaves=False)
        first["end_time"] = T0 + 53.0
        first["timeline"].append([T0 + 50.0, "gone", _xywh(PARK_LEFT)])
        second = _parked_event(arrives=False)
        second["id"] = f"{T0 + 55:.6f}-truck"
        second["start_time"] = T0 + 55.0
        second["timeline"] = [
            [T0 + 55.0, "visible", _xywh(PARK_LEFT)],
            [T0 + 60.0, "stationary", _xywh(PARK_LEFT)],
        ] + [row for row in second["timeline"] if row[1] in ("active", "gone")]
        second["data"]["path_data"] = [[_foot(PARK_LEFT), T0 + 55.0]] + [
            point for point in second["data"]["path_data"] if point[1] > T0 + 60
        ]
        units = [
            _unit(first["id"], 24, (10.0, 100.0, 210.0, 124.0), "vehicle"),
            _unit(second["id"], 24, (170.0, 100.0, 320.0, 124.0), "vehicle"),
        ]
        arrival = vehicle_pieces(first, 12)[0]
        departure = vehicle_pieces(second, 12)[0]
        tracks = {
            first["id"]: _track(
                first["id"], list(arrival), [(10.0, 100.0, 50.0, 124.0)] * 2, "car"
            ),
            second["id"]: _track(
                second["id"], list(departure), [(170.0, 100.0, 210.0, 124.0)] * 2, "car"
            ),
        }
        events = {first["id"]: first, second["id"]: second}
        (stay,) = _vehicle_stays(
            units, tracks, events, WIDTH, HEIGHT, T0 - 60, T0 + 200
        )
        self.assertEqual(stay.sources, {first["id"], second["id"]})
        self.assertEqual((stay.arrival, stay.departure), (0, 1))
        self.assertAlmostEqual(stay.until, T0 + 62.6)


class TestCacheWindow(unittest.TestCase):
    def test_each_piece_of_an_event_has_its_own_cutouts(self):
        args = ("e", 10.0, WIDTH, HEIGHT, 8.0, 12.0, "vehicle", 0.0)
        whole = cutcache.cache_key(*args)
        arrival = cutcache.cache_key(*args, window=(1.0, 5.0))
        departure = cutcache.cache_key(*args, window=(6.0, 9.0))
        self.assertEqual(len({whole, arrival, departure}), 3)


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class TestParkedCarRecap(unittest.TestCase):
    def test_it_pulls_in_onto_an_empty_curb_and_later_drives_off(self):
        event = _parked_event()

        def load_clip(ev, fps, width, height):
            count = int((ev["end_time"] - ev["start_time"]) * fps) + 1
            times = frame_times(ev["start_time"], fps, count)
            return [_scene(moment) for moment in times], times

        samples = [
            (T0 - 30, _road()),
            (T0 + 30, _scene(T0 + 30)),
            (T0 + 50, _scene(T0 + 50)),
            (T0 + 90, _road()),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            body = generate_recap(
                camera="front",
                after=T0 - 40,
                before=T0 + 100,
                settings=RecapConfig(enabled=True),
                zone=ZoneInfo("America/Chicago"),
                events=[event],
                load_clip=load_clip,
                plate_frames=samples,
                out_dir=Path(tmp),
                ffmpeg=shutil.which("ffmpeg"),
                cancel=lambda: False,
                progress=lambda *_args: None,
                width=WIDTH,
                height=HEIGHT,
            )
            video = cv2.VideoCapture(str(Path(tmp) / "video.mp4"))
            ok, first = video.read()
            video.release()
        self.assertTrue(ok)
        cars = sorted(
            (track for track in body["tracks"] if track["label"] == "car"),
            key=lambda track: track["out_start"],
        )
        self.assertEqual(
            [track["event_id"] for track in cars], [event["id"], event["id"] + LEAVES]
        )
        self.assertEqual({track["clip_event_id"] for track in cars}, {event["id"]})
        # It drives off after it has pulled in.
        self.assertGreaterEqual(
            cars[1]["out_start"], cars[0]["out_start"] + cars[0]["length"]
        )
        # While it pulls in, its parking spot is road, not a second car.
        spot = first[CAR_TOP + 4 : CAR_TOP + CAR_H - 4, 174:206]
        self.assertLess(float(spot[:, :, 2].mean()), 110.0)


if __name__ == "__main__":
    unittest.main()
