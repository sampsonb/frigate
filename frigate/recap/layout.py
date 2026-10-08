"""Pure layout for a recap: time labels, parked cars, dedupe, scheduling.

Nothing in this module reads video or talks to Frigate. The generator and
the unit tests both call these functions.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Sequence

import numpy as np

# A vehicle whose path moves less than this fraction of the frame is parked
# and is left out of the synopsis, unless someone gets in or out of it.
DEFAULT_STATIONARY = 0.015


@dataclass
class MotionTrack:
    """One tracked object, boxes in pixel xyxy, oldest first."""

    id: str
    label: str
    category: str
    start: float
    end: float
    boxes: list[tuple[float, float, float, float]]
    times: list[float]
    zone_note: str = ""


@dataclass
class ParkedLink:
    """A person getting into or out of a stationary vehicle."""

    person_id: str
    vehicle_id: str | None
    kind: str
    ts: float
    car_box: tuple[float, float, float, float]


@dataclass
class ScheduledUnit:
    """One label on the synopsis timeline."""

    event_id: str
    clip_event_id: str
    label: str
    category: str
    start_time: float
    boxes: list[tuple[float, float, float, float]]
    suffix: str | None = None
    link_index: int | None = None
    text: str = ""
    member_ids: list[str] = field(default_factory=list)


def format_clock(timestamp: float, zone: tzinfo, *, seconds: bool) -> str:
    """Local clock time. Seconds are included only when requested.

    ``5:50 PM`` or ``5:50:12 PM``. The hour has no leading zero.
    ``zone`` is ``ui.timezone``. Do not pass the container zone when that
    setting is set: the container clock is often UTC.
    """
    moment = datetime.fromtimestamp(timestamp, zone)
    pattern = "%-I:%M:%S %p" if seconds else "%-I:%M %p"
    return moment.strftime(pattern)


def schedule_due(now: datetime, schedule: str, *, window_seconds: float = 90) -> bool:
    """Whether ``HH:MM`` should fire for ``now``.

    ``now`` has to already be in ``ui.timezone``. ``02:00`` means 2 AM in
    that zone, not 02:00 UTC, even when the container clock is UTC.
    The job is due from that minute through ``window_seconds``.
    """
    hour_text, minute_text = schedule.split(":")
    scheduled = now.replace(
        hour=int(hour_text),
        minute=int(minute_text),
        second=0,
        microsecond=0,
    )
    if now < scheduled:
        return False
    return (now - scheduled).total_seconds() <= window_seconds


def assign_label_texts(
    timestamps: Sequence[float],
    suffixes: Sequence[str | None],
    zone: tzinfo,
) -> list[str]:
    """Build the on-screen label for each timestamp.

    Times omit seconds. When two or more labels fall in the same clock
    minute, each of those labels shows seconds instead of a count such
    as ``(2)``.
    """
    if len(timestamps) != len(suffixes):
        raise ValueError("timestamps and suffixes must be the same length")
    minutes = [format_clock(ts, zone, seconds=False) for ts in timestamps]
    counts = Counter(minutes)
    texts: list[str] = []
    for timestamp, minute, suffix in zip(timestamps, minutes, suffixes, strict=True):
        show_seconds = counts[minute] > 1
        text = format_clock(timestamp, zone, seconds=show_seconds)
        if suffix:
            text = f"{text} {suffix}"
        texts.append(text)
    return texts


def path_displacement(
    boxes: Sequence[tuple[float, float, float, float]],
    width: float,
    height: float,
) -> float:
    """Largest travel of the box center, as a fraction of the frame."""
    if len(boxes) < 2 or width <= 0 or height <= 0:
        return 0.0
    centers = [
        ((box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height) for box in boxes
    ]
    span = 0.0
    for left in centers:
        for right in centers:
            span = max(span, math.hypot(left[0] - right[0], left[1] - right[1]))
    return span


def is_stationary(
    boxes: Sequence[tuple[float, float, float, float]],
    width: float,
    height: float,
    threshold: float = DEFAULT_STATIONARY,
) -> bool:
    """True when the track barely moves (a parked vehicle re-trigger)."""
    return path_displacement(boxes, width, height) < threshold


def _height(box: tuple[float, float, float, float]) -> float:
    return max(1.0, box[3] - box[1])


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _overlap_fraction(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Fraction of ``inner`` covered by ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ix * iy / area


def _nearest_index(times: Sequence[float], target: float) -> int:
    return min(range(len(times)), key=lambda index: abs(times[index] - target))


def _pace(track: MotionTrack, kind: str) -> tuple[float, float]:
    """Return (pace at the junction, occlusion) in body-heights.

    Pace is how far the person moves over about one second at the start
    (got out) or the end (got in), in body heights per second. Occlusion
    is how much shorter the person is at the junction than their median
    height: someone ducking into a car gets shorter.
    """
    boxes = track.boxes
    times = track.times
    heights = [_height(box) for box in boxes]
    median_h = float(np.median(heights))
    if kind == "got out":
        junction = boxes[0]
        ref = boxes[_nearest_index(times, times[0] + 1.0)]
    else:
        junction = boxes[-1]
        ref = boxes[_nearest_index(times, times[-1] - 1.0)]
    person_h = _height(junction)
    moved = math.hypot(_center(ref)[0] - _center(junction)[0], ref[3] - junction[3])
    pace = moved / person_h
    occlusion = 1.0 - person_h / max(1.0, median_h)
    return pace, occlusion


def _time_near(vehicle: MotionTrack, moment: float, slack: float = 90.0) -> bool:
    return vehicle.start - slack <= moment <= vehicle.end + slack


def link_parked_cars(
    people: Sequence[MotionTrack],
    vehicles: Sequence[MotionTrack],
    width: float,
    height: float,
    stationary_threshold: float = DEFAULT_STATIONARY,
) -> list[ParkedLink]:
    """Link a person to a parked car they get out of or into.

    Stationary vehicles are not otherwise shown. A link is created when
    the person's track starts (got out) or ends (got in) against a
    stationary vehicle, they actually leave or approach it, and they are
    not still walking at full stride (that is someone passing the car).
    """
    parked = [
        vehicle
        for vehicle in vehicles
        if vehicle.category == "vehicle"
        and vehicle.label != "bicycle"
        and is_stationary(vehicle.boxes, width, height, stationary_threshold)
        and vehicle.boxes
    ]
    links: list[ParkedLink] = []
    for person in people:
        if person.label != "person" or len(person.boxes) < 2 or not person.times:
            continue
        for kind in ("got out", "got in"):
            junction = person.boxes[0] if kind == "got out" else person.boxes[-1]
            far = person.boxes[-1] if kind == "got out" else person.boxes[0]
            person_h = _height(junction)
            if (
                junction[0] < 0.01 * width
                or junction[2] > 0.99 * width
                or junction[3] > 0.985 * height
            ):
                continue
            away = math.hypot(
                _center(far)[0] - _center(junction)[0],
                _center(far)[1] - _center(junction)[1],
            )
            if away < 1.2 * person_h:
                continue
            pace, occlusion = _pace(person, kind)
            best: tuple[float, tuple[float, float, float, float], str] | None = None
            for car in parked:
                moment = person.start if kind == "got out" else person.end
                if not _time_near(car, moment):
                    continue
                car_box = car.boxes[len(car.boxes) // 2]
                car_w = car_box[2] - car_box[0]
                car_h = car_box[3] - car_box[1]
                if car_w < 0.8 * person_h or car_w < 0.05 * width:
                    continue
                person_w = junction[2] - junction[0]
                pad_x = max(0.12 * car_w, 0.8 * person_w)
                expanded = (
                    car_box[0] - pad_x,
                    car_box[1] - 0.1 * car_h,
                    car_box[2] + pad_x,
                    car_box[3] + 0.1 * car_h,
                )
                overlap = _overlap_fraction(junction, expanded)
                foot_ok = (
                    car_box[1] + 0.2 * car_h <= junction[3] <= car_box[3] + 0.3 * car_h
                )
                if overlap >= 0.2 and foot_ok and (best is None or overlap > best[0]):
                    best = (overlap, car_box, car.id)
            if best is None:
                continue
            # Still at a full walking pace and fully visible: they walked past.
            if pace > 1.2 and occlusion < 0.4:
                continue
            moment = person.start if kind == "got out" else person.end
            links.append(
                ParkedLink(
                    person_id=person.id,
                    vehicle_id=best[2],
                    kind=kind,
                    ts=moment,
                    car_box=best[1],
                )
            )
    return links


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = ix * iy
    area = (left[2] - left[0]) * (left[3] - left[1])
    area += (right[2] - right[0]) * (right[3] - right[1])
    area -= inter
    if area <= 0:
        return 0.0
    return inter / area


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dedupe_tracks(tracks: Sequence[MotionTrack]) -> dict[int, int]:
    """Drop a second track of the same physical object.

    Returns ``{dropped index: kept index}``. Parked-car synthetics are
    not passed in here. Two people sitting side by side are kept: only
    tracks whose boxes cover each other are merged.
    """
    dropped: dict[int, int] = {}
    order = sorted(range(len(tracks)), key=lambda index: tracks[index].start)
    for position, left_i in enumerate(order):
        if left_i in dropped:
            continue
        left = tracks[left_i]
        for right_i in order[position + 1 :]:
            if right_i in dropped:
                continue
            right = tracks[right_i]
            if right.start > left.end:
                break
            if left.category == "parked" or right.category == "parked":
                continue
            compared = 0
            same = 0
            for box, moment in zip(left.boxes, left.times, strict=True):
                other_i = _nearest_index(right.times, moment)
                if abs(right.times[other_i] - moment) > 0.4:
                    continue
                compared += 1
                if _iou(box, right.boxes[other_i]) > 0.45:
                    same += 1
            if (
                compared >= 0.4 * min(len(left.boxes), len(right.boxes))
                and same >= 0.6 * compared
            ):
                left_area = float(np.mean([_area(box) for box in left.boxes]))
                right_area = float(np.mean([_area(box) for box in right.boxes]))
                keep, lose = (
                    (left_i, right_i) if left_area >= right_area else (right_i, left_i)
                )
                dropped[lose] = keep
                if lose == left_i:
                    break
    return dropped


def repeat_for_min_show(count: int, output_fps: int, min_show_seconds: float) -> int:
    """How many times to repeat each source frame so the ghost is readable."""
    if count <= 0:
        return 1
    need = max(1, int(math.ceil(min_show_seconds * output_fps)))
    return max(1, int(math.ceil(need / count)))


def schedule_units(
    units: Sequence[ScheduledUnit],
    width: int,
    height: int,
    target_frames: int,
    *,
    cell: int = 24,
    max_delay: int = 30,
    max_active: int = 8,
) -> tuple[list[int], int]:
    """Place units on the synopsis timeline with little spatial overlap.

    Linked units (a parked car tied to a person) start on the same frame
    as the person. Returns ``(start frame per unit, total frames)``.
    """
    if not units:
        return [], 1
    grid_w = max(1, math.ceil(width / cell))
    grid_h = max(1, math.ceil(height / cell))
    cells: list[list[tuple[int, int, int, int]]] = []
    for unit in units:
        occupied = []
        for box in unit.boxes:
            x0 = max(0, int(box[0] // cell))
            y0 = max(0, int(box[1] // cell) - 2)
            x1 = min(grid_w, int(box[2] // cell) + 1)
            y1 = min(grid_h, int(box[3] // cell) + 1)
            if x1 <= x0:
                x1 = min(grid_w, x0 + 1)
            if y1 <= y0:
                y1 = min(grid_h, y0 + 1)
            occupied.append((y0, y1, x0, x1))
        cells.append(occupied)

    horizon = sum(len(unit.boxes) for unit in units) * 2 + 10 + max_delay * 2
    horizon = max(horizon, target_frames + max_delay + 5)
    link = [unit.link_index for unit in units]
    weight = [
        1 + sum(1 for other in link if other == index) for index in range(len(units))
    ]

    def place(threshold: float) -> tuple[list[int], int]:
        occupancy = np.zeros((horizon, grid_h, grid_w), np.uint16)
        active = np.zeros(horizon, np.int32)
        starts: list[int] = []
        last = 0
        for index, occupied in enumerate(cells):
            length = len(occupied)
            partner = link[index]
            if partner is not None and partner < len(starts):
                start = starts[partner]
                if start + length >= horizon:
                    start = max(0, horizon - length - 1)
                for step, bounds in enumerate(occupied):
                    occupancy[
                        start + step, bounds[0] : bounds[1], bounds[2] : bounds[3]
                    ] += 1
                starts.append(start)
                continue
            area = sum((b[1] - b[0]) * (b[3] - b[2]) for b in occupied) or 1
            limit = threshold * area
            start0 = last
            if max_active:
                while (
                    start0 + length < horizon
                    and int(active[start0 : start0 + length].max())
                    > max_active - weight[index]
                ):
                    start0 += 1
            best_start = start0
            best_cost: int | None = None
            for start in range(start0, min(horizon - length, start0 + max_delay + 1)):
                if (
                    max_active
                    and int(active[start : start + length].max())
                    > max_active - weight[index]
                ):
                    continue
                cost = 0
                for step, bounds in enumerate(occupied):
                    cost += int(
                        occupancy[
                            start + step, bounds[0] : bounds[1], bounds[2] : bounds[3]
                        ].sum()
                    )
                    if best_cost is not None and cost >= best_cost:
                        break
                if best_cost is None or cost < best_cost:
                    best_start, best_cost = start, cost
                if cost <= limit:
                    best_start = start
                    break
            start = best_start
            end = min(horizon, start + length)
            active[start:end] += weight[index]
            for step, bounds in enumerate(occupied):
                if start + step >= horizon:
                    break
                occupancy[
                    start + step, bounds[0] : bounds[1], bounds[2] : bounds[3]
                ] += 1
            starts.append(start)
            last = start
        length = max(
            (
                start + len(occupied)
                for start, occupied in zip(starts, cells, strict=True)
            ),
            default=1,
        )
        return starts, min(length, horizon)

    starts, length = place(0.0)
    if length <= target_frames:
        return starts, length
    low, high = 0.0, 0.5
    placed = place(high)
    while placed[1] > target_frames and high <= 64:
        low, high = high, high * 2
        placed = place(high)
    best = placed
    for _ in range(7):
        mid = (low + high) / 2
        trial = place(mid)
        if trial[1] <= target_frames:
            high, best = mid, trial
        else:
            low = mid
    return best


def _rects_hit(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    return (
        left[0] - margin < right[2]
        and right[0] - margin < left[2]
        and left[1] - margin < right[3]
        and right[1] - margin < left[3]
    )


def plan_label_rects(
    units: Sequence[ScheduledUnit],
    starts: Sequence[int],
    width: int,
    height: int,
    *,
    text_size: dict[int, tuple[int, int]],
    header_h: int,
    min_gap: int,
) -> list[tuple[int, int, int, int]]:
    """One stationary rectangle per unit. Rectangles of units that are
    on screen together do not overlap, and they stay ``min_gap`` apart.
    """
    gap = max(0, min_gap)
    info: list[dict[str, object]] = []
    for index, unit in enumerate(units):
        text_w, text_h = text_size.get(index, (80, 16))
        label_h = text_h + 12
        label_w = text_w + 14 + label_h
        boxes = (
            np.array(unit.boxes, np.float32)
            if unit.boxes
            else np.zeros((1, 4), np.float32)
        )
        info.append(
            {
                "lw": label_w,
                "lh": label_h,
                "boxes": boxes,
                "start": starts[index],
                "end": starts[index] + len(unit.boxes),
            }
        )
    placed: dict[int, tuple[int, int, int, int]] = {}
    order = sorted(range(len(units)), key=lambda index: starts[index])
    for index in order:
        item = info[index]
        label_w = int(item["lw"])
        label_h = int(item["lh"])
        boxes = item["boxes"]
        assert isinstance(boxes, np.ndarray)
        start = int(item["start"])
        end = int(item["end"])
        conflicts = [
            other
            for other in placed
            if int(info[other]["start"]) < end + 3
            and start < int(info[other]["end"]) + 3
        ]
        heads_x = (boxes[:, 0] + boxes[:, 2]) / 2
        xs: set[int] = set()
        count = len(boxes)
        for fraction in (0.0, 0.5, 1.0):
            head_x = float(heads_x[int(fraction * (count - 1))])
            for shift in (0.0, -0.6 * label_w, 0.6 * label_w):
                xs.add(
                    int(
                        min(
                            max(head_x + shift, label_w / 2 + 2),
                            width - label_w / 2 - 2,
                        )
                    )
                )
        top = float(boxes[:, 1].min())
        bottom = float(boxes[:, 3].max())
        ys: list[int] = []
        for band in range(14):
            y1 = top - 10 - band * (label_h + gap)
            if y1 - label_h >= header_h + 4:
                ys.append(int(y1))
        above = len(ys)
        for band in range(4):
            y1 = bottom + 10 + label_h + band * (label_h + gap)
            if y1 <= height - 4:
                ys.append(int(y1))
        best: tuple[int, int, int, int] | None = None
        best_cost: float | None = None
        for center_x in xs:
            for y_index, y1 in enumerate(ys):
                rect = (
                    int(center_x - label_w / 2),
                    int(y1 - label_h),
                    int(center_x + label_w / 2),
                    int(y1),
                )
                hard = sum(
                    1 for other in conflicts if _rects_hit(rect, placed[other], gap)
                )
                cost = 1e5 * hard + (80 if y_index >= above else 0)
                cost += 400 * sum(
                    1
                    for box in boxes[:: max(1, count // 6)]
                    if _rects_hit(rect, tuple(int(v) for v in box))
                )
                if best_cost is None or cost < best_cost:
                    best, best_cost = rect, cost
                if hard == 0 and y_index < above:
                    best = rect
                    best_cost = cost
                    break
            if best_cost is not None and best_cost < 1:
                break
        if best is None:
            head_x = float(heads_x[0])
            head_y = float(boxes[0, 1])
            y0 = int(max(header_h + 4, head_y - 10 - label_h))
            best = (
                int(head_x - label_w / 2),
                y0,
                int(head_x + label_w / 2),
                y0 + label_h,
            )
        placed[index] = best
    return [placed[index] for index in range(len(units))]
