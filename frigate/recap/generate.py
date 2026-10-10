"""Turn Frigate events into one synopsis video.

Frame loading is injected so this module does not talk to the database.
The job in ``manager`` loads recordings and calls ``generate_recap``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frigate.config.recap import RecapConfig
from frigate.recap import cutcache
from frigate.recap.align import SHIFT_HIGH, SHIFT_LOW, estimate_shift
from frigate.recap.categories import (
    CAT_COLOR,
    CAT_NAME,
    CAT_ORDER,
    DELIVERY_LABELS,
    bgr_hex,
    category_of,
)
from frigate.recap.cutout import (
    build_cutouts,
    needs_window,
    parked_car_ghost,
    window_ghost,
)
from frigate.recap.layout import (
    MotionTrack,
    ScheduledUnit,
    assign_label_texts,
    dedupe_tracks,
    is_stationary,
    link_parked_cars,
    repeat_for_min_show,
    schedule_units,
)
from frigate.recap.plates import TimedPlate, build_timed_plates, cutout_is_visible
from frigate.recap.render import GhostFrame, Tube, encode_video, layout_labels

logger = logging.getLogger(__name__)

LoadClip = Callable[
    [dict[str, Any], float, int, int],
    tuple[list[np.ndarray], list[float]] | None,
]
# Frigate's saved snapshot for an event (BGR), or None.
LoadSnapshot = Callable[[dict[str, Any]], "np.ndarray | None"]

# A snapshot held in place stays on screen at least this long.
STILL_SECONDS = 3.0
# Share of the aligned boxes that must be motion before a moving cutout
# is trusted. Below this the path did not line up with the recording.
ALIGNED_FILL = 0.25


def classify_event(
    event: dict[str, Any],
    delivery_ids: set[str],
    dog_walker_ids: set[str],
) -> str:
    """People, vehicles, animals, or deliveries.

    Frigate+ delivery attributes win. Semantic-search hits are the
    fallback the standalone prototype uses when those labels are absent.
    A person semantic-matched as walking a dog is shown as an animal.
    """
    label = str(event.get("label") or "")
    category = category_of(label)
    sub_label = str(event.get("sub_label") or "").lower()
    if label.lower() in DELIVERY_LABELS or sub_label in DELIVERY_LABELS:
        return "delivery"
    attributes = _event_data(event).get("attributes") or []
    if not isinstance(attributes, list):
        attributes = []
    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        name = str(attribute.get("label") or "").lower()
        score = float(attribute.get("score") or 0)
        if name in DELIVERY_LABELS and score >= 0.5:
            return "delivery"
    if event.get("id") in delivery_ids and category in ("person", "vehicle"):
        return "delivery"
    if label == "person" and event.get("id") in dog_walker_ids:
        return "animal"
    return category


def _path_points(event: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Normalized ``(x, bottom_y, timestamp)`` samples, oldest first."""
    raw = _event_data(event).get("path_data") or []
    if not isinstance(raw, list):
        raw = []
    points: list[tuple[float, float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        point, stamp = item[0], item[1]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1]), float(stamp)))
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda item: item[2])
    return points


def _snapshot_box(event: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Normalized xywh box, or None when the event has no box."""
    box = _event_data(event).get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    except (TypeError, ValueError):
        return None


def timeline_boxes(
    event: dict[str, Any],
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Exact detector boxes from Frigate's timeline, oldest first.

    ``event["timeline"]`` comes from ``queries.attach_timeline``. Each box
    is normalized xywh at the moment the row was written.
    """
    raw = event.get("timeline") or []
    if not isinstance(raw, list):
        return []
    rows: list[tuple[float, tuple[float, float, float, float]]] = []
    for item in raw:
        try:
            stamp = float(item[0])
            values = [float(value) for value in item[2][:4]]
        except (TypeError, ValueError, IndexError):
            continue
        if len(values) < 4 or values[2] <= 0 or values[3] <= 0:
            continue
        rows.append((stamp, (values[0], values[1], values[2], values[3])))
    rows.sort(key=lambda row: row[0])
    return rows


def _foot_points(event: dict[str, Any]) -> list[tuple[float, float, float]]:
    """``(time, x, bottom_y)`` from the path and the timeline, oldest first."""
    feet = [(stamp, x, y) for x, y, stamp in _path_points(event)]
    feet += [
        (stamp, box[0] + box[2] / 2, box[1] + box[3])
        for stamp, box in timeline_boxes(event)
    ]
    feet.sort(key=lambda item: item[0])
    return feet


def _sizes(event: dict[str, Any]) -> list[tuple[float, float, float]]:
    """``(time, w, h)`` known sizes, oldest first.

    Timeline rows carry the exact size. The snapshot size belongs to the
    best moment. A car coming up the street grows, so one size for the
    whole clip left a box far too big or too small most of the time.
    """
    sizes = [(stamp, box[2], box[3]) for stamp, box in timeline_boxes(event)]
    snapshot = _snapshot_box(event)
    if snapshot is not None and snapshot[2] > 0 and snapshot[3] > 0:
        moment = best_moment(event)
        if moment is None:
            if not sizes:
                sizes.append(
                    (float(event.get("start_time") or 0.0), snapshot[2], snapshot[3])
                )
        elif all(abs(stamp - moment) > 0.5 for stamp, _w, _h in sizes):
            # The snapshot's time is only the nearest path sample. A
            # timeline row near it is exact, so it wins.
            sizes.append((moment, snapshot[2], snapshot[3]))
    sizes.sort(key=lambda item: item[0])
    return sizes


def boxes_at(
    event: dict[str, Any],
    times: Sequence[float],
    width: int,
    height: int,
    shift: float = 0.0,
) -> list[tuple[float, float, float, float]] | None:
    """Pixel xyxy boxes at ``times``, from every exact sample Frigate kept.

    The foot point follows the path and the timeline. The size follows
    the timeline and the snapshot. ``times`` are recording times.
    ``shift`` is added to each one to get the detector time those samples
    were stamped with (see ``align``).
    """
    if not times:
        return None
    sizes = _sizes(event)
    if not sizes:
        return None
    feet = _foot_points(event)
    snapshot = _snapshot_box(event)
    if not feet:
        if snapshot is None:
            return None
        x0 = snapshot[0] * width
        y0 = snapshot[1] * height
        fixed = _clamp_box(
            (x0, y0, x0 + snapshot[2] * width, y0 + snapshot[3] * height),
            width,
            height,
        )
        return [fixed for _ in times]
    foot_t = [item[0] for item in feet]
    foot_x = [item[1] for item in feet]
    foot_y = [item[2] for item in feet]
    size_t = [item[0] for item in sizes]
    size_w = [item[1] for item in sizes]
    size_h = [item[2] for item in sizes]
    boxes: list[tuple[float, float, float, float]] = []
    for moment in times:
        when = float(moment) + shift
        center_x = float(np.interp(when, foot_t, foot_x)) * width
        bottom = float(np.interp(when, foot_t, foot_y)) * height
        box_w = max(2.0, float(np.interp(when, size_t, size_w)) * width)
        box_h = max(2.0, float(np.interp(when, size_t, size_h)) * height)
        boxes.append(
            _clamp_box(
                (center_x - box_w / 2, bottom - box_h, center_x + box_w / 2, bottom),
                width,
                height,
            )
        )
    return boxes


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    """Event ``data`` JSON, or an empty dict when it is missing."""
    data = event.get("data")
    if isinstance(data, dict):
        return data
    return {}


def snapshot_area(event: dict[str, Any]) -> float | None:
    """Area of the snapshot box as a fraction of the frame.

    Frigate keeps the best view of the object, so this is about as large
    as the object ever gets on screen.
    """
    box = _snapshot_box(event)
    if box is None:
        return None
    return max(0.0, box[2]) * max(0.0, box[3])


def too_small(event: dict[str, Any], category: str, min_area: float) -> bool:
    """True when the object is never big enough on screen to recognize.

    A car on the far street is a few dozen pixels. It gets a label and a
    leader and still cannot be told apart. People and animals are
    narrower than cars, so they use a third of the vehicle limit.
    """
    if min_area <= 0:
        return False
    area = snapshot_area(event)
    if area is None:
        return False
    limit = min_area if category in ("vehicle", "parked") else min_area / 3
    return area < limit


def best_moment(event: dict[str, Any]) -> float | None:
    """Detector time when the object stood where its snapshot was taken.

    The snapshot is Frigate's best view of the object, usually the
    closest and clearest one. Returns None when nothing places it in time.
    """
    snapshot = _snapshot_box(event)
    candidates = [(stamp, x, y) for x, y, stamp in _path_points(event)]
    candidates += [
        (stamp, box[0] + box[2] / 2, box[1] + box[3])
        for stamp, box in timeline_boxes(event)
    ]
    if snapshot is None or len(candidates) < 2:
        return None
    foot_x = snapshot[0] + snapshot[2] / 2
    foot_y = snapshot[1] + snapshot[3]
    nearest = min(
        candidates,
        key=lambda item: (item[1] - foot_x) ** 2 + (item[2] - foot_y) ** 2,
    )
    return nearest[0]


def last_seen(event: dict[str, Any]) -> float:
    """Detector time of the last sighting.

    Frigate keeps an event open for a few seconds after the object is
    gone (about 3 seconds here, up to 5 or more). Those frames are
    empty road, and a box left at the last position turned them into
    see-through ghosts.
    """
    start, end = _event_span(event)
    stamps = [stamp for stamp, _box in timeline_boxes(event)]
    stamps += [stamp for _x, _y, stamp in _path_points(event)]
    if not stamps:
        return end
    return min(end, max(max(stamps), start + 0.5))


def clip_window(event: dict[str, Any], max_seconds: float) -> tuple[float, float]:
    """Stretch of the event to cut out, at most ``max_seconds`` long.

    It ends at the last sighting, not at the event end. A long event
    (someone loitering, a car waiting at the stop sign) is trimmed to the
    part around its best view instead of its first seconds.
    """
    start, _end = _event_span(event)
    end = max(last_seen(event), start + 0.5)
    if max_seconds <= 0 or end - start <= max_seconds:
        return start, end
    anchor = best_moment(event)
    if anchor is None:
        anchor = start + max_seconds / 2
    left = min(max(start, anchor - max_seconds / 2), end - max_seconds)
    return left, left + max_seconds


def _event_span(event: dict[str, Any]) -> tuple[float, float]:
    """Start and end seconds. NULL ``end_time`` is an open event."""
    start = float(event["start_time"])
    raw = event.get("end_time")
    end = start
    if raw is not None:
        try:
            end = float(raw)
        except (TypeError, ValueError):
            end = start
    if end <= start:
        end = start + 0.5
    return start, end


def _clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x0 = min(max(0.0, box[0]), width - 2)
    y0 = min(max(0.0, box[1]), height - 2)
    x1 = min(max(x0 + 2, box[2]), float(width))
    y1 = min(max(y0 + 2, box[3]), float(height))
    return (x0, y0, x1, y1)


def preview_track(
    event: dict[str, Any],
    category: str,
    width: int,
    height: int,
) -> MotionTrack | None:
    """A cheap track used to decide parked-versus-moving before any decode."""
    start, end = _event_span(event)
    count = 8
    times = [start + (end - start) * (index + 0.5) / count for index in range(count)]
    boxes = boxes_at(event, times, width, height)
    if not boxes:
        return None
    return MotionTrack(
        id=str(event["id"]),
        label=str(event["label"]),
        category=category,
        start=start,
        end=end,
        boxes=boxes,
        times=times,
    )


def vehicle_is_parked(
    event: dict[str, Any],
    track: MotionTrack,
    width: int,
    height: int,
    threshold: float,
) -> bool:
    """Stationary vehicles are skipped unless someone later gets in or out."""
    duration = max(0.0, track.end - track.start)
    had_path = len(_path_points(event)) >= 2
    if not had_path:
        return duration >= 30
    return is_stationary(track.boxes, width, height, threshold)


def _plate_samples(
    plate_frames: Sequence[np.ndarray] | Sequence[tuple[float, np.ndarray]],
    after: float,
    before: float,
) -> list[tuple[float, np.ndarray]]:
    """Normalize bare frames or ``(time, frame)`` pairs into timed samples."""
    if not plate_frames:
        return []
    first = plate_frames[0]
    if isinstance(first, tuple):
        samples = []
        for moment, frame in plate_frames:  # type: ignore[misc]
            if frame is not None and frame.size:
                samples.append((float(moment), frame))
        return samples
    frames = [frame for frame in plate_frames if frame is not None and frame.size]  # type: ignore[union-attr]
    if not frames:
        return []
    if before <= after or len(frames) == 1:
        times = [float(after)] * len(frames)
    else:
        times = [float(item) for item in np.linspace(after, before, len(frames))]
    return list(zip(times, frames, strict=True))


def _background(
    plate_frames: Sequence[np.ndarray] | Sequence[tuple[float, np.ndarray]],
    after: float,
    before: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[TimedPlate]]:
    """Plates split by day versus infrared, sized to the synopsis."""
    timed = build_timed_plates(_plate_samples(plate_frames, after, before))
    fitted: list[TimedPlate] = []
    for plate in timed:
        image = plate.image
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        fitted.append(
            TimedPlate(
                image=image,
                time=plate.time,
                start=plate.start,
                end=plate.end,
                is_ir=plate.is_ir,
                is_dark=plate.is_dark,
            )
        )
    if not fitted:
        raise RuntimeError("No recordings to build a background from")
    return fitted[len(fitted) // 2].image, fitted


def _overlap_time(left: MotionTrack, right: MotionTrack) -> bool:
    return left.start <= right.end and right.start <= left.end


def _merge_dog_walkers(tracks: list[MotionTrack]) -> set[str]:
    """A person overlapping a dog or cat is the dog-walker, drawn in green.

    The animal's own track is dropped, because the person's cutout already
    includes whatever moves with them.
    """
    drop: set[str] = set()
    people = [track for track in tracks if track.label == "person"]
    animals = [track for track in tracks if track.category == "animal"]
    for person in people:
        for animal in animals:
            if animal.id in drop or not _overlap_time(person, animal):
                continue
            scores: list[float] = []
            for box, moment in zip(person.boxes, person.times, strict=True):
                index = min(
                    range(len(animal.times)),
                    key=lambda item: abs(animal.times[item] - moment),
                )
                if abs(animal.times[index] - moment) > 1.5:
                    continue
                animal_box = animal.boxes[index]
                ix = max(0.0, min(box[2], animal_box[2]) - max(box[0], animal_box[0]))
                iy = max(0.0, min(box[3], animal_box[3]) - max(box[1], animal_box[1]))
                area = max(
                    1.0,
                    (animal_box[2] - animal_box[0]) * (animal_box[3] - animal_box[1]),
                )
                scores.append(ix * iy / area)
            if scores and float(np.mean(scores)) >= 0.2:
                # One walker, one label. Keeping the dog's own track drew
                # the same walker twice, once moving and once held.
                person.category = "animal"
                drop.add(animal.id)
    return drop


def _clock_range(after: float, before: float, zone: tzinfo) -> str:
    start = datetime.fromtimestamp(after, zone)
    end = datetime.fromtimestamp(before, zone)
    left = start.strftime("%a %b %-d, %-I:%M %p")
    if start.strftime("%p") == end.strftime("%p") and start.date() == end.date():
        right = end.strftime("%-I:%M %p")
    else:
        right = end.strftime("%a %b %-d, %-I:%M %p")
    return f"{left} to {right}"


def generate_recap(
    *,
    camera: str,
    after: float,
    before: float,
    settings: RecapConfig,
    zone: tzinfo,
    events: list[dict[str, Any]],
    load_clip: LoadClip,
    plate_frames: list[np.ndarray] | list[tuple[float, np.ndarray]],
    out_dir: Path,
    ffmpeg: str,
    cancel: Callable[[], bool],
    progress: Callable[[float, str], None],
    delivery_ids: set[str] | None = None,
    dog_walker_ids: set[str] | None = None,
    width: int,
    height: int,
    cache_root: Path | None = None,
    cache_stats: cutcache.CacheStats | None = None,
    use_cache: bool = False,
    time_shift: float = 0.0,
    load_snapshot: LoadSnapshot | None = None,
) -> dict[str, Any]:
    """Build ``video.mp4`` and return the manifest body (without status).

    With ``use_cache`` each event's cutouts are read from, or written to,
    the per-event cache in ``cutcache`` so only new events are decoded.
    ``time_shift`` is the camera's starting guess for recording time to
    detector time (``-annotation_offset``). Each event refines it.
    ``load_snapshot`` returns Frigate's saved snapshot for an event. It is
    shown instead of a clip at night and when a clip does not line up.
    """
    delivery_ids = delivery_ids or set()
    stats = cache_stats if cache_stats is not None else cutcache.CacheStats()
    dog_walker_ids = dog_walker_ids or set()
    progress(8, "Choosing objects")
    if cancel():
        raise RecapCancelled()

    classified: list[tuple[dict[str, Any], str, MotionTrack]] = []
    excluded: list[dict[str, Any]] = []
    for event in events:
        if cancel():
            raise RecapCancelled()
        category = classify_event(event, delivery_ids, dog_walker_ids)
        track = preview_track(event, category, width, height)
        if track is None:
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "no box",
                    "start_time": event.get("start_time"),
                }
            )
            continue
        if (
            category == "vehicle"
            and settings.parked_cars
            and vehicle_is_parked(event, track, width, height, settings.stationary_path)
        ):
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "parked",
                    "start_time": event.get("start_time"),
                    "category": "vehicle",
                }
            )
            # Kept aside so a person getting in or out can still find the car.
            track.category = "vehicle"
            classified.append((event, "parked-candidate", track))
            continue
        if too_small(event, category, settings.min_object_area):
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "too small to see",
                    "start_time": event.get("start_time"),
                }
            )
            continue
        classified.append((event, category, track))

    active = [
        (event, category, track)
        for event, category, track in classified
        if category != "parked-candidate"
    ]
    parked_candidates = [
        track
        for _event, category, track in classified
        if category == "parked-candidate"
    ]
    if len(active) > settings.max_events:
        logger.warning(
            "Recap for %s has %d objects, keeping %d",
            camera,
            len(active),
            settings.max_events,
        )
        active, overflow = cap_newest(active, settings.max_events)
        for event, _category, _track in overflow:
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "over the event cap",
                    "start_time": event.get("start_time"),
                }
            )

    plate, timed_plates = _background(plate_frames, after, before, width, height)
    logger.info(
        "Recap background for %s: %d plates, %d infrared",
        camera,
        len(timed_plates),
        sum(1 for item in timed_plates if item.is_ir),
    )

    tubes: list[Tube] = []
    tracks: list[MotionTrack] = []
    total = max(1, len(active))
    for index, (event, category, preview) in enumerate(active):
        if cancel():
            raise RecapCancelled()
        progress(
            12 + 60 * index / total,
            f"Cutting out {event.get('label')} {index + 1} of {total}",
        )
        sample_fps = (
            settings.vehicle_sample_fps
            if category == "vehicle"
            else settings.sample_fps
        )
        cut = _cutouts_for(
            event,
            category,
            sample_fps,
            settings,
            width,
            height,
            load_clip,
            use_cache=use_cache,
            cache_root=cache_root,
            stats=stats,
            time_shift=time_shift,
            load_snapshot=load_snapshot,
        )
        if cut.status == "no_frames":
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "no frames",
                    "start_time": event.get("start_time"),
                }
            )
            continue
        if cut.status == "no_box":
            continue
        if cut.status != "ok":
            excluded.append(
                {
                    "id": event.get("id"),
                    "label": event.get("label"),
                    "reason": "could not separate from the background",
                    "start_time": event.get("start_time"),
                }
            )
            continue
        ghost_frames = [_ghost_from_packed(item) for item in cut.frames]
        boxes = list(cut.boxes)
        times = list(cut.times)
        context_first = cut.context_first
        context_last = cut.context_last
        count = min(len(ghost_frames), len(boxes), len(times))
        ghost_frames = ghost_frames[:count]
        boxes = boxes[:count]
        times = times[:count]
        start, end = _event_span(event)
        track = MotionTrack(
            id=str(event["id"]),
            label=str(event["label"]),
            category=category,
            start=start,
            end=end,
            boxes=boxes,
            times=list(times),
        )
        tubes.append(
            Tube(
                event_id=track.id,
                clip_event_id=track.id,
                label=track.label,
                category=category,
                start_time=track.start,
                frames=ghost_frames,
                context_first=context_first,
                context_last=context_last,
                still=cut.still,
            )
        )
        tracks.append(track)
        if settings.pause_seconds and not cut.from_cache:
            time.sleep(settings.pause_seconds)

    dropped_animals = _merge_dog_walkers(tracks)
    for track in tracks:
        if track.id in dropped_animals:
            continue
        for tube in tubes:
            if tube.event_id == track.id:
                tube.category = track.category

    keep_ids = {track.id for track in tracks if track.id not in dropped_animals}
    drop_map = dedupe_tracks([track for track in tracks if track.id in keep_ids])
    ordered = [track for track in tracks if track.id in keep_ids]
    dropped_ids = {ordered[index].id for index in drop_map}
    for event_id in dropped_ids | dropped_animals:
        excluded.append(
            {
                "id": event_id,
                "label": next(
                    (track.label for track in tracks if track.id == event_id), ""
                ),
                "reason": "same object as another track",
                "start_time": next(
                    (track.start for track in tracks if track.id == event_id), None
                ),
            }
        )
    tracks = [track for track in ordered if track.id not in dropped_ids]
    tubes = [
        tube
        for tube in tubes
        if tube.event_id not in dropped_ids and tube.event_id not in dropped_animals
    ]
    by_id = {tube.event_id: tube for tube in tubes}

    if settings.parked_cars and tracks:
        links = link_parked_cars(
            [track for track in tracks if track.label == "person"],
            parked_candidates,
            width,
            height,
            settings.stationary_path,
        )
    else:
        links = []

    progress(78, "Placing labels")
    units: list[ScheduledUnit] = []
    repeats: list[int] = []
    # Chronological, so the plate moves from afternoon into night.
    for track in sorted(tracks, key=lambda item: item.start):
        tube = by_id.get(track.id)
        if tube is None:
            continue
        hold = settings.min_show_seconds
        if tube.still:
            hold = max(hold, STILL_SECONDS)
        repeat = repeat_for_min_show(len(track.boxes), settings.output_fps, hold)
        expanded = [box for box in track.boxes for _ in range(repeat)]
        units.append(
            ScheduledUnit(
                event_id=track.id,
                clip_event_id=track.id,
                label=track.label,
                category=track.category,
                start_time=track.start,
                boxes=expanded,
                member_ids=[track.id],
            )
        )
        repeats.append(repeat)
        tube.category = track.category

    person_unit = {unit.event_id: index for index, unit in enumerate(units)}
    for link in links:
        person_index = person_unit.get(link.person_id)
        if person_index is None:
            continue
        person = units[person_index]
        source = by_id.get(link.person_id)
        which = "last" if link.kind == "got out" else "first"
        frame = source.context_image(which) if source else None
        if frame is None and source is not None:
            frame = source.context_image("first" if which == "last" else "last")
        if frame is None:
            continue
        ghost = parked_car_ghost(frame, link.car_box)
        if ghost is None:
            continue
        parked_id = f"parked-{link.person_id}-{link.kind.replace(' ', '-')}"
        ghost_frame = GhostFrame(
            x=int(ghost["x"]),
            y=int(ghost["y"]),
            box=ghost["box"],  # type: ignore[arg-type]
            crop=ghost["crop"],  # type: ignore[arg-type]
            alpha=ghost["alpha"],  # type: ignore[arg-type]
            window=bool(ghost.get("window")),
        )
        ghost_frame.pack()
        clip_id = link.vehicle_id or link.person_id
        tubes.append(
            Tube(
                event_id=parked_id,
                clip_event_id=clip_id,
                label="car",
                category="parked",
                start_time=link.ts,
                frames=[ghost_frame],
                suffix=link.kind,
                link_event_id=link.person_id,
            )
        )
        units.append(
            ScheduledUnit(
                event_id=parked_id,
                clip_event_id=clip_id,
                label="car",
                category="parked",
                start_time=link.ts,
                boxes=[link.car_box for _ in person.boxes],
                suffix=link.kind,
                link_index=person_index,
                member_ids=[parked_id],
            )
        )
        repeats.append(max(1, len(person.boxes)))

    if not units:
        return _empty_video(
            plate,
            camera,
            after,
            before,
            zone,
            settings,
            out_dir,
            ffmpeg,
            excluded,
            cancel,
            progress,
        )

    texts = assign_label_texts(
        [unit.start_time for unit in units],
        [unit.suffix for unit in units],
        zone,
    )
    for unit, text in zip(units, texts, strict=True):
        unit.text = text

    target = int(settings.target_length * settings.output_fps)
    starts, frame_count = schedule_units(
        units,
        width,
        height,
        target,
        max_delay=int(6 * settings.output_fps),
        max_active=settings.max_labels,
        max_overlap=settings.max_overlap,
    )
    header_h = max(28, int(round(0.075 * height)))
    rects = layout_labels(
        units,
        starts,
        width,
        height,
        font_scale=settings.font_scale,
        header_h=header_h,
        min_gap=settings.min_label_gap,
    )
    shown = sum(1 for unit in units if unit.category != "parked")
    header = f"{camera}  {_clock_range(after, before, zone)}  {shown} shown"
    progress(88, "Encoding video")
    thumb = encode_video(
        plate,
        units,
        tubes,
        starts,
        rects,
        frame_count,
        str(out_dir / "video.mp4"),
        ffmpeg=ffmpeg,
        fps=settings.output_fps,
        header=header,
        header_h=header_h,
        fade_frames=max(1, int(round(settings.fade_seconds * settings.output_fps))),
        label_opacity=settings.label_opacity,
        font_scale=settings.font_scale,
        repeats=repeats,
        cancel_check=cancel,
        plates=timed_plates,
        plate_fade_frames=max(1, int(round(0.5 * settings.output_fps))),
    )
    if thumb is None:
        raise RecapCancelled()
    cv2.imwrite(str(out_dir / "thumb.jpg"), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return _manifest_body(
        camera,
        after,
        before,
        settings,
        width,
        height,
        frame_count,
        units,
        starts,
        rects,
        excluded,
        links,
    )


def cap_newest(
    active: list[tuple[dict[str, Any], str, MotionTrack]], max_events: int
) -> tuple[
    list[tuple[dict[str, Any], str, MotionTrack]],
    list[tuple[dict[str, Any], str, MotionTrack]],
]:
    """Keep the newest ``max_events`` objects, oldest first.

    A busy window used to keep the first objects and drop the most recent
    ones, which is the activity a viewer most wants to see.
    """
    ordered = sorted(active, key=lambda item: item[2].start)
    if max_events <= 0:
        return [], ordered
    if len(ordered) <= max_events:
        return ordered, []
    return ordered[-max_events:], ordered[:-max_events]


@dataclass
class _Cut:
    status: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    context_first: bytes | None = None
    context_last: bytes | None = None
    from_cache: bool = False
    # Frigate's snapshot of the object, held in place, instead of a clip.
    still: bool = False


def _ghost_from_packed(item: dict[str, Any]) -> GhostFrame:
    return GhostFrame(
        x=int(item["x"]),
        y=int(item["y"]),
        box=tuple(item["box"]),  # type: ignore[arg-type]
        jpeg=item.get("jpeg"),
        alpha_shape=tuple(item["alpha_shape"]) if item.get("alpha_shape") else None,  # type: ignore[arg-type]
        alpha_bytes=item.get("alpha_bytes"),
        crop=item.get("crop"),
        alpha=item.get("alpha"),
        window=bool(item.get("window")),
    )


def _keep_visible(cut: _Cut, width: int, height: int) -> _Cut:
    """Drop empty masks and boxes that sit mostly off the frame.

    Those frames used to keep a label and a leader with nothing under them.
    """
    if cut.status != "ok":
        return cut
    frames: list[dict[str, Any]] = []
    boxes: list[tuple[float, float, float, float]] = []
    times: list[float] = []
    for item, box, moment in zip(cut.frames, cut.boxes, cut.times, strict=False):
        ghost = _ghost_from_packed(item)
        pixels = ghost.pixels()
        if pixels is None:
            continue
        _crop, alpha = pixels
        visible, _tip = cutout_is_visible(
            alpha, ghost.box, ghost.x, ghost.y, width, height
        )
        if not visible:
            continue
        frames.append(item)
        boxes.append(tuple(float(value) for value in box))
        times.append(float(moment))
    if not frames:
        return _Cut(status="no_cutout", from_cache=cut.from_cache, still=cut.still)
    return _Cut(
        status="ok",
        frames=frames,
        boxes=boxes,
        times=times,
        context_first=cut.context_first,
        context_last=cut.context_last,
        from_cache=cut.from_cache,
        still=cut.still,
    )


def fit_snapshot(
    image: np.ndarray | None, width: int, height: int
) -> np.ndarray | None:
    """Frigate's snapshot scaled to the synopsis.

    None when there is no snapshot, or when it was cropped (``snapshots.crop``)
    and the event box no longer lines up with it.
    """
    if image is None or getattr(image, "size", 0) == 0 or image.ndim != 3:
        return None
    image_h, image_w = image.shape[:2]
    if abs(image_w / image_h - width / height) > 0.02 * (width / height):
        return None
    if (image_w, image_h) != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def still_cut(
    event: dict[str, Any],
    category: str,
    image: np.ndarray,
    width: int,
    height: int,
) -> _Cut:
    """One window ghost cut from Frigate's snapshot of the object.

    The snapshot is the object's best view, and its box was measured on
    that exact frame, so it is always lined up and never motion blurred.
    """
    snapshot = _snapshot_box(event)
    if snapshot is None:
        return _Cut(status="no_cutout", still=True)
    box = _clamp_box(
        (
            snapshot[0] * width,
            snapshot[1] * height,
            (snapshot[0] + snapshot[2]) * width,
            (snapshot[1] + snapshot[3]) * height,
        ),
        width,
        height,
    )
    ghost = window_ghost(image, box, category)
    if ghost is None:
        return _Cut(status="no_cutout", still=True)
    frame = GhostFrame(
        x=int(ghost["x"]),  # type: ignore[arg-type]
        y=int(ghost["y"]),  # type: ignore[arg-type]
        box=box,
        crop=ghost["crop"],  # type: ignore[arg-type]
        alpha=ghost["alpha"],  # type: ignore[arg-type]
        window=True,
    )
    frame.pack()
    packed = cutcache.pack_frames([frame])
    if not packed:
        return _Cut(status="no_cutout", still=True)
    start, _end = _event_span(event)
    moment = best_moment(event)
    # A person's frame is also where a parked car they got in or out of
    # is cropped from.
    context = _jpeg(image) if str(event.get("label") or "") == "person" else None
    return _Cut(
        status="ok",
        frames=packed,
        boxes=[box],
        times=[start if moment is None else float(moment)],
        context_first=context,
        context_last=context,
        still=True,
    )


def _too_many_held(ghosts: list[dict[str, object]]) -> bool:
    """True when the object was missing from many frames of its clip.

    Those frames reuse a neighbor's ghost (repaired) or the nearest good
    one (held). A few are invisible. More than about a third looks like a
    sticker sliding along the road.
    """
    if not ghosts:
        return False
    filled = sum(1 for ghost in ghosts if ghost.get("held") or ghost.get("repaired"))
    return filled * 10 > 3 * len(ghosts)


def _mostly_windows(ghosts: list[dict[str, object]]) -> bool:
    """True when most drawn frames fell back to windows."""
    drawn = [
        ghost
        for ghost in ghosts
        if isinstance(ghost.get("alpha"), np.ndarray) and ghost["alpha"].size > 1  # type: ignore[union-attr]
    ]
    if not drawn:
        return True
    return sum(1 for ghost in drawn if ghost.get("window")) * 2 > len(drawn)


def _cutouts_for(
    event: dict[str, Any],
    category: str,
    sample_fps: float,
    settings: RecapConfig,
    width: int,
    height: int,
    load_clip: LoadClip,
    *,
    use_cache: bool,
    cache_root: Path | None,
    stats: cutcache.CacheStats,
    time_shift: float = 0.0,
    load_snapshot: LoadSnapshot | None = None,
) -> _Cut:
    """Ghost frames for one event, from the cache when possible.

    A moving cutout is used when the clip is well lit and the path lines
    up with the recording. Otherwise, and whenever the clip is infrared
    or dusk, the object is shown as Frigate's snapshot of it, held in
    place. That is the frame a person would pick to recognize it.
    """
    raw_end = event.get("end_time")
    end_time = None if raw_end is None else float(raw_end)
    key = ""
    if use_cache:
        key = cutcache.cache_key(
            str(event.get("id")),
            end_time,
            width,
            height,
            sample_fps,
            settings.max_object_seconds,
            category,
            time_shift,
        )
        hit = cutcache.load(key, cache_root)
        if hit is not None and hit.status in ("ok", "no_frames", "no_cutout"):
            stats.hits += 1
            return _keep_visible(
                _Cut(
                    status=hit.status,
                    frames=hit.frames,
                    boxes=list(hit.boxes),
                    times=list(hit.times),
                    context_first=hit.context_first,
                    context_last=hit.context_last,
                    from_cache=True,
                    still=hit.still,
                ),
                width,
                height,
            )
        stats.misses += 1

    def remember(cut: _Cut) -> _Cut:
        if not use_cache:
            return cut
        if cut.status != "ok" and not cutcache.should_cache_failure(end_time):
            return cut
        if cutcache.store(
            key,
            cutcache.CachedCutout(
                status=cut.status,
                frames=cut.frames,
                boxes=cut.boxes,
                times=cut.times,
                context_first=cut.context_first,
                context_last=cut.context_last,
                still=cut.still,
            ),
            cache_root,
        ):
            stats.stored += 1
        return cut

    image: np.ndarray | None = None
    if load_snapshot is not None:
        try:
            image = fit_snapshot(load_snapshot(event), width, height)
        except Exception:
            logger.debug("No usable snapshot for %s", event.get("id"), exc_info=True)
            image = None

    def still(reason: str) -> _Cut | None:
        if image is None:
            return None
        cut = still_cut(event, category, image, width, height)
        if cut.status != "ok":
            return None
        logger.debug(
            "Recap cutout %s %s: still (%s)", event.get("id"), category, reason
        )
        return remember(_keep_visible(cut, width, height))

    # Infrared or dusk: a clip is grain, glare, and motion blur.
    if image is not None and needs_window(image):
        held = still("infrared or dusk")
        if held is not None and held.status == "ok":
            return held

    # The stretch around the best view, in detector time, plus enough
    # recording on each side for any shift the search may pick.
    # ``load_clip`` returns the recording time of every frame it decoded.
    window_start, window_end = clip_window(event, settings.max_object_seconds)
    decode = dict(
        event,
        start_time=window_start - (time_shift + SHIFT_HIGH),
        end_time=window_end - (time_shift + SHIFT_LOW),
    )
    loaded = load_clip(decode, sample_fps, width, height)
    if not loaded:
        held = still("no recording")
        if held is not None and held.status == "ok":
            return held
        return remember(_Cut(status="no_frames"))
    frames, times = loaded
    aligned = estimate_shift(
        frames,
        times,
        lambda moments, w, h: boxes_at(event, moments, w, h),
        default=time_shift,
        valid=(window_start, window_end),
    )
    shift = aligned.seconds
    if not aligned.found or aligned.fill < ALIGNED_FILL:
        held = still(f"path does not line up, fill {aligned.fill:.2f}")
        if held is not None and held.status == "ok":
            return held
    keep = [
        index
        for index, moment in enumerate(times)
        if window_start <= moment + shift <= window_end
    ]
    if len(keep) < 3:
        middle = (window_start + window_end) / 2 - shift
        keep = sorted(
            sorted(range(len(times)), key=lambda index: abs(times[index] - middle))[:3]
        )
    frames = [frames[index] for index in keep]
    times = [times[index] for index in keep]
    limit = max(3, int(settings.max_object_seconds * sample_fps) + 1)
    if len(frames) > limit:
        chosen = np.linspace(0, len(frames) - 1, limit).round().astype(int)
        frames = [frames[int(item)] for item in chosen]
        times = [times[int(item)] for item in chosen]
    boxes = boxes_at(event, times, width, height, shift)
    if not boxes:
        return _Cut(status="no_box")
    ghosts = build_cutouts(frames, boxes, category)
    if ghosts and _mostly_windows(ghosts):
        held = still("mask did not hold")
        if held is not None and held.status == "ok":
            return held
    if ghosts and _too_many_held(ghosts):
        held = still("object missing from much of the clip")
        if held is not None and held.status == "ok":
            return held
    logger.debug(
        "Recap cutout %s %s: %d frames, time shift %.2fs, fill %.2f, %s",
        event.get("id"),
        category,
        len(frames),
        shift,
        aligned.fill,
        "windows" if ghosts and _mostly_windows(ghosts) else "masks",
    )
    # Only a person's clip is used to crop a parked car they get in or out of.
    wants_context = str(event.get("label") or "") == "person"
    context_first = _jpeg(frames[0]) if wants_context else None
    context_last = _jpeg(frames[-1]) if wants_context else None
    if not ghosts:
        return remember(_Cut(status="no_cutout"))
    ghost_frames = [
        GhostFrame(
            x=int(ghost["x"]),
            y=int(ghost["y"]),
            box=ghost["box"],  # type: ignore[arg-type]
            crop=ghost["crop"],  # type: ignore[arg-type]
            alpha=ghost["alpha"],  # type: ignore[arg-type]
            window=bool(ghost.get("window")),
        )
        for ghost in ghosts
    ]
    for ghost_frame in ghost_frames:
        ghost_frame.pack()
    packed: list[dict[str, Any]] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    kept_times: list[float] = []
    for ghost_frame, box, moment in zip(ghost_frames, boxes, times):
        item = cutcache.pack_frames([ghost_frame])
        if not item:
            continue
        packed.append(item[0])
        kept_boxes.append(tuple(float(v) for v in box))  # type: ignore[arg-type]
        # Detector time, like a still's best moment, so tracks compare.
        kept_times.append(float(moment) + shift)
    if not packed:
        return remember(_Cut(status="no_cutout"))
    return remember(
        _keep_visible(
            _Cut(
                status="ok",
                frames=packed,
                boxes=kept_boxes,
                times=kept_times,
                context_first=context_first,
                context_last=context_last,
            ),
            width,
            height,
        )
    )


class RecapCancelled(Exception):
    """The user or a shutdown asked the job to stop."""


def _jpeg(frame: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return encoded.tobytes()


def _empty_video(
    plate: np.ndarray,
    camera: str,
    after: float,
    before: float,
    zone: tzinfo,
    settings: RecapConfig,
    out_dir: Path,
    ffmpeg: str,
    excluded: list[dict[str, Any]],
    cancel: Callable[[], bool],
    progress: Callable[[float, str], None],
) -> dict[str, Any]:
    height, width = plate.shape[:2]
    header = f"{camera}  {_clock_range(after, before, zone)}  nothing tracked"
    progress(90, "Nothing to show, saving the background")
    thumb = encode_video(
        plate,
        [],
        [],
        [],
        [],
        settings.output_fps * 2,
        str(out_dir / "video.mp4"),
        ffmpeg=ffmpeg,
        fps=settings.output_fps,
        header=header,
        header_h=max(28, int(round(0.075 * height))),
        fade_frames=1,
        label_opacity=settings.label_opacity,
        font_scale=settings.font_scale,
        repeats=[],
        cancel_check=cancel,
    )
    if thumb is None:
        raise RecapCancelled()
    cv2.imwrite(str(out_dir / "thumb.jpg"), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return _manifest_body(
        camera,
        after,
        before,
        settings,
        width,
        height,
        settings.output_fps * 2,
        [],
        [],
        [],
        excluded,
        [],
    )


def _manifest_body(
    camera: str,
    after: float,
    before: float,
    settings: RecapConfig,
    width: int,
    height: int,
    frame_count: int,
    units: list[ScheduledUnit],
    starts: list[int],
    rects: list[tuple[int, int, int, int]],
    excluded: list[dict[str, Any]],
    links: list[Any],
) -> dict[str, Any]:
    tracks = []
    for index, unit in enumerate(units):
        rect = rects[index] if index < len(rects) else (0, 0, 0, 0)
        tracks.append(
            {
                "event_id": unit.event_id,
                "clip_event_id": unit.clip_event_id,
                "cat": unit.category,
                "label": unit.label,
                "text": unit.text,
                "start_time": unit.start_time,
                "out_start": starts[index] if index < len(starts) else 0,
                "length": len(unit.boxes),
                "label_box": [
                    round(rect[0] / width, 4),
                    round(rect[1] / height, 4),
                    round(rect[2] / width, 4),
                    round(rect[3] / height, 4),
                ],
                "suffix": unit.suffix,
                "link_event_id": None
                if unit.link_index is None
                else units[unit.link_index].event_id,
            }
        )
    counts = {
        category: sum(1 for unit in units if unit.category == category)
        for category in CAT_ORDER
    }
    return {
        "camera": camera,
        "after": after,
        "before": before,
        "fps": settings.output_fps,
        "width": width,
        "height": height,
        "seconds": round(frame_count / settings.output_fps, 2),
        "frame_count": frame_count,
        "event_count": sum(1 for unit in units if unit.category != "parked"),
        "tracks": tracks,
        "excluded": excluded,
        "parked_links": [
            {
                "person_id": link.person_id,
                "vehicle_id": link.vehicle_id,
                "kind": link.kind,
                "time": link.ts,
            }
            for link in links
        ],
        "categories": [
            {
                "key": category,
                "name": CAT_NAME[category],
                "color": bgr_hex(CAT_COLOR[category]),
                "count": counts[category],
            }
            for category in CAT_ORDER
        ],
    }


def frame_times(start: float, fps: float, count: int) -> list[float]:
    """Recording time of each frame ffmpeg's fps filter returns from ``start``."""
    if count <= 0:
        return []
    rate = max(0.5, float(fps))
    return [float(start) + index / rate for index in range(count)]


def plate_timestamps(after: float, before: float, count: int = 16) -> list[float]:
    if before <= after:
        return [after]
    return [float(item) for item in np.linspace(after, before, count)]
