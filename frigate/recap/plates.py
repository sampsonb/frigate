"""Time-matched background plates, and cutouts fitted to those plates.

A dusk window mixes color daylight and grayscale infrared. One median of
both becomes a night plate, and afternoon cutouts look pasted on. Samples
are kept apart by lighting, then grouped into plates about 30 minutes long.
The plate nearest the objects on screen is the one that is drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# Infrared frames are gray. Median saturation and the 90th percentile both
# have to be near zero, so a color frame with a gray road still counts as day.
IR_SAT_MEDIAN = 10.0
IR_SAT_P90 = 18.0
# Below this median luminance the plate is dark enough for a night outline.
DARK_LUMA = 60.0
# How often a fresh plate is built inside one lighting period.
PLATE_INTERVAL = 1800.0
# How often recordings are sampled while looking for the day/night switch.
CLASSIFY_STEP = 600.0
# Crossfade when the recap moves to another plate.
PLATE_FADE_SECONDS = 0.5
# How much saturation and brightness to take off the plate so cutouts pop.
PLATE_RECEDE = 0.18


@dataclass(frozen=True)
class TimedPlate:
    """One background, the time it represents, and the lighting it came from."""

    image: np.ndarray
    time: float
    start: float
    end: float
    is_ir: bool
    is_dark: bool


def frame_saturation(image: np.ndarray) -> np.ndarray:
    """HSV saturation for a BGR frame."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]


def frame_is_ir(image: np.ndarray) -> bool:
    """True when chroma is near zero, which is how an IR night frame looks."""
    if image.size == 0:
        return False
    sat = frame_saturation(image)
    return (
        float(np.median(sat)) <= IR_SAT_MEDIAN
        and float(np.percentile(sat, 90)) <= IR_SAT_P90
    )


def recede_plate(image: np.ndarray, amount: float = PLATE_RECEDE) -> np.ndarray:
    """Dim and desaturate a background so the cutouts read first.

    ``amount`` is the fraction removed from saturation and value, about
    15 to 20 percent. Cutouts are matched to the original plate, then
    drawn on top of this quieter copy.
    """
    if image.size == 0 or amount <= 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    keep = 1.0 - float(amount)
    hsv[..., 1] *= keep
    hsv[..., 2] *= keep
    np.clip(hsv, 0, 255, out=hsv)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def frame_is_dark(image: np.ndarray) -> bool:
    """True when the plate is dark enough that a dark object would vanish."""
    if image.size == 0:
        return False
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.median(luma)) < DARK_LUMA


def mostly_off_frame(
    box: tuple[float, float, float, float], width: int, height: int
) -> bool:
    """True when the detector box is mostly outside the frame.

    A box whose center sits on the edge leaves a sliver of pixels and a
    leader that points at the border. Those are not drawn.
    """
    x0, y0, x1, y1 = (float(value) for value in box)
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < 4 or box_h < 4 or width < 1 or height < 1:
        return True
    ix0 = max(0.0, x0)
    iy0 = max(0.0, y0)
    ix1 = min(float(width), x1)
    iy1 = min(float(height), y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return True
    if (ix1 - ix0) * (iy1 - iy0) < 0.35 * box_w * box_h:
        return True
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return (
        center_x < 1 or center_y < 1 or center_x >= width - 1 or center_y >= height - 1
    )


def cutout_is_visible(
    alpha: np.ndarray,
    box: tuple[float, float, float, float],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> tuple[bool, tuple[int, int] | None]:
    """Whether this ghost should get a label and a leader.

    Returns the tip of the leader (top of the opaque pixels) when the
    cutout is on screen. An empty mask, a box that is mostly off the
    frame, or a mask that misses the detector box returns no tip.
    """
    if alpha.size == 0 or mostly_off_frame(box, width, height):
        return False, None
    opaque = alpha >= 64
    count = int(opaque.sum())
    if count < 48:
        return False, None
    ys, xs = np.nonzero(opaque)
    span_w = int(xs.max() - xs.min() + 1)
    span_h = int(ys.max() - ys.min() + 1)
    if min(span_w, span_h) < 6:
        return False, None
    frame_x = xs + int(origin_x)
    frame_y = ys + int(origin_y)
    inside = (
        (frame_x >= box[0])
        & (frame_x < box[2])
        & (frame_y >= box[1])
        & (frame_y < box[3])
    )
    # The mask has to touch the detector box. A golf cart below a person
    # is mostly outside that box and still belongs to the cutout. A mask
    # that misses the box entirely (motion up in the trees) does not.
    if int(inside.sum()) < 24:
        return False, None
    inset = (
        (frame_x >= 3) & (frame_x < width - 3) & (frame_y >= 3) & (frame_y < height - 3)
    )
    if int(inset.sum()) < 48:
        return False, None
    top_band = ys <= int(ys.min()) + max(1, span_h // 6)
    tip_x = int(origin_x + round(float(xs[top_band].mean())))
    tip_y = int(origin_y + int(ys.min()))
    return True, (tip_x, tip_y)


def mask_centroid(
    alpha: np.ndarray, origin_x: int, origin_y: int
) -> tuple[int, int] | None:
    """Frame point at the middle of the opaque mask."""
    ys, xs = np.nonzero(alpha >= 64)
    if len(xs) == 0:
        return None
    return (
        int(origin_x + round(float(xs.mean()))),
        int(origin_y + round(float(ys.mean()))),
    )


def label_should_draw(alpha: np.ndarray) -> bool:
    """False when the painted mask is only a few pixels.

    Objects too small to recognize are left out of the recap before any
    cutting, so a ghost that is drawn keeps its time label. Skipping the
    label of a faint or small ghost left objects on screen with no time.
    """
    return int((alpha >= 64).sum()) >= 48


def plate_sample_times(
    after: float, before: float, step: float = CLASSIFY_STEP
) -> list[float]:
    """Timestamps for plate grabs, about every 10 minutes, capped at 48."""
    if before <= after:
        return [float(after)]
    span = float(before) - float(after)
    if span <= step:
        return [float(after) + span / 2]
    count = int(round(span / step)) + 1
    count = max(2, min(count, 48))
    return [float(item) for item in np.linspace(after, before, count)]


def gap_sample_times(
    samples: Sequence[tuple[float, np.ndarray]], min_gap: float = 120.0
) -> list[float]:
    """Midpoints between neighbors whose lighting disagrees.

    The 10 minute grid can step over the infrared switch. One or two
    extra grabs on each side of that step pin the change down.
    """
    ordered = sorted(samples, key=lambda item: item[0])
    extras: list[float] = []
    for (left_t, left), (right_t, right) in zip(ordered, ordered[1:], strict=False):
        if right_t - left_t < min_gap:
            continue
        if frame_is_ir(left) == frame_is_ir(right):
            continue
        extras.append((float(left_t) + float(right_t)) / 2)
    return extras


def _median_image(frames: Sequence[np.ndarray]) -> np.ndarray:
    first = frames[0]
    height, width = first.shape[:2]
    same = []
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        same.append(frame)
    if len(same) == 1:
        return same[0]
    return np.median(np.stack(same), axis=0).astype(np.uint8)


def build_timed_plates(
    samples: Sequence[tuple[float, np.ndarray]],
    interval: float = PLATE_INTERVAL,
) -> list[TimedPlate]:
    """Median plates that never mix color and infrared.

    Inside one lighting run, a new plate starts about every ``interval``
    seconds so a six hour afternoon does not keep the first frame.
    """
    usable = [
        (float(moment), frame)
        for moment, frame in samples
        if frame is not None and getattr(frame, "size", 0)
    ]
    usable.sort(key=lambda item: item[0])
    if not usable:
        return []
    runs: list[list[tuple[float, np.ndarray, bool]]] = []
    for moment, frame in usable:
        infrared = frame_is_ir(frame)
        if not runs or runs[-1][-1][2] != infrared:
            runs.append([])
        runs[-1].append((moment, frame, infrared))

    plates: list[TimedPlate] = []

    def flush(bucket: list[tuple[float, np.ndarray, bool]]) -> None:
        times = [item[0] for item in bucket]
        image = _median_image([item[1] for item in bucket])
        infrared = bucket[0][2]
        plates.append(
            TimedPlate(
                image=image,
                time=float(np.median(times)),
                start=float(times[0]),
                end=float(times[-1]),
                is_ir=infrared,
                is_dark=infrared or frame_is_dark(image),
            )
        )

    for run in runs:
        bucket: list[tuple[float, np.ndarray, bool]] = []
        bucket_start = run[0][0]
        for item in run:
            if bucket and item[0] - bucket_start >= interval:
                flush(bucket)
                bucket = []
                bucket_start = item[0]
            bucket.append(item)
        if bucket:
            flush(bucket)
    return plates


def plate_index_at(plates: Sequence[TimedPlate], moment: float) -> int:
    """Plate for ``moment``.

    Inside one lighting period the boundary is halfway between plate times,
    so the plate nearest the objects is used. A day to night boundary is
    halfway between the last daytime sample and the first infrared sample,
    which is where the camera actually switched.
    """
    if not plates:
        raise ValueError("no plates")
    when = float(moment)
    chosen = 0
    for index, plate in enumerate(plates):
        if index == 0:
            continue
        previous = plates[index - 1]
        if previous.is_ir == plate.is_ir:
            boundary = (previous.time + plate.time) / 2
        else:
            boundary = (previous.end + plate.start) / 2
        if when >= boundary:
            chosen = index
    return chosen


class PlateFade:
    """Crossfade when the chosen plate changes. ``frames`` is about 0.5 s."""

    def __init__(self, frames: int) -> None:
        self.frames = max(1, int(frames))
        self.index: int | None = None
        self.previous: int | None = None
        self.left = 0

    def step(self, index: int) -> tuple[int, int | None, float]:
        """Return ``(current, previous or None, blend toward current)``."""
        if self.index is None:
            self.index = index
            return index, None, 1.0
        if index != self.index:
            self.previous = self.index
            self.index = index
            self.left = self.frames
        if self.left > 0 and self.previous is not None:
            amount = (self.frames - self.left + 1) / self.frames
            self.left -= 1
            previous = self.previous
            if self.left <= 0:
                self.previous = None
            return self.index, previous, amount
        return self.index, None, 1.0


def _region_is_color(image: np.ndarray, mask: np.ndarray) -> bool:
    sat = frame_saturation(image)
    values = sat[mask]
    if values.size == 0:
        return False
    return float(np.percentile(values, 90)) > IR_SAT_P90


def prepare_cutout(
    crop: np.ndarray,
    alpha: np.ndarray,
    *,
    plate_is_ir: bool,
) -> np.ndarray:
    """Fit a cutout to the plate it is about to be pasted on.

    A color cutout on an infrared plate becomes gray, which happens for a
    few seconds around the switch. Nothing else changes. Brightness and
    contrast used to be matched to the plate under the object, which made
    a car on a dark road the same gray as the road.
    """
    if crop.shape[:2] != alpha.shape[:2]:
        return crop
    mask = alpha >= 32
    if not np.any(mask):
        return crop
    if plate_is_ir and _region_is_color(crop, mask):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return crop
