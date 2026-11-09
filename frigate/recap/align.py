"""Line up Frigate's object path with the recording.

Frigate stamps the path with the detect stream's clock. The recording is
a separate stream whose segments are dated to the whole second when the
file opens, so a recording frame can be anywhere from a fraction of a
second to about two seconds away from the detector's time for the same
moment, and it changes from one ten second segment to the next.
Frigate's ``annotation_offset`` is one fixed correction for the review
overlay. A car on the street crosses the view in about two seconds, so
a box placed at the raw time sits on empty road and the cutout is a
patch of pavement. Each event is checked against its own frames: the
box is slid in time until it covers the most motion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# Width the search runs at. The box only has to land on the object.
_SEARCH_WIDTH = 320
# Seconds searched around the camera's starting guess. The recording
# usually runs behind the detector, rarely ahead of it.
SHIFT_LOW = -1.0
SHIFT_HIGH = 2.6
SHIFT_STEP = 0.1
# Fewer frames than this inside the known span and a shift is not scored.
_MIN_FRAMES = 3
# Gray level difference from the clip's median frame that counts as motion.
_MOTION_LEVEL = 14
# The best shift has to cover this share of its boxes with motion, and
# beat the typical shift by this factor. Otherwise the object barely
# moved and the starting guess is kept.
_MIN_FILL = 0.06
_MIN_GAIN = 1.25
# Shifts within this share of the best score form the peak. Its middle is
# returned, so a slow car does not snap to one end of a flat top.
_PEAK = 0.95

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Alignment:
    """Result of the search.

    ``seconds`` is added to a recording time to get the detector time.
    ``fill`` is the share of the boxes covered by motion at that shift.
    ``found`` is False when no shift stood out, and ``seconds`` is then the
    starting guess, with ``fill`` measured there.
    """

    seconds: float
    fill: float
    found: bool


BoxesFor = Callable[[Sequence[float], int, int], "list[Box] | None"]


def _box_fill(integral: np.ndarray, box: Box, width: int, height: int) -> float:
    """Share of ``box`` that is motion, from an integral image."""
    x0 = int(np.clip(round(box[0]), 0, width))
    y0 = int(np.clip(round(box[1]), 0, height))
    x1 = int(np.clip(round(box[2]), 0, width))
    y1 = int(np.clip(round(box[3]), 0, height))
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0
    inside = (
        int(integral[y1, x1])
        - int(integral[y0, x1])
        - int(integral[y1, x0])
        + int(integral[y0, x0])
    )
    return inside / area


def _peak_middle(scores: np.ndarray, best: int) -> int:
    """Index in the middle of the run of near-best scores around ``best``."""
    floor = _PEAK * float(scores[best])
    left = best
    while left > 0 and scores[left - 1] >= floor:
        left -= 1
    right = best
    while right < len(scores) - 1 and scores[right + 1] >= floor:
        right += 1
    return (left + right) // 2


def estimate_shift(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    boxes_for: BoxesFor,
    *,
    default: float = 0.0,
    low: float = SHIFT_LOW,
    high: float = SHIFT_HIGH,
    step: float = SHIFT_STEP,
    valid: tuple[float, float] | None = None,
) -> Alignment:
    """Find the seconds to add to a recording time to get the detector time.

    ``boxes_for(detector_times, width, height)`` returns the object's
    boxes at those detector times, scaled to ``width`` by ``height``.
    Shifts from ``default + low`` to ``default + high`` are tried.
    ``valid`` is the detector time span the boxes are known for. A frame
    that lands outside it for a given shift is not scored, because the
    box there is only the first or last known position. ``default`` is
    kept when the object does not move enough to tell.
    """
    if len(frames) < 4 or len(frames) != len(times) or step <= 0 or high < low:
        return Alignment(default, 0.0, False)
    height, width = frames[0].shape[:2]
    scale = min(1.0, _SEARCH_WIDTH / max(1, width))
    small_w = max(8, int(round(width * scale)))
    small_h = max(8, int(round(height * scale)))
    grays = [
        cv2.cvtColor(
            cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in frames
    ]
    picks = np.linspace(0, len(grays) - 1, min(len(grays), 11)).round().astype(int)
    background = np.median(np.stack([grays[int(i)] for i in picks]), axis=0)
    background = background.astype(np.uint8)
    integrals = [
        cv2.integral((cv2.absdiff(gray, background) > _MOTION_LEVEL).astype(np.uint8))
        for gray in grays
    ]
    count = int(round((high - low) / step)) + 1
    candidates = default + np.linspace(low, high, count)
    scores = np.zeros(count, np.float64)
    for index, shift in enumerate(candidates):
        moments = [float(t) + float(shift) for t in times]
        chosen = [
            position
            for position, moment in enumerate(moments)
            if valid is None or valid[0] <= moment <= valid[1]
        ]
        if len(chosen) < _MIN_FRAMES:
            continue
        boxes = boxes_for([moments[i] for i in chosen], small_w, small_h)
        if not boxes or len(boxes) != len(chosen):
            return Alignment(default, 0.0, False)
        scores[index] = float(
            np.mean(
                [
                    _box_fill(integrals[i], box, small_w, small_h)
                    for i, box in zip(chosen, boxes, strict=True)
                ]
            )
        )
    best = int(np.argmax(scores))
    typical = float(np.median(scores))
    if scores[best] < _MIN_FILL or scores[best] < _MIN_GAIN * max(typical, 1e-6):
        # No clear winner: a slow or distant object is covered at almost
        # any shift. Report how well the starting guess covers it, so a
        # well covered object still moves instead of becoming a still.
        start_index = int(np.argmin(np.abs(candidates - default)))
        return Alignment(default, float(scores[start_index]), False)
    middle = _peak_middle(scores, best)
    return Alignment(round(float(candidates[middle]), 2), float(scores[middle]), True)
