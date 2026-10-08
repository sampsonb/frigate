"""Motion cutouts for recap ghosts.

The detector's box is only a seed. Whatever moves with that object
against a clean background plate is kept: a golf cart, bicycle, scooter,
or stroller the model does not know still belongs to the person riding
or pushing it. Edges are feathered and single-frame glitches are
replaced from neighboring frames.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from frigate.recap.layout import MotionTrack


def clean_background(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    category: str,
) -> np.ndarray:
    """Median background, ignoring pixels under the tracked object.

    People exclude a wider region so a cart they are sitting on does not
    get baked into the plate and then disappear from the motion mask.
    """
    if not frames:
        raise ValueError("clean_background requires at least one frame")
    height, width = frames[0].shape[:2]
    chosen = np.linspace(0, len(frames) - 1, min(len(frames), 25)).round().astype(int)
    stack = np.stack([frames[index] for index in chosen]).astype(np.float32)
    exclude = np.zeros((len(chosen), height, width), dtype=bool)
    for sample, index in enumerate(chosen):
        box = boxes[int(index)]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        if category == "person":
            pad_x, pad_top, pad_bottom = 0.9 * box_w, 0.2 * box_h, 0.9 * box_h
        elif category == "vehicle":
            pad_x, pad_top, pad_bottom = 0.08 * box_w, 0.08 * box_h, 0.06 * box_h
        else:
            pad_x, pad_top, pad_bottom = 0.3 * box_w, 0.15 * box_h, 0.25 * box_h
        y0 = int(max(0, box[1] - pad_top))
        y1 = int(min(height, box[3] + pad_bottom))
        x0 = int(max(0, box[0] - pad_x))
        x1 = int(min(width, box[2] + pad_x))
        if y1 > y0 and x1 > x0:
            exclude[sample, y0:y1, x0:x1] = True
    background = np.median(stack, axis=0)
    covered = exclude.mean(axis=0)
    redo = covered > 0.34
    if redo.any():
        ys, xs = np.nonzero(redo)
        values = np.array(stack[:, ys, xs], copy=True)
        values[exclude[:, ys, xs]] = np.nan
        # Pixels the object covers in every sample have no clean color.
        # Leave the plain median there instead of crashing on an empty slice.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                filled = np.nanmedian(values, axis=0)
        if filled.ndim == 1:
            good = np.isfinite(filled)
        else:
            good = np.isfinite(filled).all(axis=-1)
        if np.any(good):
            background[ys[good], xs[good]] = filled[good]
    return background.astype(np.uint8)


def _motion(crop: np.ndarray, background: np.ndarray, thresh: int) -> np.ndarray:
    """Foreground against the plate, with cast shadows removed."""
    diff = cv2.absdiff(crop, background).max(axis=2)
    foreground = diff > thresh
    current = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int16)
    plate = cv2.cvtColor(background, cv2.COLOR_BGR2HSV).astype(np.int16)
    ratio = (current[..., 2] + 1) / (plate[..., 2] + 1)
    hue_delta = np.abs(current[..., 0] - plate[..., 0])
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    shadow = (
        (ratio > 0.35)
        & (ratio < 0.92)
        & (np.abs(current[..., 1] - plate[..., 1]) < 45)
        & ((hue_delta < 14) | (plate[..., 1] < 40))
    )
    foreground &= ~shadow
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def _seed(
    shape: tuple[int, int],
    box: tuple[float, float, float, float],
    category: str,
) -> np.ndarray:
    """Soft prior inside the detector box. Motion has to touch this."""
    mask = np.zeros(shape, np.uint8)
    height, width = shape
    x0 = int(max(0, min(width - 1, round(box[0]))))
    y0 = int(max(0, min(height - 1, round(box[1]))))
    x1 = int(max(0, min(width, round(box[2]))))
    y1 = int(max(0, min(height, round(box[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return mask
    if category == "vehicle":
        inset_x = int(0.04 * (x1 - x0))
        inset_y = int(0.05 * (y1 - y0))
        cv2.rectangle(
            mask,
            (x0 + inset_x, y0 + inset_y),
            (max(x0 + inset_x + 1, x1 - inset_x), max(y0 + inset_y + 1, y1 - inset_y)),
            1,
            -1,
        )
    else:
        cv2.ellipse(
            mask,
            ((x0 + x1) // 2, (y0 + y1) // 2),
            (max(1, int((x1 - x0) * 0.42)), max(1, int((y1 - y0) * 0.42))),
            0,
            0,
            360,
            1,
            -1,
        )
    return mask


def attach_motion(
    frame: np.ndarray,
    background: np.ndarray,
    box: tuple[float, float, float, float],
    category: str,
    fg_thresh: int = 22,
) -> tuple[int, int, np.ndarray]:
    """Mask of the tracked object plus motion connected to it.

    Returns ``(x, y, mask)`` in full-frame coordinates. The mask is 0/1.
    People search well outside the detector box so a ridden or pushed
    object is included. Vehicles stay tight so the road is not smeared in.
    """
    height, width = frame.shape[:2]
    box_w = max(1.0, box[2] - box[0])
    box_h = max(1.0, box[3] - box[1])
    if category == "person":
        pad_x, pad_top, pad_bottom = (
            0.85 * max(box_w, box_h * 0.8),
            0.2 * box_h,
            0.9 * box_h,
        )
    elif category == "vehicle":
        pad_x, pad_top, pad_bottom = 0.08 * box_w, 0.08 * box_h, 0.06 * box_h
    else:
        pad_x, pad_top, pad_bottom = 0.45 * box_w, 0.15 * box_h, 0.3 * box_h
    x0 = int(max(0, box[0] - pad_x))
    y0 = int(max(0, box[1] - pad_top))
    x1 = int(min(width, box[2] + pad_x))
    y1 = int(min(height, box[3] + pad_bottom))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return x0, y0, np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    crop = frame[y0:y1, x0:x1]
    plate = background[y0:y1, x0:x1]
    motion = _motion(crop, plate, fg_thresh)
    local = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
    core = _seed(motion.shape, local, category)
    seed = cv2.dilate(core, np.ones((7, 7), np.uint8))
    _count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        (motion > 0).astype(np.uint8), connectivity=8
    )
    extra = np.zeros_like(core)
    core_area = max(1, int(core.sum()))
    touched = np.unique(labels[(seed > 0) & (motion > 0)])
    window_area = int(motion.size)
    # People: a golf cart or stroller can be several times the detector box
    # and still be only part of the search window. A full-window lighting
    # change is rejected. Vehicles stay tight so the road is not pulled in.
    if category == "person":
        extra_limit = int(0.72 * window_area)
    elif category == "vehicle":
        extra_limit = max(2 * core_area, int(0.2 * window_area))
    else:
        extra_limit = int(0.5 * window_area)
    for label in touched:
        if int(label) == 0:
            continue
        component = labels == label
        added = int((component & (core == 0)).sum())
        if added < extra_limit:
            extra |= component.astype(np.uint8)
    if category == "vehicle":
        extra = cv2.morphologyEx(extra, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = ((core > 0) | (extra > 0)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(cv2.contourArea(contour) for contour in contours)
        mask[:] = 0
        cv2.drawContours(
            mask,
            [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= 0.04 * largest
            ],
            -1,
            1,
            -1,
        )
    return x0, y0, mask


def _smooth_centers(boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    centers = np.array(
        [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes],
        np.float32,
    )
    if len(centers) < 5:
        return centers
    kernel = 5
    half = kernel // 2
    padded = np.pad(centers, ((half + 1, half), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    return (cumulative[kernel:] - cumulative[:-kernel]) / kernel


def build_cutouts(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    category: str,
    fg_thresh: int = 22,
) -> list[dict[str, object]] | None:
    """Feathered ghosts for one track. None when the object cannot be separated."""
    if len(frames) < 3 or len(frames) != len(boxes):
        return None
    background = clean_background(frames, boxes, category)
    height, width = background.shape[:2]
    pieces: list[tuple[int, int, np.ndarray] | None] = []
    for frame, box in zip(frames, boxes, strict=True):
        pieces.append(attach_motion(frame, background, box, category, fg_thresh))
    areas = np.array(
        [int(piece[2].sum()) if piece is not None else 0 for piece in pieces],
        np.float32,
    )
    bad = [area < 8 for area in areas]
    for index, area in enumerate(areas):
        if bad[index]:
            continue
        neighbors = [
            areas[other]
            for other in range(max(0, index - 3), min(len(areas), index + 4))
            if other != index and not bad[other]
        ]
        if len(neighbors) >= 2:
            median = float(np.median(neighbors))
            if median > 0 and (area < 0.45 * median or area > 2.2 * median):
                bad[index] = True
    good = [index for index, flagged in enumerate(bad) if not flagged]
    if len(good) < 3:
        return None
    centers = _smooth_centers(boxes)

    def shifted(source: int, dest: int) -> tuple[int, int, np.ndarray]:
        x, y, mask = pieces[source]
        assert mask is not None
        dx = int(round(float(centers[dest][0] - centers[source][0])))
        dy = int(round(float(centers[dest][1] - centers[source][1])))
        return x + dx, y + dy, mask

    ghosts: list[dict[str, object]] = []
    for index, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
        sources: list[tuple[tuple[int, int, np.ndarray], float]] = []
        if bad[index]:
            donor = min(good, key=lambda item: abs(item - index))
            if abs(donor - index) > 3:
                continue
            sources.append((shifted(donor, index), 1.0))
        else:
            current = pieces[index]
            assert current is not None
            sources.append((current, 2.0))
            for other in (index - 1, index + 1):
                if 0 <= other < len(pieces) and not bad[other]:
                    sources.append((shifted(other, index), 1.0))
        x0 = min(item[0][0] for item in sources)
        y0 = min(item[0][1] for item in sources)
        x1 = max(item[0][0] + item[0][2].shape[1] for item in sources)
        y1 = max(item[0][1] + item[0][2].shape[0] for item in sources)
        # Keep the accumulator inside the frame.
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, max(x1, x0 + 1))
        y1 = min(height, max(y1, y0 + 1))
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for (sx, sy, mask), weight in sources:
            dest_x = sx - x0
            dest_y = sy - y0
            src_x = 0
            src_y = 0
            if dest_x < 0:
                src_x = -dest_x
                dest_x = 0
            if dest_y < 0:
                src_y = -dest_y
                dest_y = 0
            copy_w = min(mask.shape[1] - src_x, acc.shape[1] - dest_x)
            copy_h = min(mask.shape[0] - src_y, acc.shape[0] - dest_y)
            if copy_w <= 0 or copy_h <= 0:
                continue
            acc[dest_y : dest_y + copy_h, dest_x : dest_x + copy_w] += (
                weight * mask[src_y : src_y + copy_h, src_x : src_x + copy_w]
            )
        total = sum(weight for _, weight in sources)
        mask = (acc >= 0.5 * total - 1e-3).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) < 6:
            continue
        span = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        feather = max(3, int(span * 0.045) | 1)
        pad = feather + 2
        cx0 = int(max(0, x0 + int(xs.min()) - pad))
        cy0 = int(max(0, y0 + int(ys.min()) - pad))
        cx1 = int(min(width, x0 + int(xs.max()) + 1 + pad))
        cy1 = int(min(height, y0 + int(ys.max()) + 1 + pad))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            continue
        alpha = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
        _paste(alpha, cx0, cy0, mask * 255, x0, y0)
        erode = max(1, feather // 4)
        alpha = cv2.erode(
            alpha,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)
            ),
        )
        alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
        ghosts.append(
            {
                "x": cx0,
                "y": cy0,
                "crop": frame[cy0:cy1, cx0:cx1].copy(),
                "alpha": alpha,
                "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            }
        )
    if len(ghosts) < 3:
        return None
    return ghosts


def _paste(
    dest: np.ndarray,
    dest_x: int,
    dest_y: int,
    src: np.ndarray,
    src_x: int,
    src_y: int,
) -> None:
    """Max-combine ``src`` (placed at src_x, src_y) into ``dest`` (placed at dest_x, dest_y)."""
    height, width = src.shape[:2]
    x0 = max(src_x, dest_x)
    y0 = max(src_y, dest_y)
    x1 = min(src_x + width, dest_x + dest.shape[1])
    y1 = min(src_y + height, dest_y + dest.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = dest[y0 - dest_y : y1 - dest_y, x0 - dest_x : x1 - dest_x]
    np.maximum(
        region,
        src[y0 - src_y : y1 - src_y, x0 - src_x : x1 - src_x],
        out=region,
    )


def parked_car_ghost(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> dict[str, object] | None:
    """A soft rectangle of a parked car, taken from a frame where it is visible."""
    height, width = frame.shape[:2]
    car_w = box[2] - box[0]
    car_h = box[3] - box[1]
    if car_w < 4 or car_h < 4:
        return None
    x0 = int(max(0, box[0] - 0.08 * car_w))
    y0 = int(max(0, box[1] - 0.1 * car_h))
    x1 = int(min(width, box[2] + 0.08 * car_w))
    y1 = int(min(height, box[3] + 0.08 * car_h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = frame[y0:y1, x0:x1].copy()
    alpha = np.zeros(crop.shape[:2], np.uint8)
    cv2.rectangle(
        alpha,
        (int(0.07 * crop.shape[1]), int(0.08 * crop.shape[0])),
        (int(0.93 * crop.shape[1]), int(0.94 * crop.shape[0])),
        255,
        -1,
    )
    kernel = max(3, int(min(crop.shape[:2]) * 0.16) | 1)
    alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    return {
        "x": x0,
        "y": y0,
        "crop": crop,
        "alpha": alpha,
        "box": tuple(float(v) for v in box),
    }


def trim_track_boxes(track: MotionTrack, limit: int) -> MotionTrack:
    """Keep at most ``limit`` samples, spread across the track."""
    if limit <= 0 or len(track.boxes) <= limit:
        return track
    chosen = np.linspace(0, len(track.boxes) - 1, limit).round().astype(int)
    return MotionTrack(
        id=track.id,
        label=track.label,
        category=track.category,
        start=track.start,
        end=track.end,
        boxes=[track.boxes[int(index)] for index in chosen],
        times=[track.times[int(index)] for index in chosen],
    )
