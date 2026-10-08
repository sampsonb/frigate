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

# Vehicle masks wider than this are computed on a smaller image and scaled
# back. A distant car stays at full resolution so its few pixels are not lost.
_MASK_LONG_SIDE = 160
# Median plate samples. More frames do not make a cleaner plate once the
# object has moved, and each extra frame is a full-resolution partition.
_PLATE_SAMPLES = 9


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
    chosen = np.linspace(0, len(frames) - 1, min(len(frames), _PLATE_SAMPLES)).round().astype(int)
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
    # Pixels the object covers in every sample have no clean color. The
    # plain median already includes them, and a nan-median over that region
    # is the slow part of a parked or slow car.
    excluded_count = exclude.sum(axis=0)
    samples = exclude.shape[0]
    redo = (excluded_count > 0.34 * samples) & (excluded_count < samples)
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
    return _foreground(crop, background, diff, thresh)


def _foreground(
    crop: np.ndarray, background: np.ndarray, diff: np.ndarray, thresh: int
) -> np.ndarray:
    """Foreground from a difference image, with cast shadows removed."""
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
    local = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
    work_crop, work_plate, work_box = crop, plate, local
    if category == "vehicle":
        work_crop, work_plate, work_box = _vehicle_view(crop, plate, local)
    core = _seed(work_crop.shape[:2], work_box, category)
    if category == "vehicle":
        diff = cv2.absdiff(work_crop, work_plate).max(axis=2)
        thresh = _object_threshold(diff, core, category, fg_thresh)
        motion = _foreground(work_crop, work_plate, diff, thresh)
    else:
        motion = _motion(work_crop, work_plate, fg_thresh)
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
    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(
            mask,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return x0, y0, mask


def _vehicle_view(
    crop: np.ndarray,
    plate: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Shrink a large vehicle window. Small windows are returned unchanged."""
    height, width = crop.shape[:2]
    longest = max(height, width)
    if longest <= _MASK_LONG_SIDE or height < 8 or width < 8:
        return crop, plate, box
    scale = _MASK_LONG_SIDE / longest
    work_w = max(8, int(round(width * scale)))
    work_h = max(8, int(round(height * scale)))
    small_crop = cv2.resize(crop, (work_w, work_h), interpolation=cv2.INTER_AREA)
    small_plate = cv2.resize(plate, (work_w, work_h), interpolation=cv2.INTER_AREA)
    fitted = (
        box[0] * work_w / width,
        box[1] * work_h / height,
        box[2] * work_w / width,
        box[3] * work_h / height,
    )
    return small_crop, small_plate, fitted


def _object_threshold(
    diff: np.ndarray,
    core: np.ndarray,
    category: str,
    requested: int,
) -> int:
    """How different a pixel must be before it counts as this object.

    Vehicles use a lower floor. A distant or dark car barely moves the
    plate, and a fixed threshold of 22 drops it. The value still follows
    the contrast inside this box so a bright truck does not pull in the road.
    """
    if category != "vehicle":
        return requested
    values = diff[core > 0]
    if values.size == 0:
        return min(requested, 11)
    level = float(np.percentile(values, 65))
    return int(np.clip(level * 0.5, 7, 24))


def _outline_worth_tracing(mask: np.ndarray) -> bool:
    """Skip the contour fill when the car is only a few dozen pixels.

    The detector seed is already a solid block at that size. Tracing every
    distant car is most of the extra cost and does not change the ghost.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return False
    span = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return span >= 48


def _trace_outline(mask: np.ndarray) -> np.ndarray:
    """Fill the outer contour and close small holes.

    A raw motion mask on a car is often a ring of edges. The outline
    keeps the body instead of a scatter of pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 8:
        return mask
    traced = np.zeros_like(mask)
    cv2.drawContours(traced, [largest], -1, 1, -1)
    return cv2.morphologyEx(traced, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _soft_box(
    frame: np.ndarray, box: tuple[float, float, float, float]
) -> dict[str, object] | None:
    """Feathered crop of the detector box when the motion mask fails.

    A car that has sat still long enough is in the background plate, so
    the difference mask is empty. The car is still in this frame.
    """
    height, width = frame.shape[:2]
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    if box_w < 4 or box_h < 4:
        return None
    x0 = int(max(0, box[0]))
    y0 = int(max(0, box[1]))
    x1 = int(min(width, box[2]))
    y1 = int(min(height, box[3]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = frame[y0:y1, x0:x1].copy()
    alpha = np.zeros(crop.shape[:2], np.uint8)
    cv2.rectangle(
        alpha,
        (int(0.06 * crop.shape[1]), int(0.08 * crop.shape[0])),
        (max(2, int(0.94 * crop.shape[1])), max(2, int(0.92 * crop.shape[0]))),
        255,
        -1,
    )
    kernel = max(3, int(min(crop.shape[:2]) * 0.18) | 1)
    alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    return {
        "x": x0,
        "y": y0,
        "crop": crop,
        "alpha": alpha,
        "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
    }


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


def _object_is_visible(
    frame: np.ndarray,
    background: np.ndarray,
    piece: tuple[int, int, np.ndarray],
) -> bool:
    """True when the mask covers pixels that differ from the plate.

    The detector box is always part of the mask, so an empty gap frame
    still has area. Those pixels match the plate and should be repaired
    from a neighbor instead of cropped as a blank rectangle.
    """
    origin_x, origin_y, mask = piece
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return False
    # Only the mask's box needs a difference image. A full-frame absdiff
    # repeats the same pixels for every sample of a long track.
    y0 = max(0, origin_y + int(ys.min()))
    x0 = max(0, origin_x + int(xs.min()))
    y1 = min(frame.shape[0], origin_y + int(ys.max()) + 1)
    x1 = min(frame.shape[1], origin_x + int(xs.max()) + 1)
    if y1 <= y0 or x1 <= x0:
        return False
    diff = cv2.absdiff(frame[y0:y1, x0:x1], background[y0:y1, x0:x1]).max(axis=2)
    fy = np.clip(ys + origin_y - y0, 0, diff.shape[0] - 1)
    fx = np.clip(xs + origin_x - x0, 0, diff.shape[1] - 1)
    return float(np.median(diff[fy, fx])) >= 8


def _combine_sources(
    sources: list[tuple[tuple[int, int, np.ndarray], float]],
    width: int,
    height: int,
) -> tuple[int, int, np.ndarray] | None:
    """Weighted union of 0/1 masks. Origin is the top-left of the result."""
    if not sources:
        return None
    x0 = min(item[0][0] for item in sources)
    y0 = min(item[0][1] for item in sources)
    x1 = max(item[0][0] + item[0][2].shape[1] for item in sources)
    y1 = max(item[0][1] + item[0][2].shape[0] for item in sources)
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
    if total <= 0:
        return None
    return x0, y0, (acc >= 0.5 * total - 1e-3).astype(np.uint8)


def _feather_piece(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    category: str,
) -> dict[str, object] | None:
    """Feather a 0/1 mask. The crop is taken from ``frame`` at that mask.

    Vehicles fill the outer contour first. A motion mask on a car is often
    a ring of edges, and the body would otherwise drop out.
    """
    height, width = frame.shape[:2]
    if category == "vehicle" and _outline_worth_tracing(mask):
        mask = _trace_outline(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return None
    span = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    feather = max(3, int(span * 0.045) | 1)
    pad = feather + 2
    cx0 = int(max(0, origin_x + int(xs.min()) - pad))
    cy0 = int(max(0, origin_y + int(ys.min()) - pad))
    cx1 = int(min(width, origin_x + int(xs.max()) + 1 + pad))
    cy1 = int(min(height, origin_y + int(ys.max()) + 1 + pad))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    alpha = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
    _paste(alpha, cx0, cy0, mask * 255, origin_x, origin_y)
    erode = max(1, feather // 4)
    alpha = cv2.erode(
        alpha,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)),
    )
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
    if int(alpha.max()) < 8:
        return None
    return {
        "x": cx0,
        "y": cy0,
        "crop": frame[cy0:cy1, cx0:cx1].copy(),
        "alpha": alpha,
        "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
    }


def _clip_ghost(
    ghost: dict[str, object], width: int, height: int
) -> dict[str, object] | None:
    """Keep a shifted ghost inside the frame by cropping the part that fits."""
    crop = ghost["crop"]
    alpha = ghost["alpha"]
    assert isinstance(crop, np.ndarray)
    assert isinstance(alpha, np.ndarray)
    x = int(ghost["x"])  # type: ignore[arg-type]
    y = int(ghost["y"])  # type: ignore[arg-type]
    if x >= width or y >= height:
        return None
    if x + crop.shape[1] <= 0 or y + crop.shape[0] <= 0:
        return None
    src_x = max(0, -x)
    src_y = max(0, -y)
    x = max(0, x)
    y = max(0, y)
    copy_w = min(crop.shape[1] - src_x, width - x)
    copy_h = min(crop.shape[0] - src_y, height - y)
    if copy_w < 2 or copy_h < 2:
        return None
    ghost["x"] = x
    ghost["y"] = y
    ghost["crop"] = crop[src_y : src_y + copy_h, src_x : src_x + copy_w].copy()
    ghost["alpha"] = alpha[src_y : src_y + copy_h, src_x : src_x + copy_w].copy()
    return ghost


def _blend_ghosts(
    parts: list[tuple[dict[str, object], float]],
    box: tuple[float, float, float, float],
) -> dict[str, object]:
    """Stack repaired neighbors. Color is weighted, alpha keeps the stronger one."""
    x0 = min(int(ghost["x"]) for ghost, _weight in parts)  # type: ignore[arg-type]
    y0 = min(int(ghost["y"]) for ghost, _weight in parts)  # type: ignore[arg-type]
    x1 = max(
        int(ghost["x"]) + ghost["crop"].shape[1]  # type: ignore[attr-defined]
        for ghost, _weight in parts
    )
    y1 = max(
        int(ghost["y"]) + ghost["crop"].shape[0]  # type: ignore[attr-defined]
        for ghost, _weight in parts
    )
    color = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    weight_map = np.zeros((y1 - y0, x1 - x0), np.float32)
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    for ghost, weight in parts:
        crop = ghost["crop"]
        ghost_alpha = ghost["alpha"]
        assert isinstance(crop, np.ndarray)
        assert isinstance(ghost_alpha, np.ndarray)
        ox = int(ghost["x"]) - x0  # type: ignore[arg-type]
        oy = int(ghost["y"]) - y0  # type: ignore[arg-type]
        factor = ghost_alpha.astype(np.float32) / 255.0
        color[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] += (
            crop.astype(np.float32) * factor[..., None] * weight
        )
        weight_map[oy : oy + factor.shape[0], ox : ox + factor.shape[1]] += (
            factor * weight
        )
        alpha[oy : oy + ghost_alpha.shape[0], ox : ox + ghost_alpha.shape[1]] = (
            np.maximum(
                alpha[oy : oy + ghost_alpha.shape[0], ox : ox + ghost_alpha.shape[1]],
                ghost_alpha.astype(np.float32),
            )
        )
    safe = np.maximum(weight_map, 1e-3)[..., None]
    filled = color / safe
    return {
        "x": x0,
        "y": y0,
        "crop": np.clip(filled, 0, 255).astype(np.uint8),
        "alpha": np.clip(alpha, 0, 255).astype(np.uint8),
        "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
    }


def _blank_ghost(box: tuple[float, float, float, float]) -> dict[str, object]:
    """Index-aligned placeholder when even the box crop does not fit."""
    return {
        "x": int(max(0, box[0])),
        "y": int(max(0, box[1])),
        "crop": np.zeros((1, 1, 3), np.uint8),
        "alpha": np.zeros((1, 1), np.uint8),
        "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
    }


def _boxes_are_stable(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> bool:
    """True when the detector box barely moved or changed size.

    A parked car or a slow creep does not need a new mask every sample.
    """
    previous_w = previous[2] - previous[0]
    previous_h = previous[3] - previous[1]
    current_w = current[2] - current[0]
    current_h = current[3] - current[1]
    if min(previous_w, previous_h, current_w, current_h) < 1:
        return False
    if abs(current_w - previous_w) > 0.08 * previous_w:
        return False
    if abs(current_h - previous_h) > 0.08 * previous_h:
        return False
    limit = max(3.0, 0.04 * max(previous_w, previous_h))
    dx = (current[0] + current[2]) / 2 - (previous[0] + previous[2]) / 2
    dy = (current[1] + current[3]) / 2 - (previous[1] + previous[3]) / 2
    return abs(dx) <= limit and abs(dy) <= limit


def _shift_piece(
    piece: tuple[int, int, np.ndarray],
    source: tuple[float, float, float, float],
    dest: tuple[float, float, float, float],
) -> tuple[int, int, np.ndarray]:
    """Move a mask with the box. The array itself is shared."""
    x, y, mask = piece
    dx = int(round((dest[0] + dest[2]) / 2 - (source[0] + source[2]) / 2))
    dy = int(round((dest[1] + dest[3]) / 2 - (source[1] + source[3]) / 2))
    return x + dx, y + dy, mask


def _reuse_ghost(
    cached: dict[str, object],
    source: tuple[float, float, float, float],
    dest: tuple[float, float, float, float],
    width: int,
    height: int,
) -> dict[str, object] | None:
    """Shift a finished ghost with a box that barely moved.

    Feathering and the outline trace already ran for the source frame.
    Repeating them on the same car is the slow part of a long track.
    """
    crop = cached["crop"]
    alpha = cached["alpha"]
    assert isinstance(crop, np.ndarray)
    assert isinstance(alpha, np.ndarray)
    dx = int(round((dest[0] + dest[2]) / 2 - (source[0] + source[2]) / 2))
    dy = int(round((dest[1] + dest[3]) / 2 - (source[1] + source[3]) / 2))
    x = int(cached["x"]) + dx  # type: ignore[arg-type]
    y = int(cached["y"]) + dy  # type: ignore[arg-type]
    moved: dict[str, object] = {
        "x": x,
        "y": y,
        "crop": crop,
        "alpha": alpha,
        "box": (float(dest[0]), float(dest[1]), float(dest[2]), float(dest[3])),
    }
    if x >= 0 and y >= 0 and x + crop.shape[1] <= width and y + crop.shape[0] <= height:
        return moved
    return _clip_ghost(moved, width, height)


def _motion_pieces(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    background: np.ndarray,
    category: str,
    fg_thresh: int,
) -> list[tuple[int, int, np.ndarray]]:
    """One mask per frame. Stable boxes reuse the last measured mask."""
    pieces: list[tuple[int, int, np.ndarray]] = []
    source_piece: tuple[int, int, np.ndarray] | None = None
    source_box: tuple[float, float, float, float] | None = None
    for frame, box in zip(frames, boxes, strict=True):
        if (
            source_piece is not None
            and source_box is not None
            and _boxes_are_stable(source_box, box)
        ):
            pieces.append(_shift_piece(source_piece, source_box, box))
            continue
        piece = attach_motion(frame, background, box, category, fg_thresh)
        pieces.append(piece)
        source_piece = piece
        source_box = box
    return pieces


def build_cutouts(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    category: str,
    fg_thresh: int = 22,
) -> list[dict[str, object]] | None:
    """Feathered ghosts for one track, one entry per input frame.

    A bad mask is repaired from a neighbor a few frames away. The crop
    comes from that neighbor, shifted with the box, so a short detection
    gap still shows the object. If nothing nearby is usable, the detector
    box is feathered out of the current frame. That covers a car that has
    sat still long enough to match the background plate.
    """
    if len(frames) < 3 or len(frames) != len(boxes):
        return None
    background = clean_background(frames, boxes, category)
    height, width = background.shape[:2]
    pieces = _motion_pieces(frames, boxes, background, category, fg_thresh)
    areas = np.array([int(piece[2].sum()) for piece in pieces], np.float32)
    bad = [
        bool(area < 8) or not _object_is_visible(frame, background, piece)
        for area, frame, piece in zip(areas, frames, pieces, strict=True)
    ]
    low_ratio = 0.35 if category == "vehicle" else 0.45
    for index, area in enumerate(areas):
        if bad[index]:
            continue
        neighbors = [
            areas[other]
            for other in range(max(0, index - 3), min(len(areas), index + 4))
            if other != index and not bad[other]
        ]
        if len(neighbors) < 2:
            continue
        median = float(np.median(neighbors))
        if median <= 0:
            continue
        # Vehicles are not rejected for being larger than their neighbors.
        # A stopped car's mask flickers, and the big frames are the real ones.
        if area < low_ratio * median or (category != "vehicle" and area > 2.2 * median):
            bad[index] = True
    good = [index for index, flagged in enumerate(bad) if not flagged]
    centers = _smooth_centers(boxes)
    gap = 8 if category == "vehicle" else 4

    def shifted(source: int, dest: int) -> tuple[int, int, np.ndarray]:
        x, y, mask = pieces[source]
        dx = int(round(float(centers[dest][0] - centers[source][0])))
        dy = int(round(float(centers[dest][1] - centers[source][1])))
        return x + dx, y + dy, mask

    def repair(
        index: int, box: tuple[float, float, float, float]
    ) -> dict[str, object] | None:
        nearby = [item for item in good if 0 < abs(item - index) <= gap]
        if not nearby:
            return None
        left = [item for item in nearby if item < index]
        right = [item for item in nearby if item > index]
        donors: list[int] = []
        if left:
            donors.append(max(left))
        if right:
            donors.append(min(right))
        if not donors:
            donors.append(min(nearby, key=lambda item: abs(item - index)))
        parts: list[tuple[dict[str, object], float]] = []
        for donor in donors:
            origin_x, origin_y, mask = pieces[donor]
            dx = int(round(float(centers[index][0] - centers[donor][0])))
            dy = int(round(float(centers[index][1] - centers[donor][1])))
            made = _feather_piece(
                frames[donor], box, mask, origin_x, origin_y, category
            )
            if made is None:
                continue
            made["x"] = int(made["x"]) + dx  # type: ignore[arg-type]
            made["y"] = int(made["y"]) + dy  # type: ignore[arg-type]
            clipped = _clip_ghost(made, width, height)
            if clipped is None:
                continue
            parts.append((clipped, 1.0 / (1 + abs(donor - index))))
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0][0]
        blended = _blend_ghosts(parts, box)
        return _clip_ghost(blended, width, height)

    ghosts: list[dict[str, object]] = []
    cached: dict[str, object] | None = None
    cached_box: tuple[float, float, float, float] | None = None
    for index, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
        if (
            cached is not None
            and cached_box is not None
            and _boxes_are_stable(cached_box, box)
        ):
            reused = _reuse_ghost(cached, cached_box, box, width, height)
            if reused is not None:
                ghosts.append(reused)
                continue
        ghost: dict[str, object] | None = None
        if not bad[index]:
            sources: list[tuple[tuple[int, int, np.ndarray], float]] = [
                (pieces[index], 2.0)
            ]
            for other in (index - 1, index + 1):
                if 0 <= other < len(pieces) and not bad[other]:
                    sources.append((shifted(other, index), 1.0))
            combined = _combine_sources(sources, width, height)
            if combined is not None:
                origin_x, origin_y, mask = combined
                ghost = _feather_piece(frame, box, mask, origin_x, origin_y, category)
        else:
            ghost = repair(index, box)
        if ghost is None:
            ghost = _soft_box(frame, box)
        if ghost is None:
            ghost = _blank_ghost(box)
        cached = ghost
        cached_box = box
        ghosts.append(ghost)
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
