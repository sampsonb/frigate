"""Motion cutouts for recap ghosts.

The detector's box is only a seed. Whatever moves with that object
against a clean background plate is kept: a golf cart, bicycle, scooter,
or stroller the model does not know still belongs to the person riding
or pushing it. Edges are feathered and single-frame glitches are
replaced from neighboring frames.

A motion mask needs a clean, well lit picture. In infrared, at dusk, or
when the mask spreads far past the object, the ghost is a window instead:
the padded detector box with rounded corners, copied as recorded. A
window never loses part of the object and never turns into see-through
road.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from frigate.recap.layout import MotionTrack
from frigate.recap.plates import frame_is_ir, mostly_off_frame

# Vehicle masks wider than this are computed on a smaller image and scaled
# back. A distant car stays at full resolution so its few pixels are not lost.
_MASK_LONG_SIDE = 160
# Interior of a cutout is fully opaque. Only this many pixels of edge fade.
FEATHER_PX = 3
# Median plate samples. More frames do not make a cleaner plate once the
# object has moved, and each extra frame is a full-resolution partition.
_PLATE_SAMPLES = 9
# Below this median brightness (0 to 255) the clip is dusk or night. Gain
# noise and headlights make the motion mask unreliable, so windows are used.
DIM_LUMA = 80.0
# Mask area over detector box area. Past this the mask has spread into the
# road or the yard, and the track is drawn as windows instead.
_MAX_MASK_RATIO = {"vehicle": 1.8, "person": 3.2, "animal": 3.2, "delivery": 3.2}
# Share of frames that must have a believable mask to keep the mask look.
_PLAUSIBLE_SHARE = 0.6


def _half_extents(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Widest detector box in the clip, measured from each box's center.

    One frame often misses the nose or the top of the head. Another frame
    in the same clip may include it. Placing that union back on the
    current center keeps the whole object inside the window.
    """
    left = top = right = bottom = 0.0
    for box in boxes:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        left = max(left, cx - box[0])
        right = max(right, box[2] - cx)
        top = max(top, cy - box[1])
        bottom = max(bottom, box[3] - cy)
    return left, top, right, bottom


def _horizontal_travel(boxes: list[tuple[float, float, float, float]]) -> float:
    """Median horizontal motion, in pixels per step. Positive moves right."""
    if len(boxes) < 2:
        return 0.0
    steps = [
        ((cur[0] + cur[2]) - (prev[0] + prev[2])) * 0.5
        for prev, cur in zip(boxes, boxes[1:])
    ]
    steps.sort()
    return float(steps[len(steps) // 2])


def search_window(
    box: tuple[float, float, float, float],
    category: str,
    width: int,
    height: int,
    span: tuple[float, float, float, float] | None = None,
    travel: float = 0.0,
) -> tuple[int, int, int, int]:
    """Padded region to segment. The object has to fit inside it.

    Sides grow by about 12% of the union box. People and animals also get
    about 20% above the box, and vehicles get more room on the side they
    are moving toward.
    """
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    own = (
        cx - box[0],
        cy - box[1],
        box[2] - cx,
        box[3] - cy,
    )
    if span is None:
        span = own
    left = max(span[0], own[0])
    top = max(span[1], own[1])
    right = max(span[2], own[2])
    bottom = max(span[3], own[3])
    box_w = max(1.0, left + right)
    box_h = max(1.0, top + bottom)
    pad_x = 0.12 * box_w
    lead = 0.0
    if category == "person":
        # Wide on purpose: a cart or stroller sits beside the person.
        pad_x = max(pad_x, 0.85 * max(box_w, box_h * 0.8))
        pad_top = 0.20 * box_h
        pad_bottom = max(0.12 * box_h, 0.9 * box_h)
    elif category == "animal":
        pad_x = max(pad_x, 0.45 * box_w)
        pad_top = 0.20 * box_h
        pad_bottom = max(0.12 * box_h, 0.25 * box_h)
    elif category == "vehicle":
        pad_top = 0.12 * box_h
        pad_bottom = 0.12 * box_h
        # The nose sticks out ahead of a lagging detector box.
        if abs(travel) >= 0.5:
            lead = 0.30 * box_w
    else:
        pad_top = 0.12 * box_h
        pad_bottom = 0.12 * box_h
    x0 = cx - left - pad_x
    x1 = cx + right + pad_x
    y0 = cy - top - pad_top
    y1 = cy + bottom + pad_bottom
    if lead:
        if travel > 0:
            x1 += lead
        else:
            x0 -= lead
    return (
        int(max(0, np.floor(x0))),
        int(max(0, np.floor(y0))),
        int(min(width, np.ceil(x1))),
        int(min(height, np.ceil(y1))),
    )


def clean_background(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    category: str,
    span: tuple[float, float, float, float] | None = None,
    travel: float = 0.0,
) -> np.ndarray:
    """Median background, ignoring pixels under the tracked object.

    People exclude a wider region so a cart they are sitting on does not
    get baked into the plate and then disappear from the motion mask.
    """
    if not frames:
        raise ValueError("clean_background requires at least one frame")
    height, width = frames[0].shape[:2]
    chosen = (
        np.linspace(0, len(frames) - 1, min(len(frames), _PLATE_SAMPLES))
        .round()
        .astype(int)
    )
    stack = np.stack([frames[index] for index in chosen]).astype(np.float32)
    exclude = np.zeros((len(chosen), height, width), dtype=bool)
    for sample, index in enumerate(chosen):
        box = boxes[int(index)]
        x0, y0, x1, y1 = search_window(box, category, width, height, span, travel)
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


def _smooth_contour(mask: np.ndarray) -> np.ndarray:
    """Close small gaps and simplify the outline.

    A raw motion edge is a staircase. The filled, simplified contour is
    what the night outline follows, so the ring is not wavy.
    """
    binary = (mask > 0).astype(np.uint8)
    if int(binary.max()) == 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return closed
    largest = max(cv2.contourArea(contour) for contour in contours)
    out = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) < max(8.0, 0.04 * largest):
            continue
        peri = cv2.arcLength(contour, True)
        epsilon = max(1.0, 0.012 * peri)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(out, [approx], -1, 1, -1)
    return out if int(out.max()) else closed


def attach_motion(
    frame: np.ndarray,
    background: np.ndarray,
    box: tuple[float, float, float, float],
    category: str,
    fg_thresh: int = 22,
    span: tuple[float, float, float, float] | None = None,
    travel: float = 0.0,
) -> tuple[int, int, np.ndarray]:
    """Mask of the tracked object plus motion connected to it.

    Returns ``(x, y, mask)`` in full-frame coordinates. The mask is 0/1.
    People search well outside the detector box so a ridden or pushed
    object is included. Vehicles stay tight so the road is not smeared in.
    """
    height, width = frame.shape[:2]
    box_h = max(1.0, box[3] - box[1])
    x0, y0, x1, y1 = search_window(box, category, width, height, span, travel)
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
    if category in ("person", "animal"):
        # The head sits above the detector box. Reach into that band so it
        # stays connected to the body instead of being cropped off.
        reach = max(3, int(round(0.20 * box_h)) + int(round(0.10 * box_h)))
        seed = cv2.dilate(
            seed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, reach * 2 + 1))
        )
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
    mask = _smooth_contour(((core > 0) | (extra > 0)).astype(np.uint8))
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


def solid_feather(alpha: np.ndarray, radius: int = FEATHER_PX) -> np.ndarray:
    """Opaque inside the mask, with a soft edge of about 2 to 3 pixels.

    A wide blur leaves the whole car translucent. Pixels clearly inside
    the mask become 255. Only the rim fades out.
    """
    if alpha.size == 0:
        return alpha
    radius = max(2, min(3, int(radius)))
    mask = np.where(alpha >= 32, 255, 0).astype(np.uint8)
    if int(mask.max()) == 0:
        return np.zeros_like(alpha)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    interior = cv2.erode(mask, kernel)
    soft = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0.8)
    if int(interior.max()) == 0:
        soft[mask > 0] = 255
    else:
        soft[interior > 0] = 255
    return soft


def window_rect(
    box: tuple[float, float, float, float],
    category: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Padded detector box for a window ghost, clipped to the frame.

    The path box can trail or lead the object by a few pixels, and a head
    or a side mirror sits outside a loose detector box, so the window is
    roomier than the box.
    """
    box_w = max(1.0, box[2] - box[0])
    box_h = max(1.0, box[3] - box[1])
    if category in ("person", "animal", "delivery"):
        pad_x = max(4.0, 0.16 * box_w)
        pad_top = max(4.0, 0.16 * box_h)
    else:
        pad_x = max(4.0, 0.10 * box_w)
        pad_top = max(4.0, 0.12 * box_h)
    pad_bottom = max(3.0, 0.06 * box_h)
    return (
        int(max(0, np.floor(box[0] - pad_x))),
        int(max(0, np.floor(box[1] - pad_top))),
        int(min(width, np.ceil(box[2] + pad_x))),
        int(min(height, np.ceil(box[3] + pad_bottom))),
    )


def rounded_alpha(height: int, width: int) -> np.ndarray:
    """Opaque rounded rectangle with the usual soft rim."""
    alpha = np.zeros((height, width), np.uint8)
    if height < 4 or width < 4:
        return alpha
    radius = int(max(2, min(14, 0.14 * min(height, width))))
    inset = 1
    x0, y0, x1, y1 = inset, inset, width - 1 - inset, height - 1 - inset
    cv2.rectangle(alpha, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(alpha, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in (
        (x0 + radius, y0 + radius),
        (x1 - radius, y0 + radius),
        (x0 + radius, y1 - radius),
        (x1 - radius, y1 - radius),
    ):
        cv2.circle(alpha, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return solid_feather(alpha)


def window_ghost(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    category: str,
) -> dict[str, object] | None:
    """The padded box copied straight from ``frame``, with rounded corners."""
    height, width = frame.shape[:2]
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    if mostly_off_frame(box, width, height):
        return None
    x0, y0, x1, y1 = window_rect(box, category, width, height)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return {
        "x": x0,
        "y": y0,
        "crop": frame[y0:y1, x0:x1].copy(),
        "alpha": rounded_alpha(y1 - y0, x1 - x0),
        "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        "window": True,
    }


def window_ghosts(
    frames: list[np.ndarray],
    boxes: list[tuple[float, float, float, float]],
    category: str,
) -> list[dict[str, object]]:
    """One window ghost per frame. Off-frame boxes become blanks."""
    ghosts: list[dict[str, object]] = []
    for frame, box in zip(frames, boxes, strict=True):
        ghost = window_ghost(frame, box, category)
        ghosts.append(ghost if ghost is not None else _blank_ghost(box))
    return ghosts


def needs_window(background: np.ndarray) -> bool:
    """True when a motion mask cannot be trusted for this clip.

    Infrared frames have almost no color. Dusk frames are dim and full of
    gain noise before the camera switches. Either way the difference
    image picks up glare and grain instead of the object's outline.
    """
    if background.size == 0:
        return False
    if frame_is_ir(background):
        return True
    luma = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    return float(np.median(luma)) < DIM_LUMA


def masks_plausible(
    pieces: list[tuple[int, int, np.ndarray]],
    boxes: list[tuple[float, float, float, float]],
    category: str,
) -> bool:
    """True when most masks stay about the size of the detector box.

    A mask several times the box has leaked into the road, a lawn, or a
    second object. Drawing that leak is what looks like a smeared ghost.
    """
    limit = _MAX_MASK_RATIO.get(category, 3.2)
    good = 0
    counted = 0
    for (_x, _y, mask), box in zip(pieces, boxes, strict=True):
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        counted += 1
        if int(mask.sum()) <= limit * area:
            good += 1
    if counted == 0:
        return False
    return good >= _PLAUSIBLE_SHARE * counted


def _soft_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    category: str = "vehicle",
) -> dict[str, object] | None:
    """A window ghost when the motion mask fails for one frame.

    A car that has sat still long enough is in the background plate, so
    the difference mask is empty. The car is still in this frame.
    """
    return window_ghost(frame, box, category)


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

    The interior stays fully opaque. Only about 3 pixels of edge fade.
    Vehicles fill the outer contour first. A motion mask on a car is often
    a ring of edges, and the body would otherwise drop out.
    """
    height, width = frame.shape[:2]
    if category == "vehicle" and _outline_worth_tracing(mask):
        mask = _trace_outline(mask)
    mask = _smooth_contour(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return None
    pad = FEATHER_PX + 2
    cx0 = int(max(0, origin_x + int(xs.min()) - pad))
    cy0 = int(max(0, origin_y + int(ys.min()) - pad))
    cx1 = int(min(width, origin_x + int(xs.max()) + 1 + pad))
    cy1 = int(min(height, origin_y + int(ys.max()) + 1 + pad))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    alpha = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
    _paste(alpha, cx0, cy0, mask * 255, origin_x, origin_y)
    alpha = solid_feather(alpha)
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
    span: tuple[float, float, float, float],
    travel: float,
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
        piece = attach_motion(frame, background, box, category, fg_thresh, span, travel)
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
    span = _half_extents(boxes)
    travel = _horizontal_travel(boxes)
    background = clean_background(frames, boxes, category, span, travel)
    height, width = background.shape[:2]
    if needs_window(background):
        return window_ghosts(frames, boxes, category)
    pieces = _motion_pieces(
        frames, boxes, background, category, fg_thresh, span, travel
    )
    if not masks_plausible(pieces, boxes, category):
        return window_ghosts(frames, boxes, category)
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
        if mostly_off_frame(box, width, height):
            # A box sitting on the frame edge used to keep a sliver and a
            # leader that pointed at the border. Leave a blank so the
            # caller can drop the frame instead of drawing that line.
            ghosts.append(_blank_ghost(box))
            cached = None
            cached_box = None
            continue
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
            ghost = _soft_box(frame, box, category)
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
    alpha = solid_feather(alpha)
    return {
        "x": x0,
        "y": y0,
        "crop": crop,
        "alpha": alpha,
        "box": tuple(float(v) for v in box),
        "window": True,
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
