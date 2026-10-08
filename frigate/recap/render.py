"""Draw a recap frame and encode the synopsis with ffmpeg.

Labels stay put for the whole appearance of an object. Only the thin
leader and its dot follow the ghost. Vehicles are drawn under people.
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from frigate.recap.categories import CAT_COLOR
from frigate.recap.layout import ScheduledUnit, plan_label_rects

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
DRAW_RANK = {"vehicle": 0, "parked": 1, "delivery": 2, "animal": 3, "person": 4}


@dataclass
class GhostFrame:
    """One sampled appearance of an object."""

    x: int
    y: int
    box: tuple[float, float, float, float]
    crop: np.ndarray | None = None
    alpha: np.ndarray | None = None
    jpeg: bytes | None = None
    alpha_shape: tuple[int, int] | None = None
    alpha_bytes: bytes | None = None

    def pixels(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.crop is not None and self.alpha is not None:
            return self.crop, self.alpha
        if not self.jpeg or not self.alpha_bytes or not self.alpha_shape:
            return None
        crop = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        alpha = np.frombuffer(self.alpha_bytes, np.uint8).reshape(self.alpha_shape)
        if crop is None:
            return None
        return crop, alpha

    def pack(self) -> None:
        """Replace arrays with compressed bytes so a long recap stays small."""
        if self.crop is None or self.alpha is None:
            return
        ok, encoded = cv2.imencode(
            ".jpg", self.crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not ok:
            return
        self.jpeg = encoded.tobytes()
        self.alpha_shape = (int(self.alpha.shape[0]), int(self.alpha.shape[1]))
        self.alpha_bytes = self.alpha.tobytes()
        self.crop = None
        self.alpha = None


@dataclass
class Tube:
    """Ghost frames for one event, in source order."""

    event_id: str
    clip_event_id: str
    label: str
    category: str
    start_time: float
    frames: list[GhostFrame] = field(default_factory=list)
    suffix: str | None = None
    link_event_id: str | None = None
    # Full frames from the start and end of the clip, so a parked car can
    # be cropped from a moment when the person is not standing in front of it.
    context_first: bytes | None = None
    context_last: bytes | None = None

    def context_image(self, which: str) -> np.ndarray | None:
        raw = self.context_last if which == "last" else self.context_first
        if not raw:
            return None
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height + baseline


def _leader_anchor(
    rect: tuple[int, int, int, int], head: tuple[float, float]
) -> tuple[int, int]:
    x = int(min(max(head[0], rect[0] + 6), rect[2] - 6))
    y = rect[3] if head[1] >= rect[3] else rect[1]
    return x, y


def _blend(
    canvas: np.ndarray,
    before: np.ndarray,
    box: tuple[int, int, int, int],
    amount: float,
) -> None:
    if amount >= 0.999:
        return
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = cv2.addWeighted(
        before, 1 - amount, canvas[y0:y1, x0:x1], amount, 0
    )


def draw_icon(
    image: np.ndarray,
    category: str,
    x0: int,
    y0: int,
    size: int,
    chip: tuple[int, int, int],
) -> None:
    """Tiny white glyph: person, vehicle, package, or paw."""
    white = (255, 255, 255)

    def point(fx: float, fy: float) -> tuple[int, int]:
        return int(x0 + fx * size), int(y0 + fy * size)

    def radius(fraction: float) -> int:
        return max(1, int(fraction * size))

    if category == "person":
        cv2.circle(image, point(0.5, 0.30), radius(0.15), white, -1, cv2.LINE_AA)
        cv2.ellipse(
            image,
            point(0.5, 0.86),
            (radius(0.27), radius(0.36)),
            0,
            180,
            360,
            white,
            -1,
            cv2.LINE_AA,
        )
    elif category in ("vehicle", "parked"):
        cv2.fillPoly(
            image,
            [
                np.array(
                    [
                        point(0.27, 0.48),
                        point(0.36, 0.28),
                        point(0.66, 0.28),
                        point(0.76, 0.48),
                    ]
                )
            ],
            white,
            cv2.LINE_AA,
        )
        cv2.rectangle(image, point(0.10, 0.46), point(0.90, 0.72), white, -1)
        for fx in (0.30, 0.70):
            cv2.circle(
                image, point(fx, 0.74), radius(0.11), (25, 25, 25), -1, cv2.LINE_AA
            )
    elif category == "delivery":
        cv2.rectangle(image, point(0.18, 0.30), point(0.82, 0.82), white, -1)
        cv2.line(
            image,
            point(0.18, 0.44),
            point(0.82, 0.44),
            chip,
            max(1, radius(0.06)),
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            point(0.5, 0.30),
            point(0.5, 0.82),
            chip,
            max(1, radius(0.08)),
            cv2.LINE_AA,
        )
    else:
        cv2.ellipse(
            image,
            point(0.5, 0.68),
            (radius(0.19), radius(0.15)),
            0,
            0,
            360,
            white,
            -1,
            cv2.LINE_AA,
        )
        for fx, fy in ((0.24, 0.42), (0.41, 0.27), (0.59, 0.27), (0.76, 0.42)):
            cv2.circle(image, point(fx, fy), radius(0.085), white, -1, cv2.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
    opacity: float,
    category: str,
) -> None:
    """Stationary time chip: translucent background, icon, readable text."""
    height, width = image.shape[:2]
    x0 = max(0, rect[0])
    y0 = max(0, rect[1])
    x1 = min(width, rect[2])
    y1 = min(height, rect[3])
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    dark = np.zeros_like(region)
    dark[:] = (18, 18, 18)
    image[y0:y1, x0:x1] = cv2.addWeighted(region, 1 - opacity, dark, opacity, 0)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2, cv2.LINE_AA)
    icon = max(8, (rect[3] - rect[1]) - 6)
    chip_x, chip_y = rect[0] + 3, rect[1] + 3
    cv2.rectangle(image, (chip_x, chip_y), (chip_x + icon, chip_y + icon), color, -1)
    draw_icon(image, category, chip_x, chip_y, icon, color)
    text_w, text_h = _text_size(text, scale, thickness)
    _ = text_w
    tx = rect[0] + icon + 8
    ty = rect[1] + 4 + text_h
    cv2.putText(
        image,
        text,
        (tx + 1, ty + 1),
        FONT,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (tx, ty), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _paste_ghost(canvas: np.ndarray, ghost: GhostFrame, alpha_scale: float) -> bool:
    """Paint a ghost. False when nothing visible landed on the canvas.

    Callers use that to skip the leader. A line with no object under it
    looks like the track is still there.
    """
    pixels = ghost.pixels()
    if pixels is None or alpha_scale <= 0.01:
        return False
    crop, alpha = pixels
    height, width = crop.shape[:2]
    y0, x0 = ghost.y, ghost.x
    y1 = min(canvas.shape[0], y0 + height)
    x1 = min(canvas.shape[1], x0 + width)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        return False
    region = canvas[y0:y1, x0:x1]
    crop_h = y1 - y0
    crop_w = x1 - x0
    factor = (alpha[:crop_h, :crop_w].astype(np.float32) * (alpha_scale / 255.0))[
        ..., None
    ]
    if float(factor.max()) < 0.04:
        return False
    blended = crop[:crop_h, :crop_w] * factor + region * (1 - factor)
    region[:] = blended.astype(np.uint8)
    return True


def leader_without_ghost(
    ghost_drawn: Sequence[bool], leader_drawn: Sequence[bool]
) -> int:
    """Frames that painted a leader without a ghost. Must stay 0."""
    return sum(
        1 for ghost, leader in zip(ghost_drawn, leader_drawn) if leader and not ghost
    )


def _dotted(
    canvas: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int,
) -> tuple[int, int, int, int]:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    segments = max(1, int(distance // 10))
    bounds = (
        int(max(0, min(start[0], end[0]) - 3)),
        int(max(0, min(start[1], end[1]) - 3)),
        int(min(canvas.shape[1], max(start[0], end[0]) + 3)),
        int(min(canvas.shape[0], max(start[1], end[1]) + 3)),
    )
    for step in range(0, segments, 2):
        p1 = (
            int(start[0] + (end[0] - start[0]) * step / segments),
            int(start[1] + (end[1] - start[1]) * step / segments),
        )
        p2 = (
            int(start[0] + (end[0] - start[0]) * (step + 1) / segments),
            int(start[1] + (end[1] - start[1]) * (step + 1) / segments),
        )
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    return bounds


def compose_frame(
    plate: np.ndarray,
    units: list[ScheduledUnit],
    tubes: list[Tube],
    starts: list[int],
    rects: list[tuple[int, int, int, int]],
    frame_index: int,
    *,
    header: str,
    header_h: int,
    fade_frames: int,
    label_opacity: float,
    font_scale: float,
    repeats: list[int],
) -> np.ndarray:
    """One synopsis frame. ``repeats[i]`` is how many output frames share a source frame."""
    canvas = plate.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, int(round(height / 1080 * 2)))
    scale = max(0.45, height / 1080 * font_scale)
    line_w = max(1, int(round(height / 1080 * 1.5)))
    dot_r = max(3, int(round(height / 1080 * 4)))
    order = sorted(
        range(len(units)),
        key=lambda index: (DRAW_RANK.get(units[index].category, 3), starts[index]),
    )
    active: list[int] = []
    fade: dict[int, float] = {}
    for index in order:
        length = len(units[index].boxes)
        if starts[index] <= frame_index < starts[index] + length:
            local = frame_index - starts[index]
            fade[index] = max(
                0.0,
                min(
                    1.0,
                    (local + 1) / max(1, fade_frames),
                    (length - local) / max(1, fade_frames),
                ),
            )
            active.append(index)
    tube_by_id = {tube.event_id: tube for tube in tubes}
    shown: set[int] = set()
    for index in active:
        unit = units[index]
        if unit.category == "parked":
            car_box = unit.boxes[0]
            if any(
                other != index
                and units[other].category == "parked"
                and (starts[other], other) < (starts[index], index)
                and _iou(units[other].boxes[0], car_box) > 0.4
                for other in active
            ):
                continue
        ghost_alpha = {"vehicle": 0.72, "parked": 0.95}.get(unit.category, 0.84) * fade[
            index
        ]
        tube = tube_by_id.get(unit.event_id)
        if tube is None or not tube.frames:
            continue
        repeat = max(1, repeats[index] if index < len(repeats) else 1)
        source_index = min(
            len(tube.frames) - 1, (frame_index - starts[index]) // repeat
        )
        if _paste_ghost(canvas, tube.frames[source_index], ghost_alpha):
            shown.add(index)

    for index in active:
        if index not in shown:
            continue
        local = frame_index - starts[index]
        box = units[index].boxes[min(local, len(units[index].boxes) - 1)]
        head = ((box[0] + box[2]) / 2, box[1])
        tip = (int(head[0]), int(head[1]) - 2)
        rect = rects[index]
        anchor = _leader_anchor(rect, head)
        bounds = (
            max(0, min(anchor[0], tip[0]) - dot_r - 3),
            max(0, min(anchor[1], tip[1]) - dot_r - 3),
            min(width, max(anchor[0], tip[0]) + dot_r + 3),
            min(height, max(anchor[1], tip[1]) + dot_r + 3),
        )
        before = (
            canvas[bounds[1] : bounds[3], bounds[0] : bounds[2]].copy()
            if fade[index] < 0.999
            else None
        )
        color = CAT_COLOR.get(units[index].category, (255, 255, 255))
        if not (
            rect[0] - 4 <= tip[0] <= rect[2] + 4
            and rect[1] - 4 <= tip[1] <= rect[3] + 4
        ):
            cv2.line(canvas, anchor, tip, color, line_w, cv2.LINE_AA)
        cv2.circle(canvas, tip, dot_r + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, tip, dot_r, color, -1, cv2.LINE_AA)
        if before is not None:
            _blend(canvas, before, bounds, fade[index])

    partners = {
        index: unit.link_index
        for index, unit in enumerate(units)
        if unit.link_index is not None
    }
    for index, partner in partners.items():
        if index not in fade or partner not in fade:
            continue
        left, right = rects[index], rects[partner]
        left_c = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
        right_c = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
        if abs(right_c[1] - left_c[1]) < (left[3] - left[1]) and not (
            left[0] <= right_c[0] <= left[2]
        ):
            anchor_a = (left[2] if right_c[0] > left_c[0] else left[0], left_c[1])
            anchor_b = (right[0] if right_c[0] > left_c[0] else right[2], right_c[1])
        else:
            anchor_a = _leader_anchor(left, right_c)
            anchor_b = _leader_anchor(right, left_c)
        bounds = (
            int(max(0, min(anchor_a[0], anchor_b[0]) - 3)),
            int(max(0, min(anchor_a[1], anchor_b[1]) - 3)),
            int(min(width, max(anchor_a[0], anchor_b[0]) + 3)),
            int(min(height, max(anchor_a[1], anchor_b[1]) + 3)),
        )
        amount = min(fade[index], fade[partner])
        before = (
            canvas[bounds[1] : bounds[3], bounds[0] : bounds[2]].copy()
            if amount < 0.999
            else None
        )
        _dotted(canvas, anchor_a, anchor_b, CAT_COLOR["parked"], line_w + 1)
        if before is not None:
            _blend(canvas, before, bounds, amount)

    for index in active:
        rect = rects[index]
        bounds = (
            max(0, rect[0] - 2),
            max(0, rect[1] - 2),
            min(width, rect[2] + 2),
            min(height, rect[3] + 2),
        )
        before = (
            canvas[bounds[1] : bounds[3], bounds[0] : bounds[2]].copy()
            if fade[index] < 0.999
            else None
        )
        draw_label(
            canvas,
            units[index].text,
            rect,
            CAT_COLOR.get(units[index].category, (255, 255, 255)),
            scale,
            thickness,
            label_opacity,
            units[index].category,
        )
        if before is not None:
            _blend(canvas, before, bounds, fade[index])

    cv2.rectangle(canvas, (0, 0), (width, header_h), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        header,
        (14, int(header_h * 0.68)),
        FONT,
        0.85 * height / 1080,
        (255, 255, 255),
        max(1, thickness),
        cv2.LINE_AA,
    )
    return canvas


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = ix * iy
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    area += max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    area -= inter
    return inter / area if area else 0.0


def layout_labels(
    units: list[ScheduledUnit],
    starts: list[int],
    width: int,
    height: int,
    *,
    font_scale: float,
    header_h: int,
    min_gap: int,
) -> list[tuple[int, int, int, int]]:
    """Measure label text and place one stationary rectangle per unit."""
    thickness = max(1, int(round(height / 1080 * 2)))
    scale = max(0.45, height / 1080 * font_scale)
    sizes = {
        index: _text_size(unit.text or "0:00 PM", scale, thickness)
        for index, unit in enumerate(units)
    }
    # Icon sits beside the text, so the planner's width math adds the icon
    # on top of this text size. Pad a little so the drawn chip matches.
    return plan_label_rects(
        units,
        starts,
        width,
        height,
        text_size=sizes,
        header_h=header_h,
        min_gap=int(round(min_gap * height / 1080)),
    )


def encode_video(
    plate: np.ndarray,
    units: list[ScheduledUnit],
    tubes: list[Tube],
    starts: list[int],
    rects: list[tuple[int, int, int, int]],
    frame_count: int,
    output_path: str,
    *,
    ffmpeg: str,
    fps: int,
    header: str,
    header_h: int,
    fade_frames: int,
    label_opacity: float,
    font_scale: float,
    repeats: list[int],
    cancel_check,
) -> np.ndarray | None:
    """Pipe raw frames to ffmpeg. Returns a thumbnail frame, or None if cancelled."""
    height, width = plate.shape[:2]
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    thumb: np.ndarray | None = None
    busiest = -1
    try:
        for frame_index in range(max(1, frame_count)):
            if cancel_check():
                process.kill()
                return None
            image = compose_frame(
                plate,
                units,
                tubes,
                starts,
                rects,
                frame_index,
                header=header,
                header_h=header_h,
                fade_frames=fade_frames,
                label_opacity=label_opacity,
                font_scale=font_scale,
                repeats=repeats,
            )
            visible = sum(
                1
                for start, unit in zip(starts, units, strict=True)
                if start <= frame_index < start + len(unit.boxes)
            )
            if visible >= busiest:
                busiest = visible
                thumb = image.copy()
            process.stdin.write(image.tobytes())
    except BrokenPipeError:
        err = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg closed the synopsis pipe: {err}") from None
    process.stdin.close()
    code = process.wait()
    if code != 0:
        err = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg failed to encode the recap ({code}): {err}")
    return thumb
