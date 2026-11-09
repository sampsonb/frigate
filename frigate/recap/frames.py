"""Pull a few scaled frames out of Frigate recordings with ffmpeg.

Decoding is the expensive part of a recap, so each event is sampled at a
low frame rate and scaled down before it ever becomes a numpy array.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def even(value: int) -> int:
    """yuv420 wants even dimensions."""
    return max(2, value - (value % 2))


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Fit a frame inside ``max_width`` without changing the aspect ratio."""
    if width <= 0 or height <= 0:
        return even(max_width), even(int(max_width * 9 / 16))
    scale = min(1.0, max_width / width)
    return even(int(width * scale)), even(int(height * scale))


def probe_size(ffprobe: str, path: str) -> tuple[int, int] | None:
    """Return the video width and height, or None when the file cannot be read."""
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("ffprobe failed for %s", path)
        return None
    text = (result.stdout or "").strip().splitlines()
    if not text or "x" not in text[0]:
        return None
    width_s, height_s = text[0].split("x", 1)
    try:
        return int(width_s), int(height_s)
    except ValueError:
        return None


def read_raw(
    ffmpeg: str,
    source: str,
    start_offset: float,
    duration: float,
    width: int,
    height: int,
    sample_fps: float,
    max_frames: int,
    input_format: str | None = None,
) -> list[np.ndarray]:
    """Decode ``source`` to a short list of BGR frames."""
    if duration <= 0 or max_frames <= 0:
        return []
    fps = max(0.5, sample_fps)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if input_format == "concat":
        command.extend(["-f", "concat", "-safe", "0"])
    command.extend(
        [
            "-ss",
            f"{max(0.0, start_offset):.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            source,
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:",
        ]
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=max(30, int(duration) + 20),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg could not sample %s", source)
        return []
    if result.returncode != 0:
        logger.debug(
            "ffmpeg sample failed: %s",
            result.stderr.decode("utf-8", "replace")[:400],
        )
        return []
    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        return []
    count = len(result.stdout) // frame_bytes
    frames = []
    for index in range(count):
        chunk = result.stdout[index * frame_bytes : (index + 1) * frame_bytes]
        frames.append(np.frombuffer(chunk, np.uint8).reshape((height, width, 3)).copy())
    return frames


def grab_frame(
    ffmpeg: str,
    source: str,
    start_offset: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    """One scaled frame, used to build the background plate."""
    frames = read_raw(ffmpeg, source, start_offset, 0.5, width, height, 2, 1)
    if not frames:
        return None
    return frames[0]


def concat_sample(
    ffmpeg: str,
    segments: list[tuple[str, float, float]],
    width: int,
    height: int,
    sample_fps: float,
    max_frames: int,
    work_dir: Path,
) -> list[np.ndarray]:
    """Sample an event that spans more than one recording segment.

    ``segments`` is ``(path, offset_into_file, duration)`` in order. The
    files are joined whole and the start is found with an accurate seek.
    A concat ``inpoint`` lands on the keyframe before it, so frames from
    up to a few seconds early came out first and every box was late.
    """
    if not segments:
        return []
    if len(segments) == 1:
        path, offset, duration = segments[0]
        return read_raw(
            ffmpeg, path, offset, duration, width, height, sample_fps, max_frames
        )
    playlist = work_dir / "recap-concat.txt"
    lines = []
    for path, _offset, _duration in segments:
        escaped = path.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    playlist.write_text("\n".join(lines) + "\n")
    duration = sum(item[2] for item in segments)
    return read_raw(
        ffmpeg,
        str(playlist),
        segments[0][1],
        duration,
        width,
        height,
        sample_fps,
        max_frames,
        input_format="concat",
    )
