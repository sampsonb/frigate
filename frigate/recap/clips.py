"""Playable media for a recap time label.

Safari will not play a copied camera recording inside a video element.
Live clips therefore point at Frigate's HLS VOD playlist. Archived files
are rewritten to H.264/AAC with the moov atom first, then sent inline so
a Range request can seek.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from frigate.recap.archive import ArchiveSegment

logger = logging.getLogger(__name__)


def _timestamp(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def live_clip_urls(
    camera: str, start: float, end: float, event_id: str
) -> dict[str, str]:
    """Playback and download paths for an event Frigate still has.

    ``clip`` is the HLS playlist Safari plays natively. ``download`` stays
    on the existing clip endpoint so the Download button still saves a file.
    """
    if end <= start:
        end = start + 1.0
    return {
        "clip": (
            f"vod/{camera}/start/{_timestamp(start)}/end/{_timestamp(end)}/index.m3u8"
        ),
        "download": f"events/{event_id}/clip.mp4",
        "snapshot": f"events/{event_id}/snapshot.jpg",
    }


def download_name(name: str | None) -> str:
    """A safe file name for a saved recap, ending in .mp4.

    Clock colons become dots. Only letters, digits, spaces, and ``.,_-()``
    are kept, so the name works in the Files app, Photos, and a desktop
    download folder.
    """
    cleaned = "".join(
        char
        for char in str(name or "").replace(":", ".")
        if char.isalnum() or char in " .,_-()"
    )
    cleaned = " ".join(cleaned.split()).strip(" .")[:120]
    if not cleaned:
        return "recap.mp4"
    if not cleaned.lower().endswith(".mp4"):
        cleaned = f"{cleaned}.mp4"
    return cleaned


def playable_mp4_response(
    path: Path, filename: str, *, delete_after: bool = False
) -> FileResponse:
    """Serve an MP4 inline, with Range support from FileResponse.

    The default content disposition is an attachment. Safari then refuses
    to play the file in a video element and draws a crossed-out play icon,
    while a download link still succeeds.
    """
    background = BackgroundTask(path.unlink, missing_ok=True) if delete_after else None
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        content_disposition_type="inline",
        background=background,
    )


def _concat_list(
    segments: list[ArchiveSegment], start: float | None, end: float | None
) -> str:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        for segment in segments:
            path = segment.path
            if path is None:
                raise FileNotFoundError("archived segment has no path")
            escaped = str(path).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            if (
                start is not None
                and segment.start is not None
                and segment.start < start
            ):
                handle.write(f"inpoint {max(0.0, start - segment.start):.3f}\n")
            if (
                end is not None
                and segment.start is not None
                and segment.end is not None
                and segment.end > end
            ):
                handle.write(f"outpoint {max(0.1, end - segment.start):.3f}\n")
    finally:
        handle.close()
    return handle.name


def _run(command: list[str]) -> int:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError:
        logger.exception("ffmpeg could not be started")
        return 1
    if completed.returncode != 0:
        logger.error(
            "ffmpeg failed (%s): %s",
            completed.returncode,
            completed.stderr.decode("utf-8", "replace")[-500:],
        )
    return completed.returncode


def render_faststart_mp4(
    ffmpeg: str,
    segments: list[ArchiveSegment],
    start: float | None,
    end: float | None,
) -> Path:
    """Transcode archived segments to H.264/AAC with the moov atom first.

    ``-c copy`` keeps the camera codec (often HEVC) and a fragmented moov.
    Safari cannot play that. A normal faststart file can.
    """
    playlist = _concat_list(segments, start, end)
    output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output.close()
    destination = Path(output.name)
    shared = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        playlist,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    try:
        with_audio = _run(shared + ["-c:a", "aac", "-b:a", "96k", str(destination)])
        if with_audio != 0:
            video_only = _run(shared + ["-an", str(destination)])
            if video_only != 0:
                destination.unlink(missing_ok=True)
                raise RuntimeError("archived clip could not be encoded")
        return destination
    finally:
        Path(playlist).unlink(missing_ok=True)
