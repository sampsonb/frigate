"""Recap APIs: list, start, cancel, play, and fall back to the archive."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
)
from peewee import DoesNotExist
from pydantic import BaseModel, Field

from frigate.api.auth import (
    allow_any_authenticated,
    get_allowed_cameras_for_filter,
    require_camera_access,
)
from frigate.api.defs.tags import Tags
from frigate.const import RECAP_DIR
from frigate.models import Event
from frigate.recap.archive import (
    CLIP_GONE,
    ArchivePlayback,
    archive_locations,
    archive_recap_dirs,
    lookup_playback,
    present_local_segments,
    remote_manifest,
    remote_recap_summaries,
)
from frigate.recap.clips import (
    download_name,
    live_clip_urls,
    playable_mp4_response,
    render_faststart_mp4,
)
from frigate.recap.queries import live_recordings_exist
from frigate.recap.storage import (
    delete_recap,
    find_manifest,
    list_visible_recaps,
    read_manifest,
    recap_order_key,
    safe_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.recap])


class RecapStartBody(BaseModel):
    """On-demand window. Pass ``hours``, or ``after`` and ``before`` as epoch seconds."""

    hours: float | None = Field(default=None, gt=0, le=168)
    after: float | None = None
    before: float | None = None


def _manager(request: Request):
    manager = getattr(request.app, "recap_manager", None)
    if manager is None:
        return None
    return manager


def _locations(request: Request):
    config = getattr(request.app, "frigate_config", None)
    if config is None:
        return []
    return archive_locations(config)


def _window(body: RecapStartBody) -> tuple[float, float] | None:
    import time

    if body.after is not None and body.before is not None:
        return float(body.after), float(body.before)
    if body.hours is not None:
        end = time.time()
        return end - body.hours * 3600, end
    return None


def _is_local(directory: Path | None) -> bool:
    if directory is None:
        return False
    try:
        directory.resolve().relative_to(Path(RECAP_DIR).resolve())
    except ValueError:
        return False
    return True


@router.get(
    "/recap",
    dependencies=[Depends(allow_any_authenticated())],
    summary="List recaps",
    description=(
        "Lists saved video synopses for cameras the user can view. Newest "
        "window end first, then newest creation time, across cameras and kinds. "
        "Includes older recaps stored on recap.archive when that is configured."
    ),
)
def recap_list(
    request: Request,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    allowed = set(allowed_cameras)
    locations = _locations(request)
    items = [
        item
        for item in list_visible_recaps(archive_recap_dirs(locations))
        if item.get("camera") in allowed
    ]
    seen = {str(item.get("id") or "") for item in items}
    for item in remote_recap_summaries(locations):
        recap_id = str(item.get("id") or "")
        if recap_id in seen or item.get("camera") not in allowed:
            continue
        items.append(item)
        seen.add(recap_id)
    items.sort(key=recap_order_key, reverse=True)
    return JSONResponse(content=items)


@router.get(
    "/recap/event/{event_id}",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Resolve an event clip",
    description=(
        "Tells the recap player where a time label's clip lives. Live "
        "recordings use the HLS VOD playlist Safari can play. Otherwise the "
        "configured archive is searched by event id."
    ),
)
async def recap_event_source(request: Request, event_id: str):
    if not safe_id(event_id):
        return JSONResponse(
            content={"success": False, "message": "Unknown event"},
            status_code=404,
        )
    live, camera = _live_clip(event_id)
    if camera:
        await require_camera_access(camera, request=request)
    if live is not None:
        urls = live_clip_urls(live[0], live[1], live[2], event_id)
        return JSONResponse(
            content={
                "source": "frigate",
                "camera": live[0],
                "clip": urls["clip"],
                "download": urls["download"],
                "snapshot": urls["snapshot"],
                "message": "",
            }
        )
    playback = lookup_playback(_locations(request), event_id)
    if playback is None:
        return JSONResponse(
            content={"success": False, "message": "This event is no longer available"},
            status_code=404,
        )
    if playback.segments and present_local_segments(playback.segments) is None:
        playback.segments = []
        playback.message = CLIP_GONE
    if playback.camera and playback.camera != camera:
        await require_camera_access(playback.camera, request=request)
    return JSONResponse(content=_playback_body(event_id, playback))


@router.get(
    "/recap/event/{event_id}/clip.mp4",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Play an archived event clip",
)
async def recap_event_clip(request: Request, event_id: str):
    playback, denied = await _playback(request, event_id)
    if isinstance(denied, JSONResponse):
        return denied
    assert isinstance(playback, ArchivePlayback)
    if not playback.segments:
        return _gone(playback.message)
    # Check again immediately before opening. A file can be removed after
    # the index lookup, and a missing segment must not become a 500.
    local = present_local_segments(playback.segments)
    if local is None:
        return _gone()
    remote = [segment.url for segment in playback.segments if segment.url]
    if local:
        if any(segment.path is None or not segment.path.is_file() for segment in local):
            return _gone()
        config = getattr(request.app, "frigate_config", None)
        ffmpeg = "ffmpeg"
        if config is not None:
            ffmpeg = config.ffmpeg.ffmpeg_path or ffmpeg
        try:
            rendered = await asyncio.to_thread(
                render_faststart_mp4, ffmpeg, local, playback.start, playback.end
            )
        except (RuntimeError, FileNotFoundError, OSError):
            logger.exception("Archived clip could not be prepared for playback")
            return _gone()
        return playable_mp4_response(rendered, "clip.mp4", delete_after=True)
    if len(remote) == 1:
        return RedirectResponse(remote[0])
    return JSONResponse(
        content={
            "success": False,
            "message": "Mount recap.archive.path to play a multi-file recording",
        },
        status_code=404,
    )


@router.get(
    "/recap/event/{event_id}/snapshot.jpg",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Archived event snapshot",
)
async def recap_event_snapshot(request: Request, event_id: str):
    playback, denied = await _playback(request, event_id)
    if isinstance(denied, JSONResponse):
        return denied
    assert isinstance(playback, ArchivePlayback)
    if playback.snapshot is not None:
        if not playback.snapshot.is_file():
            return _gone()
        media_type = (
            "image/webp" if playback.snapshot.suffix == ".webp" else "image/jpeg"
        )
        try:
            return FileResponse(playback.snapshot, media_type=media_type)
        except FileNotFoundError:
            return _gone()
    if playback.snapshot_url:
        return RedirectResponse(playback.snapshot_url)
    return JSONResponse(
        content={"success": False, "message": "Snapshot is not in the archive"},
        status_code=404,
    )


@router.get(
    "/recap/rolling/{camera_name}",
    dependencies=[Depends(require_camera_access), Depends(allow_any_authenticated())],
    summary="Rolling recap status",
    description=(
        "The camera's rolling recap (the last rolling_hours, rebuilt in place "
        "every rolling_interval_minutes when something changed), plus any "
        "refresh that is building."
    ),
)
def recap_rolling(request: Request, camera_name: str):
    manager = _manager(request)
    if manager is None:
        return JSONResponse(
            content={"success": False, "message": "Recap is not available"},
            status_code=503,
        )
    camera = manager.config.cameras.get(camera_name)
    if camera is None or not camera.recap.enabled:
        return JSONResponse(
            content={
                "success": False,
                "message": "Recap is not enabled for that camera",
            },
            status_code=404,
        )
    body = manager.rolling_status(camera_name)
    body["hours"] = camera.recap.rolling_hours
    body["interval_minutes"] = camera.recap.rolling_interval_minutes
    return JSONResponse(content=body)


@router.post(
    "/recap/rolling/{camera_name}/refresh",
    dependencies=[Depends(require_camera_access), Depends(allow_any_authenticated())],
    summary="Refresh the rolling recap now",
    description=(
        "Checks for new events and rebuilds the rolling recap ahead of "
        "scheduled work. Pass force=true to rebuild even when nothing changed."
    ),
)
def recap_rolling_refresh(request: Request, camera_name: str, force: bool = False):
    manager = _manager(request)
    if manager is None:
        return JSONResponse(
            content={"success": False, "message": "Recap is not available"},
            status_code=503,
        )
    try:
        result = manager.start_rolling(camera_name, manual=True, force=force)
    except ValueError as err:
        return JSONResponse(
            content={"success": False, "message": str(err)},
            status_code=400,
        )
    return JSONResponse(content={"success": True, **result})


@router.post(
    "/recap/{camera_name}/start",
    dependencies=[Depends(require_camera_access), Depends(allow_any_authenticated())],
    summary="Start a recap",
    description=(
        "Queues a synopsis for the camera. Pass hours (for example 12 or 48) or an "
        "after/before epoch range. An explicit after and before that is one "
        "local midnight-to-midnight day uses the same kind as that day's "
        "nightly recap. Any other explicit range is its own kind and is not "
        "grouped as last-N hours. One recap runs at a time. Others wait, "
        "and a request already waiting or running is returned instead of "
        "being queued twice."
    ),
)
def recap_start(request: Request, camera_name: str, body: RecapStartBody):
    manager = _manager(request)
    if manager is None:
        return JSONResponse(
            content={"success": False, "message": "Recap is not available"},
            status_code=503,
        )
    window = _window(body)
    if window is None:
        return JSONResponse(
            content={"success": False, "message": "Pass hours, or after and before"},
            status_code=400,
        )
    after, before = window
    # hours becomes a last-N window. A caller-supplied after and before
    # stays an explicit range even when the span is 1, 6, 12, or 24 hours.
    explicit_range = body.after is not None and body.before is not None
    try:
        manifest = manager.start(
            camera_name,
            after,
            before,
            reason="manual",
            explicit_range=explicit_range,
        )
    except ValueError as err:
        return JSONResponse(
            content={"success": False, "message": str(err)},
            status_code=400,
        )
    return JSONResponse(
        content={"success": True, "id": manifest["id"], "recap": manifest}
    )


@router.get(
    "/recap/{recap_id}",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Get a recap",
    description="Full manifest, including track timing so the UI can make time labels clickable.",
)
async def recap_get(request: Request, recap_id: str):
    found = await _authorized(request, recap_id)
    if isinstance(found, JSONResponse):
        return found
    _directory, manifest, _remote = found
    return JSONResponse(content=manifest)


@router.post(
    "/recap/{recap_id}/cancel",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Cancel a recap",
)
async def recap_cancel(request: Request, recap_id: str):
    found = await _authorized(request, recap_id)
    if isinstance(found, JSONResponse):
        return found
    directory, _manifest, _remote = found
    if not _is_local(directory):
        return JSONResponse(
            content={
                "success": False,
                "message": "Archived recaps are already finished",
            },
            status_code=400,
        )
    manager = _manager(request)
    if manager is None:
        return JSONResponse(
            content={"success": False, "message": "Recap is not available"},
            status_code=503,
        )
    manager.cancel(recap_id)
    return JSONResponse(content={"success": True, "message": "Cancelling"})


@router.delete(
    "/recap/{recap_id}",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Delete a recap",
)
async def recap_delete(request: Request, recap_id: str):
    found = await _authorized(request, recap_id)
    if isinstance(found, JSONResponse):
        return found
    directory, manifest, _remote = found
    if not _is_local(directory):
        return JSONResponse(
            content={
                "success": False,
                "message": "Archived recaps stay on the archive drive",
            },
            status_code=400,
        )
    manager = _manager(request)
    if manager is not None:
        manager.cancel(recap_id)
    camera = str(manifest.get("camera") or "")
    if directory is not None and not camera:
        camera = directory.parent.name
    delete_recap(camera, recap_id)
    return JSONResponse(content={"success": True, "message": "Deleted"})


@router.get(
    "/recap/{recap_id}/video.mp4",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Play a recap",
)
async def recap_video(
    request: Request,
    recap_id: str,
    download: bool = False,
    name: str | None = None,
):
    """Play a recap inline, or save it with ``?download=1&name=...``."""
    found = await _authorized(request, recap_id)
    if isinstance(found, JSONResponse):
        return found
    directory, manifest, remote = found
    if directory is not None:
        path = directory / "video.mp4"
        if path.is_file():
            if download:
                return FileResponse(
                    path,
                    media_type="video/mp4",
                    filename=download_name(name),
                    content_disposition_type="attachment",
                )
            return playable_mp4_response(path, "recap.mp4")
    if remote:
        camera = str(manifest.get("camera") or "")
        return RedirectResponse(f"{remote}/recap/{camera}/{recap_id}/video.mp4")
    return JSONResponse(
        content={"success": False, "message": "Video is not ready"},
        status_code=404,
    )


@router.get(
    "/recap/{recap_id}/thumb.jpg",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Recap thumbnail",
)
async def recap_thumb(request: Request, recap_id: str):
    found = await _authorized(request, recap_id)
    if isinstance(found, JSONResponse):
        return found
    directory, manifest, remote = found
    if directory is not None:
        path = directory / "thumb.jpg"
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")
    if remote:
        camera = str(manifest.get("camera") or "")
        return RedirectResponse(f"{remote}/recap/{camera}/{recap_id}/thumb.jpg")
    return JSONResponse(
        content={"success": False, "message": "Thumbnail is not ready"},
        status_code=404,
    )


async def _authorized(
    request: Request, recap_id: str
) -> tuple[Path | None, dict[str, Any], str | None] | JSONResponse:
    if not safe_id(recap_id):
        return JSONResponse(
            content={"success": False, "message": "Unknown recap"},
            status_code=404,
        )
    locations = _locations(request)
    found = find_manifest(recap_id, archive_recap_dirs(locations))
    remote: str | None = None
    if found is None:
        fetched = remote_manifest(locations, recap_id)
        if fetched is None:
            return JSONResponse(
                content={"success": False, "message": "Unknown recap"},
                status_code=404,
            )
        remote, manifest = fetched
        directory = None
    else:
        directory, manifest = found
        manifest = dict(manifest)
    camera = str(manifest.get("camera") or "")
    await require_camera_access(camera, request=request)
    if directory is not None and _is_local(directory):
        fresh = read_manifest(directory)
        if fresh is not None:
            manifest = fresh
        manifest = dict(manifest)
        manifest["source"] = "local"
    else:
        manifest = dict(manifest)
        manifest["source"] = "archive"
    return directory, manifest, remote


async def _playback(
    request: Request, event_id: str
) -> tuple[ArchivePlayback | None, JSONResponse | None]:
    if not safe_id(event_id):
        return None, JSONResponse(
            content={"success": False, "message": "Unknown event"},
            status_code=404,
        )
    playback = lookup_playback(_locations(request), event_id)
    if playback is None:
        return None, JSONResponse(
            content={"success": False, "message": "This event is no longer available"},
            status_code=404,
        )
    if playback.camera:
        await require_camera_access(playback.camera, request=request)
    return playback, None


def _gone(message: str | None = None) -> JSONResponse:
    """410 when an archived clip was listed and the file is now gone."""
    return JSONResponse(
        content={"success": False, "message": message or CLIP_GONE},
        status_code=410,
    )


def _playback_body(event_id: str, playback: ArchivePlayback) -> dict[str, Any]:
    clip: str | None = None
    if playback.segments:
        remote = [segment.url for segment in playback.segments if segment.url]
        local = [segment for segment in playback.segments if segment.path is not None]
        if local or len(playback.segments) > 1:
            clip = f"recap/event/{event_id}/clip.mp4"
        elif len(remote) == 1:
            clip = remote[0]
    snapshot: str | None = None
    if playback.snapshot is not None:
        snapshot = f"recap/event/{event_id}/snapshot.jpg"
    elif playback.snapshot_url:
        snapshot = playback.snapshot_url
    return {
        "source": "archive",
        "camera": playback.camera,
        "clip": clip,
        "snapshot": snapshot,
        "message": playback.message,
    }


def _live_clip(
    event_id: str,
) -> tuple[tuple[str, float, float] | None, str | None]:
    """Recordings still on disk, as ``(camera, start, end)``, plus the camera.

    The camera is returned even when the recordings are gone so the caller
    can enforce access before searching the archive. ``end`` is the wall
    clock when the event is still open.
    """
    try:
        event = Event.get(Event.id == event_id)
    except DoesNotExist:
        return None, None
    except Exception:
        logger.debug("Live event lookup failed for %s", event_id)
        return None, None
    camera = str(event.camera) if event.camera else None
    # NULL has_clip was never written. NULL end_time is still in progress.
    # Both used to look like "no clip" and skip the recordings table.
    try:
        exists = live_recordings_exist(event)
    except Exception:
        logger.debug("Live recording lookup failed for %s", event_id)
        return None, camera
    if not exists or not camera or event.start_time is None:
        return None, camera
    end = time.time() if event.end_time is None else float(event.end_time)
    return (camera, float(event.start_time), end), camera
