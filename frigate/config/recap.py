"""Video synopsis (recap) configuration."""

from typing import Optional

from pydantic import Field, field_validator

from .base import FrigateBaseModel

__all__ = ["RecapConfig", "DEFAULT_RECAP_LABELS"]

# Labels a recap considers by default. Cameras that do not track one of
# these simply contribute no events for it. Delivery company names are
# included so a Frigate+ attribute model can tag them directly.
DEFAULT_RECAP_LABELS = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "bird",
    "horse",
    "deer",
    "bear",
    "sheep",
    "cow",
    "package",
    "amazon",
    "ups",
    "fedex",
    "usps",
    "dhl",
]


class RecapArchiveConfig(FrigateBaseModel):
    """Optional copy of Frigate media kept on another disk.

    Leave both fields empty to disable the archive. ``path`` is a mount
    inside the container. ``url`` is a base URL when that same tree is
    also served over HTTP. Either one is enough for clip fallback.
    Listing archived recaps needs ``path`` or a ``recap/index.json`` on
    the URL.
    """

    path: Optional[str] = Field(
        default=None,
        title="Filesystem path of the media archive mounted into Frigate.",
        description=(
            "Root of a nightly copy that keeps Frigate's layout: "
            "recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4, clips/, and recap/. "
            "Empty disables the filesystem archive."
        ),
    )
    url: Optional[str] = Field(
        default=None,
        title="Base URL of the media archive.",
        description=(
            "Used when the archive is not mounted, or to play a single "
            "archived file directly. No trailing slash. Empty disables URL access."
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: object) -> str | None:
        """Treat a blank path as unset."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: object) -> str | None:
        """Accept an http(s) base URL, and treat blank as unset."""
        if value is None:
            return None
        text = str(value).strip().rstrip("/")
        if text == "":
            return None
        if not (text.startswith("http://") or text.startswith("https://")):
            raise ValueError("recap archive url must start with http:// or https://")
        return text


class RecapConfig(FrigateBaseModel):
    """RapidRecap-style video synopsis for one camera or the whole install."""

    enabled: bool = Field(
        default=False,
        title="Enable recap generation.",
    )
    enabled_in_config: Optional[bool] = Field(
        default=None,
        title="Keep track of original state of recap.",
    )
    schedule: Optional[str] = Field(
        default=None,
        title="Daily local time to generate a recap automatically (HH:MM).",
        description=(
            "24-hour local time. Leave empty to generate recaps only on demand. "
            "Frigate must be running within a couple of minutes of this time."
        ),
    )
    window_hours: float = Field(
        default=24,
        title="Hours of footage each scheduled recap covers.",
        gt=0,
        le=168,
    )
    max_window_hours: float = Field(
        default=48,
        title="Longest time range an on-demand recap may cover.",
        gt=0,
        le=168,
    )
    rolling_hours: float = Field(
        default=6,
        title="Hours covered by the rolling recap that is kept ready.",
        gt=0,
        le=48,
    )
    rolling_interval_minutes: int = Field(
        default=30,
        title="Minutes between rolling recap refreshes. 0 turns the rolling recap off.",
        description=(
            "The rolling recap is rebuilt in place on this interval. A refresh "
            "is skipped when no event in the window is new or changed."
        ),
        ge=0,
        le=1440,
    )
    rolling_max_age_minutes: int = Field(
        default=180,
        title="Rebuild the rolling recap after this long even with no new events.",
        description="Drops events that have aged out of the window. 0 never forces a rebuild.",
        ge=0,
        le=1440,
    )
    rolling_keep_hours: float = Field(
        default=72,
        title="Hours to keep replaced rolling recaps in the saved list. 0 deletes them on replace.",
        description=(
            "When a new rolling recap is ready, the one it replaces is saved "
            "unless it held the same events. Saved copies older than this are "
            "removed. Nightly, custom, and manual recaps are not affected."
        ),
        ge=0,
        le=720,
    )
    interval_minutes: Optional[int] = Field(
        default=None,
        title="Minutes between automatic short recaps. 0 disables them.",
        description=(
            "When set, Frigate builds a recap of the last this many minutes "
            "on that interval. The kind is rolling-<minutes>m, so the previous "
            "one is replaced when the new one is ready. A run is skipped while "
            "the previous one is still queued or rendering. This is separate "
            "from the rolling recap that stays ready."
        ),
        ge=1,
        le=1440,
    )
    labels: list[str] = Field(
        default_factory=lambda: list(DEFAULT_RECAP_LABELS),
        title="Tracked labels to include.",
    )
    target_length: int = Field(
        default=120,
        title="Target synopsis length in seconds.",
        ge=10,
        le=600,
    )
    label_opacity: float = Field(
        default=0.5,
        title="Opacity of the time-label background.",
        ge=0.0,
        le=1.0,
    )
    max_labels: int = Field(
        default=8,
        title="Maximum number of time labels on screen at once.",
        ge=1,
        le=30,
    )
    min_label_gap: int = Field(
        default=12,
        title="Minimum gap between labels, in pixels at 1080p.",
        ge=0,
        le=80,
    )
    fade_seconds: float = Field(
        default=0.2,
        title="Fade in and out duration for a ghost, its line, and its label.",
        ge=0.0,
        le=2.0,
    )
    retain_days: int = Field(
        default=14,
        title="Days to keep generated recaps. 0 keeps them until deleted.",
        ge=0,
    )
    replace_superseded: bool = Field(
        default=True,
        title="Replace an older recap of the same kind when a new one is ready.",
        description=(
            "Last-N-hours buttons group by that length. An explicit range "
            "that is one local midnight-to-midnight day groups with the "
            "nightly recap for that date. Any other explicit range is kept "
            "on its own. Running, failed, and archive copies are kept."
        ),
    )
    output_fps: int = Field(
        default=12,
        title="Frame rate of the synopsis video.",
        ge=5,
        le=30,
    )
    sample_fps: float = Field(
        default=4,
        title="Frames per second sampled from people and animals.",
        gt=0.5,
        le=15,
    )
    vehicle_sample_fps: float = Field(
        default=8,
        title="Frames per second sampled from vehicles so they stay readable.",
        gt=0.5,
        le=15,
    )
    min_show_seconds: float = Field(
        default=2.5,
        title="Shortest time a ghost stays on screen.",
        ge=0.5,
        le=15,
    )
    max_object_seconds: float = Field(
        default=12,
        title="Most source time kept for one object (loiterers are trimmed).",
        ge=1,
        le=120,
    )
    max_width: int = Field(
        default=1280,
        title="Maximum synopsis width. Frames are scaled down to this.",
        ge=320,
        le=1920,
    )
    font_scale: float = Field(
        default=0.7,
        title="Label font scale at 1080p.",
        gt=0.2,
        le=2.0,
    )
    max_events: int = Field(
        default=400,
        title="Maximum events processed in one recap.",
        ge=1,
        le=5000,
    )
    parked_cars: bool = Field(
        default=True,
        title="Show a parked car when a person gets in or out of it.",
    )
    stationary_path: float = Field(
        default=0.015,
        title="Path movement, as a fraction of the frame, below which a vehicle is parked.",
        ge=0.0,
        le=1.0,
    )
    delivery_search: bool = Field(
        default=True,
        title="Tag deliveries with semantic search when Frigate+ labels are absent.",
    )
    delivery_distance: float = Field(
        default=0.8,
        title="Maximum semantic-search distance to call an event a delivery.",
        ge=0.0,
        le=2.0,
    )
    pause_seconds: float = Field(
        default=0.02,
        title="Pause between events so live detection keeps the CPU.",
        ge=0.0,
        le=2.0,
    )
    archive: RecapArchiveConfig = Field(
        default_factory=RecapArchiveConfig,
        title="Optional external archive of recordings, events, and recaps.",
    )

    @field_validator("schedule", mode="before")
    @classmethod
    def validate_schedule(cls, value: object) -> str | None:
        """Accept HH:MM, and treat a blank value as manual only."""
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError("recap schedule must be HH:MM in 24-hour local time")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                "recap schedule must be HH:MM in 24-hour local time"
            ) from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("recap schedule must be HH:MM in 24-hour local time")
        return f"{hour:02d}:{minute:02d}"

    @field_validator("interval_minutes", mode="before")
    @classmethod
    def validate_interval(cls, value: object) -> object:
        """Blank and 0 both leave the short interval recap off."""
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return None
            value = text
        if value == 0 or value == "0":
            return None
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, value: object) -> object:
        """Accept a comma-separated string as well as a list."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value
