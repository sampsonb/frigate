import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { RecapClipSource, RecapManifest, RecapTrack } from "@/types/recap";
import axios from "axios";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

type RecapPlayerProps = {
  recap: RecapManifest;
};

type ContentRect = { x: number; y: number; w: number; h: number };

type OpenClip = {
  track: RecapTrack;
  clip?: string;
  snapshot?: string;
  source?: string;
  message?: string;
};

const FALLBACK_COLORS: Record<string, string> = {
  person: "#3cb9ff",
  vehicle: "#ffa528",
  delivery: "#eb50d7",
  animal: "#5ad750",
  parked: "#00afb9",
};

function mediaUrl(value?: string | null) {
  if (!value) {
    return undefined;
  }
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  return `${baseUrl}api/${value.replace(/^\//, "")}`;
}

export default function RecapPlayer({ recap }: RecapPlayerProps) {
  const { t } = useTranslation(["views/recap"]);
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [current, setCurrent] = useState(0);
  const [paused, setPaused] = useState(true);
  const [content, setContent] = useState<ContentRect>({
    x: 0,
    y: 0,
    w: 0,
    h: 0,
  });
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [open, setOpen] = useState<OpenClip>();

  const categories = useMemo(() => recap.categories ?? [], [recap.categories]);
  const tracks = useMemo(() => recap.tracks ?? [], [recap.tracks]);
  const fps = recap.fps || 12;

  useEffect(() => {
    const next: Record<string, boolean> = {};
    (recap.categories ?? []).forEach((category) => {
      next[category.key] = true;
    });
    (recap.tracks ?? []).forEach((track) => {
      if (next[track.cat] === undefined) {
        next[track.cat] = true;
      }
    });
    setEnabled(next);
  }, [recap]);

  const measure = useCallback(() => {
    const video = videoRef.current;
    const frame = frameRef.current;
    if (!video || !frame) {
      return;
    }
    const bounds = frame.getBoundingClientRect();
    const videoWidth = video.videoWidth || recap.width || 16;
    const videoHeight = video.videoHeight || recap.height || 9;
    const scale = Math.min(
      bounds.width / videoWidth,
      bounds.height / videoHeight,
    );
    const width = videoWidth * scale;
    const height = videoHeight * scale;
    setContent({
      x: (bounds.width - width) / 2,
      y: (bounds.height - height) / 2,
      w: width,
      h: height,
    });
  }, [recap.height, recap.width]);

  useEffect(() => {
    measure();
    const frame = frameRef.current;
    if (!frame) {
      return;
    }
    const observer = new ResizeObserver(() => measure());
    observer.observe(frame);
    return () => observer.disconnect();
  }, [measure, recap.id]);

  const colorFor = useCallback(
    (key: string) =>
      categories.find((category) => category.key === key)?.color ||
      FALLBACK_COLORS[key] ||
      "#ffffff",
    [categories],
  );

  const visibleTracks = useMemo(
    () => tracks.filter((track) => enabled[track.cat] !== false),
    [enabled, tracks],
  );

  const activeTracks = useMemo(() => {
    return visibleTracks.filter((track) => {
      const start = track.out_start / fps;
      const end = (track.out_start + Math.max(track.length, 1)) / fps;
      return current >= start && current < end;
    });
  }, [current, fps, visibleTracks]);

  const pauseAt = (track: RecapTrack) => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    video.pause();
    video.currentTime = track.out_start / fps;
    setCurrent(track.out_start / fps);
    setPaused(true);
  };

  const openClip = async (track: RecapTrack) => {
    const video = videoRef.current;
    if (video) {
      video.pause();
    }
    setPaused(true);
    const eventId = track.clip_event_id || track.event_id;
    try {
      const response = await axios.get<RecapClipSource>(
        `recap/event/${encodeURIComponent(eventId)}`,
      );
      const data = response.data;
      if (!data.clip && !data.snapshot) {
        toast.error(data.message || t("clipMissing"), {
          position: "top-center",
        });
        return;
      }
      setOpen({
        track,
        clip: mediaUrl(data.clip),
        snapshot: mediaUrl(data.snapshot),
        source: data.source,
        message: data.message,
      });
    } catch {
      toast.error(t("clipMissing"), { position: "top-center" });
    }
  };

  const openHistory = () => {
    if (!open) {
      return;
    }
    navigate("/review", {
      state: {
        severity: "detection",
        recording: {
          camera: recap.camera,
          startTime: open.track.start_time,
          severity: "detection",
        },
      },
    });
  };

  const clipIndex = open
    ? visibleTracks.findIndex((track) => track.event_id === open.track.event_id)
    : -1;

  const stepClip = (delta: number) => {
    if (clipIndex < 0) {
      return;
    }
    const next = visibleTracks[clipIndex + delta];
    if (next) {
      openClip(next);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex flex-wrap gap-3">
          {categories.map((category) => (
            <label
              key={category.key}
              className="flex items-center gap-2 text-sm"
            >
              <Checkbox
                checked={enabled[category.key] !== false}
                onCheckedChange={(checked) =>
                  setEnabled((currentEnabled) => ({
                    ...currentEnabled,
                    [category.key]: Boolean(checked),
                  }))
                }
              />
              <span
                className="inline-block size-2.5 rounded-full"
                style={{ backgroundColor: category.color }}
              />
              {category.name}
              <span className="text-muted-foreground">{category.count}</span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{t("clickHint")}</p>
        <div
          ref={frameRef}
          className="relative aspect-video max-h-[42vh] w-full overflow-hidden rounded-lg bg-black lg:max-h-[68vh]"
        >
          <video
            ref={videoRef}
            key={recap.id}
            className="size-full object-contain"
            src={`${baseUrl}api/recap/${recap.id}/video.mp4`}
            controls
            playsInline
            onLoadedMetadata={(event) => {
              measure();
              setCurrent(event.currentTarget.currentTime);
              setPaused(event.currentTarget.paused);
            }}
            onTimeUpdate={(event) =>
              setCurrent(event.currentTarget.currentTime)
            }
            onPlay={() => setPaused(false)}
            onPause={(event) => {
              setPaused(true);
              setCurrent(event.currentTarget.currentTime);
            }}
            onSeeked={(event) => setCurrent(event.currentTarget.currentTime)}
          />
          <div className="pointer-events-none absolute inset-0">
            {content.w > 0 &&
              activeTracks.map((track) => {
                const [x0, y0, x1, y1] = track.label_box;
                const boxWidth = Math.max(
                  paused ? 44 : 32,
                  (x1 - x0) * content.w,
                );
                const boxHeight = Math.max(
                  paused ? 44 : 32,
                  (y1 - y0) * content.h,
                );
                return (
                  <button
                    key={track.event_id}
                    type="button"
                    aria-label={track.text}
                    className={`pointer-events-auto absolute touch-manipulation whitespace-nowrap rounded-md border-2 text-left text-xs font-medium text-white ${paused ? "bg-black/55 px-2 py-1" : "bg-black/20"}`}
                    style={{
                      left: content.x + ((x0 + x1) / 2) * content.w,
                      top: content.y + ((y0 + y1) / 2) * content.h,
                      minWidth: boxWidth,
                      minHeight: boxHeight,
                      width: paused ? "max-content" : boxWidth,
                      height: paused ? "max-content" : boxHeight,
                      transform: "translate(-50%, -50%)",
                      borderColor: colorFor(track.cat),
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      openClip(track);
                    }}
                  >
                    {paused ? (
                      track.text
                    ) : (
                      <span className="sr-only">{track.text}</span>
                    )}
                  </button>
                );
              })}
          </div>
        </div>
      </div>
      <div className="flex max-h-[50vh] w-full flex-col gap-1 overflow-y-auto lg:max-h-none lg:w-72">
        <div className="text-sm font-medium">{t("eventList")}</div>
        {visibleTracks.length === 0 && (
          <p className="text-sm text-muted-foreground">{t("noEvents")}</p>
        )}
        {visibleTracks.map((track) => {
          const start = track.out_start / fps;
          const end = (track.out_start + Math.max(track.length, 1)) / fps;
          const active = current >= start && current < end;
          return (
            <button
              key={track.event_id}
              type="button"
              className={`rounded-md px-2 py-1 text-left text-sm ${active ? "bg-secondary" : "hover:bg-secondary/60"}`}
              onClick={() => {
                pauseAt(track);
                openClip(track);
              }}
            >
              <span style={{ color: colorFor(track.cat) }}>{track.text}</span>
            </button>
          );
        })}
      </div>
      <Dialog
        open={open !== undefined}
        onOpenChange={(isOpen) => {
          if (!isOpen) {
            setOpen(undefined);
          }
        }}
      >
        <DialogContent className="max-h-[95dvh] sm:max-w-xl md:max-w-4xl">
          <DialogTitle>{open?.track.text}</DialogTitle>
          {open?.source === "archive" && (
            <p className="text-sm text-muted-foreground">
              {open.message || t("fromArchive")}
            </p>
          )}
          {open?.clip ? (
            <video
              key={open.clip}
              className="max-h-[70dvh] w-full rounded-lg bg-black"
              src={open.clip}
              controls
              autoPlay
              playsInline
            />
          ) : (
            open?.snapshot && (
              <img
                src={open.snapshot}
                alt={open.track.text}
                className="max-h-[70dvh] w-full rounded-lg object-contain"
              />
            )
          )}
          {!open?.clip && open?.snapshot && (
            <p className="text-sm text-muted-foreground">{t("snapshotOnly")}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {open?.clip && (
              <Button variant="select" asChild>
                <a href={open.clip} download>
                  {t("download")}
                </a>
              </Button>
            )}
            <Button
              variant="outline"
              disabled={clipIndex <= 0}
              onClick={() => stepClip(-1)}
            >
              {t("previous")}
            </Button>
            <Button
              variant="outline"
              disabled={clipIndex < 0 || clipIndex >= visibleTracks.length - 1}
              onClick={() => stepClip(1)}
            >
              {t("next")}
            </Button>
            <Button variant="outline" onClick={openHistory}>
              {t("openHistory")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
