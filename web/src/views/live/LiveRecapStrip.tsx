import { baseUrl } from "@/api/baseUrl";
import { Toaster } from "@/components/ui/sonner";
import { useCameraFriendlyName } from "@/hooks/use-camera-friendly-name";
import { CameraConfig } from "@/types/frigateConfig";
import { RecapSummary } from "@/types/recap";
import { compareRecapNewest, useRecapTime } from "@/views/recap/recapUtils";
import axios from "axios";
import { useMemo } from "react";
import { LuClapperboard } from "react-icons/lu";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import useSWR from "swr";

const HOURS = [1, 6, 12, 24] as const;

type LiveRecapStripProps = {
  cameras: CameraConfig[];
};

function newestPlayable(items: RecapSummary[], camera: string) {
  return items
    .filter(
      (item) =>
        item.camera === camera &&
        item.status === "complete" &&
        (item.event_count ?? 0) > 0,
    )
    .sort(compareRecapNewest)[0];
}

export default function LiveRecapStrip({ cameras }: LiveRecapStripProps) {
  const { t } = useTranslation(["views/live"]);
  const navigate = useNavigate();
  const enabled = useMemo(
    () => cameras.filter((camera) => camera.recap?.enabled),
    [cameras],
  );
  const { data: recaps } = useSWR<RecapSummary[]>(
    enabled.length ? "recap" : null,
    { refreshInterval: 15000 },
  );

  if (!enabled.length) {
    return null;
  }

  const start = async (camera: string, hours: number) => {
    try {
      const response = await axios.post(`recap/${camera}/start`, { hours });
      toast.success(t("recapStrip.started"), { position: "top-center" });
      const id = response.data?.id ? String(response.data.id) : "";
      navigate(
        id
          ? `/recap?camera=${encodeURIComponent(camera)}&id=${encodeURIComponent(id)}`
          : `/recap?camera=${encodeURIComponent(camera)}`,
      );
    } catch (error) {
      const message =
        axios.isAxiosError(error) && error.response?.data?.message
          ? String(error.response.data.message)
          : t("recapStrip.failed");
      toast.error(message, { position: "top-center" });
    }
  };

  const hourLabel = (hours: number, short: boolean) => {
    if (hours === 1) {
      return short ? t("recapStrip.short1") : t("recapStrip.last1");
    }
    if (hours === 6) {
      return short ? t("recapStrip.short6") : t("recapStrip.last6");
    }
    if (hours === 12) {
      return short ? t("recapStrip.short12") : t("recapStrip.last12");
    }
    return short ? t("recapStrip.short24") : t("recapStrip.last24");
  };

  return (
    <div className="mb-2 flex gap-2 overflow-x-auto px-1 pb-1">
      <Toaster position="top-center" closeButton />
      {enabled.map((camera) => {
        const latest = newestPlayable(recaps ?? [], camera.name);
        const href = latest
          ? `/recap?camera=${encodeURIComponent(camera.name)}&id=${encodeURIComponent(latest.id)}`
          : `/recap?camera=${encodeURIComponent(camera.name)}`;
        return (
          <RecapStripCard
            key={camera.name}
            camera={camera}
            latest={latest}
            onOpen={() => navigate(href)}
            onStart={(hours) => start(camera.name, hours)}
            hourLabel={hourLabel}
          />
        );
      })}
    </div>
  );
}

function RecapStripCard({
  camera,
  latest,
  onOpen,
  onStart,
  hourLabel,
}: {
  camera: CameraConfig;
  latest?: RecapSummary;
  onOpen: () => void;
  onStart: (hours: number) => void;
  hourLabel: (hours: number, short: boolean) => string;
}) {
  const { t } = useTranslation(["views/live"]);
  const times = useRecapTime();
  const cameraName = useCameraFriendlyName(camera);
  const title =
    latest?.after && latest?.before
      ? times.cardTitle(latest.after, latest.before)
      : "";
  return (
    <div className="flex w-[15.5rem] shrink-0 items-center gap-2 rounded-lg border border-secondary bg-background_alt p-1.5">
      <button
        type="button"
        className="relative h-12 w-[4.5rem] shrink-0 overflow-hidden rounded-md bg-black"
        aria-label={t("recapStrip.open", { camera: cameraName })}
        onClick={onOpen}
      >
        {latest ? (
          <img
            src={`${baseUrl}api/recap/${latest.id}/thumb.jpg`}
            alt=""
            className="size-full object-cover"
          />
        ) : (
          <span className="flex size-full items-center justify-center text-muted-foreground">
            <LuClapperboard className="size-5" />
          </span>
        )}
      </button>
      <div className="min-w-0 flex-1">
        <button
          type="button"
          className="block w-full text-left text-xs font-medium"
          onClick={onOpen}
        >
          <span className="block whitespace-normal break-words [overflow-wrap:anywhere]">
            {t("recapStrip.title")}
            <span className="font-normal text-muted-foreground">
              {" "}
              · {cameraName}
            </span>
          </span>
          {title ? (
            <span className="mt-0.5 block whitespace-normal break-words text-[12px] font-medium leading-snug [overflow-wrap:anywhere]">
              {title}
            </span>
          ) : (
            !latest && (
              <span className="mt-0.5 block text-[10px] text-muted-foreground">
                {t("recapStrip.none")}
              </span>
            )
          )}
        </button>
        <div className="mt-1 flex gap-1">
          {HOURS.map((hours) => (
            <button
              key={hours}
              type="button"
              className="rounded bg-secondary px-1.5 py-0.5 text-[10px] leading-tight text-secondary-foreground hover:bg-muted"
              aria-label={hourLabel(hours, false)}
              onClick={() => onStart(hours)}
            >
              {hourLabel(hours, true)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
