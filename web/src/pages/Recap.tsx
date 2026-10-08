import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Toaster } from "@/components/ui/sonner";
import { FrigateConfig } from "@/types/frigateConfig";
import { RecapManifest, RecapSummary } from "@/types/recap";
import RecapPlayer from "@/views/recap/RecapPlayer";
import axios from "axios";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import useSWR from "swr";

const HOURS = [1, 6, 12, 24] as const;

function formatRange(after?: number, before?: number) {
  if (!after || !before) {
    return "";
  }
  const start = new Date(after * 1000);
  const end = new Date(before * 1000);
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  };
  return `${start.toLocaleString(undefined, options)} - ${end.toLocaleString(undefined, options)}`;
}

export default function Recap() {
  const { t } = useTranslation(["views/recap", "common"]);
  const { data: config } = useSWR<FrigateConfig>("config");
  const cameras = useMemo(() => {
    if (!config) {
      return [];
    }
    return Object.entries(config.cameras)
      .filter(([, camera]) => camera.recap?.enabled)
      .map(([name]) => name);
  }, [config]);
  const [camera, setCamera] = useState<string>("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [selected, setSelected] = useState<string>();

  useEffect(() => {
    document.title = t("documentTitle");
  }, [t]);

  useEffect(() => {
    if (!camera && cameras.length > 0) {
      setCamera(cameras[0]);
    }
  }, [camera, cameras]);

  const { data: recaps, mutate } = useSWR<RecapSummary[]>("recap", {
    refreshInterval: (latest) =>
      latest?.some(
        (item) => item.status === "queued" || item.status === "running",
      )
        ? 2000
        : 15000,
  });

  const { data: detail } = useSWR<RecapManifest>(
    selected ? `recap/${selected}` : null,
    {
      refreshInterval: (latest) =>
        latest && (latest.status === "queued" || latest.status === "running")
          ? 1500
          : 0,
    },
  );

  const maxHours =
    (camera && config?.cameras[camera]?.recap?.max_window_hours) ||
    config?.recap?.max_window_hours ||
    48;

  const startHours = async (hours: number) => {
    if (!camera) {
      return;
    }
    try {
      const response = await axios.post(`recap/${camera}/start`, { hours });
      setSelected(response.data.id);
      toast.success(t("started"), { position: "top-center" });
      mutate();
    } catch (error) {
      const message =
        axios.isAxiosError(error) && error.response?.data?.message
          ? String(error.response.data.message)
          : t("failed");
      toast.error(message, { position: "top-center" });
    }
  };

  const startCustom = async () => {
    if (!camera || !from || !to) {
      return;
    }
    const after = new Date(from).getTime() / 1000;
    const before = new Date(to).getTime() / 1000;
    if (!(before > after)) {
      toast.error(t("rangeInvalid"), { position: "top-center" });
      return;
    }
    if ((before - after) / 3600 > maxHours) {
      toast.error(t("rangeLong"), { position: "top-center" });
      return;
    }
    try {
      const response = await axios.post(`recap/${camera}/start`, {
        after,
        before,
      });
      setSelected(response.data.id);
      toast.success(t("started"), { position: "top-center" });
      mutate();
    } catch (error) {
      const message =
        axios.isAxiosError(error) && error.response?.data?.message
          ? String(error.response.data.message)
          : t("failed");
      toast.error(message, { position: "top-center" });
    }
  };

  const remove = async (item: RecapSummary) => {
    if (item.source === "archive") {
      toast.error(t("deleteArchive"), { position: "top-center" });
      return;
    }
    if (!window.confirm(t("deleteConfirm"))) {
      return;
    }
    await axios.delete(`recap/${item.id}`);
    if (selected === item.id) {
      setSelected(undefined);
    }
    mutate();
  };

  const cancel = async (id: string) => {
    await axios.post(`recap/${id}/cancel`);
    mutate();
  };

  const hourLabel = (hours: number) => {
    if (hours === 1) {
      return t("lastHour");
    }
    if (hours === 6) {
      return t("last6");
    }
    if (hours === 12) {
      return t("last12");
    }
    return t("last24");
  };

  return (
    <div className="flex size-full flex-col gap-3 overflow-hidden p-2 md:p-4">
      <Toaster closeButton />
      <div>
        <h1 className="text-xl font-medium">{t("title")}</h1>
        <p className="text-sm text-muted-foreground">{t("description")}</p>
      </div>
      {config && cameras.length === 0 && (
        <p className="rounded-md border border-secondary p-3 text-sm">
          {t("disabled")}
        </p>
      )}
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1 text-sm">
          {t("camera")}
          <select
            className="h-10 rounded-md border border-input bg-background px-2"
            value={camera}
            onChange={(event) => setCamera(event.target.value)}
          >
            {cameras.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        {HOURS.map((hours) => (
          <Button
            key={hours}
            variant="select"
            disabled={!camera}
            onClick={() => startHours(hours)}
          >
            {hourLabel(hours)}
          </Button>
        ))}
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1 text-sm">
          {t("from")}
          <Input
            type="datetime-local"
            value={from}
            onChange={(event) => setFrom(event.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          {t("to")}
          <Input
            type="datetime-local"
            value={to}
            onChange={(event) => setTo(event.target.value)}
          />
        </label>
        <Button variant="outline" disabled={!camera} onClick={startCustom}>
          {t("generate")}
        </Button>
      </div>
      {detail && detail.status !== "complete" && (
        <div className="rounded-md border border-secondary p-3 text-sm">
          <div className="mb-1">
            {detail.message || detail.status} ({detail.progress ?? 0}%)
          </div>
          <div className="h-1.5 overflow-hidden rounded bg-secondary">
            <div
              className="h-full bg-selected"
              style={{ width: `${detail.progress ?? 0}%` }}
            />
          </div>
          {(detail.status === "queued" || detail.status === "running") && (
            <Button
              className="mt-2"
              variant="outline"
              size="sm"
              onClick={() => cancel(detail.id)}
            >
              {t("cancel")}
            </Button>
          )}
        </div>
      )}
      {detail?.status === "complete" && <RecapPlayer recap={detail} />}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <h2 className="mb-2 text-sm font-medium">{t("saved")}</h2>
        {!recaps?.length && (
          <p className="text-sm text-muted-foreground">{t("empty")}</p>
        )}
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {recaps?.map((item) => (
            <div
              key={item.id}
              className="flex flex-col gap-1 rounded-lg border border-secondary p-3 text-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{item.camera}</span>
                {item.source === "archive" && (
                  <span className="rounded bg-secondary px-1.5 py-0.5 text-xs">
                    {t("archive")}
                  </span>
                )}
              </div>
              <div className="text-muted-foreground">
                {formatRange(item.after, item.before)}
              </div>
              <div>
                {item.status === "complete"
                  ? t("events", { count: item.event_count ?? 0 })
                  : `${item.status} ${item.progress ?? 0}%`}
                {item.seconds ? ` · ${Math.round(item.seconds)}s` : ""}
              </div>
              <div className="mt-1 flex gap-2">
                <Button
                  size="sm"
                  variant="select"
                  disabled={item.status !== "complete"}
                  onClick={() => setSelected(item.id)}
                >
                  {t("play")}
                </Button>
                {item.source !== "archive" &&
                  (item.status === "queued" || item.status === "running") && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => cancel(item.id)}
                    >
                      {t("cancel")}
                    </Button>
                  )}
                {item.source !== "archive" &&
                  item.status !== "queued" &&
                  item.status !== "running" && (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => remove(item)}
                    >
                      {t("button.delete", { ns: "common" })}
                    </Button>
                  )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
