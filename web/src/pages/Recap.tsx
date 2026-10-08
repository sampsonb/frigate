import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Toaster } from "@/components/ui/sonner";
import { resolveCameraName } from "@/hooks/use-camera-friendly-name";
import { FrigateConfig } from "@/types/frigateConfig";
import { RecapManifest, RecapSummary } from "@/types/recap";
import RecapCard from "@/views/recap/RecapCard";
import RecapPlayer from "@/views/recap/RecapPlayer";
import axios from "axios";
import { useEffect, useMemo, useRef, useState } from "react";
import { LuChevronDown } from "react-icons/lu";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

const HOURS = [1, 6, 12, 24] as const;

function playable(item: RecapSummary) {
  return item.status === "complete" && (item.event_count ?? 0) > 0;
}

export default function Recap() {
  const { t } = useTranslation(["views/recap", "common"]);
  const [searchParams] = useSearchParams();
  const requested = useRef({
    id: searchParams.get("id") || "",
    camera: searchParams.get("camera") || "",
  });
  const { data: config } = useSWR<FrigateConfig>("config");
  const cameras = useMemo(() => {
    if (!config) {
      return [];
    }
    return Object.entries(config.cameras)
      .filter(([, camera]) => camera.recap?.enabled)
      .map(([name]) => name);
  }, [config]);
  const [camera, setCamera] = useState<string>(requested.current.camera);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [selected, setSelected] = useState<string>();
  const [showEmpty, setShowEmpty] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);

  useEffect(() => {
    document.title = t("documentTitle");
  }, [t]);

  useEffect(() => {
    if (cameras.length === 0 || cameras.includes(camera)) {
      return;
    }
    const preferred = requested.current.camera;
    setCamera(
      preferred && cameras.includes(preferred) ? preferred : cameras[0],
    );
  }, [camera, cameras]);

  const { data: recaps, mutate } = useSWR<RecapSummary[]>("recap", {
    refreshInterval: (latest) =>
      latest?.some(
        (item) => item.status === "queued" || item.status === "running",
      )
        ? 2000
        : 15000,
  });

  const forCamera = useMemo(
    () => (recaps ?? []).filter((item) => item.camera === camera),
    [camera, recaps],
  );
  const featured = forCamera.filter(
    (item) =>
      playable(item) ||
      item.status === "queued" ||
      item.status === "running" ||
      item.status === "failed",
  );
  const empty = forCamera.filter(
    (item) => item.status === "complete" && (item.event_count ?? 0) === 0,
  );

  useEffect(() => {
    if (!forCamera.length) {
      return;
    }
    const requestedId = requested.current.id;
    if (requestedId && forCamera.some((item) => item.id === requestedId)) {
      setSelected(requestedId);
      requested.current.id = "";
      return;
    }
    // Keep the recap the user is watching, including one that was just queued.
    if (selected) {
      return;
    }
    const next = forCamera.find(playable);
    if (next) {
      setSelected(next.id);
    }
  }, [forCamera, selected]);

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

  const choose = (id: string) => {
    setSelected(id);
  };

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

  const running =
    detail && (detail.status === "queued" || detail.status === "running");

  return (
    <div className="flex size-full flex-col gap-3 overflow-hidden p-2 md:p-3">
      <Toaster closeButton />
      <div className="flex flex-col gap-2">
        <div>
          <h1 className="text-lg font-medium md:text-xl">{t("title")}</h1>
          <p className="text-sm text-muted-foreground">{t("description")}</p>
        </div>
        {config && cameras.length === 0 && (
          <p className="rounded-md border border-secondary p-3 text-sm">
            {t("disabled")}
          </p>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex min-w-[8rem] flex-col gap-1 text-sm">
            {t("camera")}
            <select
              className="h-10 rounded-md border border-input bg-background px-2"
              value={camera}
              onChange={(event) => {
                setSelected(undefined);
                setCamera(event.target.value);
              }}
            >
              {cameras.map((name) => (
                <option key={name} value={name}>
                  {resolveCameraName(config, name)}
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
          <Button
            variant="outline"
            type="button"
            aria-expanded={customOpen}
            onClick={() => setCustomOpen((current) => !current)}
          >
            {t("custom")}
            <LuChevronDown
              className={`ml-1 size-4 transition-transform ${customOpen ? "rotate-180" : ""}`}
            />
          </Button>
        </div>
        {customOpen && (
          <div className="flex w-full flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-end">
            <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm">
              {t("from")}
              <Input
                type="datetime-local"
                value={from}
                onChange={(event) => setFrom(event.target.value)}
              />
            </label>
            <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm">
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
        )}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-hidden lg:grid-cols-[minmax(0,1.35fr)_minmax(16rem,0.9fr)] lg:grid-rows-1">
        <div className="min-h-0 min-w-0 overflow-y-auto">
          {detail?.status === "complete" && <RecapPlayer recap={detail} />}
          {running && (
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
              <Button
                className="mt-2"
                variant="outline"
                size="sm"
                onClick={() => cancel(detail.id)}
              >
                {t("cancel")}
              </Button>
            </div>
          )}
          {!detail && !running && (
            <p className="rounded-md border border-dashed border-secondary p-4 text-sm text-muted-foreground">
              {t("emptyPlayer")}
            </p>
          )}
        </div>
        <div className="min-h-0 overflow-y-auto">
          <h2 className="mb-2 text-sm font-medium">{t("saved")}</h2>
          {!forCamera.length && (
            <p className="text-sm text-muted-foreground">{t("empty")}</p>
          )}
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-1 xl:grid-cols-2">
            {featured.map((item) => (
              <RecapCard
                key={item.id}
                item={
                  detail?.id === item.id
                    ? {
                        ...item,
                        progress: detail.progress ?? item.progress,
                        message: detail.message || item.message,
                        status: detail.status || item.status,
                      }
                    : item
                }
                active={item.id === selected}
                onOpen={() => choose(item.id)}
                onCancel={
                  item.source !== "archive" &&
                  (item.status === "queued" || item.status === "running")
                    ? () => cancel(item.id)
                    : undefined
                }
                onDelete={
                  item.source !== "archive" &&
                  item.status !== "queued" &&
                  item.status !== "running"
                    ? () => remove(item)
                    : undefined
                }
              />
            ))}
          </div>
          {empty.length > 0 && (
            <div className="mt-3">
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground"
                onClick={() => setShowEmpty((current) => !current)}
              >
                {showEmpty ? t("hideEmpty") : t("showEmpty")}
                {" · "}
                {t("emptyRecapsCount", { count: empty.length })}
              </Button>
              {showEmpty && (
                <div className="mt-2 grid grid-cols-2 gap-2 opacity-70 lg:grid-cols-1 xl:grid-cols-2">
                  {empty.map((item) => (
                    <RecapCard
                      key={item.id}
                      item={item}
                      active={item.id === selected}
                      onOpen={() => choose(item.id)}
                      onDelete={
                        item.source !== "archive"
                          ? () => remove(item)
                          : undefined
                      }
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
