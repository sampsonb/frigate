import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button, buttonVariants } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { resolveCameraName } from "@/hooks/use-camera-friendly-name";
import { cn } from "@/lib/utils";
import { FrigateConfig } from "@/types/frigateConfig";
import { RecapManifest, RecapRollingStatus, RecapSummary } from "@/types/recap";
import RecapBuildProgress from "@/views/recap/RecapBuildProgress";
import RecapListItem from "@/views/recap/RecapListItem";
import RecapPlayer from "@/views/recap/RecapPlayer";
import {
  recapDownloadUrl,
  recapFileName,
  recapSpanLabel,
  recapVideoUrl,
  useRecapDownload,
} from "@/views/recap/recapDownload";
import RecapRangePicker from "@/views/recap/RecapRangePicker";
import {
  LongerChoice,
  availableNights,
  compareRecapNewest,
  isBusy,
  isAutomatic,
  isRolling,
  nightlyPlaylist,
  useNow,
  useRecapTime,
} from "@/views/recap/recapUtils";
import axios from "axios";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { LuChevronDown, LuRefreshCw, LuSparkles } from "react-icons/lu";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

type Choice = LongerChoice | "custom";

type View =
  | { kind: "rolling" }
  | { kind: "recap"; id: string; choice?: Choice }
  | { kind: "playlist"; ids: string[]; choice: LongerChoice };

function playable(item: RecapSummary) {
  return item.status === "complete" && (item.event_count ?? 0) > 0;
}

function errorMessage(error: unknown, fallback: string) {
  return axios.isAxiosError(error) && error.response?.data?.message
    ? String(error.response.data.message)
    : fallback;
}

export default function Recap() {
  const { t } = useTranslation(["views/recap", "common"]);
  const times = useRecapTime();
  const now = useNow(30000);
  const downloads = useRecapDownload();
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
  const [view, setView] = useState<View>({ kind: "rolling" });
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<RecapSummary>();
  const [showEmpty, setShowEmpty] = useState(false);
  const [pinned, setPinned] = useState<number>();

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
      latest?.some((item) => isBusy(item) && !isRolling(item)) ? 2000 : 15000,
  });

  const { data: rolling, mutate: mutateRolling } = useSWR<RecapRollingStatus>(
    camera ? `recap/rolling/${camera}` : null,
    {
      refreshInterval: (latest) => (latest?.building ? 2000 : 30000),
      shouldRetryOnError: false,
    },
  );
  const rollingCurrent =
    rolling?.current?.status === "complete" ? rolling.current : undefined;
  const latestVersion = rollingCurrent?.finished || rollingCurrent?.created;

  // Keep the version being watched. A newer build is offered, not forced.
  useEffect(() => {
    if (latestVersion && pinned === undefined) {
      setPinned(latestVersion);
    }
  }, [latestVersion, pinned]);

  const { data: rollingDetail } = useSWR<RecapManifest>(
    view.kind === "rolling" && rollingCurrent && pinned
      ? `recap/${rollingCurrent.id}?v=${pinned}`
      : null,
  );

  const forCamera = useMemo(
    () =>
      (recaps ?? [])
        .filter((item) => item.camera === camera && !isRolling(item))
        .slice()
        .sort(compareRecapNewest),
    [camera, recaps],
  );
  const featured = forCamera.filter(
    (item) => playable(item) || isBusy(item) || item.status === "failed",
  );
  const empty = forCamera.filter(
    (item) => item.status === "complete" && (item.event_count ?? 0) === 0,
  );
  const autoItems = featured.filter(isAutomatic);
  const mineItems = featured.filter((item) => !isAutomatic(item));

  // A link from the Live page opens that recap.
  useEffect(() => {
    const requestedId = requested.current.id;
    if (!requestedId || !recaps) {
      return;
    }
    const match = recaps.find((item) => item.id === requestedId);
    if (match) {
      requested.current.id = "";
      if (!isRolling(match)) {
        setView({ kind: "recap", id: requestedId });
      }
    }
  }, [recaps]);

  const viewedId = view.kind === "recap" ? view.id : undefined;
  const { data: detail } = useSWR<RecapManifest>(
    viewedId ? `recap/${viewedId}` : null,
    {
      refreshInterval: (latest) => (isBusy(latest) ? 1500 : 0),
    },
  );

  const maxHours =
    (camera && config?.cameras[camera]?.recap?.max_window_hours) ||
    config?.recap?.max_window_hours ||
    48;
  const nights = useMemo(
    () => availableNights(recaps ?? [], camera, now),
    [camera, now, recaps],
  );

  const startBuild = useCallback(
    async (
      body: { hours: number } | { after: number; before: number },
      choice: Choice,
    ) => {
      if (!camera) {
        return;
      }
      try {
        const response = await axios.post(`recap/${camera}/start`, body);
        const id = String(response.data.id);
        setView({ kind: "recap", id, choice });
        setPickerOpen(false);
        toast.success(
          response.data?.recap?.deduped ? t("alreadyBuilding") : t("started"),
          { position: "top-center" },
        );
        mutate();
      } catch (error) {
        toast.error(errorMessage(error, t("failed")), {
          position: "top-center",
        });
      }
    },
    [camera, mutate, t],
  );

  const chooseLonger = (hours: LongerChoice) => {
    if (hours >= 24) {
      const list = nightlyPlaylist(recaps ?? [], camera, hours / 24, now);
      if (list) {
        setPickerOpen(false);
        if (list.length === 1) {
          setView({ kind: "recap", id: list[0].id, choice: hours });
        } else {
          setView({
            kind: "playlist",
            ids: list.map((item) => item.id),
            choice: hours,
          });
        }
        return;
      }
    }
    if (hours <= maxHours) {
      startBuild({ hours }, hours);
      return;
    }
    toast.error(t("picker.notYet", { count: hours / 24, have: nights }), {
      position: "top-center",
    });
  };

  const refreshNow = async () => {
    if (!camera) {
      return;
    }
    try {
      const response = await axios.post(`recap/rolling/${camera}/refresh`);
      const status = response.data?.status;
      if (status === "skipped") {
        toast.success(t("rolling.upToDate"), { position: "top-center" });
      } else {
        toast.success(t("rolling.refreshing"), { position: "top-center" });
      }
      mutateRolling();
    } catch (error) {
      toast.error(errorMessage(error, t("failed")), { position: "top-center" });
    }
  };

  const confirmDelete = async () => {
    const item = pendingDelete;
    setPendingDelete(undefined);
    if (!item) {
      return;
    }
    try {
      await axios.delete(`recap/${item.id}`);
      if (
        (view.kind === "recap" && view.id === item.id) ||
        (view.kind === "playlist" && view.ids.includes(item.id))
      ) {
        setView({ kind: "rolling" });
      }
      mutate();
    } catch (error) {
      toast.error(errorMessage(error, t("failed")), { position: "top-center" });
    }
  };

  const cancel = async (id: string) => {
    try {
      await axios.post(`recap/${id}/cancel`);
    } finally {
      if (view.kind === "recap" && view.id === id) {
        setView({ kind: "rolling" });
      }
      mutate();
    }
  };

  const choice =
    view.kind === "rolling"
      ? undefined
      : view.kind === "recap"
        ? view.choice
        : view.choice;
  const moreLabel =
    choice === undefined
      ? t("more")
      : choice === "custom"
        ? t("customShort")
        : t("lastHours", { count: choice });
  const rollingHours = rolling?.hours ?? 6;
  const building = rolling?.building;
  const updatedAt = rollingCurrent?.checked || latestVersion;
  const newerReady =
    view.kind === "rolling" &&
    latestVersion !== undefined &&
    pinned !== undefined &&
    latestVersion > pinned;

  const downloadProps = (item: RecapSummary) => {
    if (item.status !== "complete") {
      return {};
    }
    const version = item.finished || item.created || 0;
    const key = `${item.id}:${version}`;
    const name = recapFileName(
      resolveCameraName(config, item.camera),
      recapSpanLabel(times, item.after, item.before, item.created),
    );
    return {
      downloadState: downloads.stateOf(key),
      onDownload: () =>
        downloads.download({
          key,
          url: recapDownloadUrl(item.id, name, version),
          name,
          openUrl: recapVideoUrl(item.id, version),
        }),
    };
  };

  const renderItem = (item: RecapSummary) => (
    <RecapListItem
      key={item.id}
      item={item}
      now={now}
      active={
        (view.kind === "recap" && view.id === item.id) ||
        (view.kind === "playlist" && view.ids.includes(item.id))
      }
      onOpen={() => setView({ kind: "recap", id: item.id })}
      onCancel={
        item.source !== "archive" && isBusy(item)
          ? () => cancel(item.id)
          : undefined
      }
      onDelete={
        item.source !== "archive" && !isBusy(item)
          ? () => setPendingDelete(item)
          : undefined
      }
      {...downloadProps(item)}
    />
  );

  return (
    <div className="flex size-full flex-col gap-3 overflow-hidden p-2 md:p-3">
      <Toaster closeButton />
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-medium md:text-xl">{t("title")}</h1>
          {cameras.length > 1 && (
            <select
              aria-label={t("camera")}
              className="h-9 rounded-md border border-input bg-background px-2 text-sm"
              value={camera}
              onChange={(event) => {
                setView({ kind: "rolling" });
                setPinned(undefined);
                setCamera(event.target.value);
              }}
            >
              {cameras.map((name) => (
                <option key={name} value={name}>
                  {resolveCameraName(config, name)}
                </option>
              ))}
            </select>
          )}
        </div>
        {config && cameras.length === 0 && (
          <p className="rounded-md border border-secondary p-3 text-sm">
            {t("disabled")}
          </p>
        )}
        {cameras.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant={view.kind === "rolling" ? "select" : "default"}
              aria-pressed={view.kind === "rolling"}
              onClick={() => {
                setView({ kind: "rolling" });
                if (latestVersion) {
                  setPinned(latestVersion);
                }
              }}
            >
              {t("lastHours", { count: rollingHours })}
            </Button>
            <Button
              variant={choice !== undefined ? "select" : "default"}
              aria-pressed={choice !== undefined}
              aria-haspopup="dialog"
              onClick={() => setPickerOpen(true)}
            >
              {moreLabel}
              <LuChevronDown className="ml-1 size-4" />
            </Button>
            <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-x-3 gap-y-1 text-xs text-muted-foreground">
              {building ? (
                <span className="tabular-nums">
                  {building.status === "queued"
                    ? t("rolling.queued")
                    : t("rolling.refreshingProgress", {
                        progress: building.progress ?? 0,
                      })}
                </span>
              ) : updatedAt ? (
                <span title={times.dayTime(latestVersion)}>
                  {t("rolling.updated", {
                    when: times.relative(updatedAt, now),
                  })}
                </span>
              ) : null}
              {newerReady && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7"
                  onClick={() => setPinned(latestVersion)}
                >
                  <LuSparkles className="mr-1 size-3.5" />
                  {t("rolling.newer")}
                </Button>
              )}
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2"
                disabled={Boolean(building) || !camera}
                onClick={refreshNow}
              >
                <LuRefreshCw
                  className={cn("mr-1 size-3.5", building && "animate-spin")}
                />
                {t("rolling.refreshNow")}
              </Button>
            </div>
          </div>
        )}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto lg:grid-cols-[minmax(0,1fr)_20rem] lg:overflow-hidden">
        <div className="flex min-h-0 min-w-0 flex-col lg:overflow-y-auto">
          {view.kind === "rolling" && (
            <RollingPane
              status={rolling}
              detail={rollingDetail}
              version={pinned}
              hours={rollingHours}
              onRefresh={refreshNow}
            />
          )}
          {view.kind === "recap" && (
            <RecapPane
              detail={detail}
              title={
                view.choice === "custom"
                  ? t("customShort")
                  : view.choice
                    ? t("lastHours", { count: view.choice })
                    : times.span(detail?.after, detail?.before)
              }
              onCancel={() => cancel(view.id)}
            />
          )}
          {view.kind === "playlist" && (
            <PlaylistPane key={view.ids.join(",")} ids={view.ids} />
          )}
        </div>
        <div className="flex min-h-0 flex-col lg:overflow-y-auto">
          <section aria-labelledby="recap-auto" className="flex flex-col">
            <div className="mb-1 flex items-baseline justify-between gap-2 px-1">
              <h2 id="recap-auto" className="text-sm font-medium">
                {t("autoSection")}
              </h2>
              <span className="truncate text-xs text-muted-foreground">
                {t("autoSectionHint", { count: rollingHours })}
              </span>
            </div>
            <div className="flex flex-col gap-1">
              {rollingCurrent && (
                <RecapListItem
                  key={rollingCurrent.id}
                  item={rollingCurrent as RecapSummary}
                  now={now}
                  badge={t("readyToWatch")}
                  active={view.kind === "rolling"}
                  {...downloadProps(rollingCurrent as RecapSummary)}
                  onOpen={() => {
                    if (latestVersion) {
                      setPinned(latestVersion);
                    }
                    setView({ kind: "rolling" });
                  }}
                />
              )}
              {autoItems.map((item) => renderItem(item))}
            </div>
          </section>
          <section aria-labelledby="recap-mine" className="mt-4 flex flex-col">
            <h2 id="recap-mine" className="mb-1 px-1 text-sm font-medium">
              {t("mineSection")}
            </h2>
            {mineItems.length ? (
              <div className="flex flex-col gap-1">
                {mineItems.map((item) => renderItem(item))}
              </div>
            ) : (
              <p className="px-1 text-sm text-muted-foreground">
                {t("mineEmpty")}
              </p>
            )}
          </section>
          {empty.length > 0 && (
            <div className="mt-2">
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
                <div className="mt-1 flex flex-col gap-1 opacity-70">
                  {empty.map((item) => (
                    <RecapListItem
                      key={item.id}
                      item={item}
                      now={now}
                      active={view.kind === "recap" && view.id === item.id}
                      onOpen={() => setView({ kind: "recap", id: item.id })}
                      onDelete={
                        item.source !== "archive"
                          ? () => setPendingDelete(item)
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
      <RecapRangePicker
        open={pickerOpen}
        onOpenChange={setPickerOpen}
        maxHours={maxHours}
        nightsAvailable={nights}
        busy={false}
        selected={choice}
        onChoose={chooseLonger}
        onCustom={(after, before) => startBuild({ after, before }, "custom")}
      />
      <AlertDialog
        open={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(undefined);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingDelete
                ? t("deleteDescription", {
                    when: times.dayTime(
                      pendingDelete.before || pendingDelete.created,
                    ),
                  })
                : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t("button.cancel", { ns: "common" })}
            </AlertDialogCancel>
            <AlertDialogAction
              className={buttonVariants({ variant: "destructive" })}
              onClick={confirmDelete}
            >
              {t("button.delete", { ns: "common" })}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function RollingPane({
  status,
  detail,
  version,
  hours,
  onRefresh,
}: {
  status?: RecapRollingStatus;
  detail?: RecapManifest;
  version?: number;
  hours: number;
  onRefresh: () => void;
}) {
  const { t } = useTranslation(["views/recap"]);
  if (detail?.status === "complete") {
    return <RecapPlayer recap={detail} version={version} />;
  }
  if (status?.building && !status.current) {
    return (
      <RecapBuildProgress
        recap={status.building}
        title={t("rolling.firstBuild", { count: hours })}
      />
    );
  }
  if (status && !status.current) {
    return (
      <div className="flex flex-col items-start gap-2 rounded-lg border border-dashed border-secondary p-4 text-sm text-muted-foreground">
        {status.state?.last === "failed"
          ? t("rolling.lastFailed")
          : t("rolling.none", { count: hours })}
        <Button size="sm" variant="outline" onClick={onRefresh}>
          {t("rolling.buildNow")}
        </Button>
      </div>
    );
  }
  return (
    <div className="aspect-video w-full animate-pulse rounded-lg bg-secondary/60" />
  );
}

function RecapPane({
  detail,
  title,
  onCancel,
}: {
  detail?: RecapManifest;
  title: string;
  onCancel: () => void;
}) {
  const { t } = useTranslation(["views/recap"]);
  if (!detail) {
    return (
      <div className="aspect-video w-full animate-pulse rounded-lg bg-secondary/60" />
    );
  }
  if (isBusy(detail)) {
    return (
      <RecapBuildProgress
        recap={detail}
        title={t("build.title", { what: title })}
        onCancel={detail.source !== "archive" ? onCancel : undefined}
      />
    );
  }
  if (detail.status === "complete") {
    return <RecapPlayer recap={detail} version={detail.finished} />;
  }
  return (
    <p className="rounded-lg border border-secondary p-4 text-sm text-muted-foreground">
      {detail.status === "cancelled" ? t("cancelled") : t("failed")}
    </p>
  );
}

function PlaylistPane({ ids }: { ids: string[] }) {
  const { t } = useTranslation(["views/recap"]);
  const times = useRecapTime();
  const [part, setPart] = useState(0);
  const [advanced, setAdvanced] = useState(false);
  const { data: detail } = useSWR<RecapManifest>(`recap/${ids[part]}`);
  const header = (
    <div className="flex flex-wrap items-center gap-2">
      {ids.map((id, index) => (
        <PartChip
          key={id}
          id={id}
          index={index}
          active={index === part}
          onClick={() => {
            setAdvanced(false);
            setPart(index);
          }}
        />
      ))}
      <span className="text-xs text-muted-foreground">
        {t("playlist.hint", { count: ids.length })}
      </span>
    </div>
  );
  if (!detail || detail.status !== "complete") {
    return (
      <div className="flex flex-col gap-2">
        {header}
        <div className="aspect-video w-full animate-pulse rounded-lg bg-secondary/60" />
      </div>
    );
  }
  return (
    <RecapPlayer
      recap={detail}
      version={detail.finished || detail.created}
      autoPlay={advanced}
      header={
        <div className="flex flex-col gap-1">
          {header}
          <span className="text-xs text-muted-foreground">
            {t("playlist.part", {
              index: part + 1,
              count: ids.length,
              span: times.span(detail.after, detail.before),
            })}
          </span>
        </div>
      }
      onEnded={() => {
        if (part < ids.length - 1) {
          setAdvanced(true);
          setPart(part + 1);
        }
      }}
    />
  );
}

function PartChip({
  id,
  index,
  active,
  onClick,
}: {
  id: string;
  index: number;
  active: boolean;
  onClick: () => void;
}) {
  const { t } = useTranslation(["views/recap"]);
  const times = useRecapTime();
  const { data: recaps } = useSWR<RecapSummary[]>("recap");
  const item = recaps?.find((entry) => entry.id === id);
  return (
    <Button
      size="sm"
      variant={active ? "select" : "outline"}
      className="h-8"
      onClick={onClick}
    >
      {item?.before
        ? t("playlist.night", { day: times.day(item.after) })
        : t("playlist.partShort", { index: index + 1 })}
    </Button>
  );
}
