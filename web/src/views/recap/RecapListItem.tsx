import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { RecapSummary } from "@/types/recap";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { LuHistory, LuMoon, LuTrash2, LuX } from "react-icons/lu";
import { isBusy, useRecapTime } from "./recapUtils";

type RecapListItemProps = {
  item: RecapSummary;
  active: boolean;
  now: number;
  onOpen: () => void;
  onCancel?: () => void;
  onDelete?: () => void;
  badge?: string;
};

export default function RecapListItem({
  item,
  active,
  now,
  onOpen,
  onCancel,
  onDelete,
  badge,
}: RecapListItemProps) {
  const { t } = useTranslation(["views/recap"]);
  const times = useRecapTime();
  const [thumbFailed, setThumbFailed] = useState(false);
  const busy = isBusy(item);
  const nightly = item.reason === "schedule" || item.reason === "backfill";
  const savedRolling = item.reason === "rolling-archive";
  const counts = (item.categories ?? []).filter(
    (category) => category.count > 0,
  );
  const showThumb = item.status === "complete" && !thumbFailed;
  const version = item.finished || item.created || 0;
  const title =
    item.after && item.before
      ? times.range(item.after, item.before)
      : times.dayTime(item.before || item.created);
  const length = times.duration(item.after, item.before);

  return (
    <div
      className={cn(
        "group relative flex gap-3 rounded-lg border p-2 transition-colors",
        active
          ? "border-selected bg-selected/10"
          : "border-transparent hover:bg-secondary/60",
      )}
    >
      <button
        type="button"
        className="flex min-w-0 flex-1 gap-3 text-left"
        onClick={onOpen}
        aria-current={active ? "true" : undefined}
      >
        <span className="relative block aspect-video w-28 shrink-0 overflow-hidden rounded-md bg-black md:w-32">
          {showThumb ? (
            <img
              src={`${baseUrl}api/recap/${item.id}/thumb.jpg?v=${version}`}
              alt={t("thumbnail")}
              className="size-full object-cover"
              loading="lazy"
              onError={() => setThumbFailed(true)}
            />
          ) : (
            <span className="flex size-full items-center justify-center p-1 text-center text-[10px] text-muted-foreground">
              {busy
                ? `${item.progress ?? 0}%`
                : item.status === "failed"
                  ? t("failedShort")
                  : ""}
            </span>
          )}
          {busy && (
            <span className="absolute inset-x-0 bottom-0 h-1 bg-white/20">
              <span
                className="block h-full bg-selected"
                style={{ width: `${item.progress ?? 0}%` }}
              />
            </span>
          )}
          {item.source === "archive" && (
            <span className="absolute right-1 top-1 rounded bg-black/70 px-1 py-0.5 text-[9px] text-white">
              {t("archive")}
            </span>
          )}
        </span>
        <span className="flex min-w-0 flex-1 flex-col justify-center gap-0.5">
          <span className="line-clamp-2 text-[13px] font-medium leading-snug">
            {title}
          </span>
          {(length || nightly || savedRolling || badge) && (
            <span className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-muted-foreground">
              {nightly && (
                <LuMoon
                  className="size-3.5 shrink-0"
                  aria-label={t("nightly")}
                />
              )}
              {savedRolling && (
                <LuHistory
                  className="size-3.5 shrink-0"
                  aria-label={t("savedRolling")}
                />
              )}
              {length && <span>{length}</span>}
              {(badge || nightly) && (
                <span className="shrink-0 rounded bg-secondary px-1.5 py-0.5 text-[10px] font-normal text-secondary-foreground">
                  {badge || t("nightlyBadge")}
                </span>
              )}
            </span>
          )}
          <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
            {busy ? (
              <span className="truncate text-muted-foreground">
                {item.status === "queued"
                  ? t("build.waiting")
                  : item.message || t("build.starting")}
              </span>
            ) : item.status === "failed" ? (
              <span className="text-destructive">{t("failed")}</span>
            ) : counts.length ? (
              counts.map((category) => (
                <span
                  key={category.key}
                  className="inline-flex items-center gap-1"
                  title={category.name}
                >
                  <span
                    className="inline-block size-2 rounded-full"
                    style={{ backgroundColor: category.color }}
                  />
                  <span className="tabular-nums">{category.count}</span>
                </span>
              ))
            ) : (
              <span className="text-muted-foreground">
                {t("nothingTracked")}
              </span>
            )}
            <span className="ml-auto shrink-0 text-muted-foreground">
              {times.relative(item.finished || item.created, now)}
            </span>
          </span>
        </span>
      </button>
      {(onCancel || onDelete) && (
        <div className="flex shrink-0 items-start">
          {onCancel && (
            <Button
              size="xs"
              variant="ghost"
              aria-label={t("cancel")}
              title={t("cancel")}
              onClick={onCancel}
            >
              <LuX className="size-4" />
            </Button>
          )}
          {onDelete && (
            <Button
              size="xs"
              variant="ghost"
              aria-label={t("delete")}
              title={t("delete")}
              className="text-muted-foreground hover:text-destructive"
              onClick={onDelete}
            >
              <LuTrash2 className="size-4" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
