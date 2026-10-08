import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { RecapCategory, RecapSummary } from "@/types/recap";
import { useState } from "react";
import { useTranslation } from "react-i18next";

type RecapCardProps = {
  item: RecapSummary;
  active: boolean;
  onOpen: () => void;
  onCancel?: () => void;
  onDelete?: () => void;
};

function formatRange(after?: number, before?: number) {
  if (!after || !before) {
    return "";
  }
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  };
  const start = new Date(after * 1000).toLocaleString(undefined, options);
  const end = new Date(before * 1000).toLocaleString(undefined, options);
  return `${start} - ${end}`;
}

function formatRecapDay(epoch?: number) {
  if (!epoch) {
    return "";
  }
  const date = new Date(epoch * 1000);
  const weekday = date.toLocaleDateString(undefined, { weekday: "short" });
  const month = date.toLocaleDateString(undefined, { month: "short" });
  return `${weekday} ${month} ${date.getDate()}`;
}

export default function RecapCard({
  item,
  active,
  onOpen,
  onCancel,
  onDelete,
}: RecapCardProps) {
  const { t } = useTranslation(["views/recap", "common"]);
  const [thumbFailed, setThumbFailed] = useState(false);
  const busy = item.status === "queued" || item.status === "running";
  const nightly = item.reason === "schedule";
  const day = nightly ? formatRecapDay(item.before || item.created) : "";
  const counts = (item.categories ?? []).filter(
    (category) => category.count > 0,
  );
  const rounded = item.seconds ? Math.max(1, Math.round(item.seconds)) : 0;
  const lengthLabel =
    rounded === 0
      ? ""
      : rounded < 60
        ? t("secondsShort", { count: rounded })
        : t("minutesShort", { count: Math.round(rounded / 60) });
  const showThumb = item.status === "complete" && !thumbFailed;

  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-lg border bg-background_alt text-left",
        active ? "border-selected ring-1 ring-selected" : "border-secondary",
      )}
    >
      <button
        type="button"
        className="flex flex-col text-left"
        onClick={onOpen}
      >
        <span className="relative block aspect-video bg-black">
          {showThumb ? (
            <img
              src={`${baseUrl}api/recap/${item.id}/thumb.jpg`}
              alt={t("thumbnail")}
              className="size-full object-cover"
              onError={() => setThumbFailed(true)}
            />
          ) : (
            <span className="flex size-full items-center justify-center px-2 text-center text-xs text-muted-foreground">
              {busy
                ? `${item.message || item.status} (${item.progress ?? 0}%)`
                : item.camera}
            </span>
          )}
          {item.source === "archive" && (
            <span className="absolute right-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">
              {t("archive")}
            </span>
          )}
          {busy && (
            <span className="absolute inset-x-0 bottom-0 bg-black/70 p-1.5 text-[11px] text-white">
              <span className="mb-1 block truncate">
                {item.message || item.status} ({item.progress ?? 0}%)
              </span>
              <span className="block h-1 overflow-hidden rounded bg-white/20">
                <span
                  className="block h-full bg-selected"
                  style={{ width: `${item.progress ?? 0}%` }}
                />
              </span>
            </span>
          )}
        </span>
        <span className="flex flex-col gap-0.5 p-2">
          <span className="truncate text-sm font-medium">
            {day || item.camera}
          </span>
          {day && (
            <span className="truncate text-[11px] text-muted-foreground">
              {item.camera}
            </span>
          )}
          <span className="truncate text-[11px] text-muted-foreground">
            {formatRange(item.after, item.before)}
          </span>
          <span className="flex flex-wrap items-center gap-1.5 text-[11px]">
            {lengthLabel ? (
              <span className="text-muted-foreground">{lengthLabel}</span>
            ) : null}
            <CategoryDots categories={counts} />
          </span>
        </span>
      </button>
      {(onCancel || onDelete) && (
        <div className="flex gap-2 px-2 pb-2">
          {onCancel && (
            <Button size="sm" variant="outline" onClick={onCancel}>
              {t("cancel")}
            </Button>
          )}
          {onDelete && (
            <Button size="sm" variant="destructive" onClick={onDelete}>
              {t("button.delete", { ns: "common" })}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

function CategoryDots({ categories }: { categories: RecapCategory[] }) {
  if (!categories.length) {
    return null;
  }
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {categories.map((category) => (
        <span
          key={category.key}
          className="inline-flex items-center gap-1"
          title={category.name}
        >
          <span
            className="inline-block size-2 rounded-full"
            style={{ backgroundColor: category.color }}
          />
          {category.count}
        </span>
      ))}
    </span>
  );
}
