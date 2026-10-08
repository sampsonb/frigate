import { TimezoneAwareCalendar } from "@/components/overlay/ReviewActivityCalendar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import { cn } from "@/lib/utils";
import { FrigateConfig } from "@/types/frigateConfig";
import { useEffect, useMemo, useState } from "react";
import { isMobileOnly } from "react-device-detect";
import { useTranslation } from "react-i18next";
import { LuCalendarRange, LuClock, LuZap } from "react-icons/lu";
import useSWR from "swr";
import { LONGER_CHOICES, LongerChoice, useRecapTime } from "./recapUtils";

type RecapRangePickerProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  maxHours: number;
  nightsAvailable: number;
  busy: boolean;
  selected?: LongerChoice | "custom";
  onChoose: (hours: LongerChoice) => void;
  onCustom: (after: number, before: number) => void;
};

type Edge = "start" | "end";

function clockValue(epoch: number) {
  const date = new Date(epoch * 1000);
  return `${date.getHours().toString().padStart(2, "0")}:${date
    .getMinutes()
    .toString()
    .padStart(2, "0")}`;
}

function withDay(epoch: number, day: Date) {
  const date = new Date(epoch * 1000);
  date.setFullYear(day.getFullYear(), day.getMonth(), day.getDate());
  return date.getTime() / 1000;
}

function withClock(epoch: number, clock: string) {
  const [hour, minute] = clock.split(":").map((part) => parseInt(part, 10));
  if (Number.isNaN(hour) || Number.isNaN(minute)) {
    return epoch;
  }
  const date = new Date(epoch * 1000);
  date.setHours(hour, minute, 0, 0);
  return date.getTime() / 1000;
}

export default function RecapRangePicker({
  open,
  onOpenChange,
  maxHours,
  nightsAvailable,
  busy,
  selected,
  onChoose,
  onCustom,
}: RecapRangePickerProps) {
  const { t } = useTranslation(["views/recap"]);
  const { data: config } = useSWR<FrigateConfig>("config");
  const times = useRecapTime();
  const [edge, setEdge] = useState<Edge>("start");
  const [range, setRange] = useState(() => {
    const before = Math.floor(Date.now() / 60000) * 60;
    return { after: before - 3 * 3600, before };
  });

  useEffect(() => {
    if (open) {
      const before = Math.floor(Date.now() / 60000) * 60;
      setRange({ after: before - 3 * 3600, before });
      setEdge("start");
    }
  }, [open]);

  const hours = (range.before - range.after) / 3600;
  const problem =
    range.before <= range.after
      ? t("rangeInvalid")
      : hours > maxHours
        ? t("rangeLongHours", { count: maxHours })
        : range.before > Date.now() / 1000 + 60
          ? t("rangeFuture")
          : "";

  const tiles = useMemo(
    () =>
      LONGER_CHOICES.map((choice) => {
        const nights = choice === 12 ? 0 : choice / 24;
        const ready = nights > 0 && nightsAvailable >= nights;
        const canBuild = choice <= maxHours;
        let note: string;
        if (ready) {
          note =
            nights === 1
              ? t("picker.readyNightly")
              : t("picker.readyNights", { count: nights });
        } else if (canBuild) {
          note = t("picker.buildsNow");
        } else {
          note = t("picker.notYet", { count: nights, have: nightsAvailable });
        }
        return {
          choice,
          ready,
          disabled: !ready && !canBuild,
          note,
        };
      }),
    [maxHours, nightsAvailable, t],
  );

  const active = edge === "start" ? range.after : range.before;

  const body = (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3">
        {tiles.map((tile) => (
          <button
            key={tile.choice}
            type="button"
            disabled={tile.disabled}
            onClick={() => onChoose(tile.choice)}
            className={cn(
              "flex min-h-24 flex-col items-start justify-between rounded-xl border p-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
              selected === tile.choice
                ? "border-selected bg-selected/15"
                : "border-secondary bg-background_alt hover:border-selected/60 hover:bg-secondary/60",
            )}
          >
            <span className="text-2xl font-semibold tabular-nums">
              {t("picker.hoursShort", { count: tile.choice })}
            </span>
            <span
              className={cn(
                "flex items-center gap-1 text-xs",
                tile.ready ? "text-selected" : "text-muted-foreground",
              )}
            >
              {tile.ready ? (
                <LuZap className="size-3.5" />
              ) : (
                <LuClock className="size-3.5" />
              )}
              {tile.note}
            </span>
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-3 rounded-xl border border-secondary p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <LuCalendarRange className="size-4" />
          {t("picker.custom")}
        </div>
        <div className="grid grid-cols-2 gap-2">
          {(["start", "end"] as Edge[]).map((which) => {
            const value = which === "start" ? range.after : range.before;
            return (
              <button
                key={which}
                type="button"
                onClick={() => setEdge(which)}
                className={cn(
                  "flex flex-col rounded-lg border px-3 py-2 text-left",
                  edge === which
                    ? "border-selected bg-selected/10"
                    : "border-secondary hover:bg-secondary/60",
                )}
              >
                <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  {which === "start" ? t("from") : t("to")}
                </span>
                <span className="text-sm font-medium">
                  {times.dayTime(value)}
                </span>
              </button>
            );
          })}
        </div>
        <div className="flex flex-col items-center gap-2 sm:flex-row sm:items-start sm:justify-center">
          <TimezoneAwareCalendar
            timezone={config?.ui?.timezone}
            selectedDay={new Date(active * 1000)}
            onSelect={(day) => {
              if (!day) {
                return;
              }
              setRange((current) =>
                edge === "start"
                  ? { ...current, after: withDay(current.after, day) }
                  : { ...current, before: withDay(current.before, day) },
              );
            }}
          />
          <label className="flex w-full flex-col gap-1 text-xs text-muted-foreground sm:w-36 sm:pt-3">
            {edge === "start" ? t("picker.startTime") : t("picker.endTime")}
            <input
              type="time"
              step="60"
              value={clockValue(active)}
              className="h-10 rounded-md border border-input bg-background px-2 text-sm text-primary dark:[color-scheme:dark]"
              onChange={(event) => {
                const clock = event.target.value;
                setRange((current) =>
                  edge === "start"
                    ? { ...current, after: withClock(current.after, clock) }
                    : { ...current, before: withClock(current.before, clock) },
                );
              }}
            />
          </label>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span
            className={cn(
              "text-xs",
              problem ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {problem || times.duration(range.after, range.before)}
          </span>
          <Button
            variant="select"
            disabled={Boolean(problem) || busy}
            onClick={() => onCustom(range.after, range.before)}
          >
            {t("picker.buildCustom")}
          </Button>
        </div>
      </div>
    </div>
  );

  if (isMobileOnly) {
    return (
      <Drawer open={open} onOpenChange={onOpenChange}>
        <DrawerContent className="max-h-[92dvh] px-4 pb-6">
          <DrawerHeader className="px-0 text-left">
            <DrawerTitle>{t("picker.title")}</DrawerTitle>
            <DrawerDescription>{t("picker.description")}</DrawerDescription>
          </DrawerHeader>
          <div className="scrollbar-container overflow-y-auto">{body}</div>
        </DrawerContent>
      </Drawer>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92dvh] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("picker.title")}</DialogTitle>
          <DialogDescription>{t("picker.description")}</DialogDescription>
        </DialogHeader>
        {body}
      </DialogContent>
    </Dialog>
  );
}
