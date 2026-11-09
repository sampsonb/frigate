import { FrigateConfig } from "@/types/frigateConfig";
import { RecapSummary } from "@/types/recap";
import { formatUnixTimestampToDateTime } from "@/utils/dateUtil";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import useSWR from "swr";
import {
  compactClock,
  compactDay,
  compactDuration,
  compactRange,
  compactStamp,
  sameLocalDay,
} from "./compactTime";

export const LONGER_CHOICES = [12, 24, 48, 72] as const;
export type LongerChoice = (typeof LONGER_CHOICES)[number];

const HOUR = 3600;
// A nightly recap counts as "latest" when it ended within this long.
const NIGHTLY_FRESH_SECONDS = 26 * HOUR;
// Largest gap or overlap allowed between back-to-back nightly windows.
const NIGHTLY_JOIN_SLACK_SECONDS = 3 * HOUR;

export function compareRecapNewest(a: RecapSummary, b: RecapSummary) {
  return (
    (b.before ?? 0) - (a.before ?? 0) || (b.created ?? 0) - (a.created ?? 0)
  );
}

export function isBusy(item?: { status?: string } | null) {
  return item?.status === "queued" || item?.status === "running";
}

export function isRolling(item?: RecapSummary | null) {
  return Boolean(
    item && (item.reason === "rolling" || item.id.endsWith("_rolling")),
  );
}

const AUTOMATIC_REASONS = new Set([
  "rolling-archive",
  "schedule",
  "backfill",
  "interval",
]);

/** Made by Frigate on its own: earlier rolling recaps, nightly, interval. */
export function isAutomatic(item: RecapSummary) {
  return AUTOMATIC_REASONS.has(item.reason || "") || isNightly(item);
}

function isNightly(item: RecapSummary) {
  const kind = item.kind || "";
  return (
    item.reason === "schedule" ||
    item.reason === "backfill" ||
    kind.startsWith("day:") ||
    kind.startsWith("backfill-day:")
  );
}

/**
 * Finished nightly recaps for a camera, newest first, one per night.
 */
export function nightlyRecaps(items: RecapSummary[], camera: string) {
  const seen = new Set<string>();
  return items
    .filter(
      (item) =>
        item.camera === camera &&
        item.status === "complete" &&
        isNightly(item) &&
        item.after &&
        item.before,
    )
    .sort(
      (a, b) =>
        (b.before ?? 0) - (a.before ?? 0) ||
        (b.created ?? 0) - (a.created ?? 0),
    )
    .filter((item) => {
      const night = new Date((item.before ?? 0) * 1000).toDateString();
      if (seen.has(night)) {
        return false;
      }
      seen.add(night);
      return true;
    });
}

/**
 * The last ``nights`` nightly recaps as a playlist, oldest first, or
 * undefined when the newest is stale or the nights do not join up.
 */
export function nightlyPlaylist(
  items: RecapSummary[],
  camera: string,
  nights: number,
  now: number,
) {
  const list = nightlyRecaps(items, camera);
  if (list.length < nights) {
    return undefined;
  }
  const chosen = list.slice(0, nights);
  if (now - (chosen[0].before ?? 0) > NIGHTLY_FRESH_SECONDS) {
    return undefined;
  }
  for (let index = 1; index < chosen.length; index++) {
    const newer = chosen[index - 1];
    const older = chosen[index];
    const gap = Math.abs((newer.after ?? 0) - (older.before ?? 0));
    if (gap > NIGHTLY_JOIN_SLACK_SECONDS) {
      return undefined;
    }
  }
  return chosen.reverse();
}

export function availableNights(
  items: RecapSummary[],
  camera: string,
  now: number,
) {
  for (let nights = 3; nights >= 1; nights--) {
    if (nightlyPlaylist(items, camera, nights, now)) {
      return nights;
    }
  }
  return 0;
}

/** Re-render every ``ms`` so relative times stay current. */
export function useNow(ms = 30000) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), ms);
    return () => window.clearInterval(timer);
  }, [ms]);
  return now;
}

/**
 * Date and time strings that follow Frigate's ui.time_format and
 * ui.timezone settings.
 */
export function useRecapTime() {
  const { t } = useTranslation(["views/recap"]);
  const { data: config } = useSWR<FrigateConfig>("config");
  const timezone = config?.ui?.timezone || undefined;
  const hour24 = config?.ui?.time_format === "24hour";
  const clock = hour24 ? "HH:mm" : "h:mm aaa";

  const format = useCallback(
    (epoch: number, pattern: string) =>
      formatUnixTimestampToDateTime(epoch, {
        timezone,
        date_format: pattern,
      }),
    [timezone],
  );

  const dayTime = useCallback(
    (epoch?: number) => (epoch ? format(epoch, `EEE MMM d, ${clock}`) : ""),
    [clock, format],
  );

  const time = useCallback(
    (epoch?: number) => (epoch ? format(epoch, clock) : ""),
    [clock, format],
  );

  const day = useCallback(
    (epoch?: number) => (epoch ? format(epoch, "EEE MMM d") : ""),
    [format],
  );

  const span = useCallback(
    (after?: number, before?: number) => {
      if (!after || !before) {
        return "";
      }
      const sameDay =
        format(after, "yyyy-MM-dd") === format(before, "yyyy-MM-dd");
      return sameDay
        ? t("spanSameDay", { start: time(after), end: time(before) })
        : t("spanDays", { start: dayTime(after), end: dayTime(before) });
    },
    [dayTime, format, t, time],
  );

  const range = useCallback(
    (after?: number, before?: number) => {
      if (!after || !before) {
        return "";
      }
      const start = compactClock(after, timezone, hour24);
      const end = compactClock(before, timezone, hour24);
      const startDay = compactDay(after, timezone);
      if (sameLocalDay(after, before, timezone)) {
        return t("rangeSameDay", { day: startDay, start, end });
      }
      return t("rangeCrossDay", {
        startDay,
        start,
        endDay: compactDay(before, timezone),
        end,
      });
    },
    [hour24, t, timezone],
  );

  const lengthOf = useCallback(
    (after?: number, before?: number) => {
      if (!after || !before) {
        return "";
      }
      const parts = compactDuration(after, before);
      if (parts.kind === "minutes") {
        return t("minutesShort", { count: parts.count });
      }
      if (parts.kind === "hours") {
        return t("hoursShort", { count: parts.count });
      }
      return t("durationHoursMinutes", {
        hours: parts.hours,
        minutes: parts.minutes,
      });
    },
    [t],
  );

  const cardTitle = useCallback(
    (after?: number, before?: number) => {
      if (!after || !before) {
        return "";
      }
      const window = compactRange(after, before, timezone, hour24);
      const length = lengthOf(after, before);
      if (!length) {
        return window;
      }
      return t("rangeWithLength", { range: window, length });
    },
    [hour24, lengthOf, t, timezone],
  );

  const stamp = useCallback(
    (epoch?: number) => (epoch ? compactStamp(epoch, timezone, hour24) : ""),
    [hour24, timezone],
  );

  const relative = useCallback(
    (epoch: number | undefined, now: number) => {
      if (!epoch) {
        return "";
      }
      const seconds = Math.max(0, now - epoch);
      if (seconds < 60) {
        return t("relative.justNow");
      }
      if (seconds < HOUR) {
        return t("relative.minutes", { count: Math.floor(seconds / 60) });
      }
      if (seconds < 24 * HOUR) {
        return t("relative.hours", { count: Math.floor(seconds / HOUR) });
      }
      return t("relative.days", { count: Math.floor(seconds / (24 * HOUR)) });
    },
    [t],
  );

  return useMemo(
    () => ({
      dayTime,
      time,
      day,
      span,
      range,
      duration: lengthOf,
      cardTitle,
      stamp,
      relative,
      hour24,
      timezone,
    }),
    [
      dayTime,
      time,
      day,
      span,
      range,
      lengthOf,
      cardTitle,
      stamp,
      relative,
      hour24,
      timezone,
    ],
  );
}

/** "About 3 min left" once there is enough progress to guess. */
export function estimateRemaining(
  elapsed: number,
  progress: number,
): number | undefined {
  if (progress < 15 || progress >= 100 || elapsed <= 0) {
    return undefined;
  }
  return (elapsed * (100 - progress)) / progress;
}
