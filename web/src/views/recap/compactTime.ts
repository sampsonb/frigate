import { formatInTimeZone } from "date-fns-tz";

function zoneOf(timezone?: string) {
  if (timezone) {
    return timezone;
  }
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

/** Clock with no ":00", and a single a/p in 12-hour mode. */
export function compactClock(
  epoch: number,
  timezone: string | undefined,
  hour24: boolean,
) {
  const date = new Date(epoch * 1000);
  const zone = zoneOf(timezone);
  if (hour24) {
    const value = formatInTimeZone(date, zone, "H:mm");
    return value.endsWith(":00") ? value.slice(0, -3) : value;
  }
  const value = formatInTimeZone(date, zone, "h:mm a");
  const match = value.match(/^(\d{1,2}):(\d{2})\s*([AaPp])/);
  if (!match) {
    return value;
  }
  const suffix = match[3].toLowerCase();
  return match[2] === "00"
    ? `${match[1]}${suffix}`
    : `${match[1]}:${match[2]}${suffix}`;
}

export function compactDay(epoch: number, timezone: string | undefined) {
  return formatInTimeZone(new Date(epoch * 1000), zoneOf(timezone), "MMM d");
}

const RANGE_DASH = "\u2013";

/** "Oct 9 1:06a–7:06a", or both dates when the window crosses midnight. */
export function compactRange(
  after: number,
  before: number,
  timezone: string | undefined,
  hour24: boolean,
) {
  const start = compactClock(after, timezone, hour24);
  const end = compactClock(before, timezone, hour24);
  const startDay = compactDay(after, timezone);
  if (sameLocalDay(after, before, timezone)) {
    return `${startDay} ${start}${RANGE_DASH}${end}`;
  }
  return `${startDay} ${start}${RANGE_DASH}${compactDay(before, timezone)} ${end}`;
}

/** "6h", "30m", or "1h 30m". */
export function formatShortDuration(after: number, before: number) {
  const parts = compactDuration(after, before);
  if (parts.kind === "minutes") {
    return `${parts.count}m`;
  }
  if (parts.kind === "hours") {
    return `${parts.count}h`;
  }
  return `${parts.hours}h ${parts.minutes}m`;
}

/** Full card title: "Oct 9 1:06a–7:06a · 6h". */
export function compactCardTitle(
  after: number,
  before: number,
  timezone: string | undefined,
  hour24: boolean,
) {
  return `${compactRange(after, before, timezone, hour24)} \u00b7 ${formatShortDuration(after, before)}`;
}

/** One moment, when a recap has no window: "Oct 9 1:06a". */
export function compactStamp(
  epoch: number,
  timezone: string | undefined,
  hour24: boolean,
) {
  return `${compactDay(epoch, timezone)} ${compactClock(epoch, timezone, hour24)}`;
}

export function sameLocalDay(
  after: number,
  before: number,
  timezone: string | undefined,
) {
  const zone = zoneOf(timezone);
  const key = (epoch: number) =>
    formatInTimeZone(new Date(epoch * 1000), zone, "yyyy-MM-dd");
  return key(after) === key(before);
}

export type CompactDuration =
  | { kind: "minutes"; count: number }
  | { kind: "hours"; count: number }
  | { kind: "both"; hours: number; minutes: number };

/** Whole hours as "6h", otherwise "30m" or "1h 30m". */
export function compactDuration(
  after: number,
  before: number,
): CompactDuration {
  const minutes = Math.max(1, Math.round((before - after) / 60));
  if (minutes < 60) {
    return { kind: "minutes", count: minutes };
  }
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (rest === 0) {
    return { kind: "hours", count: hours };
  }
  return { kind: "both", hours, minutes: rest };
}
