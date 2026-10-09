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
