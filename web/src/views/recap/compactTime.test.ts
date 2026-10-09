import { describe, expect, it } from "vitest";
import {
  compactClock,
  compactDuration,
  compactDay,
  sameLocalDay,
} from "./compactTime";

const TZ = "America/Chicago";

// 2026-10-08 16:05 CDT and 22:05 CDT.
const START = Date.UTC(2026, 9, 8, 21, 5) / 1000;
const END = Date.UTC(2026, 9, 9, 3, 5) / 1000;
// 2026-10-08 22:00 CDT through 2026-10-09 04:00 CDT.
const NIGHT = Date.UTC(2026, 9, 9, 3, 0) / 1000;
const MORNING = Date.UTC(2026, 9, 9, 9, 0) / 1000;

function range(after: number, before: number, hour24 = false) {
  const start = compactClock(after, TZ, hour24);
  const end = compactClock(before, TZ, hour24);
  const startDay = compactDay(after, TZ);
  if (sameLocalDay(after, before, TZ)) {
    return `${startDay} ${start} to ${end}`;
  }
  return `${startDay} ${start} to ${compactDay(before, TZ)} ${end}`;
}

describe("compact recap ranges", () => {
  it("uses a short same-day range in the ui timezone", () => {
    expect(range(START, END)).toBe("Oct 8 4:05p to 10:05p");
    expect(compactDuration(START, END)).toEqual({ kind: "hours", count: 6 });
  });

  it("drops :00 and shows both dates across midnight", () => {
    expect(range(NIGHT, MORNING)).toBe("Oct 8 10p to Oct 9 4a");
    expect(compactDuration(NIGHT, MORNING)).toEqual({
      kind: "hours",
      count: 6,
    });
  });

  it("keeps 24-hour clocks and short minute spans", () => {
    expect(range(START, END, true)).toBe("Oct 8 16:05 to 22:05");
    expect(compactClock(NIGHT, TZ, true)).toBe("22");
    expect(compactDuration(START, START + 30 * 60)).toEqual({
      kind: "minutes",
      count: 30,
    });
    expect(compactDuration(START, START + 90 * 60)).toEqual({
      kind: "both",
      hours: 1,
      minutes: 30,
    });
  });
});
