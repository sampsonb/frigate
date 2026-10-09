import { describe, expect, it } from "vitest";
import {
  compactCardTitle,
  compactClock,
  compactDuration,
  compactStamp,
} from "./compactTime";

const TZ = "America/Chicago";

// 2026-10-08 16:05 CDT and 22:05 CDT.
const START = Date.UTC(2026, 9, 8, 21, 5) / 1000;
const END = Date.UTC(2026, 9, 9, 3, 5) / 1000;
// 2026-10-09 01:06 CDT through 07:06 CDT.
const EARLY = Date.UTC(2026, 9, 9, 6, 6) / 1000;
const LATER = Date.UTC(2026, 9, 9, 12, 6) / 1000;
// 2026-10-08 22:00 CDT through 2026-10-09 04:00 CDT.
const NIGHT = Date.UTC(2026, 9, 9, 3, 0) / 1000;
const MORNING = Date.UTC(2026, 9, 9, 9, 0) / 1000;

describe("compact recap ranges", () => {
  it("joins the window and a short duration on one title", () => {
    expect(compactCardTitle(EARLY, LATER, TZ, false)).toBe(
      "Oct 9 1:06a–7:06a · 6h",
    );
    expect(compactCardTitle(START, END, TZ, false)).toBe(
      "Oct 8 4:05p–10:05p · 6h",
    );
    expect(compactDuration(START, END)).toEqual({ kind: "hours", count: 6 });
  });

  it("uses 30m and drops :00 across midnight", () => {
    expect(compactCardTitle(EARLY, EARLY + 30 * 60, TZ, false)).toBe(
      "Oct 9 1:06a–1:36a · 30m",
    );
    expect(compactCardTitle(NIGHT, MORNING, TZ, false)).toBe(
      "Oct 8 10p–Oct 9 4a · 6h",
    );
  });

  it("keeps 24-hour clocks and a single-moment stamp", () => {
    expect(compactCardTitle(START, END, TZ, true)).toBe(
      "Oct 8 16:05–22:05 · 6h",
    );
    expect(compactClock(NIGHT, TZ, true)).toBe("22");
    expect(compactStamp(EARLY, TZ, false)).toBe("Oct 9 1:06a");
    expect(compactDuration(START, START + 90 * 60)).toEqual({
      kind: "both",
      hours: 1,
      minutes: 30,
    });
  });
});
