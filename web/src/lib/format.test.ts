import { describe, expect, it } from "vitest";
import { formatDate, formatDateTime, formatScore, formatTime } from "./format";

describe("formatDate", () => {
  it("reads a service timestamp as a date", () => {
    expect(formatDate("2026-09-27T10:00:00Z")).toBe("27 Sept 2026");
  });

  it("shows a timestamp it cannot read as sent, rather than throwing", () => {
    expect(formatDate("not a date")).toBe("not a date");
    expect(formatDate("")).toBe("");
  });
});

describe("formatDateTime", () => {
  it("reads a service timestamp as a date and a time to the second", () => {
    // Tests run in UTC (vite.config.ts); the browser shows local time.
    expect(formatDateTime("2026-09-27T10:00:05.250Z")).toBe(
      "27 Sept 2026, 10:00:05",
    );
  });

  it("shows a timestamp it cannot read as sent, rather than throwing", () => {
    expect(formatDateTime("not a date")).toBe("not a date");
  });
});

describe("formatTime", () => {
  it("reads a service timestamp as a time to the second", () => {
    expect(formatTime("2026-09-27T21:04:59Z")).toBe("21:04:59");
  });

  it("shows a timestamp it cannot read as sent, rather than throwing", () => {
    expect(formatTime("not a date")).toBe("not a date");
  });
});

describe("formatScore", () => {
  it("shows three decimals", () => {
    expect(formatScore(0.5)).toBe("0.500");
  });
});
