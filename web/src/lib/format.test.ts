import { describe, expect, it } from "vitest";
import { formatDate, formatScore } from "./format";

describe("formatDate", () => {
  it("reads a service timestamp as a date", () => {
    expect(formatDate("2026-09-27T10:00:00Z")).toBe("27 Sept 2026");
  });

  it("shows a timestamp it cannot read as sent, rather than throwing", () => {
    expect(formatDate("not a date")).toBe("not a date");
    expect(formatDate("")).toBe("");
  });
});

describe("formatScore", () => {
  it("shows three decimals", () => {
    expect(formatScore(0.5)).toBe("0.500");
  });
});
