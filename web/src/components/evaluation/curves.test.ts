import { describe, expect, it } from "vitest";
import { falseAlarmFloor, logCurve, rateFloor, valueBefore } from "./curves";

describe("logCurve", () => {
  it("starts at the floor with the rate reached before the first false alarm, as figures.py does", () => {
    expect(
      logCurve([0, 0, 0.0003, 0.001, 1], [0, 0.97, 0.98, 0.99, 1], 1e-4),
    ).toEqual([
      [1e-4, 0.97],
      [0.0003, 0.98],
      [0.001, 0.99],
      [1, 1],
    ]);
  });

  it("keeps a curve with no point at 0 as it is, from the floor", () => {
    expect(logCurve([0.01, 1], [0.5, 1], 1e-3)).toEqual([
      [1e-3, 0.5],
      [0.01, 0.5],
      [1, 1],
    ]);
  });

  it("is empty with no points", () => {
    expect(logCurve([], [], 1e-4)).toEqual([]);
  });
});

describe("valueBefore", () => {
  it("is the highest rate at a false-alarm rate of 0", () => {
    expect(valueBefore([0, 0, 0.1], [0, 0.6, 0.9])).toBe(0.6);
  });

  it("is null when the curve has no point at 0", () => {
    expect(valueBefore([0.1, 1], [0.2, 1])).toBeNull();
  });
});

describe("falseAlarmFloor", () => {
  it("is the power of ten at or under one false alarm", () => {
    expect(falseAlarmFloor(4072)).toBe(1e-4);
    expect(falseAlarmFloor(3929)).toBe(1e-4);
    expect(falseAlarmFloor(1000)).toBe(1e-3);
    expect(falseAlarmFloor(999)).toBe(1e-3);
  });

  it("falls back to the LFW axis's floor with no probes", () => {
    expect(falseAlarmFloor(0)).toBe(1e-4);
  });
});

describe("rateFloor", () => {
  it("rounds the lowest rate down to a twentieth, never under 0", () => {
    expect(rateFloor([0.973, 0.86, 0.99])).toBe(0.85);
    expect(rateFloor([0.04])).toBe(0);
    expect(rateFloor([])).toBe(0);
  });
});
