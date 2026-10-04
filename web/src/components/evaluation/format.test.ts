import { describe, expect, it } from "vitest";
import {
  fineDigitsFor,
  formatCount,
  formatGain,
  formatPercent,
  formatRate,
  formatReading,
  formatSigned,
  formatTarget,
  formatThreshold,
} from "./format";

describe("evaluation numbers", () => {
  it("shows a rate in percent, one decimal from 10% up and two below", () => {
    expect(formatPercent(0.9503)).toBe("95.0");
    expect(formatPercent(0.0091)).toBe("0.91");
    expect(formatPercent(0.99376, 2)).toBe("99.38");
  });

  it("shows a rate's interval at the value's precision", () => {
    expect(
      formatRate({
        value: 0.0091,
        ci: { low: 0.0062, high: 0.0124 },
        adjustedWilson: null,
      }),
    ).toBe("0.91 [0.62–1.24]");
  });

  it("gives a rate far under 0.1% a third decimal", () => {
    expect(fineDigitsFor(0.00054)).toBe(3);
    expect(fineDigitsFor(0.0015)).toBe(2);
    expect(fineDigitsFor(0.5)).toBe(1);
  });

  it("signs a difference with a true minus, and a rounded zero not at all", () => {
    expect(formatSigned(1.36, 1)).toBe("+1.4");
    expect(formatSigned(-3.93, 1)).toBe("−3.9");
    expect(formatSigned(-0.02, 1)).toBe("0.0");
    expect(
      formatGain(
        {
          targetFpir: 0.01,
          value: 0.0013,
          ci: { low: -0.0013, high: 0.0051 },
          improves: false,
        },
        1,
      ),
    ).toBe("+0.1 [−0.1, +0.5]");
  });

  it("shows a target as a round percentage and a reading to two figures", () => {
    expect(formatTarget(0.01)).toBe("1%");
    expect(formatTarget(0.001)).toBe("0.1%");
    expect(formatTarget(1e-4)).toBe("0.01%");
    expect(formatReading(0.00068)).toBe("0.068%");
    expect(formatReading(0.0098)).toBe("0.98%");
  });

  it("shows counts with separators and thresholds to three decimals", () => {
    expect(formatCount(5707000)).toBe("5,707,000");
    expect(formatThreshold(0.4979689)).toBe("0.498");
    expect(formatThreshold(-0.4019)).toBe("−0.402");
  });
});
