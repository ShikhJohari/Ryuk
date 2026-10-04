import { describe, expect, it } from "vitest";
import {
  linearScale,
  linearTicks,
  logScale,
  logTicks,
  spreadLabels,
  stepPath,
} from "./scale";

describe("linearScale", () => {
  it("maps the domain's ends to the range's ends and keeps proportion", () => {
    const x = linearScale([0, 10], [100, 300]);

    expect(x(0)).toBe(100);
    expect(x(10)).toBe(300);
    expect(x(2.5)).toBe(150);
    expect(x.domain).toEqual([0, 10]);
    expect(x.range).toEqual([100, 300]);
  });

  it("maps upwards on a y axis whose range runs bottom to top", () => {
    const y = linearScale([0.8, 1], [400, 0]);

    expect(y(0.8)).toBe(400);
    expect(y(1)).toBe(0);
    expect(y(0.9)).toBeCloseTo(200);
  });

  it("extrapolates outside the domain", () => {
    const x = linearScale([0, 1], [0, 100]);

    expect(x(-0.02)).toBeCloseTo(-2);
    expect(x(1.5)).toBe(150);
  });

  it("refuses an empty domain, which has no proportion", () => {
    expect(() => linearScale([1, 1], [0, 100])).toThrow(RangeError);
  });
});

describe("logScale", () => {
  it("gives each decade the same length", () => {
    const x = logScale([1e-4, 1], [0, 400]);

    expect(x(1e-4)).toBe(0);
    expect(x(1e-3)).toBeCloseTo(100);
    expect(x(1e-2)).toBeCloseTo(200);
    expect(x(1e-1)).toBeCloseTo(300);
    expect(x(1)).toBe(400);
    expect(x(10 ** -2.5)).toBeCloseTo(150);
  });

  it("has no place for zero or a negative value", () => {
    const x = logScale([1e-4, 1], [0, 400]);

    expect(x(0)).toBeNaN();
    expect(x(-1)).toBeNaN();
  });

  it("refuses a domain that is not positive, or is empty", () => {
    expect(() => logScale([0, 1], [0, 100])).toThrow(RangeError);
    expect(() => logScale([-1, 1], [0, 100])).toThrow(RangeError);
    expect(() => logScale([1e-3, 1e-3], [0, 100])).toThrow(RangeError);
  });
});

describe("linearTicks", () => {
  it("steps by 1, 2 or 5 times a power of ten, inside the domain", () => {
    expect(linearTicks([0.8, 1.0005], 4)).toEqual([0.8, 0.85, 0.9, 0.95, 1]);
    expect(linearTicks([0, 10], 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(linearTicks([0, 1.3], 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]);
  });

  it("gives exact decimals, without floating-point dust", () => {
    expect(linearTicks([0.1, 0.4], 3)).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it("starts at the first step at or after a domain that starts below zero", () => {
    expect(linearTicks([-0.03, 1.5], 5)).toEqual([0, 0.5, 1, 1.5]);
  });

  it("gives the one value of an empty domain", () => {
    expect(linearTicks([3, 3], 5)).toEqual([3]);
  });

  it("refuses a reversed domain or fewer than one tick, as linearScale does", () => {
    expect(() => linearTicks([1, 0], 5)).toThrow(RangeError);
    expect(() => linearTicks([0, 1], 0)).toThrow(RangeError);
  });
});

describe("logTicks", () => {
  it("gives every power of ten in the domain", () => {
    expect(logTicks([1e-4, 1])).toEqual([1e-4, 1e-3, 1e-2, 1e-1, 1]);
  });

  it("leaves out a power of ten just outside the domain", () => {
    expect(logTicks([2.5e-4, 0.5])).toEqual([1e-3, 1e-2, 1e-1]);
  });

  it("gives none for a domain within one decade", () => {
    expect(logTicks([2e-3, 8e-3])).toEqual([]);
  });

  it("refuses a domain that is not positive", () => {
    expect(() => logTicks([0, 1])).toThrow(RangeError);
  });
});

describe("stepPath", () => {
  const x = linearScale([0, 10], [0, 100]);
  const y = linearScale([0, 1], [100, 0]);

  it("holds each value until the next point, then steps (step-post)", () => {
    expect(
      stepPath(
        [
          [0, 0.5],
          [5, 0.8],
          [10, 1],
        ],
        x,
        y,
      ),
    ).toBe("M0,50H50V20H100V0");
  });

  it("skips points that have no place on the axes", () => {
    const logX = logScale([1, 100], [0, 100]);

    expect(
      stepPath(
        [
          [0, 0.2],
          [1, 0.5],
          [100, 1],
        ],
        logX,
        y,
      ),
    ).toBe("M0,50H100V0");
  });

  it("is empty with no points", () => {
    expect(stepPath([], x, y)).toBe("");
  });
});

describe("spreadLabels", () => {
  it("leaves labels that are already far enough apart where they are", () => {
    expect(spreadLabels([10, 40, 90], 14)).toEqual([10, 40, 90]);
  });

  it("pushes a crowded label down until it clears the one above", () => {
    expect(spreadLabels([50, 10, 55], 14)).toEqual([50, 10, 64]);
  });

  it("pulls the stack back up when it would run past the bottom", () => {
    expect(spreadLabels([90, 95, 100], 10, 100)).toEqual([80, 90, 100]);
  });
});
