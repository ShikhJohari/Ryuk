import type { DataPoint } from "./scale";

/*
 * Curves made ready for a log false-alarm axis, as `ryuk.evaluation.figures`
 * makes them for the report: `lfw_roc` and `openset_curves`.
 */

/** The left edge of LFW's FAR axis: View 2's 3,000 mismatched pairs cannot resolve below 1/3,000. */
export const MIN_FAR = 1e-4;

/** Where the open-set figure labels each curve: at FPIR 0.1% the curves stand furthest apart. */
export const LABEL_FPIR = 1e-3;

/**
 * The rate reached before the first false alarm: the highest at a
 * false-alarm rate of 0, or null when the curve has no such point.
 */
export function valueBefore(
  falseAlarms: ReadonlyArray<number>,
  rates: ReadonlyArray<number>,
): number | null {
  const atZero = rates.filter((_, index) => falseAlarms[index] === 0);
  return atZero.length === 0 ? null : Math.max(...atZero);
}

/**
 * A curve paired by index, from `floor`: a log axis cannot show a
 * false-alarm rate of 0, so the curve starts at the floor with the rate it
 * reached before its first false alarm, then follows every point above 0.
 */
export function logCurve(
  falseAlarms: ReadonlyArray<number>,
  rates: ReadonlyArray<number>,
  floor: number,
): DataPoint[] {
  const points = falseAlarms.flatMap((falseAlarm, index) => {
    const rate = rates[index];
    return falseAlarm > 0 && rate !== undefined
      ? [[falseAlarm, rate] as const]
      : [];
  });
  const first = valueBefore(falseAlarms, rates) ?? points[0]?.[1];
  return first === undefined ? [] : [[floor, first], ...points];
}

/** The power of ten at or under one false alarm in `nonMatedProbes`: the least the draw can show. */
export function falseAlarmFloor(nonMatedProbes: number): number {
  if (!(nonMatedProbes > 0)) {
    throw new RangeError(
      `a draw needs non-mated probes for a false-alarm floor: ${nonMatedProbes}`,
    );
  }
  return Number(`1e${Math.floor(Math.log10(1 / nonMatedProbes) + 1e-9)}`);
}

/** The bottom of a rate axis: the lowest rate drawn, down to a twentieth, never under 0. */
export function rateFloor(rates: ReadonlyArray<number>): number {
  if (rates.length === 0) {
    return 0;
  }
  return Math.max(0, Math.floor(Math.min(...rates) * 20) / 20);
}
