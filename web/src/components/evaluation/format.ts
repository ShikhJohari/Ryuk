import type { Gain, Interval, Rate } from "@/api/evaluation";

/*
 * Numbers as the evaluation page shows them, at the report's precision
 * (`ryuk.evaluation.tables`): rates in percent, one decimal from 10% up and
 * two below, where a tenth is a big step; intervals in brackets with an en
 * dash; differences signed with a true minus.
 */

const counts = new Intl.NumberFormat("en-GB");

/** A count with thousands separators: 5,917. */
export function formatCount(count: number): string {
  return counts.format(count);
}

/** Decimals for a rate in percent: one from 10% up, two below. */
export function digitsFor(rate: number): number {
  return rate >= 0.1 ? 1 : 2;
}

/** Decimals for a rate in percent that may be far under 0.1%, such as a small watchlist's FPIR. */
export function fineDigitsFor(rate: number): number {
  return rate < 0.001 ? 3 : digitsFor(rate);
}

/** A fraction in percent, without the sign: 0.9503 as "95.0". */
export function formatPercent(
  fraction: number,
  digits: number = digitsFor(fraction),
): string {
  return withMinus((fraction * 100).toFixed(digits));
}

/** An interval in percent: "[94.2–95.8]". */
export function formatInterval(interval: Interval, digits: number): string {
  return `[${formatPercent(interval.low, digits)}–${formatPercent(interval.high, digits)}]`;
}

/** A rate and its interval, both at the value's precision: "95.0 [94.2–95.8]". */
export function formatRate(
  rate: Rate,
  digits: number = digitsFor(rate.value),
): string {
  return `${formatPercent(rate.value, digits)} ${formatInterval(rate.ci, digits)}`;
}

/** A difference in percentage points with its sign; a rounded zero has none. */
export function formatSigned(points: number, digits: number): string {
  const text = points.toFixed(digits);
  if (Number(text) === 0) {
    return (0).toFixed(digits);
  }
  return withMinus(points > 0 ? `+${text}` : text);
}

/** A gain over the baseline in points with its interval, each signed: "+1.4 [+0.6, +2.0]". */
export function formatGain(gain: Gain, digits: number): string {
  return `${formatSigned(gain.value * 100, digits)} [${formatSigned(gain.ci.low * 100, digits)}, ${formatSigned(gain.ci.high * 100, digits)}]`;
}

/** A target rate as a round percentage: 0.001 as "0.1%". */
export function formatTarget(fraction: number): string {
  return `${Number((fraction * 100).toPrecision(12))}%`;
}

/** A measured rate in a readout, to two significant figures: 0.00068 as "0.068%". */
export function formatReading(fraction: number): string {
  return `${Number((fraction * 100).toPrecision(2))}%`;
}

/** Milliseconds per face: "5.4". */
export function formatMs(ms: number): string {
  return ms.toFixed(1);
}

/** A cosine or other match score, as the rest of the client shows one: "0.498". */
export function formatThreshold(score: number): string {
  return withMinus(score.toFixed(3));
}

/** A hyphen-minus as the true minus sign. */
function withMinus(text: string): string {
  return text.replace("-", "−");
}
