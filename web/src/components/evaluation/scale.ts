/**
 * The arithmetic under the evaluation page's charts: scales from data to SVG
 * user units, their ticks, a step-post path and label spreading. Pure, so
 * each is tested on its own; the SVG primitives in ./chart draw with them.
 */

/** The data values an axis spans, low end first. */
export type Domain = readonly [number, number];

/** The SVG user units a domain maps to; a y range runs bottom to top. */
export type Range = readonly [number, number];

/** Data to SVG user units along one axis. */
export interface Scale {
  (value: number): number;
  readonly domain: Domain;
  readonly range: Range;
}

/** A point in data coordinates, x then y. */
export type DataPoint = readonly [number, number];

function scale(
  domain: Domain,
  range: Range,
  transform: (value: number) => number,
): Scale {
  const [d0, d1] = [transform(domain[0]), transform(domain[1])];
  const [r0, r1] = range;
  const map = (value: number) =>
    r0 + ((transform(value) - d0) / (d1 - d0)) * (r1 - r0);
  return Object.assign(map, { domain, range });
}

/** A linear scale. Values outside the domain extrapolate. */
export function linearScale(domain: Domain, range: Range): Scale {
  if (!(domain[0] < domain[1])) {
    throw new RangeError(`a linear domain must increase: ${domain}`);
  }
  return scale(domain, range, (value) => value);
}

/**
 * A base-10 log scale, each decade the same length. A value of 0 or less has
 * no place on it and maps to NaN, which `stepPath` leaves out.
 */
export function logScale(domain: Domain, range: Range): Scale {
  assertLogDomain(domain);
  return scale(domain, range, (value) => (value > 0 ? Math.log10(value) : NaN));
}

function assertLogDomain(domain: Domain): void {
  if (!(domain[0] > 0 && domain[0] < domain[1])) {
    throw new RangeError(
      `a log domain must be positive and increase: ${domain}`,
    );
  }
}

/** Decimal places that show a multiple of `step` exactly. */
function decimalsOf(step: number): number {
  return Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
}

/**
 * Round numbers across `domain`, about `count` of them: steps of 1, 2 or 5
 * times a power of ten, from the first multiple of the step in the domain.
 */
export function linearTicks(domain: Domain, count: number): number[] {
  const [low, high] = domain;
  if (!(low <= high) || !(count >= 1)) {
    throw new RangeError(
      `linear ticks need an increasing domain and a count of at least 1: ${domain}, ${count}`,
    );
  }
  if (low === high) {
    return [low];
  }
  const raw = (high - low) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalised = raw / magnitude;
  const step =
    (normalised < 1.5 ? 1 : normalised < 3 ? 2 : normalised < 7 ? 5 : 10) *
    magnitude;
  const decimals = decimalsOf(step);
  const ticks: number[] = [];
  // A little tolerance, so a domain's own round ends are ticks.
  const first = Math.ceil(low / step - 1e-9);
  const last = Math.floor(high / step + 1e-9);
  for (let index = first; index <= last; index += 1) {
    // Fixed decimals, so 3 × 0.1 is 0.3 and not 0.30000000000000004.
    ticks.push(Number((index * step).toFixed(decimals)) + 0);
  }
  return ticks;
}

/** Every power of ten in `domain`, ends included. */
export function logTicks(domain: Domain): number[] {
  assertLogDomain(domain);
  const first = Math.ceil(Math.log10(domain[0]) - 1e-9);
  const last = Math.floor(Math.log10(domain[1]) + 1e-9);
  const ticks: number[] = [];
  for (let exponent = first; exponent <= last; exponent += 1) {
    // Parsed rather than computed, so 1e-3 is exactly 0.001.
    ticks.push(Number(`1e${exponent}`));
  }
  return ticks;
}

/**
 * A step-post line through `points`: each value holds until the next
 * point's x, then steps to its y, as matplotlib's `step(where="post")`.
 * Points with no place on the axes, such as 0 on a log axis, are left out.
 */
export function stepPath(
  points: ReadonlyArray<DataPoint>,
  x: Scale,
  y: Scale,
): string {
  const placed = points
    .map(([px, py]) => [x(px), y(py)] as const)
    .filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py));
  return placed
    .map(([px, py], index) =>
      index === 0 ? `M${round(px)},${round(py)}` : `H${round(px)}V${round(py)}`,
    )
    .join("");
}

/** SVG coordinates to a hundredth of a unit: finer is invisible. */
function round(value: number): number {
  return Math.round(value * 100) / 100 + 0;
}

/**
 * Label positions along one axis, each at least `gap` from the next, in the
 * order given: crowded labels are pushed down (to larger values), then the
 * stack is pulled back up if it would run past `bottom`.
 */
export function spreadLabels(
  positions: ReadonlyArray<number>,
  gap: number,
  bottom: number = Number.POSITIVE_INFINITY,
): number[] {
  const order = positions
    .map((position, index) => ({ position, index }))
    .sort((a, b) => a.position - b.position);
  const placed = order.map(({ position }) => position);
  for (let i = 1; i < placed.length; i += 1) {
    placed[i] = Math.max(placed[i] as number, (placed[i - 1] as number) + gap);
  }
  for (let i = placed.length - 1; i >= 0; i -= 1) {
    const limit =
      i === placed.length - 1 ? bottom : (placed[i + 1] as number) - gap;
    placed[i] = Math.min(placed[i] as number, limit);
  }
  const spread = new Array<number>(positions.length);
  order.forEach(({ index }, rank) => {
    spread[index] = placed[rank] as number;
  });
  return spread;
}
