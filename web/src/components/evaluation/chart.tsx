import { type ReactNode, useId } from "react";
import { cn } from "@/lib/utils";
import type { SeriesStyle } from "./model-styles";
import type { Scale } from "./scale";

/*
 * The evaluation page's SVG primitives, in #13's lab-notebook style: a
 * hairline grid, ink axes with muted tick labels, a colour and dash per
 * model, labels set beside the data instead of a legend. Coordinates are SVG
 * user units in the chart's viewBox, which scales with the page.
 */

/** A plot area inside a chart's viewBox. */
export type PlotArea = {
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
};

const TICK = 4;
const TICK_LABEL = 11;
const AXIS_TITLE = 12;
/** Line width of a model's series, as the report draws it. */
export const SERIES_WIDTH = 1.6;

/**
 * A numbered figure: its chart, then a serif caption like a table's. The
 * caption's ID is handed to the chart, whose SVG it names.
 */
export function Figure({
  number,
  caption,
  children,
}: {
  readonly number: number;
  readonly caption: ReactNode;
  readonly children: (captionId: string) => ReactNode;
}) {
  const captionId = useId();
  return (
    <figure className="flex flex-col">
      {children(captionId)}
      <figcaption
        id={captionId}
        className="mt-3 max-w-[80ch] font-serif text-base text-muted-foreground"
      >
        <span className="text-ink">Figure {number}.</span> {caption}
      </figcaption>
    </figure>
  );
}

/** Something on a chart that shows its value on hover or focus, in viewBox units. */
export type ReadoutSpec = {
  readonly key: string;
  /** The value in words: the readout's accessible name and its visible text. */
  readonly label: string;
  readonly x: number;
  readonly y: number;
  /** A box's size; a point when left out. */
  readonly width?: number;
  readonly height?: number;
};

/**
 * An SVG chart named by its figure's caption, with a readout over each
 * point or bar in `readouts`. The SVG is one image to assistive technology;
 * the readouts sit over it as their own focusable images, so a keyboard or
 * screen reader reaches each value the chart marks.
 */
export function Chart({
  width,
  height,
  labelledBy,
  readouts = [],
  children,
}: {
  readonly width: number;
  readonly height: number;
  readonly labelledBy: string;
  readonly readouts?: ReadonlyArray<ReadoutSpec>;
  readonly children: ReactNode;
}) {
  return (
    <div className="relative w-full" style={{ maxWidth: width * 1.25 }}>
      {/* biome-ignore lint/a11y/noSvgWithoutTitle: named by its figure's caption through aria-labelledby, which the rule does not recognise */}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={labelledBy}
        className="block h-auto w-full overflow-visible font-sans"
      >
        {children}
      </svg>
      {readouts.map((readout) => (
        <Readout
          key={readout.key}
          readout={readout}
          width={width}
          height={height}
        />
      ))}
    </div>
  );
}

function Readout({
  readout,
  width,
  height,
}: {
  readonly readout: ReadoutSpec;
  readonly width: number;
  readonly height: number;
}) {
  const point = readout.width === undefined || readout.height === undefined;
  const across = readout.x / width;
  const percent = (value: number, of: number) => `${(value / of) * 100}%`;
  return (
    <span
      role="img"
      aria-label={readout.label}
      // biome-ignore lint/a11y/noNoninteractiveTabindex: a readout is reached by keyboard to read its value
      tabIndex={0}
      className={cn(
        "group absolute hover:z-10 focus:z-10",
        point
          ? "-translate-x-1/2 -translate-y-1/2 size-5 rounded-full"
          : "rounded-sm hover:bg-ink/5 focus:bg-ink/5",
      )}
      style={{
        left: percent(readout.x, width),
        top: percent(readout.y, height),
        ...(point
          ? {}
          : {
              width: percent(readout.width ?? 0, width),
              height: percent(readout.height ?? 0, height),
            }),
      }}
    >
      <span
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute bottom-full mb-1.5 hidden whitespace-nowrap rounded-sm border border-ink bg-raised px-2 py-1 text-ink text-xs group-hover:block group-focus:block",
          // Kept inside the chart: opened towards its middle.
          across > 0.6
            ? "right-0"
            : across < 0.4
              ? "left-0"
              : "-translate-x-1/2 left-1/2",
        )}
      >
        {readout.label}
      </span>
    </span>
  );
}

/** Hairline grid lines at each tick, behind the data. */
export function Grid({
  area,
  x,
  xTicks = [],
  y,
  yTicks = [],
}: {
  readonly area: PlotArea;
  readonly x?: Scale;
  readonly xTicks?: ReadonlyArray<number>;
  readonly y?: Scale;
  readonly yTicks?: ReadonlyArray<number>;
}) {
  return (
    <g className="stroke-rule" strokeWidth={0.6}>
      {x === undefined
        ? null
        : xTicks.map((tick) => (
            <line
              key={`x${tick}`}
              x1={x(tick)}
              x2={x(tick)}
              y1={area.top}
              y2={area.bottom}
            />
          ))}
      {y === undefined
        ? null
        : yTicks.map((tick) => (
            <line
              key={`y${tick}`}
              x1={area.left}
              x2={area.right}
              y1={y(tick)}
              y2={y(tick)}
            />
          ))}
    </g>
  );
}

/** An ink x axis along `area`'s bottom, with ticks, their labels and a title under it. */
export function AxisBottom({
  area,
  scale,
  ticks,
  format,
  title,
}: {
  readonly area: PlotArea;
  readonly scale: Scale;
  readonly ticks: ReadonlyArray<number>;
  readonly format: (tick: number) => string;
  readonly title: string;
}) {
  const y = area.bottom;
  return (
    <g>
      <line
        x1={area.left}
        x2={area.right}
        y1={y}
        y2={y}
        className="stroke-ink"
        strokeWidth={0.8}
      />
      {ticks.map((tick) => (
        <g key={tick} transform={`translate(${scale(tick)},${y})`}>
          <line y2={TICK} className="stroke-ink" strokeWidth={0.8} />
          <text
            y={TICK + TICK_LABEL + 1}
            textAnchor="middle"
            fontSize={TICK_LABEL}
            className="fill-muted-ink"
          >
            {format(tick)}
          </text>
        </g>
      ))}
      <text
        x={(area.left + area.right) / 2}
        y={y + TICK + TICK_LABEL + AXIS_TITLE + 10}
        textAnchor="middle"
        fontSize={AXIS_TITLE}
        className="fill-ink"
      >
        {title}
      </text>
    </g>
  );
}

/** An ink y axis along `area`'s left, with ticks, their labels and a turned title. */
export function AxisLeft({
  area,
  scale,
  ticks,
  format,
  title,
}: {
  readonly area: PlotArea;
  readonly scale: Scale;
  readonly ticks: ReadonlyArray<number>;
  readonly format: (tick: number) => string;
  readonly title: string;
}) {
  const x = area.left;
  return (
    <g>
      <line
        x1={x}
        x2={x}
        y1={area.top}
        y2={area.bottom}
        className="stroke-ink"
        strokeWidth={0.8}
      />
      {ticks.map((tick) => (
        <g key={tick} transform={`translate(${x},${scale(tick)})`}>
          <line x2={-TICK} className="stroke-ink" strokeWidth={0.8} />
          <text
            x={-TICK - 3}
            dominantBaseline="central"
            textAnchor="end"
            fontSize={TICK_LABEL}
            className="fill-muted-ink"
          >
            {format(tick)}
          </text>
        </g>
      ))}
      <text
        transform={`translate(${x - 46},${(area.top + area.bottom) / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={AXIS_TITLE}
        className="fill-ink"
      >
        {title}
      </text>
    </g>
  );
}

/** A model's line in its colour and dash, the dash scaled by the line's width as matplotlib does. */
export function Series({
  d,
  series,
  style,
  width = SERIES_WIDTH,
}: {
  readonly d: string;
  /** What the line is of, such as a network, for whoever inspects the SVG. */
  readonly series: string;
  readonly style: SeriesStyle;
  readonly width?: number;
}) {
  return (
    <path
      d={d}
      data-series={series}
      fill="none"
      stroke={style.colour}
      strokeWidth={width}
      strokeDasharray={
        style.dashes === null
          ? undefined
          : style.dashes.map((length) => roundUnit(length * width)).join(" ")
      }
      strokeLinejoin="miter"
    />
  );
}

/** A marker in a model's colour and shape, centred on (x, y). */
export function Marker({
  x,
  y,
  style,
  size = 7,
}: {
  readonly x: number;
  readonly y: number;
  readonly style: SeriesStyle;
  readonly size?: number;
}) {
  const half = size / 2;
  const common = {
    fill: style.colour,
    className: "stroke-paper",
    strokeWidth: 1,
  };
  switch (style.marker) {
    case "circle":
      return <circle cx={x} cy={y} r={half} {...common} />;
    case "square":
      return (
        <rect
          x={x - half}
          y={y - half}
          width={size}
          height={size}
          {...common}
        />
      );
    case "diamond":
      return (
        <path
          d={`M${x},${y - half * 1.25}L${x + half},${y}L${x},${y + half * 1.25}L${x - half},${y}Z`}
          {...common}
        />
      );
  }
}

/**
 * A label set where its series is drawn (#13), on a paper halo so a line
 * behind it does not cross the letters.
 */
export function DirectLabel({
  x,
  y,
  colour,
  anchor = "start",
  children,
}: {
  readonly x: number;
  readonly y: number;
  /** A model's colour; ink when left out. */
  readonly colour?: string;
  readonly anchor?: "start" | "middle" | "end";
  readonly children: ReactNode;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="central"
      fontSize={12}
      fill={colour}
      className={cn("stroke-paper", colour === undefined && "fill-ink")}
      strokeWidth={3}
      strokeLinejoin="round"
      paintOrder="stroke"
    >
      {children}
    </text>
  );
}

/** A horizontal bar from `x0` to `x1`, centred on `y`. */
export function Bar({
  x0,
  x1,
  y,
  thickness,
  colour,
}: {
  readonly x0: number;
  readonly x1: number;
  readonly y: number;
  readonly thickness: number;
  readonly colour: string;
}) {
  return (
    <rect
      x={Math.min(x0, x1)}
      y={y - thickness / 2}
      width={Math.abs(x1 - x0)}
      height={thickness}
      fill={colour}
      fillOpacity={0.3}
      stroke={colour}
      strokeWidth={0.8}
    />
  );
}

/** An interval from `x0` to `x1` at `y`, capped at both ends. */
export function Whisker({
  x0,
  x1,
  y,
  colour,
  cap = 6,
}: {
  readonly x0: number;
  readonly x1: number;
  readonly y: number;
  readonly colour: string;
  readonly cap?: number;
}) {
  return (
    <g stroke={colour} strokeWidth={1.2}>
      <line x1={x0} x2={x1} y1={y} y2={y} />
      <line x1={x0} x2={x0} y1={y - cap / 2} y2={y + cap / 2} />
      <line x1={x1} x2={x1} y1={y - cap / 2} y2={y + cap / 2} />
    </g>
  );
}

function roundUnit(value: number): number {
  return Math.round(value * 100) / 100;
}
