import type { LfwResult } from "@/api/evaluation";
import {
  AxisBottom,
  AxisLeft,
  Chart,
  DirectLabel,
  Grid,
  Marker,
  type PlotArea,
  type ReadoutSpec,
  Series,
} from "./chart";
import { logCurve, MIN_FAR, rateFloor } from "./curves";
import { formatPercent, formatReading, formatTarget } from "./format";
import { MODEL_STYLES } from "./model-styles";
import {
  linearScale,
  linearTicks,
  logScale,
  logTicks,
  spreadLabels,
  stepPath,
} from "./scale";

const WIDTH = 640;
const HEIGHT = 400;
const AREA: PlotArea = { left: 64, right: 616, top: 14, bottom: 344 };
const LABEL_GAP = 15;

/**
 * Every model's TAR against FAR on LFW View 2, log FAR from 0.01%, stepping
 * at each false accept (`figures.lfw_roc`), each operating point marked.
 * Each curve is labelled with its model and accuracy where it starts.
 */
export function RocChart({
  models,
  labelledBy,
}: {
  readonly models: ReadonlyArray<LfwResult>;
  readonly labelledBy: string;
}) {
  const curves = models.map((result) => ({
    result,
    style: MODEL_STYLES[result.model.network],
    points: logCurve(result.roc.far, result.roc.tar, MIN_FAR),
  }));
  const x = logScale([MIN_FAR, 1], [AREA.left, AREA.right]);
  const yDomain = [
    rateFloor(curves.flatMap((curve) => curve.points.map(([, tar]) => tar))),
    1.0005,
  ] as const;
  const y = linearScale(yDomain, [AREA.bottom, AREA.top]);
  const xTicks = logTicks(x.domain);
  const yTicks = linearTicks(yDomain, 5);
  // Under the start of each curve, clear of one another.
  const labelYs = spreadLabels(
    curves.map((curve) => y(curve.points[0]?.[1] ?? yDomain[0]) + 10),
    LABEL_GAP,
    AREA.bottom - 8,
  );
  const readouts: ReadonlyArray<ReadoutSpec> = curves.flatMap(({ result }) =>
    result.operatingPoints.map((point) => ({
      key: `${result.model.id}-${point.targetFar}`,
      label: `${result.model.name}: TAR ${formatPercent(point.tar, 2)}% at FAR ${formatReading(point.far)}, the operating point for FAR ${formatTarget(point.targetFar)}${point.indicative ? " (indicative)" : ""}`,
      x: x(Math.max(point.far, MIN_FAR)),
      y: y(point.tar),
    })),
  );

  return (
    <Chart
      width={WIDTH}
      height={HEIGHT}
      labelledBy={labelledBy}
      readouts={readouts}
    >
      <Grid area={AREA} x={x} xTicks={xTicks} y={y} yTicks={yTicks} />
      <AxisBottom
        area={AREA}
        scale={x}
        ticks={xTicks}
        format={formatTarget}
        title="False accept rate (log scale)"
      />
      <AxisLeft
        area={AREA}
        scale={y}
        ticks={yTicks}
        format={formatTarget}
        title="True accept rate"
      />
      {curves.map(({ result, style, points }) => (
        <Series
          key={result.model.id}
          d={stepPath(points, x, y)}
          series={result.model.network}
          style={style}
        />
      ))}
      {curves.map(({ result, style }) =>
        result.operatingPoints.map((point) => (
          <Marker
            key={`${result.model.id}-${point.targetFar}`}
            x={x(Math.max(point.far, MIN_FAR))}
            y={y(point.tar)}
            style={style}
          />
        )),
      )}
      {curves.map(({ result, style }, index) => (
        <DirectLabel
          key={result.model.id}
          x={AREA.left + 6}
          y={labelYs[index] ?? AREA.bottom}
          colour={style.colour}
        >
          {`${result.model.name}, ${formatPercent(result.accuracy, 2)}%`}
        </DirectLabel>
      ))}
    </Chart>
  );
}
