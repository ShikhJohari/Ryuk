import type { OpenSetResult } from "@/api/evaluation";
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
import { LABEL_FPIR, logCurve, rateFloor } from "./curves";
import { formatPercent, formatTarget, formatThreshold } from "./format";
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
 * Every model's TPIR against FPIR on the test draw, log FPIR from `floor`,
 * each frozen threshold marked (`figures.openset_curves`). Each curve is
 * labelled just under itself at FPIR 0.1%, where the curves stand furthest
 * apart.
 */
export function OpenSetChart({
  models,
  floor,
  labelledBy,
}: {
  readonly models: ReadonlyArray<OpenSetResult>;
  /** The power of ten at or under one false alarm on the test draw. */
  readonly floor: number;
  readonly labelledBy: string;
}) {
  const curves = models.map((result) => {
    const points = logCurve(result.curve.fpir, result.curve.tpir, floor);
    const frozen = {
      x: Math.max(result.atThreshold.fpir.value, floor),
      y: result.atThreshold.tpir.value,
    };
    return {
      result,
      style: MODEL_STYLES[result.model.network],
      points,
      frozen,
    };
  });
  const x = logScale([floor, 1], [AREA.left, AREA.right]);
  const yDomain = [
    rateFloor(curves.flatMap((curve) => curve.points.map(([, tpir]) => tpir))),
    1.0005,
  ] as const;
  const y = linearScale(yDomain, [AREA.bottom, AREA.top]);
  const xTicks = logTicks(x.domain);
  const yTicks = linearTicks(yDomain, 5);
  const labelAt = Math.max(LABEL_FPIR, floor);
  const labelYs = spreadLabels(
    curves.map(({ points }) => {
      const reached = points
        .filter(([fpir]) => fpir <= labelAt)
        .map(([, tpir]) => tpir);
      return y(reached.length === 0 ? yDomain[0] : Math.max(...reached)) + 10;
    }),
    LABEL_GAP,
    AREA.bottom - 8,
  );
  const readouts: ReadonlyArray<ReadoutSpec> = curves.map(
    ({ result, frozen }) => ({
      key: result.model.id,
      label: `${result.model.name} at its frozen threshold, ${formatThreshold(result.threshold)}: TPIR ${formatPercent(result.atThreshold.tpir.value)}%, FPIR ${formatPercent(result.atThreshold.fpir.value)}%`,
      x: x(frozen.x),
      y: y(frozen.y),
    }),
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
        title="False positive identification rate (log scale)"
      />
      <AxisLeft
        area={AREA}
        scale={y}
        ticks={yTicks}
        format={formatTarget}
        title="True positive identification rate"
      />
      {curves.map(({ result, style, points }) => (
        <Series
          key={result.model.id}
          d={stepPath(points, x, y)}
          series={result.model.network}
          style={style}
        />
      ))}
      {curves.map(({ result, style, frozen }) => (
        <Marker
          key={result.model.id}
          x={x(frozen.x)}
          y={y(frozen.y)}
          style={style}
          size={9}
        />
      ))}
      {curves.map(({ result, style }, index) => (
        <DirectLabel
          key={result.model.id}
          x={x(labelAt) + 4}
          y={labelYs[index] ?? AREA.bottom}
          colour={style.colour}
        >
          {result.model.name}
        </DirectLabel>
      ))}
    </Chart>
  );
}
