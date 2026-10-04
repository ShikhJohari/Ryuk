import type {
  AttributeBreakdown,
  BiasAttribute,
  BiasBreakdown,
} from "@/api/evaluation";
import {
  AxisBottom,
  Bar,
  Chart,
  DirectLabel,
  type PlotArea,
  type ReadoutSpec,
  Whisker,
} from "./chart";
import {
  digitsFor,
  formatCount,
  formatInterval,
  formatPercent,
} from "./format";
import { MODEL_STYLES } from "./model-styles";
import { linearScale, linearTicks } from "./scale";

/** Height of one group's row. */
const ROW = 22;
/** Space between one attribute's groups and the next's, in rows. */
const ATTRIBUTE_GAP = 0.7;
/** Width of the column of group labels, left of the first panel. */
const LABELS = 150;
const PANEL = 200;
const PANEL_GAP = 20;
/** Room right of the last panel for the "indicative" label. */
const RIGHT = 26;
/** The panels' top: under their titles and the overall line's label. */
const TOP = 46;
const AXIS = 52;

/** Where one group sits, top to bottom, in rows from the first group's centre. */
type GroupRow = {
  readonly at: number;
  readonly attribute: BiasAttribute;
  readonly label: string;
  readonly indicative: boolean;
};

/** Every group of the attributes, top to bottom, a gap between one attribute and the next. */
function groupRows(
  attributes: ReadonlyArray<AttributeBreakdown>,
): ReadonlyArray<GroupRow> {
  let at = 0;
  return attributes.flatMap((attribute, position) => {
    if (position > 0) {
      at += ATTRIBUTE_GAP;
    }
    return attribute.groups.map((group) => {
      const row = {
        at,
        attribute: attribute.attribute,
        label: group.label,
        indicative: attribute.indicative,
      };
      at += 1;
      return row;
    });
  });
}

/**
 * Each group's test FPIR at the model's single frozen threshold, a panel per
 * model under best photo, with a bar to the rate and its interval; the dashed
 * line is the model's overall test FPIR (`figures.group_fpir`). A group with
 * too few held-out identities keeps its row, labelled with its count, and
 * indicative rows are shaded.
 */
export function GroupFpirChart({
  breakdowns,
  overall,
  labelledBy,
}: {
  /** The breakdowns under best photo, one panel each. */
  readonly breakdowns: ReadonlyArray<BiasBreakdown>;
  /** Each model's overall test FPIR at its frozen threshold, by model ID. */
  readonly overall: ReadonlyMap<string, number>;
  readonly labelledBy: string;
}) {
  const rows = groupRows(breakdowns[0]?.attributes ?? []);
  const lastRow = rows.at(-1)?.at ?? 0;
  const rowY = (at: number) => TOP + (at + 0.5) * ROW;
  const plotTop = rowY(-0.5);
  const plotBottom = rowY(lastRow + 0.5);
  const width =
    LABELS +
    breakdowns.length * PANEL +
    (breakdowns.length - 1) * PANEL_GAP +
    RIGHT;
  const height = plotBottom + AXIS;
  const measured = breakdowns.flatMap((breakdown) =>
    breakdown.attributes.flatMap((attribute) =>
      attribute.groups.flatMap((group) =>
        group.fpir === null ? [] : [group.fpir.ci.high, group.fpir.value],
      ),
    ),
  );
  // In percent; a little room left of 0, so a group with no false alarm shows.
  const reach = Math.max(...measured, ...overall.values(), 0) * 100 * 1.08 || 1;
  const domain = [-0.02 * reach, reach] as const;
  const indicative = rows.filter((row) => row.indicative).map((row) => row.at);
  const shading =
    indicative.length === 0
      ? null
      : {
          top: rowY(Math.min(...indicative) - 0.5),
          bottom: rowY(Math.max(...indicative) + 0.5),
        };

  const panels = breakdowns.map((breakdown, index) => {
    const left = LABELS + index * (PANEL + PANEL_GAP);
    const area: PlotArea = {
      left,
      right: left + PANEL,
      top: plotTop,
      bottom: plotBottom,
    };
    return {
      breakdown,
      area,
      x: linearScale(domain, [area.left, area.right]),
      style: MODEL_STYLES[breakdown.model.network],
      groups: new Map(
        breakdown.attributes.flatMap((attribute) =>
          attribute.groups.map(
            (group) =>
              [`${attribute.attribute}/${group.label}`, group] as const,
          ),
        ),
      ),
    };
  });
  const ticks = linearTicks(domain, 4);

  const readouts: ReadonlyArray<ReadoutSpec> = panels.flatMap(
    ({ breakdown, area, groups }) =>
      rows.flatMap((row) => {
        const fpir = groups.get(`${row.attribute}/${row.label}`)?.fpir ?? null;
        if (fpir === null) {
          return [];
        }
        const digits = digitsFor(fpir.value);
        return [
          {
            key: `${breakdown.model.id}/${row.attribute}/${row.label}`,
            label: `${breakdown.model.name}, ${row.label}: FPIR ${formatPercent(fpir.value, digits)}% ${formatInterval(fpir.ci, digits)}`,
            x: area.left,
            y: rowY(row.at) - ROW / 2,
            width: PANEL,
            height: ROW,
          },
        ];
      }),
  );

  return (
    <Chart
      width={width}
      height={height}
      labelledBy={labelledBy}
      readouts={readouts}
    >
      {rows.map((row) => (
        <text
          key={`${row.attribute}/${row.label}`}
          x={LABELS - 10}
          y={rowY(row.at)}
          textAnchor="end"
          dominantBaseline="central"
          fontSize={12}
          className="fill-ink"
        >
          {row.label}
        </text>
      ))}
      {panels.map(({ breakdown, area, x, style, groups }, index) => {
        const total = overall.get(breakdown.model.id);
        return (
          <g key={breakdown.model.id} data-panel={breakdown.model.name}>
            {shading === null ? null : (
              <rect
                data-indicative=""
                x={area.left}
                y={shading.top}
                width={PANEL}
                height={shading.bottom - shading.top}
                className="fill-rule"
                fillOpacity={0.5}
              />
            )}
            <g className="stroke-rule" strokeWidth={0.6}>
              {ticks.map((tick) => (
                <line
                  key={tick}
                  x1={x(tick)}
                  x2={x(tick)}
                  y1={area.top}
                  y2={area.bottom}
                />
              ))}
            </g>
            <text
              x={area.left}
              y={16}
              fontSize={14}
              className="fill-ink font-serif"
            >
              {breakdown.model.name}
            </text>
            {rows.map((row) => {
              const group = groups.get(`${row.attribute}/${row.label}`);
              if (group === undefined) {
                return null;
              }
              const y = rowY(row.at);
              const key = `${row.attribute}/${row.label}`;
              if (group.fpir === null) {
                return (
                  <text
                    key={key}
                    x={area.left + 4}
                    y={y}
                    dominantBaseline="central"
                    fontSize={11}
                    className="fill-muted-ink"
                  >
                    {`too few (n = ${formatCount(group.heldOutIdentities)})`}
                  </text>
                );
              }
              return (
                <g key={key}>
                  <Bar
                    x0={x(0)}
                    x1={x(group.fpir.value * 100)}
                    y={y}
                    thickness={ROW * 0.45}
                    colour={style.colour}
                  />
                  <Whisker
                    x0={x(group.fpir.ci.low * 100)}
                    x1={x(group.fpir.ci.high * 100)}
                    y={y}
                    colour={style.colour}
                  />
                </g>
              );
            })}
            {total === undefined ? null : (
              <>
                <line
                  data-overall=""
                  x1={x(total * 100)}
                  x2={x(total * 100)}
                  y1={area.top}
                  y2={area.bottom}
                  className="stroke-ink"
                  strokeWidth={0.9}
                  strokeDasharray="4 2"
                />
                {index === 0 ? (
                  <DirectLabel x={x(total * 100) + 3} y={area.top - 8}>
                    overall
                  </DirectLabel>
                ) : null}
              </>
            )}
            <AxisBottom
              area={area}
              scale={x}
              ticks={ticks}
              format={(tick) => String(Number(tick.toFixed(2)))}
              title="Test FPIR (%)"
            />
          </g>
        );
      })}
      {shading === null || panels.length === 0 ? null : (
        <text
          transform={`translate(${width - RIGHT + 14},${(shading.top + shading.bottom) / 2}) rotate(90)`}
          textAnchor="middle"
          fontSize={11}
          className="fill-muted-ink"
        >
          indicative
        </text>
      )}
    </Chart>
  );
}
