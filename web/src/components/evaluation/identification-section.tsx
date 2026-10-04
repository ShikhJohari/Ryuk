import type { ReactNode } from "react";
import type {
  IdentificationReport,
  OpenSetResult,
  TpirAtFpir,
} from "@/api/evaluation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Figure } from "./chart";
import { falseAlarmFloor } from "./curves";
import { formatCount, formatMs, formatTarget, formatThreshold } from "./format";
import { RateText } from "./labels";
import {
  EvaluationSection,
  Lede,
  Notes,
  NotMeasured,
  NumberedCaption,
  numberClass,
  RowHeader,
} from "./layout";
import { OpenSetChart } from "./open-set-chart";

/** How many tables and figures the section numbers. */
export function identificationCounts(
  identification: IdentificationReport | null,
) {
  return identification === null
    ? { tables: 0, figures: 0 }
    : { tables: 1, figures: 1 };
}

/** A row of Table 2: a measure, and each model's cell. */
type Measure = {
  readonly key: string;
  readonly name: string;
  readonly cell: (result: OpenSetResult) => ReactNode;
};

function operatingPoint(
  result: OpenSetResult,
  target: number,
): TpirAtFpir | undefined {
  return result.operatingPoints.find((point) => point.targetFpir === target);
}

/**
 * Watchlist search on CelebA's test draw: Table 2 of the report, a row per
 * measure and a column per model, and TPIR against FPIR.
 */
export function IdentificationSection({
  identification,
  firstActiveId,
  lfwTable,
  table,
  figure,
}: {
  readonly identification: IdentificationReport | null;
  /** The first active model's ID, starred; null when there is none. */
  readonly firstActiveId: string | null;
  /** The LFW table's number, whose int8 footnote this section's units are set against. */
  readonly lfwTable: number;
  readonly table: number;
  readonly figure: number;
}) {
  return (
    <EvaluationSection title="Watchlist search on CelebA">
      {identification === null ? (
        <NotMeasured command="ryuk evaluate celeba" />
      ) : (
        <Identification
          identification={identification}
          firstActiveId={firstActiveId}
          lfwTable={lfwTable}
          table={table}
          figure={figure}
        />
      )}
    </EvaluationSection>
  );
}

function Identification({
  identification,
  firstActiveId,
  lfwTable,
  table,
  figure,
}: {
  readonly identification: IdentificationReport;
  readonly firstActiveId: string | null;
  readonly lfwTable: number;
  readonly table: number;
  readonly figure: number;
}) {
  const { models } = identification;
  const draw =
    identification.draws.find((candidate) => candidate.draw === "test") ?? null;
  const floor = falseAlarmFloor(draw?.nonMatedProbes ?? 0);
  const targetFpirs = [...new Set(models.map((result) => result.targetFpir))];
  const frozenAt =
    targetFpirs.length === 1 && targetFpirs[0] !== undefined
      ? `at FPIR ${formatTarget(targetFpirs[0])}`
      : "at each model's own target FPIR";
  const pointTargets = [
    ...new Set(
      models.flatMap((result) =>
        result.operatingPoints.map((point) => point.targetFpir),
      ),
    ),
  ];
  const starred = models.some((result) => result.model.id === firstActiveId);
  const measures: ReadonlyArray<Measure> = [
    {
      key: "rank-1",
      name: "Rank-1",
      cell: (result) => <RateText rate={result.rank1} />,
    },
    {
      key: "threshold",
      name: "Frozen threshold",
      cell: (result) => formatThreshold(result.threshold),
    },
    {
      key: "tpir",
      name: "TPIR",
      cell: (result) => <RateText rate={result.atThreshold.tpir} />,
    },
    {
      key: "fpir",
      name: "FPIR",
      cell: (result) => <RateText rate={result.atThreshold.fpir} />,
    },
    {
      key: "misidentification",
      name: "Misidentification",
      cell: (result) => (
        <RateText rate={result.atThreshold.misidentification} />
      ),
    },
    ...pointTargets.map((target) => ({
      key: `at-${target}`,
      name: `TPIR @ FPIR ${formatTarget(target)}${
        models.some((result) => operatingPoint(result, target)?.indicative)
          ? " (indicative)"
          : ""
      }`,
      cell: (result: OpenSetResult) => {
        const point = operatingPoint(result, target);
        return point === undefined ? "—" : <RateText rate={point.tpir} />;
      },
    })),
    {
      key: "ms",
      name: "ms per face (end to end)",
      cell: (result) => formatMs(result.msPerFace),
    },
  ];

  return (
    <>
      <Lede>
        Each model rehearses the watchlist: whose face is this, if anyone's? Its
        threshold is frozen on the validation draw, then the test draw is scored
        once.
      </Lede>
      <Table>
        <NumberedCaption number={table}>
          Watchlist search on the CelebA test draw. TPIR, FPIR and
          misidentification are at each model's threshold, frozen on the
          validation draw {frozenAt}; TPIR @ FPIR is read off the test curve
          instead, and an indicative one rests on a handful of false alarms.
          Rates in percent with 95% identity-level bootstrap intervals in
          brackets, and the dependence-adjusted Wilson interval under a rate
          within 1% of 0 or 100%
          {starred ? "; ★ marks the first active model." : "."}
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>
              <span className="sr-only">Measure</span>
            </TableHead>
            {models.map((result) => (
              <TableHead key={result.model.id} className={numberClass}>
                {result.model.name}
                {result.model.id === firstActiveId ? " ★" : ""}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {measures.map((measure) => (
            <TableRow key={measure.key}>
              <RowHeader>{measure.name}</RowHeader>
              {models.map((result) => (
                <TableCell key={result.model.id} className={numberClass}>
                  {measure.cell(result)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Notes>
        {draw === null ? null : (
          <li>
            CelebA test draw: {formatCount(draw.galleryIdentities)} gallery
            identities with {formatCount(draw.matedProbes)} mated probes, and{" "}
            {formatCount(draw.heldOutIdentities)} held-out identities with{" "}
            {formatCount(draw.nonMatedProbes)} non-mated probes. The gallery
            enrols {formatCount(draw.enrolledPhotos)} photos; intervals from{" "}
            {formatCount(identification.bootstrapResamples)} bootstrap resamples
            of the identities.
          </li>
        )}
        <li>
          ms per face runs from pixels to top candidate, detection included: a
          warm median on the machine evaluation ran on. SFace int8's figure
          under Table {lfwTable} times the embedding alone.
        </li>
      </Notes>

      <Figure
        number={figure}
        caption={`TPIR against FPIR on the CelebA test draw. FPIR is on a log axis from ${formatTarget(floor)}, the decade of one false alarm in ${formatCount(draw?.nonMatedProbes ?? 0)} non-mated probes, the least the draw can show. Each marker is the model's frozen threshold, where the validation draw put it.`}
      >
        {(captionId) => (
          <OpenSetChart models={models} floor={floor} labelledBy={captionId} />
        )}
      </Figure>
    </>
  );
}
