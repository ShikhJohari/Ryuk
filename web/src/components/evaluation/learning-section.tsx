import type {
  LearningComparison,
  LearningReport,
  MethodComparison,
} from "@/api/evaluation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { digitsFor, formatGain, formatTarget, formatThreshold } from "./format";
import { METHOD_NAMES, RateText, RULE_IN_PROSE } from "./labels";
import {
  EvaluationSection,
  Lede,
  Notes,
  NotMeasured,
  NumberedCaption,
  numberClass,
  RowHeader,
} from "./layout";

/** How many tables the section numbers: one per model. */
export function learningTables(learning: LearningReport | null): number {
  return learning === null ? 0 : learning.models.length;
}

/**
 * The operating point methods are compared at: TPIR at the target FPIR read
 * off the method's own test curve, not at its frozen threshold.
 */
function headline(method: MethodComparison, targetFpir: number) {
  return method.operatingPoints.find(
    (point) => Math.abs(point.targetFpir - targetFpir) < 1e-12,
  );
}

/** Learning on embeddings (#10): a methods table per model, then its live rule and why. */
export function LearningSection({
  learning,
  table,
}: {
  readonly learning: LearningReport | null;
  readonly table: number;
}) {
  return (
    <EvaluationSection title="Learning on embeddings">
      {learning === null ? (
        <NotMeasured command="ryuk evaluate learn" />
      ) : (
        <>
          <Lede>
            Can learning on the frozen embeddings beat matching a face to each
            person's best photo? Every method is compared at FPIR{" "}
            {formatTarget(learning.targetFpir)} on the test draw, and only one
            that needs no retraining when the watchlist changes can run live.
          </Lede>
          {learning.models.map((compared, index) => (
            <ModelMethods
              key={compared.model.id}
              compared={compared}
              targetFpir={learning.targetFpir}
              table={table + index}
            />
          ))}
          <Notes>
            <li>★ The rule the model runs live.</li>
            {learning.models.some((compared) =>
              compared.methods.some((method) => method.needsRetraining),
            ) ? (
              <li>
                † Needs retraining whenever the watchlist changes, so never runs
                live, whatever its gain.
              </li>
            ) : null}
          </Notes>
        </>
      )}
    </EvaluationSection>
  );
}

function ModelMethods({
  compared,
  targetFpir,
  table,
}: {
  readonly compared: LearningComparison;
  readonly targetFpir: number;
  readonly table: number;
}) {
  const target = formatTarget(targetFpir);
  const baseline = compared.methods.find(
    (method) => method.method === "best-photo",
  );
  const baselinePoint =
    baseline === undefined ? undefined : headline(baseline, targetFpir);
  // A gain is shown at the precision of the baseline's TPIR.
  const gainDigits =
    baselinePoint === undefined ? 1 : digitsFor(baselinePoint.tpir.value);

  return (
    <div className="flex flex-col gap-3">
      <Table>
        <NumberedCaption number={table}>
          Learning on embeddings for {compared.model.name}. Each method was
          fitted and frozen on the validation draw, then scored on the test
          draw. TPIR @ FPIR {target} is read off the method's own test curve;
          its gain is the difference from best photo in points, with the 95%
          paired bootstrap interval, in bold where that interval lies wholly
          above zero. The rest are at the method's frozen threshold. Rates in
          percent.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Method</TableHead>
            <TableHead>Hyperparameter</TableHead>
            <TableHead className={numberClass}>Rank-1</TableHead>
            <TableHead className={numberClass}>TPIR @ FPIR {target}</TableHead>
            <TableHead className={numberClass}>Gain (points)</TableHead>
            <TableHead className={numberClass}>Frozen threshold</TableHead>
            <TableHead className={numberClass}>TPIR</TableHead>
            <TableHead className={numberClass}>FPIR</TableHead>
            <TableHead className={numberClass}>Misidentification</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {compared.methods.map((method) => {
            const point = headline(method, targetFpir);
            const gain =
              method.gain === null ? null : formatGain(method.gain, gainDigits);
            return (
              <TableRow key={method.method}>
                <RowHeader>
                  {METHOD_NAMES[method.method]}
                  {method.method === compared.liveRule ? " ★" : ""}
                  {method.needsRetraining ? " †" : ""}
                </RowHeader>
                <TableCell>
                  {method.hyperparameter === null
                    ? "—"
                    : `${method.hyperparameter.name} = ${Number(method.hyperparameter.value.toPrecision(6))}`}
                </TableCell>
                <TableCell className={numberClass}>
                  <RateText rate={method.rank1} />
                </TableCell>
                <TableCell className={numberClass}>
                  {point === undefined ? (
                    "—"
                  ) : (
                    <>
                      <RateText rate={point.tpir} />
                      {point.indicative ? " (indicative)" : null}
                    </>
                  )}
                </TableCell>
                <TableCell className={numberClass}>
                  {gain === null ? (
                    "—"
                  ) : method.gain?.improves ? (
                    <strong className="font-semibold text-ink">{gain}</strong>
                  ) : (
                    gain
                  )}
                </TableCell>
                <TableCell className={numberClass}>
                  {formatThreshold(method.threshold)}
                </TableCell>
                <TableCell className={numberClass}>
                  <RateText rate={method.atThreshold.tpir} />
                </TableCell>
                <TableCell className={numberClass}>
                  <RateText rate={method.atThreshold.fpir} />
                </TableCell>
                <TableCell className={numberClass}>
                  <RateText rate={method.atThreshold.misidentification} />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <p className="max-w-[80ch]">
        <strong className="font-semibold text-ink">
          {compared.model.name} runs {RULE_IN_PROSE[compared.liveRule]} live.
        </strong>{" "}
        <span className="text-muted-foreground">{compared.liveReason}</span>
      </p>
    </div>
  );
}
