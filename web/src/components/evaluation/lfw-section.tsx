import type { LfwResult, VerificationReport } from "@/api/evaluation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Figure } from "./chart";
import { MIN_FAR } from "./curves";
import { formatCount, formatMs, formatPercent, formatTarget } from "./format";
import {
  EvaluationSection,
  Lede,
  Notes,
  NumberedCaption,
  numberClass,
  RowHeader,
} from "./layout";
import { RocChart } from "./roc-chart";

/** How many tables and figures the section numbers. */
export const LFW_TABLES = 1;
export const LFW_FIGURES = 1;

/** The operating points' target FARs, in the order the results list them. */
function targets(models: ReadonlyArray<LfwResult>): ReadonlyArray<number> {
  return [
    ...new Set(
      models.flatMap((result) =>
        result.operatingPoints.map((point) => point.targetFar),
      ),
    ),
  ];
}

/**
 * Verification on LFW View 2: the report's LFW table with both accuracies,
 * its notes, the SFace int8 footnote, and the ROC.
 */
export function LfwSection({
  verification,
  folds,
  table,
  figure,
}: {
  readonly verification: VerificationReport;
  /** View 2's fold count, from the dataset summary; null if it is not there. */
  readonly folds: number | null;
  readonly table: number;
  readonly figure: number;
}) {
  const { models } = verification;
  const unscored = verification.pairs - verification.scoredPairs;
  const columns = targets(models).map((target) => ({
    target,
    indicative: models.some((result) =>
      result.operatingPoints.some(
        (point) => point.targetFar === target && point.indicative,
      ),
    ),
  }));
  const flagged = models.some((result) => !result.reproducesPublished);
  const int8 = verification.sfaceInt8;

  return (
    <EvaluationSection title="Verification on LFW">
      <Lede>
        Each model decides whether two faces are the same person on LFW View 2,
        and must reproduce its published accuracy before evaluation trusts the
        rest of its pipeline.
      </Lede>
      <Table>
        <NumberedCaption number={table}>
          Verification on LFW View 2. {formatCount(verification.scoredPairs)} of{" "}
          {formatCount(verification.pairs)} pairs scored
          {folds === null ? "" : `, in ${folds} folds`}, the threshold of each
          fold chosen on the other nine. Accuracy and TAR in percent; our ± is
          the standard error over the folds. An indicative operating point rests
          on too few mismatched pairs to pin down.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead className={numberClass}>Embedding size</TableHead>
            <TableHead className={numberClass}>Published LFW</TableHead>
            <TableHead className={numberClass}>Our LFW (± SE)</TableHead>
            <TableHead className={numberClass}>
              Unscored pairs as errors
            </TableHead>
            <TableHead className={numberClass}>AUC</TableHead>
            {columns.map((column) => (
              <TableHead key={column.target} className={numberClass}>
                TAR @ FAR {formatTarget(column.target)}
                {column.indicative ? " (indicative)" : ""}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((result) => (
            <TableRow key={result.model.id}>
              <RowHeader>{result.model.name}</RowHeader>
              <TableCell className={numberClass}>
                {result.model.dimension}
              </TableCell>
              <TableCell className={numberClass}>
                {formatPercent(result.published.accuracy, 2)}
              </TableCell>
              <TableCell className={numberClass}>
                {formatPercent(result.accuracy, 2)} ±{" "}
                {formatPercent(result.standardError, 2)}
                {result.reproducesPublished ? "" : " ⚑"}
              </TableCell>
              <TableCell className={numberClass}>
                {formatPercent(result.accuracyIfExcludedWereErrors, 2)}
              </TableCell>
              <TableCell className={numberClass}>
                {result.auc.toFixed(4)}
              </TableCell>
              {columns.map((column) => {
                const point = result.operatingPoints.find(
                  (candidate) => candidate.targetFar === column.target,
                );
                return (
                  <TableCell key={column.target} className={numberClass}>
                    {point === undefined ? "—" : formatPercent(point.tar, 2)}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Notes>
        {flagged ? (
          <li>
            ⚑ More than {verification.tolerancePoints} points from the published
            figure: a pipeline bug to find, not a result.
          </li>
        ) : null}
        {unscored > 0 ? (
          <li>
            Unscored pairs as errors: the same accuracy with each of the{" "}
            {formatCount(unscored)} unscored pairs counted as an error, the
            worst the exclusions could hide. It is reported and never decides:
            the first active model is judged on the scored pairs.
          </li>
        ) : null}
        {models.map((result) =>
          result.published.note === null ? null : (
            <li key={result.model.id}>
              {result.model.name}'s published figure: {result.published.note}
              {result.published.note.includes("±")
                ? " Its ± is a standard deviation across the source's folds, not a standard error like ours."
                : ""}
            </li>
          ),
        )}
        <li>
          Published figures from{" "}
          {models.map((result, index) => (
            <span key={result.model.id}>
              {index === 0 ? "" : index === models.length - 1 ? " and " : ", "}
              <a
                href={result.published.source}
                target="_blank"
                rel="noreferrer"
                className="text-ink underline underline-offset-2"
              >
                {result.model.name}'s source
              </a>
            </span>
          ))}
          .
        </li>
        <li>
          SFace int8 is not a row: Ryuk runs SFace fp32. On the same pairs int8
          scores {formatPercent(int8.accuracy, 2)} ±{" "}
          {formatPercent(int8.standardError, 2)}, and takes{" "}
          {formatMs(int8.msPerFaceInt8)} ms per face against fp32's{" "}
          {formatMs(int8.msPerFaceFp32)} ms, timing the embedding alone, not end
          to end like every other ms per face on this page. Its embeddings also
          drift from fp32's: cosine {int8.cosineToFp32Mean.toFixed(3)} on
          average, {int8.cosineToFp32Min.toFixed(3)} at worst, over{" "}
          {formatCount(int8.facesCompared)} faces.
        </li>
      </Notes>

      <Figure
        number={figure}
        caption={`TAR against FAR on LFW View 2. FAR is on a log axis from ${formatTarget(MIN_FAR)}; each curve steps at a false accept, and is labelled with its model's accuracy. Each marker is an operating point of Table ${table}, where an indicative one is drawn hollow.`}
      >
        {(captionId) => <RocChart models={models} labelledBy={captionId} />}
      </Figure>
    </EvaluationSection>
  );
}
