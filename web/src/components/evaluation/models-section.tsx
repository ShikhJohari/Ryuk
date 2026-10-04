import { Link } from "@tanstack/react-router";
import type { Eligibility, FirstActiveModel } from "@/api/evaluation";
import type { ModelState, RecognitionModelInfo } from "@/api/models";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  formatCount,
  formatMs,
  formatPercent,
  formatRate,
  formatSigned,
  formatTarget,
  formatThreshold,
} from "./format";
import {
  EvaluationSection,
  Lede,
  NotMeasured,
  NumberedCaption,
  numberClass,
  RowHeader,
  Subsection,
} from "./layout";

const STATE_NAMES: Readonly<Record<ModelState, string>> = {
  active: "Active",
  available: "Available",
  not_evaluated: "Not evaluated",
  unavailable: "Weights missing",
};

/** How many tables the section numbers. */
export function modelsTables(first: FirstActiveModel | null): number {
  return first === null ? 1 : 2;
}

/**
 * Every recognition model, read-only (#20): switching the active model
 * happens in the live monitor's toolbar alone. Then the first active model
 * and how each model was judged for it.
 */
export function ModelsSection({
  models,
  first,
  table,
}: {
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly first: FirstActiveModel | null;
  /** This section's first table number. */
  readonly table: number;
}) {
  return (
    <EvaluationSection title="Recognition models">
      <Lede>
        Every recognition model the service knows, as evaluation left it. This
        page only reads them: the active model is switched in the{" "}
        <Link to="/monitor" className="text-ink underline underline-offset-2">
          live monitor
        </Link>
        .
      </Lede>
      <Table>
        <NumberedCaption number={table}>
          Recognition models. Each threshold is on the match score of the
          model's live rule, frozen by evaluation. ms per face is end to end,
          from pixels to top candidate, detection included: a warm median on the
          machine evaluation ran on.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>State</TableHead>
            <TableHead className={numberClass}>Embedding size</TableHead>
            <TableHead className={numberClass}>Threshold</TableHead>
            <TableHead className={numberClass}>
              ms per face (end to end)
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => (
            <TableRow key={model.id}>
              <RowHeader>{model.name}</RowHeader>
              <TableCell>{STATE_NAMES[model.state]}</TableCell>
              <TableCell className={numberClass}>{model.dimension}</TableCell>
              <TableCell className={numberClass}>
                {model.threshold === null
                  ? "—"
                  : formatThreshold(model.threshold)}
              </TableCell>
              <TableCell className={numberClass}>
                {model.msPerFace === null ? "—" : formatMs(model.msPerFace)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Subsection title="The first active model">
        {first === null ? (
          <NotMeasured command="ryuk evaluate celeba" />
        ) : (
          <FirstActive first={first} table={table + 1} />
        )}
      </Subsection>
    </EvaluationSection>
  );
}

function FirstActive({
  first,
  table,
}: {
  readonly first: FirstActiveModel;
  readonly table: number;
}) {
  const gate = first.lfwGate;
  return (
    <>
      <dl className="grid max-w-[72ch] grid-cols-[max-content_1fr] gap-x-6 gap-y-2">
        <dt className="section-label pt-1">First active model</dt>
        <dd className="font-serif text-[20px] text-ink leading-tight">
          {first.model === null
            ? "None: no model is eligible"
            : first.model.name}
        </dd>
        <dt className="section-label pt-0.5">Why</dt>
        <dd className="text-muted-foreground">{first.reason}</dd>
      </dl>
      <Table>
        <NumberedCaption number={table}>
          Eligibility for the first active model. A model is eligible with LFW
          accuracy within {gate.tolerancePoints} points of its published
          accuracy over the {formatCount(gate.scoredPairs)} scored pairs of{" "}
          {formatCount(gate.pairs)}, test FPIR at the frozen threshold at most{" "}
          {formatTarget(first.maxTestFpir)}, and at most {first.maxMsPerFace} ms
          per face end to end; each is judged under its live rule. TPIR and FPIR
          in percent, on the CelebA test draw.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead className={numberClass}>LFW gap (points)</TableHead>
            <TableHead className={numberClass}>Test TPIR</TableHead>
            <TableHead className={numberClass}>Test FPIR</TableHead>
            <TableHead className={numberClass}>
              ms per face (end to end)
            </TableHead>
            <TableHead>Eligible</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {first.eligibility.map((judged) => (
            <EligibilityRow key={judged.model.id} judged={judged} />
          ))}
        </TableBody>
      </Table>
    </>
  );
}

function EligibilityRow({ judged }: { readonly judged: Eligibility }) {
  return (
    <TableRow>
      <RowHeader>{judged.model.name}</RowHeader>
      <TableCell className={numberClass}>
        {judged.lfwGapPoints === null ? (
          "not scored"
        ) : (
          <>
            {formatSigned(judged.lfwGapPoints, 2)}
            {judged.reproducesLfw ? null : <Fails>outside the tolerance</Fails>}
          </>
        )}
      </TableCell>
      <TableCell className={numberClass}>
        {formatRate(judged.testTpir)}
      </TableCell>
      <TableCell className={numberClass}>
        {formatPercent(judged.testFpir)}
        {judged.fpirWithinLimit ? null : <Fails>over the limit</Fails>}
      </TableCell>
      <TableCell className={numberClass}>
        {formatMs(judged.msPerFace)}
        {judged.fastEnough ? null : <Fails>over the limit</Fails>}
      </TableCell>
      <TableCell>{judged.eligible ? "Yes" : "No"}</TableCell>
    </TableRow>
  );
}

/** Beside a number that fails one of the rule's tests. */
function Fails({ children }: { readonly children: string }) {
  return (
    <>
      {" "}
      <span className="text-destructive">{children}</span>
    </>
  );
}
