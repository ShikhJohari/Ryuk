import type { ReactNode } from "react";
import type {
  LiveOperatingPoints,
  LiveReport,
  SamePersonRates,
} from "@/api/evaluation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  fineDigitsFor,
  formatCount,
  formatTarget,
  formatThreshold,
} from "./format";
import { METHOD_NAMES, RateText } from "./labels";
import {
  EvaluationSection,
  Lede,
  NotMeasured,
  NumberedCaption,
  numberClass,
  RowHeader,
  Subsection,
} from "./layout";

/** How many tables the section numbers. */
export function liveTables(live: LiveReport | null): number {
  return live === null ? 0 : 2;
}

function photos(count: number): string {
  return `${count} photo${count === 1 ? "" : "s"}`;
}

/** Each distinct value of `key` over `items`, in the order they first appear. */
function distinct<T>(
  items: ReadonlyArray<T>,
  key: (item: T) => number,
): ReadonlyArray<number> {
  return [...new Set(items.map(key))];
}

/**
 * Where enrollment and the live monitor actually operate (#49): the live
 * rules on smaller watchlists and with one photo, and the same-person
 * warning's one-to-one threshold.
 */
export function LiveSection({
  live,
  table,
}: {
  readonly live: LiveReport | null;
  readonly table: number;
}) {
  return (
    <EvaluationSection title="Away from the rehearsal">
      {live === null ? (
        <NotMeasured command="ryuk evaluate live" />
      ) : (
        <>
          <Lede>
            The rehearsal's gallery is larger, and holds more photos per
            identity, than a live watchlist usually does. These measure the live
            rules where they operate: on smaller watchlists, with one photo
            each, and one to one when a photo is added to a person of interest.
          </Lede>
          <Subsection title="Smaller watchlists and one photo">
            <SmallGalleries models={live.models} table={table} />
          </Subsection>
          <Subsection title="The same-person warning">
            <SamePerson models={live.models} table={table + 1} />
          </Subsection>
        </>
      )}
    </EvaluationSection>
  );
}

function SmallGalleries({
  models,
  table,
}: {
  readonly models: ReadonlyArray<LiveOperatingPoints>;
  readonly table: number;
}) {
  const cells = models.flatMap((model) => model.smallGalleries);
  const enrolled = distinct(cells, (cell) => cell.enrolledPhotos);
  const sizes = distinct(cells, (cell) => cell.identities);
  const [largest, ...smaller] = sizes;
  return (
    <Table>
      <NumberedCaption number={table}>
        The live rules on smaller watchlists. CelebA test draw: the{" "}
        {formatCount(largest ?? 0)}-identity gallery is the rehearsal's own;
        each smaller one is that gallery split at random into disjoint galleries
        of {smaller.map(formatCount).join(", ")} identities, each scored as a
        watchlist of its own. 1 photo enrols each identity's first photo alone.
        Each model runs its live rule at its frozen threshold. Rates in percent
        with 95% identity-level bootstrap intervals in brackets.
      </NumberedCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead>Live rule</TableHead>
          <TableHead className={numberClass}>Identities</TableHead>
          <TableHead className={numberClass}>Galleries</TableHead>
          {enrolled.flatMap((count) =>
            ["TPIR", "FPIR"].map((rate) => (
              <TableHead key={`${rate}-${count}`} className={numberClass}>
                {rate}, {photos(count)}
              </TableHead>
            )),
          )}
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.flatMap((model) =>
          sizes.map((identities, index) => {
            const row = model.smallGalleries.filter(
              (cell) => cell.identities === identities,
            );
            return (
              <TableRow key={`${model.model.id}-${identities}`}>
                {index === 0 ? (
                  <>
                    <RowHeader className="align-top" rowSpan={sizes.length}>
                      {model.model.name}
                    </RowHeader>
                    <TableCell className="align-top" rowSpan={sizes.length}>
                      {METHOD_NAMES[model.rule]}
                    </TableCell>
                  </>
                ) : null}
                <TableCell className={numberClass}>
                  {formatCount(identities)}
                </TableCell>
                <TableCell className={numberClass}>
                  {row[0] === undefined ? "—" : formatCount(row[0].galleries)}
                </TableCell>
                {enrolled.flatMap((count) => {
                  const cell = row.find(
                    (candidate) => candidate.enrolledPhotos === count,
                  );
                  return [
                    <TableCell key={`tpir-${count}`} className={numberClass}>
                      {cell === undefined ? "—" : <RateText rate={cell.tpir} />}
                    </TableCell>,
                    <TableCell key={`fpir-${count}`} className={numberClass}>
                      {cell === undefined ? (
                        "—"
                      ) : (
                        <RateText
                          rate={cell.fpir}
                          digits={fineDigitsFor(cell.fpir.value)}
                        />
                      )}
                    </TableCell>,
                  ];
                })}
              </TableRow>
            );
          }),
        )}
      </TableBody>
    </Table>
  );
}

/** A row of the same-person table: a measure, and each model's cell. */
type Measure = {
  readonly key: string;
  readonly name: string;
  readonly cell: (model: LiveOperatingPoints) => ReactNode;
};

function rates(
  model: LiveOperatingPoints,
  enrolledPhotos: number,
): SamePersonRates | undefined {
  return model.samePerson.test.find(
    (candidate) => candidate.enrolledPhotos === enrolledPhotos,
  );
}

function SamePerson({
  models,
  table,
}: {
  readonly models: ReadonlyArray<LiveOperatingPoints>;
  readonly table: number;
}) {
  const first = models[0]?.samePerson;
  const enrolled = distinct(
    models.flatMap((model) => model.samePerson.test),
    (rates) => rates.enrolledPhotos,
  );
  const perPhotos = (
    key: string,
    name: string,
    read: (rates: SamePersonRates) => ReactNode,
  ): ReadonlyArray<Measure> =>
    enrolled.map((count) => ({
      key: `${key}-${count}`,
      name: `${name}, ${photos(count)}`,
      cell: (model) => {
        const found = rates(model, count);
        return found === undefined ? "—" : read(found);
      },
    }));
  const measures: ReadonlyArray<Measure> = [
    {
      key: "same-person",
      name: "Same-person threshold",
      cell: (model) => formatThreshold(model.samePerson.threshold),
    },
    {
      key: "one-to-many",
      name: "1:N threshold",
      cell: (model) =>
        `${formatThreshold(model.threshold)} (${METHOD_NAMES[model.rule].toLowerCase()})`,
    },
    ...perPhotos("warned", "Own photos warned", (found) => (
      <RateText rate={found.warningRate} />
    )),
    ...perPhotos("at-one-to-many", "At the 1:N threshold", (found) =>
      found.warningRateAtLiveThreshold === null ? (
        "—"
      ) : (
        <RateText rate={found.warningRateAtLiveThreshold} />
      ),
    ),
    ...perPhotos("far", "FAR", (found) => <RateText rate={found.far} />),
  ];
  const one = first?.test[0];

  return (
    <Table>
      <NumberedCaption number={table}>
        The same-person warning. CelebA test draw
        {one === undefined
          ? ""
          : `, with ${formatCount(one.matedPairs)} mated and ${formatCount(one.impostorPairs)} impostor pairs`}
        . Each model's same-person threshold is a cosine
        {first === undefined
          ? ""
          : ` frozen at FAR ${formatTarget(first.targetFar)} on the validation draw's ${formatCount(first.validationImpostorPairs)} impostor pairs with one photo enrolled`}
        . Own photos warned is the share of mated pairs under it; FAR the share
        of impostor pairs at or above it, which would not warn. The 1:N rows
        apply the model's match threshold under its live rule instead; a dash is
        a learned rule, whose threshold is not a cosine. Rates in percent with
        95% identity-level bootstrap intervals in brackets.
      </NumberedCaption>
      <TableHeader>
        <TableRow>
          <TableHead>
            <span className="sr-only">Measure</span>
          </TableHead>
          {models.map((model) => (
            <TableHead key={model.model.id} className={numberClass}>
              {model.model.name}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {measures.map((measure) => (
          <TableRow key={measure.key}>
            <RowHeader>{measure.name}</RowHeader>
            {models.map((model) => (
              <TableCell key={model.model.id} className={numberClass}>
                {measure.cell(model)}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
