import type {
  AttributeBreakdown,
  BiasBreakdown,
  BiasReport,
  GroupRates,
  IdentificationReport,
  Rate,
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
import { formatCount, formatTarget, formatThreshold } from "./format";
import { GroupFpirChart } from "./group-fpir-chart";
import { ATTRIBUTE_NAMES, RateText } from "./labels";
import {
  EvaluationSection,
  Lede,
  Notes,
  NotMeasured,
  NumberedCaption,
  numberClass,
  RowHeader,
} from "./layout";

/** Every model's breakdown under best photo: the page compares models on equal terms. */
function underBestPhoto(bias: BiasReport): ReadonlyArray<BiasBreakdown> {
  return bias.models.filter((breakdown) => breakdown.rule === "best-photo");
}

/** How many tables and figures the section numbers. */
export function biasCounts(bias: BiasReport | null) {
  const breakdowns = bias === null ? [] : underBestPhoto(bias);
  return {
    tables: breakdowns.length,
    figures: breakdowns.length === 0 ? 0 : 1,
  };
}

const COLUMNS = 8;

function listed(names: ReadonlyArray<string>): string {
  return names.length < 2
    ? (names[0] ?? "")
    : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/** What sets an attribute's groups apart, after its name. */
function qualifiers(attribute: AttributeBreakdown): string {
  return [
    ATTRIBUTE_NAMES[attribute.attribute],
    attribute.basis === "identity" ? "by identity" : "per photo",
    ...(attribute.indicative ? ["indicative"] : []),
    `worst-to-best FPIR ${
      attribute.fpirRatio === null ? "—" : `${attribute.fpirRatio.toFixed(2)}×`
    }`,
  ].join(" · ");
}

function Estimate({ rate }: { readonly rate: Rate | null }) {
  return rate === null ? "too few" : <RateText rate={rate} />;
}

/**
 * The bias breakdown under best photo only (the service sends every rule):
 * group FPIR against each model's overall FPIR, then a table of rates by
 * group per model, each attribute's basis labelled.
 */
export function BiasSection({
  bias,
  identification,
  table,
  figure,
}: {
  readonly bias: BiasReport | null;
  /** For each model's overall test FPIR; the dashed line is left out without it. */
  readonly identification: IdentificationReport | null;
  readonly table: number;
  readonly figure: number;
}) {
  return (
    <EvaluationSection title="Bias breakdown">
      {bias === null ? (
        <NotMeasured command="ryuk evaluate bias" />
      ) : (
        <Breakdowns
          bias={bias}
          identification={identification}
          table={table}
          figure={figure}
        />
      )}
    </EvaluationSection>
  );
}

function Breakdowns({
  bias,
  identification,
  table,
  figure,
}: {
  readonly bias: BiasReport;
  readonly identification: IdentificationReport | null;
  readonly table: number;
  readonly figure: number;
}) {
  const breakdowns = underBestPhoto(bias);
  const first = breakdowns[0];
  if (first === undefined) {
    return (
      <p className="text-muted-foreground">
        No model was broken down under best photo.
      </p>
    );
  }
  const overall = new Map(
    (identification?.models ?? []).map(
      (result) => [result.model.id, result.atThreshold.fpir.value] as const,
    ),
  );
  const byIdentity = first.attributes.filter(
    (attribute) => attribute.basis === "identity",
  );
  const mixed = byIdentity.filter(
    (attribute) =>
      attribute.mixedGalleryIdentities > 0 ||
      attribute.mixedHeldOutIdentities > 0,
  );
  const byPhoto = first.attributes.filter(
    (attribute) => attribute.basis === "photo",
  );
  const indicative = first.attributes.filter(
    (attribute) => attribute.indicative,
  );
  const tooFew = breakdowns.some((breakdown) =>
    breakdown.attributes.some((attribute) =>
      attribute.groups.some(
        (group) =>
          group.tpir === null ||
          group.misidentification === null ||
          group.fpir === null,
      ),
    ),
  );

  return (
    <>
      <Lede>
        Each model's rates by group on the test draw, at its single frozen
        threshold: no group has its own. Every model is drawn under best photo
        here, so the models compare on equal terms.
      </Lede>
      <Figure
        number={figure}
        caption={`Each group's test FPIR at the model's single frozen threshold under best photo, with its 95% identity-level interval. ${
          overall.size === 0
            ? "Each model's overall test FPIR is not drawn: the CelebA test draw is not measured yet. Indicative"
            : "The dashed line is the model's overall test FPIR; indicative"
        } rows are shaded, and a group too small to estimate shows its count of held-out identities.`}
      >
        {(captionId) => (
          <GroupFpirChart
            breakdowns={breakdowns}
            overall={overall}
            labelledBy={captionId}
          />
        )}
      </Figure>

      {breakdowns.map((breakdown, index) => (
        <GroupTable
          key={breakdown.model.id}
          breakdown={breakdown}
          table={table + index}
        />
      ))}
      <Notes>
        {byIdentity.length === 0 ? null : (
          <li>
            By identity: each identity goes by its majority label over all its
            images, where at least {formatTarget(bias.agreement)} of them agree;
            an identity with no such label is left out of that attribute
            {mixed.length === 0
              ? "."
              : `: ${mixed
                  .map(
                    (attribute) =>
                      `${ATTRIBUTE_NAMES[attribute.attribute]} ${formatCount(attribute.mixedGalleryIdentities)} gallery and ${formatCount(attribute.mixedHeldOutIdentities)} held-out`,
                  )
                  .join(", ")}.`}
          </li>
        )}
        {byPhoto.length === 0 ? null : (
          <li>
            Per photo: each probe goes by its own label, so one identity's
            probes can fall in both groups; a group's identities are those with
            a probe in it.
          </li>
        )}
        {indicative.length === 0 ? null : (
          <li>
            Indicative:{" "}
            {listed(
              indicative.map(
                (attribute) => ATTRIBUTE_NAMES[attribute.attribute],
              ),
            )}
            's rates are indicative only, and shaded in Figure {figure}.
          </li>
        )}
        {tooFew ? (
          <li>
            Too few: fewer than {bias.minIdentities} identities stand behind the
            rate, so it is not estimated.
          </li>
        ) : null}
        <li>
          Worst-to-best FPIR: the highest group FPIR over the lowest; a dash
          where fewer than two groups have one, or the lowest is 0.
        </li>
      </Notes>
    </>
  );
}

function GroupTable({
  breakdown,
  table,
}: {
  readonly breakdown: BiasBreakdown;
  readonly table: number;
}) {
  return (
    <Table>
      <NumberedCaption number={table}>
        Rates by group for {breakdown.model.name} under best photo. CelebA test
        draw, at its single frozen threshold,{" "}
        {formatThreshold(breakdown.threshold)}, frozen on the validation draw.
        TPIR and misidentification are over the mated probes of each group's
        gallery identities, FPIR over the non-mated probes of its held-out
        identities. Rates in percent with 95% identity-level bootstrap intervals
        in brackets.
      </NumberedCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Group</TableHead>
          <TableHead className={numberClass}>Gallery identities</TableHead>
          <TableHead className={numberClass}>Mated probes</TableHead>
          <TableHead className={numberClass}>TPIR</TableHead>
          <TableHead className={numberClass}>Misidentification</TableHead>
          <TableHead className={numberClass}>Held-out identities</TableHead>
          <TableHead className={numberClass}>Non-mated probes</TableHead>
          <TableHead className={numberClass}>FPIR</TableHead>
        </TableRow>
      </TableHeader>
      {breakdown.attributes.map((attribute) => (
        <TableBody key={attribute.attribute}>
          <TableRow className="hover:bg-transparent">
            <th
              scope="rowgroup"
              colSpan={COLUMNS}
              className="px-3 pt-4 pb-1.5 text-left font-normal text-muted-foreground italic"
            >
              {qualifiers(attribute)}
            </th>
          </TableRow>
          {attribute.groups.map((group) => (
            <GroupRow key={group.label} group={group} />
          ))}
        </TableBody>
      ))}
    </Table>
  );
}

function GroupRow({ group }: { readonly group: GroupRates }) {
  return (
    <TableRow>
      <RowHeader className="pl-6">{group.label}</RowHeader>
      <TableCell className={numberClass}>
        {formatCount(group.galleryIdentities)}
      </TableCell>
      <TableCell className={numberClass}>
        {formatCount(group.matedProbes)}
      </TableCell>
      <TableCell className={numberClass}>
        <Estimate rate={group.tpir} />
      </TableCell>
      <TableCell className={numberClass}>
        <Estimate rate={group.misidentification} />
      </TableCell>
      <TableCell className={numberClass}>
        {formatCount(group.heldOutIdentities)}
      </TableCell>
      <TableCell className={numberClass}>
        {formatCount(group.nonMatedProbes)}
      </TableCell>
      <TableCell className={numberClass}>
        <Estimate rate={group.fpir} />
      </TableCell>
    </TableRow>
  );
}
