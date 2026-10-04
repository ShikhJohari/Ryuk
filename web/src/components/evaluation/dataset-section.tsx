import type {
  CelebaDataset,
  DatasetSummary,
  DetectionCounts,
} from "@/api/evaluation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatCount } from "./format";
import {
  EvaluationSection,
  Lede,
  NumberedCaption,
  numberClass,
  RowHeader,
} from "./layout";

/** How many tables the section numbers. */
export const DATASET_TABLES = 2;

const DRAW_NAMES: Readonly<Record<CelebaDataset["draw"], string>> = {
  validation: "CelebA validation draw",
  test: "CelebA test draw",
};

/** The datasets as the EDA summarised them: counts, each share of images worked out here. */
export function DatasetSection({
  dataset,
  table,
}: {
  readonly dataset: DatasetSummary;
  /** This section's first table number. */
  readonly table: number;
}) {
  const rows: ReadonlyArray<{
    readonly name: string;
    readonly identities: number;
    readonly images: number;
    readonly detection: DetectionCounts;
    readonly gallery: CelebaDataset | null;
  }> = [
    {
      name: "LFW",
      identities: dataset.lfw.identities,
      images: dataset.lfw.images,
      detection: dataset.lfw.detection,
      gallery: null,
    },
    ...dataset.celeba.map((draw) => ({
      name: `${DRAW_NAMES[draw.draw]} (${draw.split} split)`,
      identities: draw.identities,
      images: draw.images,
      detection: draw.detection,
      gallery: draw,
    })),
  ];
  return (
    <EvaluationSection title="Datasets">
      <Lede>
        LFW measures verification, one face against another. CelebA's two draws
        rehearse the watchlist: the validation draw sets every threshold, the
        test draw reports the results.
      </Lede>
      <Table>
        <NumberedCaption number={table}>
          The datasets, from images to usable faces. A usable face is at least{" "}
          {dataset.minUsableFaceSize} px on the detection box's short side; an
          image without one is excluded from evaluation. A CelebA identity is a
          gallery candidate with at least {dataset.minGalleryImages} images, and
          eligible for a gallery with at least {dataset.minGalleryImages} usable
          ones. Shares are of the images.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Dataset</TableHead>
            <TableHead className={numberClass}>Identities</TableHead>
            <TableHead className={numberClass}>Images</TableHead>
            <TableHead className={numberClass}>Face found</TableHead>
            <TableHead className={numberClass}>More than one face</TableHead>
            <TableHead className={numberClass}>Usable face</TableHead>
            <TableHead className={numberClass}>Gallery candidates</TableHead>
            <TableHead className={numberClass}>Eligible identities</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.name}>
              <RowHeader>{row.name}</RowHeader>
              <TableCell className={numberClass}>
                {formatCount(row.identities)}
              </TableCell>
              <TableCell className={numberClass}>
                {formatCount(row.images)}
              </TableCell>
              <TableCell className={numberClass}>
                {share(row.detection.detected, row.detection.images)}
              </TableCell>
              <TableCell className={numberClass}>
                {share(row.detection.multipleFaces, row.detection.images)}
              </TableCell>
              <TableCell className={numberClass}>
                {share(row.detection.usable, row.detection.images)}
              </TableCell>
              <TableCell className={numberClass}>
                {row.gallery === null
                  ? "—"
                  : formatCount(row.gallery.galleryCandidates)}
              </TableCell>
              <TableCell className={numberClass}>
                {row.gallery === null
                  ? "—"
                  : formatCount(row.gallery.eligibleIdentities)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Table>
        <NumberedCaption number={table + 1}>
          LFW's pairs lists. View 1's lists are for choosing a pipeline, View
          2's for reporting it. A pair with an image that has no usable face is
          not scored.
        </NumberedCaption>
        <TableHeader>
          <TableRow>
            <TableHead>List</TableHead>
            <TableHead className={numberClass}>View</TableHead>
            <TableHead className={numberClass}>Folds</TableHead>
            <TableHead className={numberClass}>Matched</TableHead>
            <TableHead className={numberClass}>Mismatched</TableHead>
            <TableHead className={numberClass}>Identities</TableHead>
            <TableHead className={numberClass}>Images</TableHead>
            <TableHead className={numberClass}>Pairs not scored</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {dataset.lfw.pairs.map((list) => (
            <TableRow key={list.name}>
              <RowHeader>
                <code>{list.name}</code>
              </RowHeader>
              {(
                [
                  ["view", list.view],
                  ["folds", list.folds],
                  ["matched", list.matched],
                  ["mismatched", list.mismatched],
                  ["identities", list.identities],
                  ["images", list.images],
                  ["not-scored", list.pairsWithExcludedImage],
                ] as const
              ).map(([column, value]) => (
                <TableCell key={column} className={numberClass}>
                  {formatCount(value)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </EvaluationSection>
  );
}

/** A count and its share of `of`: "13,161 (99.5%)". */
function share(count: number, of: number): string {
  const percent = of === 0 ? "—" : `${((count / of) * 100).toFixed(1)}%`;
  return `${formatCount(count)} (${percent})`;
}
