import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { modelNameOf, type RecognitionModelInfo } from "@/api/models";
import { type SightingSummary, sightingCropSrc } from "@/api/sightings";
import { formatDateTime, formatScore, formatTime } from "@/lib/format";
import { PersonOfInterestLink, SightingState } from "./sighting-labels";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";

type SightingsTableProps = {
  readonly sightings: ReadonlyArray<SightingSummary>;
  /** Every recognition model the service knows, to name each sighting's. */
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly caption: ReactNode;
  /** Shown in place of the rows when there are none. */
  readonly empty: string;
  /** False on a person of interest's own page, where every sighting is theirs. */
  readonly showPerson?: boolean;
};

/** A column: its heading, and each sighting's cell. */
type Column = {
  readonly key: string;
  readonly heading: ReactNode;
  readonly headClassName?: string;
  readonly cellClassName?: string;
  readonly cell: (
    sighting: SightingSummary,
    models: ReadonlyArray<RecognitionModelInfo>,
  ) => ReactNode;
};

const cropColumn: Column = {
  key: "crop",
  heading: <span className="sr-only">Crop</span>,
  headClassName: "w-16",
  cell: (sighting) => (
    <img
      src={sightingCropSrc(sighting)}
      alt=""
      loading="lazy"
      className="size-10 rounded-sm object-cover"
    />
  ),
};

const personColumn: Column = {
  key: "person",
  heading: "Person of interest",
  cell: (sighting) => <PersonOfInterestLink person={sighting.person} />,
};

const detailColumns: ReadonlyArray<Column> = [
  {
    key: "started",
    heading: "Started",
    cell: (sighting) => (
      <Link
        to="/sightings/$sightingId"
        params={{ sightingId: sighting.id }}
        className="text-ink underline underline-offset-2"
      >
        <time dateTime={sighting.startedAt}>
          {formatDateTime(sighting.startedAt)}
        </time>
      </Link>
    ),
  },
  {
    key: "last-seen",
    heading: "Last seen",
    cell: (sighting) => (
      <time dateTime={sighting.lastSeenAt}>
        {formatTime(sighting.lastSeenAt)}
      </time>
    ),
  },
  {
    key: "state",
    heading: "State",
    cell: (sighting) => <SightingState endedAt={sighting.endedAt} />,
  },
  {
    key: "best-score",
    heading: "Best match score",
    headClassName: "text-right",
    cellClassName: "text-right",
    cell: (sighting) => formatScore(sighting.bestScore),
  },
  {
    key: "model",
    heading: "Model",
    cell: (sighting, models) => {
      const modelName = modelNameOf(models, sighting.modelKey);
      return modelName === sighting.modelKey ? (
        // A key is long; the whole of it is kept for copying.
        <span className="block max-w-[16ch] truncate" title={modelName}>
          {modelName}
        </span>
      ) : (
        modelName
      );
    },
  },
];

/** Sightings in the order given, one row each, linked to their detail. */
export function SightingsTable({
  sightings,
  models,
  caption,
  empty,
  showPerson = true,
}: SightingsTableProps) {
  const columns = [
    cropColumn,
    ...(showPerson ? [personColumn] : []),
    ...detailColumns,
  ];
  return (
    <Table>
      <TableCaption>{caption}</TableCaption>
      <TableHeader>
        <TableRow>
          {columns.map((column) => (
            <TableHead key={column.key} className={column.headClassName}>
              {column.heading}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sightings.length === 0 ? (
          <TableRow>
            <TableCell
              colSpan={columns.length}
              className="py-6 text-muted-foreground"
            >
              {empty}
            </TableCell>
          </TableRow>
        ) : (
          sightings.map((sighting) => (
            <TableRow key={sighting.id}>
              {columns.map((column) => (
                <TableCell key={column.key} className={column.cellClassName}>
                  {column.cell(sighting, models)}
                </TableCell>
              ))}
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );
}
