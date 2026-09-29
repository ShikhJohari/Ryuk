import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import type { RecognitionModelInfo } from "@/api/models";
import { formatDateTime, formatScore, formatTime } from "@/lib/format";
import { PersonOfInterestLink, SightingState } from "./sighting-labels";
import {
  modelNameOf,
  type SightingSummary,
  sightingCropSrc,
} from "./sighting-types";
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

/** Sightings in the order given, one row each, linked to their detail. */
export function SightingsTable({
  sightings,
  models,
  caption,
  empty,
  showPerson = true,
}: SightingsTableProps) {
  return (
    <Table>
      <TableCaption>{caption}</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead className="w-16">
            <span className="sr-only">Crop</span>
          </TableHead>
          {showPerson ? <TableHead>Person of interest</TableHead> : null}
          <TableHead>Started</TableHead>
          <TableHead>Last seen</TableHead>
          <TableHead>State</TableHead>
          <TableHead className="text-right">Best score</TableHead>
          <TableHead>Model</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sightings.length === 0 ? (
          <TableRow>
            <TableCell
              colSpan={showPerson ? 7 : 6}
              className="py-6 text-muted-foreground"
            >
              {empty}
            </TableCell>
          </TableRow>
        ) : (
          sightings.map((sighting) => (
            <SightingRow
              key={sighting.id}
              sighting={sighting}
              models={models}
              showPerson={showPerson}
            />
          ))
        )}
      </TableBody>
    </Table>
  );
}

/** One sighting in a `SightingsTable`. */
export function SightingRow({
  sighting,
  models,
  showPerson,
}: {
  readonly sighting: SightingSummary;
  readonly models: ReadonlyArray<RecognitionModelInfo>;
  readonly showPerson: boolean;
}) {
  const modelName = modelNameOf(models, sighting.modelKey);
  return (
    <TableRow>
      <TableCell>
        <img
          src={sightingCropSrc(sighting)}
          alt=""
          loading="lazy"
          className="size-10 rounded-sm object-cover"
        />
      </TableCell>
      {showPerson ? (
        <TableCell>
          <PersonOfInterestLink person={sighting.person} />
        </TableCell>
      ) : null}
      <TableCell>
        <Link
          to="/sightings/$sightingId"
          params={{ sightingId: sighting.id }}
          className="text-ink underline underline-offset-2"
        >
          <time dateTime={sighting.startedAt}>
            {formatDateTime(sighting.startedAt)}
          </time>
        </Link>
      </TableCell>
      <TableCell>
        <time dateTime={sighting.lastSeenAt}>
          {formatTime(sighting.lastSeenAt)}
        </time>
      </TableCell>
      <TableCell>
        <SightingState endedAt={sighting.endedAt} />
      </TableCell>
      <TableCell className="text-right">
        {formatScore(sighting.bestScore)}
      </TableCell>
      <TableCell>
        {modelName === sighting.modelKey ? (
          // A key is long; the whole of it is kept for copying.
          <span className="block max-w-[16ch] truncate" title={modelName}>
            {modelName}
          </span>
        ) : (
          modelName
        )}
      </TableCell>
    </TableRow>
  );
}
