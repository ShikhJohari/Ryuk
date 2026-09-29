import { Link } from "@tanstack/react-router";
import { useId } from "react";
import { type SightingSummary, sightingCropSrc } from "@/api/sightings";
import { formatScore, formatTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import { SightingState } from "./sighting-labels";

type SightingsRailProps = {
  /** Newest first, as the service pages them. */
  readonly sightings: ReadonlyArray<SightingSummary>;
  /** Sightings opened while the page was showing: each slides in highlighted. */
  readonly highlighted: ReadonlySet<string>;
  /** Sightings whose person of interest is matched in the latest frame. */
  readonly inView?: ReadonlySet<string>;
  /** Shown in place of the sightings, such as while they load. */
  readonly message?: string;
};

/** The recent sightings beside the live monitor, each linked to its detail. */
export function SightingsRail({
  sightings,
  highlighted,
  inView = new Set(),
  message,
}: SightingsRailProps) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-4">
        <h2 id={headingId} className="section-label">
          Sightings
        </h2>
        <Link
          to="/sightings"
          className="text-muted-foreground text-sm hover:text-ink"
        >
          All sightings
        </Link>
      </div>
      {message !== undefined ? (
        <p className="border-rule border-y py-4 text-muted-foreground">
          {message}
        </p>
      ) : sightings.length === 0 ? (
        <p className="border-rule border-y py-4 text-muted-foreground">
          No sightings yet. One opens here once a person of interest is matched
          steadily.
        </p>
      ) : (
        <ol className="flex flex-col border-rule border-y">
          {sightings.map((sighting) => (
            <SightingsRailItem
              key={sighting.id}
              sighting={sighting}
              highlighted={highlighted.has(sighting.id)}
              inView={inView.has(sighting.id)}
            />
          ))}
        </ol>
      )}
    </section>
  );
}

/**
 * One sighting in the rail. A highlighted one slides in and fades from the
 * highlight colour once, as it mounts; with reduced motion it only fades.
 */
export function SightingsRailItem({
  sighting,
  highlighted,
  inView = false,
}: {
  readonly sighting: SightingSummary;
  readonly highlighted: boolean;
  readonly inView?: boolean;
}) {
  return (
    <li
      data-highlighted={highlighted ? "" : undefined}
      className={cn(
        "border-rule border-b last:border-b-0",
        highlighted &&
          "motion-safe:animate-sighting-in motion-reduce:animate-sighting-highlight",
      )}
    >
      <Link
        to="/sightings/$sightingId"
        params={{ sightingId: sighting.id }}
        className="flex items-center gap-3 px-1 py-2 hover:bg-highlight"
      >
        <img
          src={sightingCropSrc(sighting)}
          alt=""
          className="size-12 shrink-0 rounded-sm object-cover"
        />
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate font-medium text-ink">
            {sighting.person.name}
          </span>{" "}
          <span className="text-sm">
            <time dateTime={sighting.startedAt} className="text-ink">
              {formatTime(sighting.startedAt)}
            </time>
            {" · "}
            <SightingState endedAt={sighting.endedAt} />
            {inView ? " · in view" : null}
          </span>
        </span>{" "}
        <span className="font-serif text-[18px] text-ink">
          <span className="sr-only">best match score </span>
          {formatScore(sighting.bestScore)}
        </span>
      </Link>
    </li>
  );
}
