import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import type { SightingPerson } from "./sighting-types";

/**
 * Whether a sighting is still open, in words; an open one is marked with a
 * dot too, in the match colour, so it stands out without reading.
 */
export function SightingState({
  endedAt,
}: {
  readonly endedAt: string | null;
}) {
  return endedAt === null ? (
    <span className="inline-flex items-center gap-1.5 font-medium text-match">
      <span aria-hidden="true" className="size-2 rounded-full bg-match" />
      Open
    </span>
  ) : (
    <span className="text-muted-foreground">Ended</span>
  );
}

/** A person of interest's name, linked to their page, and whether they were removed. */
export function PersonOfInterestLink({
  person,
  className,
}: {
  readonly person: SightingPerson;
  readonly className?: string;
}) {
  return (
    <>
      <Link
        to="/watchlist/$personId"
        params={{ personId: person.id }}
        className={cn(
          "font-medium text-ink underline-offset-2 hover:underline",
          className,
        )}
      >
        {person.name}
      </Link>
      {person.status === "removed" ? (
        <>
          {" "}
          <span className="block text-muted-foreground text-xs">
            Removed from the watchlist
          </span>
        </>
      ) : null}
    </>
  );
}
