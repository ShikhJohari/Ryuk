import { Link } from "@tanstack/react-router";
import type { SightingPerson } from "@/api/sightings";
import { formatScore } from "@/lib/format";

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

/** A match score under the name it belongs to. */
export function MatchScoreNote({ score }: { readonly score: number }) {
  return (
    <span className="block text-muted-foreground text-sm">
      match score {formatScore(score)}
    </span>
  );
}

/**
 * A person of interest's name, linked to their page, with a match score of
 * theirs when given, and whether they were removed.
 */
export function PersonOfInterestLink({
  person,
  score,
}: {
  readonly person: SightingPerson;
  readonly score?: number;
}) {
  return (
    <>
      <Link
        to="/watchlist/$personId"
        params={{ personId: person.id }}
        className="font-medium text-ink underline-offset-2 hover:underline"
      >
        {person.name}
      </Link>
      {score === undefined ? null : (
        <>
          {" "}
          <MatchScoreNote score={score} />
        </>
      )}
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
