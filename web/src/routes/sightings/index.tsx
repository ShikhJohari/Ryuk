import {
  useSuspenseInfiniteQuery,
  useSuspenseQuery,
} from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Schema } from "effect";
import { useId } from "react";
import { modelsQueryOptions } from "@/api/models.queries";
import { personsQueryOptions } from "@/api/persons.queries";
import { sightingsQueryOptions } from "@/api/sightings.queries";
import { PageHeader } from "@/components/page-header";
import { SightingsTable } from "@/components/sightings-table";
import { Button } from "@/components/ui/button";
import { problemMessage } from "@/lib/problems";

/** One person of interest's sightings: their ID, as the service gives it. */
const PersonFilter = Schema.NonEmptyString;

export const Route = createFileRoute("/sightings/")({
  // Undefined rather than left out: the router passes on any search key a
  // route does not set, as it arrived.
  validateSearch: (
    search: Record<string, unknown>,
  ): { readonly personId?: string | undefined } => ({
    personId: Schema.is(PersonFilter)(search.personId)
      ? search.personId
      : undefined,
  }),
  loaderDeps: ({ search }) => ({ personId: search.personId }),
  loader: ({ context, deps }) =>
    Promise.all([
      context.queryClient.ensureInfiniteQueryData(
        sightingsQueryOptions(deps.personId),
      ),
      context.queryClient.ensureQueryData(modelsQueryOptions),
      // Everyone ever enrolled and not purged, to filter by.
      context.queryClient.ensureQueryData(personsQueryOptions("all")),
    ]),
  component: SightingsPage,
});

function SightingsPage() {
  const { personId } = Route.useSearch();
  const navigate = Route.useNavigate();
  const filterId = useId();
  const history = useSuspenseInfiniteQuery(sightingsQueryOptions(personId));
  const models = useSuspenseQuery(modelsQueryOptions).data;
  const persons = useSuspenseQuery(personsQueryOptions("all")).data;
  const sightings = history.data.pages.flatMap((page) => page.items);
  const name =
    personId === undefined
      ? undefined
      : persons.find((person) => person.id === personId)?.name;

  return (
    <section className="flex flex-col gap-8">
      <PageHeader
        title="Sightings"
        description="Every person of interest the live monitor saw steadily, newest first, with the face crop of their best match."
      />

      <div className="flex flex-col gap-1 self-start">
        <label htmlFor={filterId} className="section-label">
          Person of interest
        </label>
        <select
          id={filterId}
          value={personId ?? ""}
          onChange={(event) =>
            void navigate({
              search:
                event.target.value === ""
                  ? {}
                  : { personId: event.target.value },
            })
          }
          className="h-11 min-w-[240px] rounded-[22px] border border-ink bg-transparent px-4 text-ink"
        >
          <option value="">Everyone</option>
          {persons.map((person) => (
            <option key={person.id} value={person.id}>
              {person.status === "removed"
                ? `${person.name} (removed)`
                : person.name}
            </option>
          ))}
          {personId !== undefined && name === undefined ? (
            // Purged, or never here: the service answers with no sightings.
            <option value={personId}>Nobody on record</option>
          ) : null}
        </select>
      </div>

      <SightingsTable
        sightings={sightings}
        models={models}
        caption={
          <>
            <span className="text-ink">Table 1.</span>{" "}
            {personId === undefined
              ? "Every sighting, newest first."
              : name === undefined
                ? "Sightings of nobody on record."
                : `${name}'s sightings, newest first.`}
          </>
        }
        empty={
          personId === undefined
            ? "No sightings yet. One opens on the live monitor once a person of interest is matched steadily."
            : name === undefined
              ? "No person of interest has this ID; they may have been purged."
              : `${name} has not been sighted.`
        }
      />

      {history.isFetchNextPageError ? (
        <p role="alert" className="text-destructive">
          {problemMessage(history.error)}
        </p>
      ) : null}
      {history.hasNextPage ? (
        <Button
          variant="secondary"
          className="self-start"
          disabled={history.isFetchingNextPage}
          onClick={() => void history.fetchNextPage()}
        >
          {history.isFetchingNextPage ? "Loading…" : "Load more"}
        </Button>
      ) : null}
    </section>
  );
}
