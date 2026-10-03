import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { Schema } from "effect";
import { useState } from "react";
import { photoImageUrl, StatusFilter } from "@/api/persons";
import { personsQueryOptions } from "@/api/persons.queries";
import { EnrollDialog } from "@/components/enroll-dialog";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDate } from "@/lib/format";

const DEFAULT_STATUS: StatusFilter = "on_watchlist";

const filters: ReadonlyArray<{
  readonly status: StatusFilter;
  readonly label: string;
  readonly caption: string;
}> = [
  {
    status: "on_watchlist",
    label: "On watchlist",
    caption: "Persons of interest on the watchlist.",
  },
  {
    status: "removed",
    label: "Removed",
    caption: "Persons of interest removed from the watchlist.",
  },
  { status: "all", label: "All", caption: "Every person of interest." },
];

const statusLabels = { on_watchlist: "On watchlist", removed: "Removed" };

export const Route = createFileRoute("/watchlist/")({
  // Undefined rather than left out: the router passes on any search key a
  // route does not set, as it arrived.
  validateSearch: (
    search: Record<string, unknown>,
  ): { readonly status?: StatusFilter | undefined } => ({
    status: Schema.is(StatusFilter)(search.status) ? search.status : undefined,
  }),
  loaderDeps: ({ search }) => ({ status: search.status ?? DEFAULT_STATUS }),
  loader: ({ context, deps }) =>
    context.queryClient.ensureQueryData(personsQueryOptions(deps.status)),
  component: WatchlistPage,
});

function WatchlistPage() {
  const status = Route.useSearch().status ?? DEFAULT_STATUS;
  const persons = useSuspenseQuery(personsQueryOptions(status)).data;
  const [enrolling, setEnrolling] = useState(false);
  const filter = filters.find((f) => f.status === status) ?? filters[0];

  return (
    <section className="flex flex-col gap-8">
      <div className="flex items-end justify-between gap-6">
        <PageHeader
          title="Watchlist"
          description="The persons of interest the live monitor compares faces against."
        />
        <Button onClick={() => setEnrolling(true)}>Enroll</Button>
      </div>

      <nav aria-label="Status" className="flex gap-6 border-rule border-b">
        {filters.map((f) => (
          <Link
            key={f.status}
            to="/watchlist"
            search={f.status === DEFAULT_STATUS ? {} : { status: f.status }}
            aria-current={f.status === status ? "page" : undefined}
            className={
              f.status === status
                ? "-mb-px border-ink border-b-2 pb-2 font-semibold text-ink"
                : "-mb-px border-transparent border-b-2 pb-2 text-muted-foreground hover:text-ink"
            }
          >
            {f.label}
          </Link>
        ))}
      </nav>

      <Table>
        <TableCaption>
          <span className="text-ink">Table 1.</span> {filter?.caption}
        </TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead className="w-16">
              <span className="sr-only">Photo</span>
            </TableHead>
            <TableHead>Name</TableHead>
            <TableHead className="text-right">Photos</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Enrolled</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {persons.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="py-6 text-muted-foreground">
                {status === "removed"
                  ? "Nobody has been removed from the watchlist."
                  : "Nobody is on the watchlist yet. Enroll someone to start."}
              </TableCell>
            </TableRow>
          ) : (
            persons.map((person) => (
              <TableRow key={person.id}>
                <TableCell>
                  <img
                    src={photoImageUrl(person.id, person.coverPhotoId)}
                    alt=""
                    className="size-10 rounded-sm object-cover"
                  />
                </TableCell>
                <TableCell>
                  <Link
                    to="/watchlist/$personId"
                    params={{ personId: person.id }}
                    className="font-medium text-ink underline-offset-2 hover:underline"
                  >
                    {person.name}
                  </Link>
                </TableCell>
                <TableCell className="text-right">
                  {person.photoCount}
                </TableCell>
                <TableCell>{statusLabels[person.status]}</TableCell>
                <TableCell>{formatDate(person.createdAt)}</TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      {enrolling ? <EnrollDialog onClose={() => setEnrolling(false)} /> : null}
    </section>
  );
}
