import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { modelsQueryOptions } from "@/api/models.queries";
import type { Sighting } from "@/api/sightings";
import { sightingQueryOptions } from "@/api/sightings.queries";
import { PageHeader } from "@/components/page-header";
import { SightingDetail } from "@/components/sighting-detail";
import { formatDateTime, formatTime } from "@/lib/format";
import { orNotFound } from "@/lib/route-loading";

export const Route = createFileRoute("/sightings/$sightingId")({
  loader: ({ context, params }) =>
    // Never there, or purged with its person of interest.
    orNotFound(
      Promise.all([
        context.queryClient.ensureQueryData(
          sightingQueryOptions(params.sightingId),
        ),
        context.queryClient.ensureQueryData(modelsQueryOptions),
      ]),
    ),
  component: SightingPage,
});

function SightingPage() {
  const { sightingId } = Route.useParams();
  const sighting = useSuspenseQuery(sightingQueryOptions(sightingId)).data;
  const models = useSuspenseQuery(modelsQueryOptions).data;

  return (
    <section className="flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <Link
          to="/sightings"
          className="text-muted-foreground text-sm hover:text-ink"
        >
          ← Sightings
        </Link>
        <PageHeader
          label="Sighting"
          title={sighting.person.name}
          description={span(sighting)}
        />
      </div>
      <SightingDetail sighting={sighting} models={models} />
    </section>
  );
}

function span(sighting: Sighting): string {
  return sighting.endedAt === null
    ? `Open since ${formatDateTime(sighting.startedAt)}.`
    : `Seen from ${formatDateTime(sighting.startedAt)} to ${formatTime(sighting.endedAt)}.`;
}
