import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/sightings/")({
  component: SightingsPage,
});

function SightingsPage() {
  return (
    <PageHeader
      title="Sightings"
      description="The history of sightings, filterable by person of interest."
    />
  );
}
