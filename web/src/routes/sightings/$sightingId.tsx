import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/sightings/$sightingId")({
  component: SightingPage,
});

function SightingPage() {
  return (
    <PageHeader
      title="Sighting"
      description="The best face crop, its match score, the runner-up, and the recognition model and threshold that produced it."
    />
  );
}
