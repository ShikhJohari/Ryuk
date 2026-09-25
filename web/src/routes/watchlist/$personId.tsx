import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/watchlist/$personId")({
  component: PersonOfInterestPage,
});

function PersonOfInterestPage() {
  return (
    <PageHeader
      title="Person of interest"
      description="Their enrolled photos and their sightings."
    />
  );
}
