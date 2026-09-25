import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/monitor")({
  component: LiveMonitorPage,
});

function LiveMonitorPage() {
  return (
    <PageHeader
      title="Live monitor"
      description="Your webcam with a box on every face, beside the rail of sightings as they open."
    />
  );
}
