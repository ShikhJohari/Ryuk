import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/watchlist/")({
  component: WatchlistPage,
});

function WatchlistPage() {
  return (
    <PageHeader
      title="Watchlist"
      description="A ruled table of the persons of interest, filtered by status, where new ones are enrolled from a dialog."
    />
  );
}
