import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";

export const Route = createFileRoute("/evaluation")({
  component: EvaluationPage,
});

function EvaluationPage() {
  return (
    <PageHeader
      title="Evaluation"
      description="Results on LFW and CelebA, and the state of each recognition model."
    />
  );
}
