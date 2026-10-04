import { useSuspenseQuery } from "@tanstack/react-query";
import {
  createFileRoute,
  type ErrorComponentProps,
  useRouter,
} from "@tanstack/react-router";
import { useEffect } from "react";
import { evaluationQueryOptions } from "@/api/evaluation.queries";
import { modelsQueryOptions } from "@/api/models.queries";
import { BiasSection } from "@/components/evaluation/bias-section";
import { DatasetSection } from "@/components/evaluation/dataset-section";
import { IdentificationSection } from "@/components/evaluation/identification-section";
import { LearningSection } from "@/components/evaluation/learning-section";
import { LfwSection } from "@/components/evaluation/lfw-section";
import { LiveSection } from "@/components/evaluation/live-section";
import { ModelsSection } from "@/components/evaluation/models-section";
import { sectionNumbers } from "@/components/evaluation/numbering";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { problemMessage } from "@/lib/problems";

const TITLE = "Evaluation";
const DESCRIPTION =
  "Results on LFW and CelebA, and the state of each recognition model.";

export const Route = createFileRoute("/evaluation")({
  loader: ({ context }) =>
    Promise.all([
      context.queryClient.ensureQueryData(evaluationQueryOptions),
      context.queryClient.ensureQueryData(modelsQueryOptions),
    ]),
  errorComponent: EvaluationError,
  component: EvaluationPage,
});

/**
 * The committed evaluation as tables and figures, in the order of the
 * report's Section 5 (#32). Read-only: the active model is switched in the
 * live monitor alone (#20).
 */
function EvaluationPage() {
  const report = useSuspenseQuery(evaluationQueryOptions).data;
  const models = useSuspenseQuery(modelsQueryOptions).data;
  const numbers = sectionNumbers(report);
  const folds =
    report.dataset.lfw.pairs.find((list) => list.name === "pairs")?.folds ??
    null;
  const firstActiveId = report.firstActiveModel?.model?.id ?? null;

  return (
    <section className="flex flex-col gap-12">
      <PageHeader title={TITLE} description={DESCRIPTION} />
      <ModelsSection
        models={models}
        first={report.firstActiveModel}
        table={numbers.models.table}
      />
      <DatasetSection dataset={report.dataset} table={numbers.dataset.table} />
      <LfwSection
        verification={report.verification}
        folds={folds}
        table={numbers.lfw.table}
        figure={numbers.lfw.figure}
      />
      <IdentificationSection
        identification={report.identification}
        firstActiveId={firstActiveId}
        lfwTable={numbers.lfw.table}
        table={numbers.identification.table}
        figure={numbers.identification.figure}
      />
      <LearningSection
        learning={report.learning}
        table={numbers.learning.table}
      />
      <BiasSection
        bias={report.bias}
        identification={report.identification}
        table={numbers.bias.table}
        figure={numbers.bias.figure}
      />
      <LiveSection live={report.live} table={numbers.live.table} />
    </section>
  );
}

/** Why the evaluation could not be shown, such as a service started without it. */
function EvaluationError({ error, reset }: ErrorComponentProps) {
  const router = useRouter();

  // The operator sees the client's copy; the developer console keeps the cause.
  useEffect(() => {
    console.error("Evaluation failed to load", error);
  }, [error]);

  return (
    <section className="flex flex-col items-start gap-6">
      <PageHeader title={TITLE} description={DESCRIPTION} />
      <p role="alert" className="max-w-[64ch] text-destructive">
        {problemMessage(error)}
      </p>
      <Button
        onClick={() => {
          reset();
          void router.invalidate();
        }}
      >
        Try again
      </Button>
    </section>
  );
}
