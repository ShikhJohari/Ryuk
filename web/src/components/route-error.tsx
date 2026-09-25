import { type ErrorComponentProps, useRouter } from "@tanstack/react-router";
import { PageHeader } from "./page-header";
import { Button } from "./ui/button";

/** Default error boundary for every route. */
export function RouteError({ reset }: ErrorComponentProps) {
  const router = useRouter();

  return (
    <section className="flex flex-col items-start gap-6">
      <PageHeader
        label="Error"
        title="This page could not be loaded"
        description="Something went wrong while loading it. Trying again usually helps; if it keeps happening, check that the service is running."
      />
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
