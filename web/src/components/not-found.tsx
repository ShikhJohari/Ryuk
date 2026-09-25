import { Link } from "@tanstack/react-router";
import { PageHeader } from "./page-header";
import { buttonVariants } from "./ui/button";

/** Shown for any URL that matches no route. */
export function NotFound() {
  return (
    <section className="flex flex-col items-start gap-6">
      <PageHeader
        label="Not found"
        title="Nothing here"
        description="No page lives at this address."
      />
      <Link to="/monitor" className={buttonVariants({ variant: "secondary" })}>
        Go to the Live monitor
      </Link>
    </section>
  );
}
