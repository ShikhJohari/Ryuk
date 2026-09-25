import { Link, type LinkProps } from "@tanstack/react-router";
import { ServiceStatus } from "./service-status";

const sections: ReadonlyArray<{
  readonly to: LinkProps["to"];
  readonly label: string;
}> = [
  { to: "/monitor", label: "Live monitor" },
  { to: "/watchlist", label: "Watchlist" },
  { to: "/sightings", label: "Sightings" },
  { to: "/evaluation", label: "Evaluation" },
];

export function SiteHeader() {
  return (
    <header className="flex h-16 items-stretch gap-10 border-rule border-b px-14">
      <Link
        to="/"
        // "/" only redirects, so never mark the wordmark as the current page.
        activeOptions={{ exact: true }}
        className="self-center font-serif text-[26px] text-ink italic leading-none"
      >
        Ryuk
      </Link>
      <nav aria-label="Sections" className="flex items-stretch gap-7">
        {sections.map((section) => (
          // Fuzzy matching keeps a section active on its detail routes.
          <Link
            key={section.label}
            to={section.to}
            className="-mb-px flex items-center whitespace-nowrap border-b-2"
            activeProps={{ className: "border-ink font-semibold text-ink" }}
            inactiveProps={{
              className:
                "border-transparent text-muted-foreground hover:text-ink",
            }}
          >
            {section.label}
          </Link>
        ))}
      </nav>
      <div className="ml-auto self-center">
        <ServiceStatus />
      </div>
    </header>
  );
}
