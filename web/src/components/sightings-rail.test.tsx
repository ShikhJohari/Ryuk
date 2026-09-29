import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithRouter } from "@/test/render";
import { ada, grace, sightingSummary } from "@/test/sightings";
import { SightingsRail } from "./sightings-rail";

async function rail() {
  return screen.findByRole("region", { name: "Sightings" });
}

describe("SightingsRail", () => {
  it("lists sightings in the order given, each linked to its detail", async () => {
    renderWithRouter(
      <SightingsRail
        sightings={[
          sightingSummary("s2", grace, {
            startedAt: "2026-09-27T11:30:00Z",
            endedAt: null,
            bestScore: 0.8,
          }),
          sightingSummary("s1", ada),
        ]}
        highlighted={new Set()}
      />,
    );

    const items = within(await rail()).getAllByRole("listitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Grace Hopper 11:30:00 · Open best match score 0.800",
      "Ada Lovelace 10:00:05 · Ended best match score 0.874",
    ]);
    const [newest] = items;
    expect(within(newest as HTMLElement).getByRole("link")).toHaveAttribute(
      "href",
      "/sightings/s2",
    );
    expect(
      within(newest as HTMLElement).getByRole("presentation"),
    ).toHaveAttribute("src", "/api/sightings/s2/crop?score=0.8");
  });

  it("highlights a sighting opened while the page was showing", async () => {
    renderWithRouter(
      <SightingsRail
        sightings={[
          sightingSummary("s2", grace, { endedAt: null }),
          sightingSummary("s1", ada),
        ]}
        highlighted={new Set(["s2"])}
      />,
    );

    const [opened, earlier] = within(await rail()).getAllByRole("listitem");
    expect(opened).toHaveAttribute("data-highlighted");
    expect(opened).toHaveClass(
      "motion-safe:animate-sighting-in",
      "motion-reduce:animate-sighting-highlight",
    );
    expect(earlier).not.toHaveAttribute("data-highlighted");
    expect(earlier).not.toHaveClass("motion-safe:animate-sighting-in");
  });

  it("says so when nobody has been sighted, and links to every sighting", async () => {
    renderWithRouter(<SightingsRail sightings={[]} highlighted={new Set()} />);

    const region = await rail();
    expect(region).toHaveTextContent("No sightings yet.");
    expect(within(region).queryByRole("list")).not.toBeInTheDocument();
    expect(
      within(region).getByRole("link", { name: "All sightings" }),
    ).toHaveAttribute("href", "/sightings");
  });
});
