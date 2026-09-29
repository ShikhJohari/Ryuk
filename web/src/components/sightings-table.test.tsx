import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { facenet, sface } from "@/test/monitor";
import { renderWithRouter } from "@/test/render";
import { ada, grace, sightingSummary } from "@/test/sightings";
import { SightingsTable } from "./sightings-table";

const models = [sface, facenet];

function cellsOf(row: HTMLElement) {
  return within(row)
    .getAllByRole("cell")
    .map((cell) => cell.textContent);
}

describe("SightingsTable", () => {
  it("lists each sighting with its person, times, state, best score and model", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings, newest first."
        empty="No sightings yet."
        models={models}
        sightings={[
          sightingSummary("s2", grace, {
            modelKey: facenet.id,
            threshold: 0.709,
            startedAt: "2026-09-27T11:30:00Z",
            lastSeenAt: "2026-09-27T11:30:04Z",
            endedAt: null,
            bestScore: 0.8,
          }),
          sightingSummary("s1", ada),
        ]}
      />,
    );

    const table = await screen.findByRole("table", {
      name: "Sightings, newest first.",
    });
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((th) => th.textContent),
    ).toEqual([
      "Crop",
      "Person of interest",
      "Started",
      "Last seen",
      "State",
      "Best score",
      "Model",
    ]);
    const [, open, ended] = within(table).getAllByRole("row");
    if (open === undefined || ended === undefined) {
      throw new Error("expected two sightings");
    }
    expect(cellsOf(open)).toEqual([
      "",
      "Grace Hopper",
      "27 Sept 2026, 11:30:00",
      "11:30:04",
      "Open",
      "0.800",
      "FaceNet",
    ]);
    expect(cellsOf(ended)).toEqual([
      "",
      "Ada Lovelace",
      "27 Sept 2026, 10:00:05",
      "10:00:17",
      "Ended",
      "0.874",
      "SFace",
    ]);
  });

  it("links each sighting to its detail and its person of interest", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings."
        empty="No sightings yet."
        models={models}
        sightings={[sightingSummary("s1", ada)]}
      />,
    );

    expect(
      await screen.findByRole("link", { name: "27 Sept 2026, 10:00:05" }),
    ).toHaveAttribute("href", "/sightings/s1");
    expect(screen.getByRole("link", { name: "Ada Lovelace" })).toHaveAttribute(
      "href",
      "/watchlist/ada",
    );
  });

  it("shows the best face crop, fetched again when a higher score replaces it", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings."
        empty="No sightings yet."
        models={models}
        sightings={[sightingSummary("s/1", ada, { bestScore: 0.9 })]}
      />,
    );

    const table = await screen.findByRole("table");
    // Decorative beside the name, as on the watchlist.
    const [crop] = within(table).getAllByRole("presentation");
    expect(crop).toHaveAttribute("src", "/api/sightings/s%2F1/crop?score=0.9");
  });

  it("says when a person of interest has been removed from the watchlist", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings."
        empty="No sightings yet."
        models={models}
        sightings={[sightingSummary("s1", { ...ada, status: "removed" })]}
      />,
    );

    const person = await screen.findByRole("link", { name: "Ada Lovelace" });
    expect(person.closest("td")).toHaveTextContent(
      "Ada Lovelace Removed from the watchlist",
    );
  });

  it("names a model the service no longer knows by its key", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings."
        empty="No sightings yet."
        models={models}
        sightings={[sightingSummary("s1", ada, { modelKey: "sface-cpu-old" })]}
      />,
    );

    expect(await screen.findByText("sface-cpu-old")).toBeInTheDocument();
  });

  it("leaves out the person on a person of interest's own page", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Ada Lovelace's sightings."
        empty="Ada Lovelace has not been sighted."
        models={models}
        showPerson={false}
        sightings={[sightingSummary("s1", ada)]}
      />,
    );

    const table = await screen.findByRole("table");
    expect(
      within(table).queryByRole("columnheader", {
        name: "Person of interest",
      }),
    ).not.toBeInTheDocument();
    expect(
      within(table).queryByRole("link", { name: "Ada Lovelace" }),
    ).not.toBeInTheDocument();
  });

  it("says so when there are no sightings", async () => {
    renderWithRouter(
      <SightingsTable
        caption="Sightings."
        empty="No sightings yet."
        models={models}
        sightings={[]}
      />,
    );

    const table = await screen.findByRole("table");
    expect(within(table).getByRole("cell")).toHaveTextContent(
      "No sightings yet.",
    );
  });
});
