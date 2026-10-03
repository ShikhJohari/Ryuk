import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Sighting } from "@/api/sightings";
import { facenet, sface } from "@/test/monitor";
import { renderWithRouter } from "@/test/render";
import { ada, grace, sighting } from "@/test/sightings";
import { SightingDetail } from "./sighting-detail";

const models = [sface, facenet];

/** Each reading's value, by its term. */
async function readings(of: Sighting): Promise<Record<string, HTMLElement>> {
  renderWithRouter(<SightingDetail sighting={of} models={models} />);
  const terms = await screen.findAllByRole("term");
  return Object.fromEntries(
    terms.map((term) => [
      term.textContent ?? "",
      term.nextElementSibling as HTMLElement,
    ]),
  );
}

describe("SightingDetail", () => {
  it("shows the best face crop", async () => {
    renderWithRouter(
      <SightingDetail sighting={sighting("s1")} models={models} />,
    );

    const crop = await screen.findByRole("img", {
      name: "Ada Lovelace's face at the best match, match score 0.874",
    });
    expect(crop).toHaveAttribute("src", "/api/sightings/s1/crop?score=0.874");
    expect(screen.getByText("Figure 1.")).toBeInTheDocument();
  });

  it("reads out the match, the runner-up, the model, the threshold and the times", async () => {
    const values = await readings(sighting("s1"));

    expect(
      Object.fromEntries(
        Object.entries(values).map(([term, value]) => [
          term,
          value.textContent,
        ]),
      ),
    ).toEqual({
      "Person of interest": "Ada Lovelace",
      "Best match score": "0.874",
      "Runner-up": "Grace Hopper match score 0.612",
      "Recognition model": "SFace",
      Threshold: "0.498",
      Started: "27 Sept 2026, 10:00:05",
      "Last seen": "27 Sept 2026, 10:00:17",
      Ended: "27 Sept 2026, 10:00:17",
    });
    expect(
      within(values["Person of interest"] as HTMLElement).getByRole("link"),
    ).toHaveAttribute("href", "/watchlist/ada");
    expect(
      within(values["Runner-up"] as HTMLElement).getByRole("link"),
    ).toHaveAttribute("href", "/watchlist/grace");
  });

  it("says a sighting is still open", async () => {
    const values = await readings(sighting("s1", ada, { endedAt: null }));

    expect(values.Ended).toHaveTextContent("Still open");
  });

  it("keeps a purged runner-up's score, and nothing else of them", async () => {
    const values = await readings(
      sighting("s1", ada, { runnerUp: { person: null, score: 0.612 } }),
    );

    expect(values["Runner-up"]).toHaveTextContent("Purged match score 0.612");
    expect(
      within(values["Runner-up"] as HTMLElement).queryByRole("link"),
    ).not.toBeInTheDocument();
  });

  it("says when nobody else was on the watchlist to rank second", async () => {
    const values = await readings(sighting("s1", ada, { runnerUp: null }));

    expect(values["Runner-up"]).toHaveTextContent(
      "None Nobody else was on the watchlist.",
    );
  });

  it("says who has since been removed from the watchlist", async () => {
    const values = await readings(
      sighting("s1", ada, {
        person: { ...ada, status: "removed" },
        runnerUp: { person: { ...grace, status: "removed" }, score: 0.612 },
      }),
    );

    expect(values["Person of interest"]).toHaveTextContent(
      "Ada Lovelace Removed from the watchlist",
    );
    expect(values["Runner-up"]).toHaveTextContent(
      "Grace Hopper match score 0.612 Removed from the watchlist",
    );
  });

  it("names a model the service no longer knows by its key", async () => {
    const values = await readings(
      sighting("s1", ada, { modelKey: "sface-cpu-old" }),
    );

    expect(values["Recognition model"]).toHaveTextContent("sface-cpu-old");
  });
});
