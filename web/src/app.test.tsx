import { screen, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";
import { healthy, mockService, unavailable } from "./test/api-server";
import { sface } from "./test/monitor";
import { personOfInterest } from "./test/persons";
import { renderAt } from "./test/render";
import { sightingsService } from "./test/sightings";

const server = mockService(
  healthy,
  // Nobody has been sighted.
  ...sightingsService().handlers,
  http.get("*/api/models", () => HttpResponse.json([sface])),
  http.get("*/api/persons/42", () =>
    HttpResponse.json(personOfInterest("42", "Ada Lovelace")),
  ),
);

describe("app shell", () => {
  it("redirects / to the Live monitor and shows the service as connected", async () => {
    const router = renderAt("/");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Live monitor" }),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/monitor");

    const nav = screen.getByRole("navigation", { name: "Sections" });
    for (const section of [
      "Live monitor",
      "Watchlist",
      "Sightings",
      "Evaluation",
    ]) {
      expect(
        within(nav).getByRole("link", { name: section }),
      ).toBeInTheDocument();
    }
    expect(
      within(nav).getByRole("link", { name: "Live monitor" }),
    ).toHaveAttribute("aria-current", "page");

    expect(await screen.findByText("Service connected")).toBeInTheDocument();
    expect(screen.getByText("v0.1.0")).toBeInTheDocument();
  });

  it("shows the service as unreachable when health returns a problem", async () => {
    server.use(unavailable);

    renderAt("/");

    expect(await screen.findByText("Service unreachable")).toBeInTheDocument();
    expect(screen.queryByText("Service connected")).not.toBeInTheDocument();
  });

  it("keeps the parent section active on a detail route", async () => {
    renderAt("/watchlist/42");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "Sections" });
    expect(
      within(nav).getByRole("link", { name: "Watchlist" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(nav).getByRole("link", { name: "Live monitor" }),
    ).not.toHaveAttribute("aria-current");
  });

  it("renders the not-found page inside the shell for an unknown URL", async () => {
    renderAt("/no-such-page");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Nothing here" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Sections" }),
    ).toBeInTheDocument();
  });
});
