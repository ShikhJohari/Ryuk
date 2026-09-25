import { QueryClient } from "@tanstack/react-query";
import { createMemoryHistory } from "@tanstack/react-router";
import { render, screen, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { App } from "./app";
import { createAppRouter } from "./router";

const server = setupServer(
  http.get("*/api/health", () =>
    HttpResponse.json({ status: "ok", version: "0.1.0" }),
  ),
);

beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => {
  server.resetHandlers();
});
afterAll(() => {
  server.close();
});

function renderAt(path: string) {
  const queryClient = new QueryClient();
  const router = createAppRouter({
    queryClient,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(<App queryClient={queryClient} router={router} />);
  return router;
}

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
    server.use(
      http.get("*/api/health", () =>
        HttpResponse.json(
          {
            type: "about:blank",
            title: "Service Unavailable",
            status: 503,
            detail: "The service is starting up.",
            code: "unavailable",
          },
          {
            status: 503,
            headers: { "content-type": "application/problem+json" },
          },
        ),
      ),
    );

    renderAt("/");

    expect(await screen.findByText("Service unreachable")).toBeInTheDocument();
    expect(screen.queryByText("Service connected")).not.toBeInTheDocument();
  });

  it("keeps the parent section active on a detail route", async () => {
    renderAt("/watchlist/42");

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Person of interest",
      }),
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
