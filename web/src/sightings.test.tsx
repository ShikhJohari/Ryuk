import {
  cleanup,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { healthy, mockService } from "./test/api-server";
import { fakeCamera } from "./test/camera";
import { facenet, mockMonitor, sface } from "./test/monitor";
import { personOfInterest, summary } from "./test/persons";
import { renderAt } from "./test/render";
import {
  ada,
  grace,
  sighting,
  sightingsService,
  summaryOf,
} from "./test/sightings";

const sightings = sightingsService({ pageSize: 2 });

const first = sighting("s1", ada, {
  startedAt: "2026-09-27T09:00:00Z",
  lastSeenAt: "2026-09-27T09:00:12Z",
  endedAt: "2026-09-27T09:00:12Z",
});
const second = sighting("s2", grace, {
  modelKey: facenet.id,
  threshold: 0.709,
  startedAt: "2026-09-27T10:00:00Z",
  lastSeenAt: "2026-09-27T10:00:04Z",
  endedAt: "2026-09-27T10:00:04Z",
  bestScore: 0.8,
  runnerUp: { person: ada, score: 0.55 },
});
const third = sighting("s3", ada, {
  startedAt: "2026-09-27T11:00:00Z",
  lastSeenAt: "2026-09-27T11:00:30Z",
  endedAt: null,
  bestScore: 0.91,
});

beforeEach(() => {
  sightings.reset([first, second, third]);
});

const server = mockService(
  healthy,
  http.get("*/api/models", () => HttpResponse.json([sface, facenet])),
  http.get("*/api/persons", () =>
    HttpResponse.json(
      [
        personOfInterest("ada", "Ada Lovelace"),
        personOfInterest("grace", "Grace Hopper"),
      ].map(summary),
    ),
  ),
  ...sightings.handlers,
);

/** Each row of the history: the person and when the sighting started. */
async function rows() {
  const table = await screen.findByRole("table");
  return within(table)
    .getAllByRole("row")
    .slice(1)
    .map((row) =>
      within(row)
        .getAllByRole("cell")
        .slice(1, 3)
        .map((cell) => cell.textContent)
        .join(" "),
    );
}

describe("sightings history", () => {
  it("lists every sighting newest first, a page at a time", async () => {
    renderAt("/sightings");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Sightings" }),
    ).toBeInTheDocument();
    await waitFor(async () =>
      expect(await rows()).toEqual([
        "Ada Lovelace 27 Sept 2026, 11:00:00",
        "Grace Hopper 27 Sept 2026, 10:00:00",
      ]),
    );
    expect(
      screen.getByRole("table", {
        name: "Table 1. Every sighting, newest first.",
      }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));

    await waitFor(async () =>
      expect(await rows()).toEqual([
        "Ada Lovelace 27 Sept 2026, 11:00:00",
        "Grace Hopper 27 Sept 2026, 10:00:00",
        "Ada Lovelace 27 Sept 2026, 09:00:00",
      ]),
    );
    expect(
      screen.queryByRole("button", { name: "Load more" }),
    ).not.toBeInTheDocument();
    expect(sightings.requests.at(-1)?.get("cursor")).toBe("after-s2");
  });

  it("filters by person of interest, kept in the address", async () => {
    const router = renderAt("/sightings");

    const filter = await screen.findByLabelText("Person of interest");
    expect(
      within(filter)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["Everyone", "Ada Lovelace", "Grace Hopper"]);
    fireEvent.change(filter, { target: { value: "ada" } });

    await waitFor(() =>
      expect(router.state.location.search).toEqual({ personId: "ada" }),
    );
    await waitFor(async () =>
      expect(await rows()).toEqual([
        "Ada Lovelace 27 Sept 2026, 11:00:00",
        "Ada Lovelace 27 Sept 2026, 09:00:00",
      ]),
    );
    expect(
      screen.getByRole("table", {
        name: "Table 1. Ada Lovelace's sightings, newest first.",
      }),
    ).toBeInTheDocument();
    expect(sightings.requests.at(-1)?.get("personId")).toBe("ada");

    fireEvent.change(screen.getByLabelText("Person of interest"), {
      target: { value: "" },
    });
    await waitFor(() => expect(router.state.location.search).toEqual({}));
  });

  it.each([
    ["empty", "/sightings?personId="],
    ["not an ID", "/sightings?personId=42"],
  ])(
    "shows everyone's sightings for a person filter that is %s",
    async (_, path) => {
      const router = renderAt(path);

      expect(
        await screen.findByRole("table", {
          name: "Table 1. Every sighting, newest first.",
        }),
      ).toBeInTheDocument();
      expect(router.state.location.search).toEqual({});
      expect(screen.getByLabelText("Person of interest")).toHaveValue("");
      expect(sightings.requests.at(-1)?.has("personId")).toBe(false);
    },
  );

  it("says so when a person of interest has not been sighted", async () => {
    sightings.reset([second]);
    renderAt("/sightings?personId=ada");

    expect(
      await screen.findByText("Ada Lovelace has not been sighted."),
    ).toBeInTheDocument();
  });

  it("agrees with the rail on the live monitor, which a reload keeps", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    // A page as long as the service's, so a reload loads the whole rail.
    sightings.reset([first], { pageSize: 50 });
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    const railOf = async () =>
      within(await screen.findByRole("region", { name: "Sightings" }))
        .getAllByRole("listitem")
        .map((item) => within(item).getByRole("link").getAttribute("href"));
    await waitFor(async () =>
      expect(await railOf()).toEqual(["/sightings/s1"]),
    );

    // Grace was confirmed from an earlier frame than Ada, though after her.
    const ada2 = sighting("s5", ada, {
      startedAt: "2026-09-27T12:00:02Z",
      lastSeenAt: "2026-09-27T12:00:02Z",
      endedAt: null,
    });
    const grace2 = sighting("s4", grace, {
      startedAt: "2026-09-27T12:00:01Z",
      lastSeenAt: "2026-09-27T12:00:03Z",
      endedAt: null,
    });
    for (const opened of [ada2, grace2]) {
      sightings.record(opened);
      monitor.send({ type: "sighting_opened", sighting: summaryOf(opened) });
    }
    const live = ["/sightings/s5", "/sightings/s4", "/sightings/s1"];
    await waitFor(async () => expect(await railOf()).toEqual(live));

    cleanup();
    renderAt("/monitor");
    await waitFor(async () => expect(await railOf()).toEqual(live));

    cleanup();
    renderAt("/sightings");
    await waitFor(async () => {
      const table = await screen.findByRole("table");
      expect(
        within(table)
          .getAllByRole("link", { name: /27 Sept 2026/ })
          .map((link) => link.getAttribute("href")),
      ).toEqual(live);
    });
  });
});

describe("sighting", () => {
  it("shows everything the sighting keeps", async () => {
    renderAt("/sightings/s2");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Grace Hopper" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Seen from 27 Sept 2026, 10:00:00 to 10:00:04."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: "Grace Hopper's face at the best match, match score 0.800",
      }),
    ).toHaveAttribute("src", "/api/sightings/s2/crop?score=0.8");
    const runnerUp = screen.getByText("Runner-up").nextElementSibling;
    expect(runnerUp).toHaveTextContent("Ada Lovelace match score 0.550");
    expect(
      screen.getByText("Recognition model").nextElementSibling,
    ).toHaveTextContent("FaceNet");
    expect(screen.getByText("Threshold").nextElementSibling).toHaveTextContent(
      "0.709",
    );
  });

  it("says an open sighting is still open", async () => {
    renderAt("/sightings/s3");

    expect(
      await screen.findByText("Open since 27 Sept 2026, 11:00:00."),
    ).toBeInTheDocument();
  });

  it("keeps a purged runner-up's match score, and nothing else of them", async () => {
    sightings.record({ ...second, runnerUp: { person: null, score: 0.55 } });
    renderAt("/sightings/s2");

    await screen.findByRole("heading", { level: 1, name: "Grace Hopper" });
    const runnerUp = screen.getByText("Runner-up").nextElementSibling;
    expect(runnerUp).toHaveTextContent("Purged match score 0.550");
    expect(
      within(runnerUp as HTMLElement).queryByRole("link"),
    ).not.toBeInTheDocument();
  });

  it("shows the not-found page for a sighting that does not exist, or was purged", async () => {
    renderAt("/sightings/nothing");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Nothing here" }),
    ).toBeInTheDocument();
  });
});
