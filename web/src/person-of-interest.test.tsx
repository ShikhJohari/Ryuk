import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import type { PersonOfInterest } from "./api/persons";
import { healthy, mockService, problemResponse } from "./test/api-server";
import { gate } from "./test/gate";
import { sface } from "./test/monitor";
import { enrolledPhoto, personOfInterest, summary } from "./test/persons";
import { renderAt } from "./test/render";
import {
  ada as adaSighted,
  grace as graceSighted,
  sighting,
  sightingsService,
} from "./test/sightings";

/** The service's copy of Ada, which the handlers change as the page acts. */
let ada: PersonOfInterest;

/** Whether the service has purged Ada. */
let purged: boolean;

const sightings = sightingsService();

beforeEach(() => {
  ada = personOfInterest("ada", "Ada Lovelace", ["p1"]);
  purged = false;
  sightings.reset();
});

const server = mockService(
  healthy,
  http.get("*/api/models", () => HttpResponse.json([sface])),
  ...sightings.handlers,
  http.get("*/api/persons/ada", () =>
    purged ? notFoundResponse() : HttpResponse.json(ada),
  ),
  http.get("*/api/persons", ({ request }) => {
    const status = new URL(request.url).searchParams.get("status") ?? "all";
    return HttpResponse.json(
      purged || (status !== "all" && status !== ada.status)
        ? []
        : [summary(ada)],
    );
  }),
  http.get("*/api/persons/nobody", () => notFoundResponse()),
);

function notFoundResponse() {
  return problemResponse({
    type: "about:blank",
    title: "Not Found",
    status: 404,
    detail: "No person of interest has that ID.",
    code: "not_found",
  });
}

/** Changes Ada's status as the service does, recording each PATCH body. */
function statusChanges() {
  const changes: Array<unknown> = [];
  server.use(
    http.patch("*/api/persons/ada", async ({ request }) => {
      const body = (await request.json()) as {
        status: PersonOfInterest["status"];
      };
      changes.push(body);
      ada = {
        ...ada,
        status: body.status,
        statusChangedAt: "2026-09-28T12:00:00Z",
      };
      return HttpResponse.json(ada);
    }),
  );
  return changes;
}

/** The toasts' status region. */
function toasts() {
  return screen
    .getAllByRole("status")
    .find((region) => region.closest("header") === null) as HTMLElement;
}

function chooseFile() {
  fireEvent.change(screen.getByLabelText("Add photo"), {
    target: {
      files: [new File(["jpeg"], "ada-2.jpg", { type: "image/jpeg" })],
    },
  });
}

/** Adds each posted photo to Ada unless `respond` answers otherwise first. */
function photoUploads(respond?: (attempt: number) => Response | undefined) {
  const acknowledged: unknown[][] = [];
  server.use(
    http.post("*/api/persons/ada/photos", async ({ request }) => {
      const form = await request.formData();
      acknowledged.push(form.getAll("acknowledgedWarnings"));
      const answer = respond?.(acknowledged.length);
      if (answer !== undefined) {
        return answer;
      }
      const added = enrolledPhoto(`p${ada.photos.length + 1}`);
      ada = { ...ada, photos: [...ada.photos, added] };
      return HttpResponse.json(added, { status: 201 });
    }),
  );
  return acknowledged;
}

describe("person of interest", () => {
  it("shows their name and enrolled photos", async () => {
    ada = personOfInterest("ada", "Ada Lovelace", ["p1", "p2"]);
    renderAt("/watchlist/ada");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    const photos = within(
      screen.getByRole("region", { name: "Enrolled photos" }),
    ).getAllByRole("img");
    expect(photos.map((photo) => photo.getAttribute("src"))).toEqual([
      "/api/persons/ada/photos/p1/image",
      "/api/persons/ada/photos/p2/image",
    ]);
    expect(screen.getByText("Figure 2.")).toBeInTheDocument();
  });

  it("never offers to delete the last photo", async () => {
    renderAt("/watchlist/ada");

    expect(
      await screen.findByRole("button", { name: "Delete photo 1" }),
    ).toBeDisabled();
    expect(
      screen.getByText(/last enrolled photo cannot be deleted/),
    ).toBeInTheDocument();
  });

  it("deletes one of several photos", async () => {
    ada = personOfInterest("ada", "Ada Lovelace", ["p1", "p2"]);
    const deleted: string[] = [];
    server.use(
      http.delete("*/api/persons/ada/photos/:photoId", ({ params }) => {
        deleted.push(String(params.photoId));
        ada = {
          ...ada,
          photos: ada.photos.filter((p) => p.id !== params.photoId),
        };
        return new HttpResponse(null, { status: 204 });
      }),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(
      await screen.findByRole("button", { name: "Delete photo 1" }),
    );

    await waitFor(() => {
      expect(screen.getAllByRole("img")).toHaveLength(1);
    });
    expect(deleted).toEqual(["p1"]);
    expect(
      screen.getByRole("button", { name: "Delete photo 1" }),
    ).toBeDisabled();
  });

  it("renames them", async () => {
    const renamed: unknown[] = [];
    server.use(
      http.patch("*/api/persons/ada", async ({ request }) => {
        const body = (await request.json()) as { name: string };
        renamed.push(body);
        ada = { ...ada, name: body.name };
        return HttpResponse.json(ada);
      }),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(await screen.findByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name"), {
      target: { value: "Augusta Ada King" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Augusta Ada King",
      }),
    ).toBeInTheDocument();
    expect(renamed).toEqual([{ name: "Augusta Ada King" }]);
    expect(screen.queryByLabelText("New name")).not.toBeInTheDocument();
  });

  it("cannot cancel a rename in flight", async () => {
    const held = gate();
    const renamed: unknown[] = [];
    server.use(
      http.patch("*/api/persons/ada", async ({ request }) => {
        const body = (await request.json()) as { name: string };
        renamed.push(body);
        await held.opened;
        ada = { ...ada, name: body.name };
        return HttpResponse.json(ada);
      }),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(await screen.findByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name"), {
      target: { value: "Augusta Ada King" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(renamed).toHaveLength(1));

    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByLabelText("New name")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByLabelText("New name")).toBeInTheDocument();

    held.open();
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Augusta Ada King",
      }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("New name")).not.toBeInTheDocument();
  });

  it("adds a photo", async () => {
    const acknowledged = photoUploads();
    renderAt("/watchlist/ada");
    await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" });

    chooseFile();

    await waitFor(() => {
      expect(screen.getAllByRole("img")).toHaveLength(2);
    });
    expect(acknowledged).toEqual([[]]);
  });

  it("adds a photo that may not be them once the warning is acknowledged", async () => {
    const acknowledged = photoUploads((attempt) =>
      attempt === 1
        ? HttpResponse.json(
            {
              type: "about:blank",
              title: "Conflict",
              status: 409,
              detail: "This photo may not be Ada Lovelace.",
              code: "warnings",
              warnings: [
                {
                  code: "may_not_be_same_person",
                  detail: "This photo may not be Ada Lovelace.",
                  personId: null,
                },
              ],
            },
            {
              status: 409,
              headers: { "content-type": "application/problem+json" },
            },
          )
        : undefined,
    );
    renderAt("/watchlist/ada");
    await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" });

    chooseFile();
    const warnings = await screen.findByRole("dialog", {
      name: "Check before you continue",
    });
    expect(
      within(warnings).getByText("This photo may not be Ada Lovelace."),
    ).toBeInTheDocument();
    expect(within(warnings).queryByRole("link")).not.toBeInTheDocument();
    fireEvent.click(
      within(warnings).getByRole("button", { name: "Add anyway" }),
    );

    await waitFor(() => {
      expect(screen.getAllByRole("img")).toHaveLength(2);
    });
    expect(acknowledged).toEqual([[], ["may_not_be_same_person"]]);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("says why a photo was rejected", async () => {
    photoUploads(() =>
      problemResponse({
        type: "about:blank",
        title: "Unprocessable Content",
        status: 422,
        detail: "The photo has 2 faces large enough to use.",
        code: "multiple_faces",
      }),
    );
    renderAt("/watchlist/ada");
    await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" });

    chooseFile();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The photo has more than one face large enough to use.",
    );
  });

  it("says why a rename was refused, and keeps the form", async () => {
    server.use(
      http.patch("*/api/persons/ada", () =>
        problemResponse({
          type: "about:blank",
          title: "Unprocessable Content",
          status: 422,
          detail: "A name can be at most 200 characters.",
          code: "invalid_name",
        }),
      ),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(await screen.findByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("New name"), {
      target: { value: "Augusta Ada King" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter a name of at most 200 characters.",
    );
    expect(screen.getByLabelText("New name")).toBeEnabled();
    expect(
      screen.getByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
  });

  it("says why a photo could not be deleted, beside that photo", async () => {
    ada = personOfInterest("ada", "Ada Lovelace", ["p1", "p2"]);
    server.use(
      http.delete("*/api/persons/ada/photos/:photoId", () =>
        problemResponse({
          type: "about:blank",
          title: "Conflict",
          status: 409,
          detail:
            "A person of interest's last enrolled photo cannot be deleted.",
          code: "last_photo",
        }),
      ),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(
      await screen.findByRole("button", { name: "Delete photo 2" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/last enrolled photo cannot be deleted/);
    const [, second] = screen.getAllByRole("figure");
    expect(second).toContainElement(alert);
    expect(screen.getAllByRole("img")).toHaveLength(2);
  });

  it("says when the service could not be reached", async () => {
    photoUploads(() => HttpResponse.error());
    renderAt("/watchlist/ada");
    await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" });

    chooseFile();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The service could not be reached.",
    );
  });

  it("shows their sightings, linked to all of them", async () => {
    sightings.reset([
      sighting("s1", adaSighted, { runnerUp: null }),
      sighting("s2", graceSighted),
    ]);
    renderAt("/watchlist/ada");

    const section = await screen.findByRole("region", { name: "Sightings" });
    const table = await within(section).findByRole("table", {
      name: "Table 1. Ada Lovelace's sightings, newest first.",
    });
    expect(
      within(table).getAllByRole("link", { name: /27 Sept 2026/ }),
    ).toHaveLength(1);
    expect(
      within(table).queryByRole("columnheader", { name: "Person of interest" }),
    ).not.toBeInTheDocument();
    expect(
      within(section).getByRole("link", { name: "All their sightings" }),
    ).toHaveAttribute("href", "/sightings?personId=ada");
  });

  it("removes them from the watchlist, and undoes it from the toast", async () => {
    const changes = statusChanges();
    renderAt("/watchlist/ada");

    fireEvent.click(
      await screen.findByRole("button", { name: "Remove from watchlist" }),
    );

    expect(
      await screen.findByText("Removed from the watchlist on 28 Sept 2026."),
    ).toBeInTheDocument();
    expect(toasts()).toHaveTextContent("Removed Ada Lovelace.");
    fireEvent.click(within(toasts()).getByRole("button", { name: "Undo" }));

    expect(
      await screen.findByText("On the watchlist since 28 Sept 2026."),
    ).toBeInTheDocument();
    expect(changes).toEqual([
      { status: "removed" },
      { status: "on_watchlist" },
    ]);
    expect(toasts()).toBeEmptyDOMElement();
  });

  it("undoes a removal after the page has gone", async () => {
    const changes = statusChanges();
    renderAt("/watchlist/ada");
    fireEvent.click(
      await screen.findByRole("button", { name: "Remove from watchlist" }),
    );
    await screen.findByText("Removed from the watchlist on 28 Sept 2026.");

    fireEvent.click(screen.getByRole("link", { name: "← Watchlist" }));
    expect(
      await screen.findByText(
        "Nobody is on the watchlist yet. Enroll someone to start.",
      ),
    ).toBeInTheDocument();
    fireEvent.click(within(toasts()).getByRole("button", { name: "Undo" }));

    // The watchlist is fetched again, with Ada back on it.
    expect(
      await screen.findByRole("link", { name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    expect(changes).toEqual([
      { status: "removed" },
      { status: "on_watchlist" },
    ]);
  });

  it("says why an undo failed, wherever the operator is", async () => {
    statusChanges();
    renderAt("/watchlist/ada");
    fireEvent.click(
      await screen.findByRole("button", { name: "Remove from watchlist" }),
    );
    await screen.findByText("Removed from the watchlist on 28 Sept 2026.");
    server.use(http.patch("*/api/persons/ada", () => notFoundResponse()));

    fireEvent.click(within(toasts()).getByRole("button", { name: "Undo" }));

    await waitFor(() =>
      expect(toasts()).toHaveTextContent(
        "Ada Lovelace could not be put back on the watchlist. It is no longer there",
      ),
    );
  });

  it("restores someone removed", async () => {
    ada = { ...ada, status: "removed" };
    const changes = statusChanges();
    renderAt("/watchlist/ada");

    fireEvent.click(
      await screen.findByRole("button", { name: "Restore to watchlist" }),
    );

    expect(
      await screen.findByText("On the watchlist since 28 Sept 2026."),
    ).toBeInTheDocument();
    expect(changes).toEqual([{ status: "on_watchlist" }]);
    expect(
      screen.getByRole("button", { name: "Remove from watchlist" }),
    ).toBeInTheDocument();
  });

  it("says why a removal was refused", async () => {
    server.use(http.patch("*/api/persons/ada", () => notFoundResponse()));
    renderAt("/watchlist/ada");

    fireEvent.click(
      await screen.findByRole("button", { name: "Remove from watchlist" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "It is no longer there",
    );
    expect(toasts()).toBeEmptyDOMElement();
  });

  it("purges them only once their name is typed, then shows the watchlist", async () => {
    const purges: Array<string> = [];
    const changes = statusChanges();
    server.use(
      http.delete("*/api/persons/ada", ({ request }) => {
        purges.push(request.url);
        purged = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const router = renderAt("/watchlist/ada");
    // Removed first, so an Undo is showing when the purge lands.
    fireEvent.click(
      await screen.findByRole("button", { name: "Remove from watchlist" }),
    );
    await screen.findByText("Removed from the watchlist on 28 Sept 2026.");

    fireEvent.click(screen.getByRole("button", { name: "Purge" }));
    const dialog = await screen.findByRole("dialog", {
      name: "Purge Ada Lovelace?",
    });
    const purge = within(dialog).getByRole("button", {
      name: "Purge permanently",
    });
    const name = within(dialog).getByLabelText("Type Ada Lovelace to confirm");
    expect(purge).toBeDisabled();
    fireEvent.change(name, { target: { value: "ada lovelace" } });
    expect(purge).toBeDisabled();
    fireEvent.change(name, { target: { value: "Ada Lovelace" } });
    fireEvent.click(purge);

    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/watchlist"),
    );
    expect(
      await screen.findByText(
        "Nobody is on the watchlist yet. Enroll someone to start.",
      ),
    ).toBeInTheDocument();
    expect(purges).toHaveLength(1);
    expect(toasts()).toHaveTextContent("Purged Ada Lovelace.");
    expect(
      within(toasts()).queryByRole("button", { name: "Undo" }),
    ).not.toBeInTheDocument();
    expect(changes).toEqual([{ status: "removed" }]);
  });

  it("says why a purge failed, and keeps the dialog open", async () => {
    server.use(
      http.delete("*/api/persons/ada", () =>
        problemResponse({
          type: "about:blank",
          title: "Service Unavailable",
          status: 503,
          detail: "The watchlist is not running.",
          code: "watchlist_unavailable",
        }),
      ),
    );
    renderAt("/watchlist/ada");

    fireEvent.click(await screen.findByRole("button", { name: "Purge" }));
    const dialog = await screen.findByRole("dialog", {
      name: "Purge Ada Lovelace?",
    });
    fireEvent.change(
      within(dialog).getByLabelText("Type Ada Lovelace to confirm"),
      { target: { value: "Ada Lovelace" } },
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Purge permanently" }),
    );

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "The watchlist is not running.",
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
  });

  it("shows the not-found page for someone who does not exist", async () => {
    renderAt("/watchlist/nobody");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Nothing here" }),
    ).toBeInTheDocument();
  });
});
