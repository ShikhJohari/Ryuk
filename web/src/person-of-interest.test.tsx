import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import type { PersonOfInterest } from "./api/persons";
import { healthy, mockService, problemResponse } from "./test/api-server";
import { gate } from "./test/gate";
import { enrolledPhoto, personOfInterest, summary } from "./test/persons";
import { renderAt } from "./test/render";

/** The service's copy of Ada, which the handlers change as the page acts. */
let ada: PersonOfInterest;

beforeEach(() => {
  ada = personOfInterest("ada", "Ada Lovelace", ["p1"]);
});

const server = mockService(
  healthy,
  http.get("*/api/persons/ada", () => HttpResponse.json(ada)),
  http.get("*/api/persons", () => HttpResponse.json([summary(ada)])),
  http.get("*/api/persons/nobody", () =>
    problemResponse({
      type: "about:blank",
      title: "Not Found",
      status: 404,
      detail: "No person of interest has that ID.",
      code: "not_found",
    }),
  ),
);

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

  it("shows the not-found page for someone who does not exist", async () => {
    renderAt("/watchlist/nobody");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Nothing here" }),
    ).toBeInTheDocument();
  });
});
