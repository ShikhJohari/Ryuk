import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";
import type { PersonOfInterest } from "./api/persons";
import { healthy, mockService, problemResponse } from "./test/api-server";
import { gate } from "./test/gate";
import { personOfInterest, summary } from "./test/persons";
import { renderAt } from "./test/render";

const ada = personOfInterest("ada", "Ada Lovelace");
const grace = personOfInterest("grace", "Grace Hopper", ["g1", "g2"]);

const server = mockService(
  healthy,
  http.get("*/api/persons", ({ request }) => {
    const status = new URL(request.url).searchParams.get("status");
    return HttpResponse.json(
      status === "removed" ? [] : [ada, grace].map(summary),
    );
  }),
);

function photoFile() {
  return new File(["jpeg bytes"], "ada.jpg", { type: "image/jpeg" });
}

async function openEnrollDialog() {
  const enroll = await screen.findByRole("button", { name: "Enroll" });
  // A real click focuses the button, which the dialog returns focus to.
  enroll.focus();
  fireEvent.click(enroll);
  const dialog = await screen.findByRole("dialog", {
    name: "Enroll a person of interest",
  });
  fireEvent.change(within(dialog).getByLabelText("Name"), {
    target: { value: "Ada Lovelace" },
  });
  fireEvent.change(within(dialog).getByLabelText("Photo"), {
    target: { files: [photoFile()] },
  });
  return dialog;
}

function warningsResponse() {
  return HttpResponse.json(
    {
      type: "about:blank",
      title: "Conflict",
      status: 409,
      detail: "A person of interest named Ada Lovelace already exists.",
      code: "warnings",
      warnings: [
        {
          code: "duplicate_name",
          detail: "A person of interest named Ada Lovelace already exists.",
          personId: "ada-1",
        },
      ],
    },
    {
      status: 409,
      headers: { "content-type": "application/problem+json" },
    },
  );
}

/** Answers enrollment with `respond`, recording each request's form fields. */
function enrollment(
  respond: (attempt: number) => Response | Promise<Response>,
): Array<{ name: unknown; photo: unknown; acknowledged: unknown[] }> {
  const seen: Array<{
    name: unknown;
    photo: unknown;
    acknowledged: unknown[];
  }> = [];
  server.use(
    http.post("*/api/persons", async ({ request }) => {
      const form = await request.formData();
      seen.push({
        name: form.get("name"),
        photo: form.get("photo"),
        acknowledged: form.getAll("acknowledgedWarnings"),
      });
      return respond(seen.length);
    }),
    http.get("*/api/persons/ada", () => HttpResponse.json(ada)),
  );
  return seen;
}

describe("watchlist", () => {
  it("lists the persons of interest on the watchlist in a ruled table", async () => {
    renderAt("/watchlist");

    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(3);
    expect(
      within(rows[1] as HTMLElement).getByRole("link", {
        name: "Ada Lovelace",
      }),
    ).toHaveAttribute("href", "/watchlist/ada");
    expect(within(rows[2] as HTMLElement).getByText("2")).toBeInTheDocument();
    expect(
      within(screen.getByRole("navigation", { name: "Status" })).getByRole(
        "link",
        { name: "On watchlist" },
      ),
    ).toHaveAttribute("aria-current", "page");
  });

  it("filters by status from the URL", async () => {
    const router = renderAt("/watchlist");

    fireEvent.click(await screen.findByRole("link", { name: "Removed" }));

    expect(
      await screen.findByText("Nobody has been removed from the watchlist."),
    ).toBeInTheDocument();
    expect(router.state.location.search).toEqual({ status: "removed" });
    expect(screen.getByRole("link", { name: "Removed" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("enrolls a person of interest and opens their page", async () => {
    const seen = enrollment(() => HttpResponse.json(ada, { status: 201 }));
    const router = renderAt("/watchlist");

    const dialog = await openEnrollDialog();
    fireEvent.click(within(dialog).getByRole("button", { name: "Enroll" }));

    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/watchlist/ada");
    expect(seen).toHaveLength(1);
    expect(seen[0]?.name).toBe("Ada Lovelace");
    expect(seen[0]?.photo).toBeInstanceOf(File);
    expect(seen[0]?.acknowledged).toEqual([]);
  });

  it("shows why a photo was rejected and keeps the dialog open", async () => {
    enrollment(() =>
      problemResponse({
        type: "about:blank",
        title: "Unprocessable Content",
        status: 422,
        detail: "No face was found in the photo.",
        code: "no_face",
      }),
    );
    renderAt("/watchlist");

    const dialog = await openEnrollDialog();
    fireEvent.click(within(dialog).getByRole("button", { name: "Enroll" }));

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "No face was found in the photo.",
    );
    expect(dialog).toBeInTheDocument();
  });

  it("lists warnings and resends with them acknowledged", async () => {
    const warned: PersonOfInterest = ada;
    const seen = enrollment((attempt) =>
      attempt === 1
        ? HttpResponse.json(
            {
              type: "about:blank",
              title: "Conflict",
              status: 409,
              detail: "A person of interest named Ada Lovelace already exists.",
              code: "warnings",
              warnings: [
                {
                  code: "duplicate_name",
                  detail:
                    "A person of interest named Ada Lovelace already exists.",
                  personId: "ada-1",
                },
                {
                  code: "looks_like_other",
                  detail: "This photo looks like Grace Hopper.",
                  personId: "grace",
                },
              ],
            },
            {
              status: 409,
              headers: { "content-type": "application/problem+json" },
            },
          )
        : HttpResponse.json(warned, { status: 201 }),
    );
    renderAt("/watchlist");

    const form = await openEnrollDialog();
    fireEvent.click(within(form).getByRole("button", { name: "Enroll" }));

    const warnings = await screen.findByRole("dialog", {
      name: "Check before you continue",
    });
    expect(within(warnings).getByText(/already exists/)).toBeInTheDocument();
    expect(
      within(warnings).getAllByRole("link", { name: "Open their record" })[1],
    ).toHaveAttribute("href", "/watchlist/grace");
    fireEvent.click(
      within(warnings).getByRole("button", { name: "Enroll anyway" }),
    );

    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    expect(seen.map((request) => request.acknowledged)).toEqual([
      [],
      ["duplicate_name", "looks_like_other"],
    ]);
  });

  it("goes back from the warnings to the enroll form", async () => {
    enrollment(() =>
      HttpResponse.json(
        {
          type: "about:blank",
          title: "Conflict",
          status: 409,
          detail: "dup",
          code: "warnings",
          warnings: [
            { code: "duplicate_name", detail: "dup", personId: "ada-1" },
          ],
        },
        {
          status: 409,
          headers: { "content-type": "application/problem+json" },
        },
      ),
    );
    renderAt("/watchlist");

    const form = await openEnrollDialog();
    fireEvent.click(within(form).getByRole("button", { name: "Enroll" }));
    fireEvent.click(await screen.findByRole("button", { name: "Go back" }));

    expect(
      await screen.findByRole("dialog", {
        name: "Enroll a person of interest",
      }),
    ).toBeInTheDocument();
  });

  it("keeps the enroll dialog, every control disabled, while the upload is in flight", async () => {
    const held = gate();
    const seen = enrollment(async () => {
      await held.opened;
      return HttpResponse.json(ada, { status: 201 });
    });
    const router = renderAt("/watchlist");

    const dialog = await openEnrollDialog();
    fireEvent.click(within(dialog).getByRole("button", { name: "Enroll" }));
    await waitFor(() => expect(seen).toHaveLength(1));

    const cancel = within(dialog).getByRole("button", { name: "Cancel" });
    expect(cancel).toBeDisabled();
    expect(within(dialog).getByLabelText("Photo")).toBeDisabled();
    expect(within(dialog).getByLabelText("Name")).toBeDisabled();
    expect(
      within(dialog).getByRole("button", { name: "Enrolling…" }),
    ).toBeDisabled();
    fireEvent.click(cancel);
    fireEvent.keyDown(dialog, { key: "Escape" });
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute("open");

    held.open();
    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/watchlist/ada");
    expect(seen).toHaveLength(1);
  });

  it("never navigates to the person when the operator has left mid-upload", async () => {
    const held = gate();
    const seen = enrollment(async () => {
      await held.opened;
      return HttpResponse.json(ada, { status: 201 });
    });
    server.use(http.get("*/api/models", () => HttpResponse.json([])));
    const router = renderAt("/watchlist");
    const dialog = await openEnrollDialog();
    fireEvent.click(within(dialog).getByRole("button", { name: "Enroll" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    const answered = new Promise((resolve) =>
      server.events.on("response:mocked", resolve),
    );

    // Browser back, say: the modal leaves nothing else on the page to click.
    await router.navigate({ to: "/monitor" });
    await screen.findByRole("heading", { level: 1, name: "Live monitor" });
    held.open();
    await answered;
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(router.state.location.pathname).toBe("/monitor");
    expect(seen).toHaveLength(1);
  });

  it("ignores Escape and Go back while Enroll anyway is in flight", async () => {
    const held = gate();
    const seen = enrollment(async (attempt) => {
      if (attempt === 1) {
        return warningsResponse();
      }
      await held.opened;
      return HttpResponse.json(ada, { status: 201 });
    });
    renderAt("/watchlist");

    const form = await openEnrollDialog();
    fireEvent.click(within(form).getByRole("button", { name: "Enroll" }));
    const warnings = await screen.findByRole("dialog", {
      name: "Check before you continue",
    });
    fireEvent.click(
      within(warnings).getByRole("button", { name: "Enroll anyway" }),
    );
    await waitFor(() => expect(seen).toHaveLength(2));

    expect(
      within(warnings).getByRole("button", { name: "Go back" }),
    ).toBeDisabled();
    fireEvent.keyDown(warnings, { key: "Escape" });
    expect(warnings).toBeInTheDocument();

    held.open();
    expect(
      await screen.findByRole("heading", { level: 1, name: "Ada Lovelace" }),
    ).toBeInTheDocument();
    expect(seen).toHaveLength(2);
  });

  it("moves focus into the warnings when they replace the form", async () => {
    enrollment(() => warningsResponse());
    renderAt("/watchlist");

    const form = await openEnrollDialog();
    fireEvent.click(within(form).getByRole("button", { name: "Enroll" }));
    const warnings = await screen.findByRole("dialog", {
      name: "Check before you continue",
    });

    expect(warnings).toContainElement(document.activeElement as HTMLElement);
  });

  it("closes the enroll dialog on Escape when nothing is in flight", async () => {
    renderAt("/watchlist");

    const dialog = await openEnrollDialog();
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(dialog).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enroll" })).toHaveFocus();
  });
});
