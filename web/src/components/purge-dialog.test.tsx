import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiProblem } from "@/api/api-client";
import { namesMatch, PurgeDialog } from "./purge-dialog";

function renderDialog({
  name = "Ada Lovelace",
  pending = false,
  error = null,
}: {
  readonly name?: string;
  readonly pending?: boolean;
  readonly error?: Error | null;
} = {}) {
  const onConfirm = vi.fn();
  const onCancel = vi.fn();
  render(
    <PurgeDialog
      name={name}
      pending={pending}
      error={error}
      onConfirm={onConfirm}
      onCancel={onCancel}
    />,
  );
  const dialog = screen.getByRole("dialog", { name: `Purge ${name}?` });
  return {
    onConfirm,
    onCancel,
    dialog,
    input: within(dialog).getByLabelText(`Type ${name} to confirm`),
    // Renamed while the purge is in flight.
    get purge() {
      return within(dialog).getByRole("button", { name: "Purge permanently" });
    },
  };
}

function type(input: HTMLElement, text: string) {
  fireEvent.change(input, { target: { value: text } });
}

describe("PurgeDialog", () => {
  it("says what a purge erases, and that it cannot be undone", () => {
    const { dialog } = renderDialog();

    expect(dialog).toHaveAccessibleDescription(
      "Purging permanently erases their enrolled photos, embeddings and sightings, and clears them as runner-up on anyone else's sightings. It cannot be undone.",
    );
  });

  it("starts in the name field, with the purge refused", () => {
    const { input, purge } = renderDialog();

    expect(input).toHaveFocus();
    expect(purge).toBeDisabled();
  });

  it("purges only once the person's name is typed exactly", () => {
    const { input, purge, onConfirm } = renderDialog();

    for (const wrong of [
      "Ada",
      "ada lovelace",
      "Ada Lovelace.",
      "AdaLovelace",
    ]) {
      type(input, wrong);
      expect(purge).toBeDisabled();
    }
    type(input, "Ada Lovelace");
    expect(purge).toBeEnabled();
    fireEvent.click(purge);

    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("purges on Enter only once the name matches", () => {
    const { input, onConfirm } = renderDialog();

    type(input, "Ada");
    fireEvent.submit(input);
    expect(onConfirm).not.toHaveBeenCalled();

    type(input, "Ada Lovelace");
    fireEvent.submit(input);
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("goes back without purging", () => {
    const { dialog, onCancel, onConfirm } = renderDialog();

    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledOnce();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("holds everything while the purge is in flight", () => {
    const { dialog, input, onCancel } = renderDialog({ pending: true });

    expect(input).toBeDisabled();
    expect(
      within(dialog).getByRole("button", { name: "Cancel" }),
    ).toBeDisabled();
    expect(
      within(dialog).getByRole("button", { name: "Purging…" }),
    ).toBeDisabled();
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onCancel).not.toHaveBeenCalled();
    expect(dialog).toHaveAttribute("open");
  });

  it("says why a purge failed", () => {
    const { dialog } = renderDialog({
      error: new ApiProblem({
        type: "about:blank",
        title: "Not Found",
        status: 404,
        detail: "No person of interest has that ID.",
        code: "not_found",
      }),
    });

    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "It is no longer there",
    );
  });
});

describe("namesMatch", () => {
  it("forgives the space the service trims and collapses in a name", () => {
    expect(namesMatch("  Ada \t Lovelace ", "Ada Lovelace")).toBe(true);
  });

  it("is otherwise exact, case and punctuation included", () => {
    expect(namesMatch("ada lovelace", "Ada Lovelace")).toBe(false);
    expect(namesMatch("Ada-Lovelace", "Ada Lovelace")).toBe(false);
    expect(namesMatch("", "Ada Lovelace")).toBe(false);
  });

  it("matches however an accented letter was composed", () => {
    // "é" as one code point, and as "e" with a combining accent.
    expect(namesMatch("José", "José")).toBe(true);
    expect(namesMatch("José", "José")).toBe(true);
  });
});
