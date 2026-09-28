import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Button } from "./button";
import { Dialog } from "./dialog";

/** A page with a button that opens a dialog, which Escape or Close closes. */
function Page({ dismissible = true }: { readonly dismissible?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button onClick={() => setOpen(true)}>Open</Button>
      {open ? (
        <Dialog
          title="Check this"
          description="Something to confirm."
          dismissible={dismissible}
          onClose={() => setOpen(false)}
        >
          <Button onClick={() => setOpen(false)}>Close</Button>
          <Button>Confirm</Button>
        </Dialog>
      ) : null}
    </>
  );
}

function open() {
  const opener = screen.getByRole("button", { name: "Open" });
  opener.focus();
  fireEvent.click(opener);
  return { opener, dialog: screen.getByRole("dialog", { name: "Check this" }) };
}

describe("Dialog", () => {
  it("opens modal, described, with focus on its first control", () => {
    render(<Page />);

    const { dialog } = open();

    expect(dialog).toHaveAttribute("open");
    expect(dialog).toHaveAccessibleDescription("Something to confirm.");
    expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  });

  it("returns focus to what opened it when it goes", () => {
    render(<Page />);
    const { opener } = open();

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("asks the page to close it on Escape", () => {
    render(<Page />);
    const { opener, dialog } = open();

    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("ignores Escape while it is not dismissible", () => {
    render(<Page dismissible={false} />);
    const { dialog } = open();

    fireEvent.keyDown(dialog, { key: "Escape" });
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute("open");
  });

  it("stays open if the browser closes it while the page still shows it", async () => {
    const onClose = vi.fn();
    render(
      <Dialog title="Pending" dismissible={false} onClose={onClose}>
        <Button>Wait</Button>
      </Dialog>,
    );
    const dialog = screen.getByRole("dialog", {
      name: "Pending",
    }) as HTMLDialogElement;

    dialog.close();

    await vi.waitFor(() => expect(dialog).toHaveAttribute("open"));
    expect(onClose).not.toHaveBeenCalled();
  });
});
