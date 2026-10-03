import { fireEvent, render, screen } from "@testing-library/react";
import { StrictMode, useRef, useState } from "react";
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

/**
 * A page whose dialog is opened with nothing focused, as after a request
 * whose controls were disabled while it was in flight; the opener can also
 * be taken away while the dialog is open.
 */
function PageWithFallback() {
  const [open, setOpen] = useState(false);
  const [openerShown, setOpenerShown] = useState(true);
  const fallback = useRef<HTMLInputElement>(null);
  return (
    <>
      <input ref={fallback} aria-label="Fallback" />
      {openerShown ? <Button onClick={() => setOpen(true)}>Open</Button> : null}
      {open ? (
        <Dialog
          title="Check this"
          fallbackFocus={fallback}
          onClose={() => setOpen(false)}
        >
          <Button onClick={() => setOpen(false)}>Close</Button>
          <Button onClick={() => setOpenerShown(false)}>Remove opener</Button>
        </Dialog>
      ) : null}
    </>
  );
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

  it("returns focus to its fallback when nothing had focus before it", () => {
    render(<PageWithFallback />);
    // Clicked without focus, so nothing outside the dialog had it.
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.getByLabelText("Fallback")).toHaveFocus();
  });

  it("returns focus to its fallback when what opened it has gone", () => {
    render(<PageWithFallback />);
    const { opener } = open();

    fireEvent.click(screen.getByRole("button", { name: "Remove opener" }));
    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(opener).not.toBeInTheDocument();
    expect(screen.getByLabelText("Fallback")).toHaveFocus();
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

  it("stays open under StrictMode, whose first cleanup closes it", async () => {
    render(
      <StrictMode>
        <Page />
      </StrictMode>,
    );

    const { dialog } = open();
    // The first cleanup's close event arrives after the dialog reopened.
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute("open");
    expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  });
});
