import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Button } from "./button";
import { TOAST_DURATION_MS, ToastProvider, useToast } from "./toast";

/** A page whose buttons raise toasts, as a removal does. */
function Page({ onUndo }: { readonly onUndo: (name: string) => void }) {
  const toast = useToast();
  const removed = (name: string) =>
    toast.show({
      message: `Removed ${name}.`,
      action: { label: "Undo", onAction: () => onUndo(name) },
      tag: name,
    });
  return (
    <>
      <Button onClick={() => removed("Ada Lovelace")}>Remove Ada</Button>
      <Button onClick={() => removed("Grace Hopper")}>Remove Grace</Button>
      <Button onClick={() => toast.show({ message: "Saved." })}>Save</Button>
      <Button onClick={() => toast.dismiss("Ada Lovelace")}>Purge Ada</Button>
    </>
  );
}

function renderPage() {
  const onUndo = vi.fn<(name: string) => void>();
  render(
    <StrictMode>
      <ToastProvider>
        <Page onUndo={onUndo} />
      </ToastProvider>
    </StrictMode>,
  );
  return { onUndo, region: screen.getByRole("status") };
}

function click(name: string) {
  fireEvent.click(screen.getByRole("button", { name }));
}

function elapse(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Toast", () => {
  it("announces its message in a status region, with its action", () => {
    const { region } = renderPage();
    expect(region).toBeEmptyDOMElement();

    click("Remove Ada");

    expect(region).toHaveTextContent("Removed Ada Lovelace.");
    expect(
      within(region).getByRole("button", { name: "Undo" }),
    ).toBeInTheDocument();
  });

  it("runs its action once, and goes", () => {
    const { onUndo, region } = renderPage();
    click("Remove Ada");

    fireEvent.click(within(region).getByRole("button", { name: "Undo" }));

    expect(onUndo).toHaveBeenCalledExactlyOnceWith("Ada Lovelace");
    expect(region).toBeEmptyDOMElement();
  });

  it("goes by itself once its time is up, its action not taken", () => {
    const { onUndo, region } = renderPage();
    click("Remove Ada");

    elapse(TOAST_DURATION_MS - 1);
    expect(region).toHaveTextContent("Removed Ada Lovelace.");
    elapse(1);

    expect(region).toBeEmptyDOMElement();
    expect(onUndo).not.toHaveBeenCalled();
  });

  it("waits while the pointer is on it, then starts its time again", () => {
    const { region } = renderPage();
    click("Remove Ada");
    const toast = within(region).getByText("Removed Ada Lovelace.");

    fireEvent.mouseEnter(toast);
    elapse(TOAST_DURATION_MS * 3);
    expect(region).toHaveTextContent("Removed Ada Lovelace.");

    fireEvent.mouseLeave(toast);
    elapse(TOAST_DURATION_MS - 1);
    expect(region).toHaveTextContent("Removed Ada Lovelace.");
    elapse(1);
    expect(region).toBeEmptyDOMElement();
  });

  it("waits while it has focus, so a keyboard can reach its action", () => {
    const { onUndo, region } = renderPage();
    click("Remove Ada");
    const undo = within(region).getByRole("button", { name: "Undo" });

    act(() => undo.focus());
    elapse(TOAST_DURATION_MS * 3);
    fireEvent.click(undo);

    expect(onUndo).toHaveBeenCalledOnce();
  });

  it("gives focus back to where it was when it goes", () => {
    const { region } = renderPage();
    const remove = screen.getByRole("button", { name: "Remove Ada" });
    act(() => remove.focus());
    fireEvent.click(remove);

    act(() => within(region).getByRole("button", { name: "Undo" }).focus());
    fireEvent.click(within(region).getByRole("button", { name: "Undo" }));

    expect(remove).toHaveFocus();
  });

  it("is replaced by a newer one, whose time starts afresh", () => {
    const { onUndo, region } = renderPage();
    click("Remove Ada");
    elapse(TOAST_DURATION_MS - 1);

    click("Remove Grace");
    elapse(TOAST_DURATION_MS - 1);
    expect(region).toHaveTextContent("Removed Grace Hopper.");
    expect(region).not.toHaveTextContent("Ada Lovelace");
    fireEvent.click(within(region).getByRole("button", { name: "Undo" }));

    expect(onUndo).toHaveBeenCalledExactlyOnceWith("Grace Hopper");
  });

  it("can be dismissed, with its button or Escape, its action not taken", () => {
    const { onUndo, region } = renderPage();
    click("Remove Ada");

    fireEvent.click(within(region).getByRole("button", { name: "Dismiss" }));
    expect(region).toBeEmptyDOMElement();

    click("Remove Grace");
    fireEvent.keyDown(within(region).getByRole("button", { name: "Undo" }), {
      key: "Escape",
    });
    expect(region).toBeEmptyDOMElement();
    expect(onUndo).not.toHaveBeenCalled();
  });

  it("can be dismissed by its tag, and no other", () => {
    const { region } = renderPage();
    click("Remove Grace");

    click("Purge Ada");
    expect(region).toHaveTextContent("Removed Grace Hopper.");

    click("Remove Ada");
    click("Purge Ada");
    expect(region).toBeEmptyDOMElement();
  });

  it("may have no action", () => {
    const { region } = renderPage();

    click("Save");

    expect(region).toHaveTextContent("Saved.");
    expect(
      within(region).queryByRole("button", { name: "Undo" }),
    ).not.toBeInTheDocument();
  });

  it("needs a provider", () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(() => render(<Page onUndo={() => undefined} />)).toThrow(
      "useToast must be used inside a ToastProvider",
    );
    vi.restoreAllMocks();
  });
});
