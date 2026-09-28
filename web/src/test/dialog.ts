/**
 * jsdom has `<dialog>` but not its modal API. This is the part of the HTML
 * spec the client relies on, and no more:
 * - `showModal()` opens the dialog on top of any other and focuses its first
 *   focusable control (the spec's focus delegate);
 * - `close()` closes it and fires `close`;
 * - Escape is a close request to the topmost modal dialog, unless its
 *   keydown was cancelled: `cancel` first, closing only if that was not
 *   cancelled.
 * jsdom has no inert or top layer, so the page behind stays reachable in
 * tests; that part is the browser's.
 */
export function polyfillDialog(): void {
  if (typeof HTMLDialogElement.prototype.showModal === "function") {
    return;
  }
  const topLayer: Array<HTMLDialogElement> = [];
  const topmost = () =>
    topLayer.filter((dialog) => dialog.isConnected && dialog.open).at(-1);

  HTMLDialogElement.prototype.showModal = function showModal(
    this: HTMLDialogElement,
  ) {
    if (this.open) {
      throw new DOMException(
        "The dialog is already open.",
        "InvalidStateError",
      );
    }
    if (!this.isConnected) {
      throw new DOMException(
        "The dialog is not connected.",
        "InvalidStateError",
      );
    }
    this.setAttribute("open", "");
    topLayer.push(this);
    focusDelegate(this)?.focus();
  };

  HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
    if (!this.open) {
      return;
    }
    this.removeAttribute("open");
    topLayer.splice(topLayer.indexOf(this), 1);
    // The spec queues the event as a task.
    queueMicrotask(() => this.dispatchEvent(new Event("close")));
  };

  document.addEventListener("keydown", (event) => {
    const dialog = topmost();
    if (event.key !== "Escape" || event.defaultPrevented || !dialog) {
      return;
    }
    if (dialog.dispatchEvent(new Event("cancel", { cancelable: true }))) {
      dialog.close();
    }
  });
}

const FOCUSABLE = [
  "button",
  "input",
  "select",
  "textarea",
  "a[href]",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

function focusDelegate(dialog: HTMLDialogElement): HTMLElement | undefined {
  return [...dialog.querySelectorAll<HTMLElement>(FOCUSABLE)].find(
    (element) => !element.matches(":disabled"),
  );
}
