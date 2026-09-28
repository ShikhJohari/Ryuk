import { type ReactNode, useId, useLayoutEffect, useRef } from "react";

type DialogProps = {
  readonly title: string;
  readonly description?: string;
  /**
   * Called on Escape while the dialog is dismissible; the page decides
   * whether it goes, by unmounting it.
   */
  readonly onClose: () => void;
  /**
   * False while the dialog's request is in flight: Escape does nothing, so
   * the operator cannot leave a change half made.
   */
  readonly dismissible?: boolean;
  readonly children: ReactNode;
};

/**
 * A modal panel on a dimmed page, labelled by its title. Built on the native
 * `<dialog>` opened with `showModal()`, so the page behind is inert, focus
 * starts on the dialog's first control and stays inside it, and returns to
 * whatever had it when the dialog goes.
 */
export function Dialog({
  title,
  description,
  onClose,
  dismissible = true,
  children,
}: DialogProps) {
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef<HTMLDialogElement>(null);
  // Read when the event arrives, so new props never reopen the dialog.
  const latest = useRef({ onClose, dismissible });
  latest.current = { onClose, dismissible };

  // A layout effect, so its cleanup closes the dialog and restores focus
  // before React removes it from the page.
  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    if (dialog === null) {
      return;
    }
    // What had focus before the dialog opened; an `autoFocus` inside it
    // may already have moved focus in.
    const focused = document.activeElement;
    const opener =
      focused instanceof HTMLElement && !dialog.contains(focused)
        ? focused
        : null;
    let unmounting = false;

    // Escape asks to close; the dialog itself never closes in place.
    const onCancel = (event: Event) => {
      event.preventDefault();
      if (latest.current.dismissible) {
        latest.current.onClose();
      }
    };
    // A browser may close a dialog anyway after repeated Escapes without
    // user activation. It stays open for as long as the page renders it.
    const onNativeClose = () => {
      if (unmounting) {
        return;
      }
      if (latest.current.dismissible) {
        latest.current.onClose();
      }
      if (!dialog.open && dialog.isConnected) {
        dialog.showModal();
      }
    };

    dialog.addEventListener("cancel", onCancel);
    dialog.addEventListener("close", onNativeClose);
    if (!dialog.open) {
      dialog.showModal();
    }
    return () => {
      unmounting = true;
      dialog.removeEventListener("cancel", onCancel);
      dialog.removeEventListener("close", onNativeClose);
      dialog.close();
      if (opener?.isConnected) {
        opener.focus();
      }
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby={titleId}
      aria-describedby={description === undefined ? undefined : descriptionId}
      onKeyDown={(event) => {
        // Cancelling the key stops the browser's close request altogether.
        if (event.key === "Escape" && !dismissible) {
          event.preventDefault();
        }
      }}
      className="m-auto w-[calc(100%-3rem)] max-w-[520px] rounded-md border border-rule bg-raised p-8 text-ink backdrop:bg-ink/40"
    >
      <h2
        id={titleId}
        className="font-serif text-[26px] text-ink leading-tight"
      >
        {title}
      </h2>
      {description === undefined ? null : (
        <p id={descriptionId} className="mt-2 text-muted-foreground">
          {description}
        </p>
      )}
      <div className="mt-6">{children}</div>
    </dialog>
  );
}
