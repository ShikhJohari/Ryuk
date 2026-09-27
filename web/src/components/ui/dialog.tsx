import { type ReactNode, useEffect, useId } from "react";

type DialogProps = {
  readonly title: string;
  readonly description?: string;
  /** Called on Escape; the dialog itself decides nothing. */
  readonly onClose: () => void;
  readonly children: ReactNode;
};

/** A modal panel on a dimmed page, labelled by its title. */
export function Dialog({ title, description, onClose, children }: DialogProps) {
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-6">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description === undefined ? undefined : descriptionId}
        className="w-full max-w-[520px] rounded-md border border-rule bg-raised p-8"
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
      </div>
    </div>
  );
}
