import { Link } from "@tanstack/react-router";
import type { EnrollmentWarning } from "@/api/problem";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";

type WarningsDialogProps = {
  readonly warnings: ReadonlyArray<EnrollmentWarning>;
  readonly confirmLabel: string;
  readonly pending: boolean;
  readonly onConfirm: () => void;
  readonly onCancel: () => void;
};

/** The service's enrollment warnings, confirmed by resending with them acknowledged. */
export function WarningsDialog({
  warnings,
  confirmLabel,
  pending,
  onConfirm,
  onCancel,
}: WarningsDialogProps) {
  return (
    <Dialog
      title="Check before you continue"
      description="Ryuk found something worth a second look. Continue only if you are sure."
      onClose={onCancel}
      dismissible={!pending}
    >
      <ul className="flex flex-col gap-3 border-rule border-y py-4">
        {warnings.map((warning) => (
          <li key={warning.code} className="text-warning">
            {warning.detail}
            {warning.personId === null ? null : (
              <>
                {" "}
                <Link
                  to="/watchlist/$personId"
                  params={{ personId: warning.personId }}
                  target="_blank"
                  className="text-ink underline underline-offset-2"
                >
                  Open their record
                </Link>
              </>
            )}
          </li>
        ))}
      </ul>
      <div className="mt-6 flex justify-end gap-3">
        <Button variant="secondary" onClick={onCancel} disabled={pending}>
          Go back
        </Button>
        <Button onClick={onConfirm} disabled={pending}>
          {confirmLabel}
        </Button>
      </div>
    </Dialog>
  );
}
