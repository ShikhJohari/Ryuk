import { useId, useState } from "react";
import { problemMessage } from "@/lib/problems";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";
import { Input } from "./ui/input";

type PurgeDialogProps = {
  /** The person of interest's name, as the service stores it. */
  readonly name: string;
  /** True while the purge is in flight: nothing can change or close it. */
  readonly pending: boolean;
  /** Why the last purge failed, or null. */
  readonly error: Error | null;
  readonly onConfirm: () => void;
  readonly onCancel: () => void;
};

/**
 * The last step before a purge, which cannot be undone: the operator types
 * the person's name, so a purge is never one slip of the pointer away.
 */
export function PurgeDialog({
  name,
  pending,
  error,
  onConfirm,
  onCancel,
}: PurgeDialogProps) {
  const inputId = useId();
  const [typed, setTyped] = useState("");
  const confirmed = namesMatch(typed, name);

  return (
    <Dialog
      title={`Purge ${name}?`}
      description="Purging permanently erases their enrolled photos, embeddings and sightings, and clears them as runner-up on anyone else's sightings. It cannot be undone."
      onClose={onCancel}
      dismissible={!pending}
    >
      <form
        className="flex flex-col gap-5"
        onSubmit={(event) => {
          event.preventDefault();
          if (confirmed && !pending) {
            onConfirm();
          }
        }}
      >
        <div className="flex flex-col gap-2">
          <label htmlFor={inputId} className="font-medium">
            Type <span className="font-semibold">{name}</span> to confirm
          </label>
          <Input
            id={inputId}
            value={typed}
            disabled={pending}
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            data-autofocus
            onChange={(event) => setTyped(event.target.value)}
          />
        </div>
        {error === null ? null : (
          <p role="alert" className="text-destructive">
            {problemMessage(error)}
          </p>
        )}
        <div className="flex justify-end gap-3">
          <Button variant="secondary" onClick={onCancel} disabled={pending}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="destructive"
            disabled={!confirmed || pending}
          >
            {pending ? "Purging…" : "Purge permanently"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}

/**
 * Whether the operator typed `name`. Exact, case and punctuation included,
 * except for what could never be part of a stored name: the service trims a
 * name and collapses its runs of whitespace to one space, so the typed text
 * is read the same way. Both are compared in Unicode NFC, so an accented
 * letter matches however the keyboard or the enrolled name composed it.
 */
export function namesMatch(typed: string, name: string): boolean {
  const cleaned = typed.trim().split(/\s+/).join(" ");
  return cleaned !== "" && cleaned.normalize("NFC") === name.normalize("NFC");
}
