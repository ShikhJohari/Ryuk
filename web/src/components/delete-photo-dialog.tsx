import type { RefObject } from "react";
import { problemMessage } from "@/lib/problems";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";

type DeletePhotoDialogProps = {
  /** The photo's figure number on the page, as its Delete button names it. */
  readonly figure: number;
  /** Where the photo loads from, so the operator sees which one goes. */
  readonly src: string;
  readonly alt: string;
  /** True while the delete is in flight: nothing can change or close it. */
  readonly pending: boolean;
  /** Why the last delete failed, or null. */
  readonly error: Error | null;
  readonly onConfirm: () => void;
  readonly onCancel: () => void;
  /** Where focus goes if what opened the dialog cannot take it back. */
  readonly fallbackFocus?: RefObject<HTMLElement | null>;
};

/**
 * The last step before an enrolled photo is deleted, which cannot be undone:
 * keeping it is the first choice, so a slip of the pointer deletes nothing.
 */
export function DeletePhotoDialog({
  figure,
  src,
  alt,
  pending,
  error,
  onConfirm,
  onCancel,
  fallbackFocus,
}: DeletePhotoDialogProps) {
  return (
    <Dialog
      title={`Delete photo ${figure}?`}
      description="Deleting erases this enrolled photo and the embeddings made from it. It cannot be undone."
      onClose={onCancel}
      dismissible={!pending}
      fallbackFocus={fallbackFocus}
    >
      <img
        src={src}
        alt={alt}
        className="aspect-square w-40 rounded-sm border border-rule object-cover"
      />
      {error === null ? null : (
        <p role="alert" className="mt-4 text-destructive">
          {problemMessage(error)}
        </p>
      )}
      <div className="mt-6 flex justify-end gap-3">
        <Button
          variant="secondary"
          onClick={onCancel}
          disabled={pending}
          data-autofocus
        >
          Keep photo
        </Button>
        <Button variant="destructive" onClick={onConfirm} disabled={pending}>
          {pending ? "Deleting…" : "Delete photo"}
        </Button>
      </div>
    </Dialog>
  );
}
