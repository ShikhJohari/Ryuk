import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useId, useState } from "react";
import { createPerson, PHOTO_TYPES } from "@/api/persons";
import { personsKey } from "@/api/persons.queries";
import { useAcknowledgedMutation } from "@/hooks/use-acknowledged-mutation";
import { problemMessage } from "@/lib/problems";
import { runQuery } from "@/lib/runtime";
import { Button } from "./ui/button";
import { Dialog } from "./ui/dialog";
import { Input } from "./ui/input";
import { WarningsDialog } from "./warnings-dialog";

/** Create a person of interest from a name and one photo. */
export function EnrollDialog({ onClose }: { readonly onClose: () => void }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const nameId = useId();
  const photoId = useId();
  const [name, setName] = useState("");
  const [photo, setPhoto] = useState<File | null>(null);

  const enroll = useAcknowledgedMutation({
    mutationFn: (
      upload: { readonly name: string; readonly photo: File },
      acknowledgedWarnings,
    ) => runQuery(createPerson({ ...upload, acknowledgedWarnings })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: personsKey }),
    // Never after the dialog has gone: the operator may be elsewhere by then.
    onSuccessWhileMounted: (person) => {
      onClose();
      void navigate({
        to: "/watchlist/$personId",
        params: { personId: person.id },
      });
    },
  });
  // Once sent, an enrollment cannot be taken back, so nothing in the dialog
  // can change or close it until the service answers.
  const pending = enroll.isPending;

  if (enroll.warnings !== null) {
    return (
      <WarningsDialog
        warnings={enroll.warnings}
        confirmLabel="Enroll anyway"
        pending={pending}
        onConfirm={enroll.acknowledge}
        onCancel={enroll.dismiss}
      />
    );
  }

  return (
    <Dialog
      title="Enroll a person of interest"
      description="A name and one clear photo with only their face large enough to use."
      onClose={onClose}
      dismissible={!pending}
    >
      <form
        className="flex flex-col gap-5"
        onSubmit={(event) => {
          event.preventDefault();
          if (photo !== null) {
            enroll.submit({ name, photo });
          }
        }}
      >
        <div className="flex flex-col gap-2">
          <label htmlFor={nameId} className="font-medium">
            Name
          </label>
          <Input
            id={nameId}
            value={name}
            required
            disabled={pending}
            maxLength={200}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <label htmlFor={photoId} className="font-medium">
            Photo
          </label>
          <input
            id={photoId}
            type="file"
            accept={PHOTO_TYPES}
            disabled={pending}
            className="text-sm file:mr-4 file:h-9 file:rounded-[18px] file:border file:border-ink file:bg-transparent file:px-4 file:text-ink"
            onChange={(event) => {
              setPhoto(event.target.files?.[0] ?? null);
              enroll.reset();
            }}
          />
          <p className="text-muted-foreground text-sm">
            JPEG, PNG or WebP, up to 10 MB.
          </p>
        </div>
        {enroll.error === null ? null : (
          <p role="alert" className="text-destructive">
            {problemMessage(enroll.error)}
          </p>
        )}
        <div className="flex justify-end gap-3">
          <Button variant="secondary" onClick={onClose} disabled={pending}>
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={pending || photo === null || !name.trim()}
          >
            {pending ? "Enrolling…" : "Enroll"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
