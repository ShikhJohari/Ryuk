import {
  useMutation,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query";
import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { useId, useState } from "react";
import {
  addPhoto,
  deletePhoto,
  type PersonOfInterest,
  photoImageUrl,
  renamePerson,
} from "@/api/persons";
import { personQueryOptions, personsKey } from "@/api/persons.queries";
import { PageHeader } from "@/components/page-header";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { WarningsDialog } from "@/components/warnings-dialog";
import { useAcknowledgedMutation } from "@/hooks/use-acknowledged-mutation";
import { formatDate, PHOTO_TYPES } from "@/lib/format";
import { isNotFound, problemMessage } from "@/lib/problems";
import { runQuery } from "@/lib/runtime";

export const Route = createFileRoute("/watchlist/$personId")({
  loader: async ({ context, params }) => {
    try {
      await context.queryClient.ensureQueryData(
        personQueryOptions(params.personId),
      );
    } catch (error) {
      throw isNotFound(error) ? notFound() : error;
    }
  },
  component: PersonOfInterestPage,
});

function PersonOfInterestPage() {
  const { personId } = Route.useParams();
  const person = useSuspenseQuery(personQueryOptions(personId)).data;
  const status =
    person.status === "on_watchlist"
      ? `On the watchlist since ${formatDate(person.statusChangedAt)}.`
      : `Removed from the watchlist on ${formatDate(person.statusChangedAt)}.`;

  return (
    <section className="flex flex-col gap-10">
      <div className="flex flex-col gap-4">
        <Link
          to="/watchlist"
          className="text-muted-foreground text-sm hover:text-ink"
        >
          ← Watchlist
        </Link>
        <PageHeader
          label="Person of interest"
          title={person.name}
          description={status}
        />
        <Rename person={person} />
      </div>
      <Photos person={person} />
    </section>
  );
}

function Rename({ person }: { readonly person: PersonOfInterest }) {
  const queryClient = useQueryClient();
  const inputId = useId();
  const [name, setName] = useState<string | null>(null);
  const rename = useMutation({
    mutationFn: (newName: string) => runQuery(renamePerson(person.id, newName)),
    onSuccess: async (renamed) => {
      queryClient.setQueryData(personQueryOptions(person.id).queryKey, renamed);
      await queryClient.invalidateQueries({ queryKey: personsKey });
      setName(null);
    },
  });

  if (name === null) {
    return (
      <div>
        <Button variant="secondary" onClick={() => setName(person.name)}>
          Rename
        </Button>
      </div>
    );
  }

  return (
    <form
      className="flex max-w-[560px] flex-col gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        rename.mutate(name);
      }}
    >
      <label htmlFor={inputId} className="font-medium">
        New name
      </label>
      <div className="flex gap-3">
        <Input
          id={inputId}
          value={name}
          required
          autoFocus
          maxLength={200}
          onChange={(event) => setName(event.target.value)}
        />
        <Button type="submit" disabled={rename.isPending || !name.trim()}>
          Save
        </Button>
        <Button
          variant="secondary"
          onClick={() => {
            setName(null);
            rename.reset();
          }}
        >
          Cancel
        </Button>
      </div>
      {rename.error === null ? null : (
        <p role="alert" className="text-destructive">
          {problemMessage(rename.error)}
        </p>
      )}
    </form>
  );
}

function Photos({ person }: { readonly person: PersonOfInterest }) {
  const queryClient = useQueryClient();
  const refresh = () => queryClient.invalidateQueries({ queryKey: personsKey });

  const add = useAcknowledgedMutation({
    mutationFn: (photo: File, acknowledgedWarnings) =>
      runQuery(addPhoto(person.id, { photo, acknowledgedWarnings })),
    onSuccess: refresh,
  });
  const remove = useMutation({
    mutationFn: (photoId: string) => runQuery(deletePhoto(person.id, photoId)),
    onSuccess: refresh,
  });
  const onlyOne = person.photos.length === 1;
  const error = add.error ?? remove.error;

  return (
    <section aria-labelledby="enrolled-photos" className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-6">
        <h2 id="enrolled-photos" className="section-label">
          Enrolled photos
        </h2>
        <label className={buttonVariants({ variant: "secondary" })}>
          {add.isPending ? "Adding photo…" : "Add photo"}
          <input
            type="file"
            accept={PHOTO_TYPES}
            className="sr-only"
            disabled={add.isPending}
            onChange={(event) => {
              const chosen = event.target.files?.[0];
              // Cleared, so choosing the same file again still uploads it.
              event.target.value = "";
              if (chosen !== undefined) {
                add.submit(chosen);
              }
            }}
          />
        </label>
      </div>
      {error === null ? null : (
        <p role="alert" className="text-destructive">
          {problemMessage(error)}
        </p>
      )}
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-6">
        {person.photos.map((enrolled, index) => (
          <li key={enrolled.id}>
            <figure className="flex flex-col gap-2">
              <img
                src={photoImageUrl(person.id, enrolled.id)}
                alt={`${person.name}, enrolled ${formatDate(enrolled.createdAt)} (figure ${index + 1})`}
                className="aspect-square w-full rounded-sm border border-rule object-cover"
              />
              <figcaption className="font-serif text-muted-foreground">
                <span className="text-ink">Figure {index + 1}.</span> Enrolled{" "}
                {formatDate(enrolled.createdAt)}
              </figcaption>
              <Button
                variant="destructive"
                className="self-start"
                disabled={onlyOne || remove.isPending}
                title={
                  onlyOne
                    ? "A person of interest's last photo cannot be deleted."
                    : undefined
                }
                aria-label={`Delete photo ${index + 1}`}
                onClick={() => remove.mutate(enrolled.id)}
              >
                Delete
              </Button>
            </figure>
          </li>
        ))}
      </ul>
      {onlyOne ? (
        <p className="text-muted-foreground text-sm">
          The last enrolled photo cannot be deleted; add another first.
        </p>
      ) : null}
      {add.warnings === null ? null : (
        <WarningsDialog
          warnings={add.warnings}
          confirmLabel="Add anyway"
          pending={add.isPending}
          onConfirm={add.acknowledge}
          onCancel={add.dismiss}
        />
      )}
    </section>
  );
}
