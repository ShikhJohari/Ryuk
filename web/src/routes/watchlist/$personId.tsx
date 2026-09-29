import {
  type QueryClient,
  useMutation,
  useQueryClient,
  useSuspenseInfiniteQuery,
  useSuspenseQuery,
} from "@tanstack/react-query";
import {
  createFileRoute,
  Link,
  notFound,
  useNavigate,
} from "@tanstack/react-router";
import { useId, useState } from "react";
import { modelsQueryOptions } from "@/api/models.queries";
import {
  addPhoto,
  deletePhoto,
  type PersonOfInterest,
  type PersonStatus,
  PHOTO_TYPES,
  photoImageUrl,
  purgePerson,
  renamePerson,
  setPersonStatus,
} from "@/api/persons";
import { personQueryOptions, personsKey } from "@/api/persons.queries";
import { sightingsKey, sightingsQueryOptions } from "@/api/sightings.queries";
import { PageHeader } from "@/components/page-header";
import { PurgeDialog } from "@/components/purge-dialog";
import { SightingsTable } from "@/components/sightings-table";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { WarningsDialog } from "@/components/warnings-dialog";
import { useAcknowledgedMutation } from "@/hooks/use-acknowledged-mutation";
import { formatDate } from "@/lib/format";
import { isNotFound, problemMessage } from "@/lib/problems";
import { runQuery } from "@/lib/runtime";

export const Route = createFileRoute("/watchlist/$personId")({
  loader: async ({ context, params }) => {
    try {
      await Promise.all([
        context.queryClient.ensureQueryData(
          personQueryOptions(params.personId),
        ),
        context.queryClient.ensureInfiniteQueryData(
          sightingsQueryOptions(params.personId),
        ),
        context.queryClient.ensureQueryData(modelsQueryOptions),
      ]);
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
        <div className="flex flex-wrap items-start gap-3">
          <Rename person={person} />
          <Lifecycle person={person} />
        </div>
      </div>
      <Photos person={person} />
      <PersonSightings person={person} />
    </section>
  );
}

type StatusChange = {
  readonly status: PersonStatus;
  /** Taken from a removal's toast, perhaps after the page has gone. */
  readonly undo: boolean;
};

/** Removal with an Undo, restoring someone removed, and purge. */
function Lifecycle({ person }: { readonly person: PersonOfInterest }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const toast = useToast();
  const [purging, setPurging] = useState(false);

  // Its callbacks are the mutation's own, not per call: an Undo from the
  // toast may run after this page has gone, and must still refresh the lists.
  const change = useMutation({
    mutationFn: ({ status }: StatusChange) =>
      runQuery(setPersonStatus(person.id, status)),
    onSuccess: async (changed, { undo }) => {
      queryClient.setQueryData(
        personQueryOptions(changed.id).queryKey,
        changed,
      );
      if (changed.status === "removed" && !undo) {
        toast.show({
          message: `Removed ${changed.name}.`,
          tag: changed.id,
          action: {
            label: "Undo",
            onAction: () =>
              change.mutate({ status: "on_watchlist", undo: true }),
          },
        });
      }
      // A removal ends their open sighting; the watchlist tabs change.
      await refresh(queryClient);
    },
    onError: (error, { undo }) => {
      if (undo) {
        toast.show({
          message: `${person.name} could not be put back on the watchlist. ${problemMessage(error)}`,
        });
      }
    },
  });

  const purge = useMutation({
    mutationFn: () => runQuery(purgePerson(person.id)),
    onSuccess: async () => {
      // An Undo for someone purged could only fail.
      toast.dismiss(person.id);
      toast.show({ message: `Purged ${person.name}.` });
      const detail = queryClient.getQueryCache().find({
        queryKey: personQueryOptions(person.id).queryKey,
        exact: true,
      });
      // A page still showing them goes first, below; one already gone must
      // not come back from the cache.
      if (detail !== undefined && detail.getObserversCount() === 0) {
        queryClient.removeQueries({ queryKey: detail.queryKey, exact: true });
      }
      await refresh(queryClient, "none");
    },
  });

  return (
    <>
      {person.status === "on_watchlist" ? (
        <Button
          variant="secondary"
          disabled={change.isPending}
          onClick={() => change.mutate({ status: "removed", undo: false })}
        >
          Remove from watchlist
        </Button>
      ) : (
        <Button
          variant="secondary"
          disabled={change.isPending}
          onClick={() => change.mutate({ status: "on_watchlist", undo: false })}
        >
          Restore to watchlist
        </Button>
      )}
      <Button variant="destructive" onClick={() => setPurging(true)}>
        Purge
      </Button>
      {change.error !== null && change.variables?.undo === false ? (
        <p role="alert" className="basis-full text-destructive">
          {problemMessage(change.error)}
        </p>
      ) : null}
      {purging ? (
        <PurgeDialog
          name={person.name}
          pending={purge.isPending}
          error={purge.error}
          onCancel={() => {
            setPurging(false);
            purge.reset();
          }}
          onConfirm={() =>
            // Per call, so it never navigates after the page has gone.
            purge.mutate(undefined, {
              onSuccess: async () => {
                await navigate({ to: "/watchlist" });
                queryClient.removeQueries({
                  queryKey: personQueryOptions(person.id).queryKey,
                  exact: true,
                });
                await refresh(queryClient);
              },
            })
          }
        />
      ) : null}
    </>
  );
}

/**
 * The watchlist and the sightings after a person of interest changed:
 * fetched again at once if shown, or when next shown with `"none"`.
 */
function refresh(
  queryClient: QueryClient,
  refetchType: "active" | "none" = "active",
) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: personsKey, refetchType }),
    queryClient.invalidateQueries({ queryKey: sightingsKey, refetchType }),
  ]);
}

/** Their newest sightings; the history has the rest. */
function PersonSightings({ person }: { readonly person: PersonOfInterest }) {
  const headingId = useId();
  const history = useSuspenseInfiniteQuery(sightingsQueryOptions(person.id));
  const models = useSuspenseQuery(modelsQueryOptions).data;

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-5">
      <div className="flex items-baseline justify-between gap-6">
        <h2 id={headingId} className="section-label">
          Sightings
        </h2>
        <Link
          to="/sightings"
          search={{ personId: person.id }}
          className="text-muted-foreground text-sm hover:text-ink"
        >
          All their sightings
        </Link>
      </div>
      <SightingsTable
        sightings={history.data.pages[0]?.items ?? []}
        models={models}
        showPerson={false}
        caption={
          <>
            <span className="text-ink">Table 1.</span> {person.name}'s
            sightings, newest first.
          </>
        }
        empty={`${person.name} has not been sighted.`}
      />
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
      // Sightings show their person's name as it is now.
      await refresh(queryClient);
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
        // Per call, so it never runs after the page has gone.
        rename.mutate(name, { onSuccess: () => setName(null) });
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
          disabled={rename.isPending}
          onChange={(event) => setName(event.target.value)}
        />
        <Button type="submit" disabled={rename.isPending || !name.trim()}>
          Save
        </Button>
        <Button
          variant="secondary"
          // A rename in flight cannot be taken back.
          disabled={rename.isPending}
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

  return (
    <section aria-labelledby="enrolled-photos" className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-6">
        <h2 id="enrolled-photos" className="section-label">
          Enrolled photos
        </h2>
        <label
          className={buttonVariants({
            variant: "secondary",
            // The input is visually hidden, so its label shows its focus.
            className:
              "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring has-[:focus-visible]:outline-offset-2",
          })}
        >
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
      {add.error === null ? null : (
        <p role="alert" className="text-destructive">
          {problemMessage(add.error)}
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
              {remove.error !== null && remove.variables === enrolled.id ? (
                <p role="alert" className="text-destructive">
                  {problemMessage(remove.error)}
                </p>
              ) : null}
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
