import {
  type InfiniteData,
  infiniteQueryOptions,
  type QueryClient,
  queryOptions,
} from "@tanstack/react-query";
import { runQuery } from "@/lib/runtime";
import type { SightingMessage } from "./monitor";
import {
  getSighting,
  listSightings,
  type SightingPage,
  type SightingSummary,
} from "./sightings";

/** Every sightings query starts with this key, so one invalidation refreshes them all. */
export const sightingsKey = ["sightings"] as const;

/** Pages of sightings, newest first: everyone's, or one person of interest's. */
export const sightingsQueryOptions = (personId?: string) =>
  infiniteQueryOptions({
    queryKey: [...sightingsKey, "list", personId ?? null],
    queryFn: ({ pageParam, signal }) =>
      runQuery(
        listSightings({
          ...(personId === undefined ? {} : { personId }),
          ...(pageParam === null ? {} : { cursor: pageParam }),
        }),
        { signal },
      ),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.nextCursor,
  });

export const sightingQueryOptions = (sightingId: string) =>
  queryOptions({
    queryKey: [...sightingsKey, "detail", sightingId],
    queryFn: ({ signal }) => runQuery(getSighting(sightingId), { signal }),
  });

/**
 * Applies what the live monitor said about a sighting to the cached history,
 * so the rail, a reload and the history page all show the same thing.
 *
 * Everyone's history is changed in place: the service writes a change before
 * it sends it, so the cache then holds what a fetch would. When that cannot
 * be done safely (nothing cached yet, a fetch in flight that may have read
 * the service before the change, or a sighting outside the pages loaded) it
 * is fetched again instead. Every other view of the sighting, a person's
 * history or its detail, is fetched again when next shown.
 */
export function followSighting(
  queryClient: QueryClient,
  message: SightingMessage,
): void {
  const { sighting } = message;
  const everyone = sightingsQueryOptions().queryKey;
  const data = queryClient.getQueryData(everyone);
  const fetching =
    queryClient.getQueryState(everyone)?.fetchStatus === "fetching";
  const followed =
    data === undefined || fetching
      ? null
      : withSighting(data, sighting, message.type === "sighting_opened");
  if (followed === null) {
    void queryClient.invalidateQueries({ queryKey: everyone, exact: true });
  } else {
    queryClient.setQueryData(everyone, followed);
  }
  void queryClient.invalidateQueries({
    queryKey: sightingsQueryOptions(sighting.person.id).queryKey,
    exact: true,
  });
  void queryClient.invalidateQueries({
    queryKey: sightingQueryOptions(sighting.id).queryKey,
    exact: true,
  });
}

/**
 * The pages with `sighting` in them: replaced where it already is, or, when
 * it has just opened, inserted where the service orders it (newest start
 * first, then highest ID). Null when it belongs outside the pages loaded.
 */
function withSighting<Cursor>(
  data: InfiniteData<SightingPage, Cursor>,
  sighting: SightingSummary,
  opened: boolean,
): InfiniteData<SightingPage, Cursor> | null {
  const found = data.pages.findIndex((page) =>
    page.items.some((item) => item.id === sighting.id),
  );
  if (found !== -1) {
    return {
      ...data,
      pages: data.pages.map((page, index) =>
        index === found
          ? {
              ...page,
              items: page.items.map((item) =>
                item.id === sighting.id ? sighting : item,
              ),
            }
          : page,
      ),
    };
  }
  if (!opened) {
    return null;
  }
  const into = data.pages.findIndex(
    (page, index) =>
      page.items.some((item) => isNewer(sighting, item)) ||
      // After every item of the last page, and no page follows it.
      (index === data.pages.length - 1 && page.nextCursor === null),
  );
  if (into === -1) {
    return null;
  }
  return {
    ...data,
    pages: data.pages.map((page, index) => {
      if (index !== into) {
        return page;
      }
      const at = page.items.findIndex((item) => isNewer(sighting, item));
      const items = [...page.items];
      items.splice(at === -1 ? items.length : at, 0, sighting);
      return { ...page, items };
    }),
  };
}

/** Whether `a` comes before `b` in the service's order. */
function isNewer(a: SightingSummary, b: SightingSummary): boolean {
  const started = Date.parse(a.startedAt) - Date.parse(b.startedAt);
  return started === 0 ? a.id > b.id : started > 0;
}
