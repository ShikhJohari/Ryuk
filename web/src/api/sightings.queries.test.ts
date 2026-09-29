import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { ada, grace, sightingSummary } from "@/test/sightings";
import type { SightingPage, SightingSummary } from "./sightings";
import {
  followSighting,
  sightingQueryOptions,
  sightingsQueryOptions,
} from "./sightings.queries";

const everyone = sightingsQueryOptions().queryKey;

/** A client holding everyone's history as `pages`. */
function cached(...pages: ReadonlyArray<SightingPage>) {
  const queryClient = new QueryClient();
  queryClient.setQueryData(everyone, {
    pages: [...pages],
    pageParams: pages.map((_, index) => (index === 0 ? null : `c${index}`)),
  });
  return queryClient;
}

function ids(queryClient: QueryClient) {
  return queryClient
    .getQueryData(everyone)
    ?.pages.map((page) => page.items.map((item) => item.id));
}

const at = (id: string, startedAt: string, changes = {}) =>
  sightingSummary(id, ada, { startedAt, ...changes });

describe("followSighting", () => {
  it("puts a newly opened sighting where the service orders it", () => {
    const queryClient = cached({
      items: [
        at("s3", "2026-09-27T10:00:03Z"),
        at("s1", "2026-09-27T10:00:01Z"),
      ],
      nextCursor: null,
    });

    // Opened after s3, but confirmed from an earlier first match.
    followSighting(queryClient, {
      type: "sighting_opened",
      sighting: at("s2", "2026-09-27T10:00:02Z", { endedAt: null }),
    });
    followSighting(queryClient, {
      type: "sighting_opened",
      sighting: at("s4", "2026-09-27T10:00:04Z", { endedAt: null }),
    });

    expect(ids(queryClient)).toEqual([["s4", "s3", "s2", "s1"]]);
  });

  it("replaces a sighting it has, on whichever page", () => {
    const queryClient = cached(
      { items: [at("s2", "2026-09-27T10:00:02Z")], nextCursor: "c1" },
      { items: [at("s1", "2026-09-27T10:00:01Z")], nextCursor: null },
    );
    const ended: SightingSummary = at("s1", "2026-09-27T10:00:01Z", {
      bestScore: 0.95,
      endedAt: "2026-09-27T10:00:09Z",
    });

    followSighting(queryClient, { type: "sighting_ended", sighting: ended });

    expect(queryClient.getQueryData(everyone)?.pages[1]?.items).toEqual([
      ended,
    ]);
    expect(ids(queryClient)).toEqual([["s2"], ["s1"]]);
  });

  it("never adds a sighting it only heard progress of, and fetches again instead", () => {
    const queryClient = cached({
      items: [at("s1", "2026-09-27T10:00:01Z")],
      nextCursor: null,
    });

    followSighting(queryClient, {
      type: "sighting_updated",
      sighting: at("s9", "2026-09-27T10:00:09Z"),
    });

    expect(ids(queryClient)).toEqual([["s1"]]);
    expect(queryClient.getQueryState(everyone)?.isInvalidated).toBe(true);
  });

  it("leaves a sighting older than every page loaded to the pages not loaded", () => {
    const queryClient = cached({
      items: [at("s5", "2026-09-27T10:00:05Z")],
      nextCursor: "c1",
    });

    followSighting(queryClient, {
      type: "sighting_opened",
      sighting: at("s1", "2026-09-27T10:00:01Z"),
    });

    expect(ids(queryClient)).toEqual([["s5"]]);
  });

  it("marks the person's history and the sighting's detail as out of date", () => {
    const queryClient = cached({ items: [], nextCursor: null });
    const theirs = sightingsQueryOptions(grace.id).queryKey;
    const detail = sightingQueryOptions("s1").queryKey;
    queryClient.setQueryData(theirs, { pages: [], pageParams: [] });
    queryClient.setQueryData(detail, {
      ...at("s1", "2026-09-27T10:00:01Z"),
      runnerUp: null,
    });

    followSighting(queryClient, {
      type: "sighting_opened",
      sighting: sightingSummary("s1", grace),
    });

    expect(ids(queryClient)).toEqual([["s1"]]);
    expect(queryClient.getQueryState(theirs)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(detail)?.isInvalidated).toBe(true);
  });
});
