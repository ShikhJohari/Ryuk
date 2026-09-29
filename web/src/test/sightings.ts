import { HttpResponse, http } from "msw";
import type {
  Sighting,
  SightingPerson,
  SightingSummary,
} from "@/api/sightings";
import { problemResponse } from "./api-server";
import { sface } from "./monitor";

export const ada: SightingPerson = {
  id: "ada",
  name: "Ada Lovelace",
  status: "on_watchlist",
};

export const grace: SightingPerson = {
  id: "grace",
  name: "Grace Hopper",
  status: "on_watchlist",
};

/** An ended sighting of `person` under SFace, seen for 12 s. */
export function sightingSummary(
  id: string,
  person: SightingPerson = ada,
  changes: Partial<SightingSummary> = {},
): SightingSummary {
  return {
    id,
    person,
    modelKey: sface.id,
    threshold: 0.498,
    startedAt: "2026-09-27T10:00:05Z",
    lastSeenAt: "2026-09-27T10:00:17Z",
    endedAt: "2026-09-27T10:00:17Z",
    bestScore: 0.874,
    ...changes,
  };
}

/** `sightingSummary`, with Grace as runner-up at 0.612. */
export function sighting(
  id: string,
  person: SightingPerson = ada,
  changes: Partial<Sighting> = {},
): Sighting {
  return {
    ...sightingSummary(id, person),
    runnerUp: { person: grace, score: 0.612 },
    ...changes,
  };
}

/** A sighting as the history lists it: without its runner-up. */
export function summaryOf(of: Sighting): SightingSummary {
  const { runnerUp: _, ...summary } = of;
  return summary;
}

/**
 * The service's sightings, answering the list and the detail as it does:
 * newest first by start then ID, one person's with `personId`, `pageSize`
 * to a page with an opaque cursor. A test changes them with `record`, as
 * the service writes a change before announcing it.
 */
export function sightingsService({
  pageSize: defaultPageSize = 50,
}: {
  readonly pageSize?: number;
} = {}) {
  let sightings: Array<Sighting> = [];
  let pageSize = defaultPageSize;
  const requests: Array<URLSearchParams> = [];
  const ordered = () =>
    [...sightings].sort((a, b) => {
      const started = Date.parse(b.startedAt) - Date.parse(a.startedAt);
      return started !== 0 ? started : b.id < a.id ? -1 : 1;
    });
  const handlers = [
    http.get("*/api/sightings", ({ request }) => {
      const params = new URL(request.url).searchParams;
      requests.push(params);
      const personId = params.get("personId");
      const cursor = params.get("cursor");
      const theirs = ordered().filter(
        (item) => personId === null || item.person.id === personId,
      );
      const from =
        cursor === null
          ? 0
          : theirs.findIndex((item) => `after-${item.id}` === cursor) + 1;
      const items = theirs.slice(from, from + pageSize);
      const last = items.at(-1);
      return HttpResponse.json({
        items: items.map(summaryOf),
        nextCursor:
          last !== undefined && from + pageSize < theirs.length
            ? `after-${last.id}`
            : null,
      });
    }),
    http.get("*/api/sightings/:sightingId", ({ params }) => {
      const found = sightings.find((item) => item.id === params.sightingId);
      return found === undefined
        ? problemResponse({
            type: "about:blank",
            title: "Not Found",
            status: 404,
            detail: "No sighting has that ID.",
            code: "not_found",
          })
        : HttpResponse.json(found);
    }),
  ];
  return {
    handlers,
    /** The query of every list request, in order. */
    requests,
    /** Adds a sighting, or replaces the one with its ID. */
    record(sighting: Sighting) {
      sightings = [
        ...sightings.filter((item) => item.id !== sighting.id),
        sighting,
      ];
    },
    /** Starts again with `initial`, a page `pageSize` long. */
    reset(
      initial: ReadonlyArray<Sighting> = [],
      options: { readonly pageSize?: number } = {},
    ) {
      sightings = [...initial];
      pageSize = options.pageSize ?? defaultPageSize;
      requests.length = 0;
    },
  };
}
