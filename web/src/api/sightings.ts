import { Effect, Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { ApiClient } from "./api-client";
import { API_BASE_URL } from "./base-url";
import { PersonStatus } from "./persons";
import type { components, operations } from "./schema.gen";

type Schemas = components["schemas"];

/** A sighting's person of interest, or its runner-up, as they are now. */
export const SightingPerson = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  status: PersonStatus,
}).annotations({ identifier: "SightingPerson" });
export type SightingPerson = typeof SightingPerson.Type;
export type SightingPersonMatchesContract = Assert<
  Equals<SightingPerson, Schemas["SightingPerson"]>
>;

/**
 * A sighting as the history lists it and the live monitor announces it.
 * `endedAt` is null while it is open. Never carries the runner-up.
 */
export const SightingSummary = Schema.Struct({
  id: Schema.String,
  person: SightingPerson,
  modelKey: Schema.String,
  threshold: Schema.Number,
  startedAt: Schema.String,
  lastSeenAt: Schema.String,
  endedAt: Schema.NullOr(Schema.String),
  bestScore: Schema.Number,
}).annotations({ identifier: "SightingSummary" });
export type SightingSummary = typeof SightingSummary.Type;
export type SightingSummaryMatchesContract = Assert<
  Equals<SightingSummary, Schemas["SightingSummary"]>
>;

/** The second-ranked candidate at the best match; `person` is null once purged. */
export const RunnerUp = Schema.Struct({
  person: Schema.NullOr(SightingPerson),
  score: Schema.Number,
}).annotations({ identifier: "RunnerUp" });
export type RunnerUp = typeof RunnerUp.Type;
export type RunnerUpMatchesContract = Assert<
  Equals<RunnerUp, Schemas["RunnerUp"]>
>;

/** One sighting with the runner-up at its best match: null when nobody else was on the watchlist. */
export const Sighting = Schema.Struct({
  ...SightingSummary.fields,
  runnerUp: Schema.NullOr(RunnerUp),
}).annotations({ identifier: "Sighting" });
export type Sighting = typeof Sighting.Type;
export type SightingMatchesContract = Assert<
  Equals<Sighting, Schemas["Sighting"]>
>;

/** Sightings newest first; `nextCursor` is null on the last page. */
export const SightingPage = Schema.Struct({
  items: Schema.Array(SightingSummary),
  nextCursor: Schema.NullOr(Schema.String),
}).annotations({ identifier: "SightingPage" });
export type SightingPage = typeof SightingPage.Type;
export type SightingPageMatchesContract = Assert<
  Equals<SightingPage, Schemas["SightingPage"]>
>;

/** Which page of which sightings: everyone's unless `personId` is given. */
export type SightingsQuery = {
  readonly personId?: string;
  /** The previous page's `nextCursor`; the newest page without one. */
  readonly cursor?: string;
  readonly limit?: number;
};
export type SightingsQueryMatchesContract = Assert<
  Equals<
    keyof SightingsQuery,
    keyof NonNullable<operations["listSightings"]["parameters"]["query"]>
  >
>;

const sightings = "/api/sightings";
const sighting = (sightingId: string) =>
  `${sightings}/${encodeURIComponent(sightingId)}`;

/**
 * Where the browser loads a sighting's best face crop from; never cached by
 * the service, so a purge cannot leave it in the browser.
 */
const sightingCropUrl = (sightingId: string) =>
  `${API_BASE_URL}${sighting(sightingId)}/crop`;

/**
 * The crop as an image source. The crop of an open sighting is replaced when
 * a higher match score arrives, and only then, so the score tells the
 * browser that the image at the same address has changed; the service
 * ignores the parameter.
 */
export const sightingCropSrc = (summary: SightingSummary) =>
  `${sightingCropUrl(summary.id)}?${new URLSearchParams({ score: String(summary.bestScore) })}`;

export const listSightings = (query: SightingsQuery) => {
  const params = new URLSearchParams();
  if (query.personId !== undefined) {
    params.set("personId", query.personId);
  }
  if (query.cursor !== undefined) {
    params.set("cursor", query.cursor);
  }
  if (query.limit !== undefined) {
    params.set("limit", String(query.limit));
  }
  const search = params.size === 0 ? "" : `?${params}`;
  return Effect.flatMap(ApiClient, (api) =>
    api.get(`${sightings}${search}`, SightingPage),
  );
};

export const getSighting = (sightingId: string) =>
  Effect.flatMap(ApiClient, (api) => api.get(sighting(sightingId), Sighting));
