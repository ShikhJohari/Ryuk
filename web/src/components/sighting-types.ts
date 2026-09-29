import { API_BASE_URL } from "@/api/base-url";
import type { RecognitionModelInfo } from "@/api/models";
import type { PersonStatus } from "@/api/persons";

// The sighting shapes of the #31 contract (#12), typed here until the
// generated contract has them; `api/sightings.ts` then replaces this module
// with Effect schemas asserted equal to the generated types.

/** The person of interest a sighting is of, or its runner-up. */
export type SightingPerson = {
  readonly id: string;
  readonly name: string;
  readonly status: PersonStatus;
};

export type SightingSummary = {
  readonly id: string;
  readonly person: SightingPerson;
  readonly modelKey: string;
  readonly threshold: number;
  readonly startedAt: string;
  readonly lastSeenAt: string;
  /** Null while the sighting is open. */
  readonly endedAt: string | null;
  readonly bestScore: number;
};

/** The second-ranked candidate at the best match; `person` is null once purged. */
export type RunnerUp = {
  readonly person: SightingPerson | null;
  readonly score: number;
};

export type Sighting = SightingSummary & {
  /** Null when nobody else was on the watchlist to rank second. */
  readonly runnerUp: RunnerUp | null;
};

/**
 * Where the browser loads a sighting's best face crop from; never cached by
 * the service, so a purge cannot leave it in the browser.
 */
export const sightingCropUrl = (sightingId: string) =>
  `${API_BASE_URL}/api/sightings/${encodeURIComponent(sightingId)}/crop`;

/**
 * The crop as an image source. The crop of an open sighting is replaced when
 * a higher score arrives, and only then, so the score tells the browser that
 * the image at the same address has changed.
 */
export const sightingCropSrc = (sighting: SightingSummary) =>
  `${sightingCropUrl(sighting.id)}?${new URLSearchParams({ score: String(sighting.bestScore) })}`;

/**
 * The name of the recognition model that produced a sighting, or its key
 * when the service no longer knows that model: new weights for a network are
 * a new model, and the old one is forgotten.
 */
export function modelNameOf(
  models: ReadonlyArray<RecognitionModelInfo>,
  modelKey: string,
): string {
  return models.find((model) => model.id === modelKey)?.name ?? modelKey;
}
