import type {
  Sighting,
  SightingPerson,
  SightingSummary,
} from "@/components/sighting-types";
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
