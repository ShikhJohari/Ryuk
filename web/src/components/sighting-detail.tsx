import type { ReactNode } from "react";
import { modelNameOf, type RecognitionModelInfo } from "@/api/models";
import { type RunnerUp, type Sighting, sightingCropSrc } from "@/api/sightings";
import { formatDateTime, formatScore } from "@/lib/format";
import { MatchScoreNote, PersonOfInterestLink } from "./sighting-labels";

type SightingDetailProps = {
  readonly sighting: Sighting;
  /** Every recognition model the service knows, to name the sighting's. */
  readonly models: ReadonlyArray<RecognitionModelInfo>;
};

/**
 * Everything a sighting keeps: the face crop of its best match with its
 * match score, the runner-up at that match, the model and threshold that
 * produced it, and when it started, was last seen and ended.
 */
export function SightingDetail({ sighting, models }: SightingDetailProps) {
  const { person, bestScore } = sighting;
  return (
    <div className="grid items-start gap-10 md:grid-cols-[minmax(0,320px)_1fr]">
      <figure className="flex flex-col gap-2">
        <img
          src={sightingCropSrc(sighting)}
          alt={`${person.name}'s face at the best match, match score ${formatScore(bestScore)}`}
          className="aspect-square w-full rounded-sm border border-rule bg-ink object-contain"
        />
        <figcaption className="font-serif text-muted-foreground">
          <span className="text-ink">Figure 1.</span> The face at the best
          match. The whole frame is never kept.
        </figcaption>
      </figure>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-10 gap-y-4 border-rule border-y py-5">
        <Reading label="Person of interest">
          <PersonOfInterestLink person={person} />
        </Reading>
        <Reading label="Best match score">{formatScore(bestScore)}</Reading>
        <Reading label="Runner-up">
          <RunnerUpReading runnerUp={sighting.runnerUp} />
        </Reading>
        <Reading label="Recognition model">
          <span className="break-all">
            {modelNameOf(models, sighting.modelKey)}
          </span>
        </Reading>
        <Reading label="Threshold">{formatScore(sighting.threshold)}</Reading>
        <Reading label="Started">
          <Timestamp value={sighting.startedAt} />
        </Reading>
        <Reading label="Last seen">
          <Timestamp value={sighting.lastSeenAt} />
        </Reading>
        <Reading label="Ended">
          {sighting.endedAt === null ? (
            "Still open"
          ) : (
            <Timestamp value={sighting.endedAt} />
          )}
        </Reading>
      </dl>
    </div>
  );
}

function Reading({
  label,
  children,
}: {
  readonly label: string;
  readonly children: ReactNode;
}) {
  return (
    <>
      <dt className="section-label pt-1">{label}</dt>
      <dd className="text-ink">{children}</dd>
    </>
  );
}

function Timestamp({ value }: { readonly value: string }) {
  return <time dateTime={value}>{formatDateTime(value)}</time>;
}

/**
 * The second-ranked candidate at the best match. A purged runner-up keeps
 * only its match score; nobody ranks second when nobody else was on the
 * watchlist.
 */
function RunnerUpReading({ runnerUp }: { readonly runnerUp: RunnerUp | null }) {
  if (runnerUp === null) {
    return (
      <>
        None{" "}
        <span className="block text-muted-foreground text-sm">
          Nobody else was on the watchlist.
        </span>
      </>
    );
  }
  if (runnerUp.person === null) {
    return (
      <>
        Purged <MatchScoreNote score={runnerUp.score} />
      </>
    );
  }
  return (
    <PersonOfInterestLink person={runnerUp.person} score={runnerUp.score} />
  );
}
