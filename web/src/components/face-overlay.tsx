import type { CSSProperties, ReactNode } from "react";
import type { Face, FaceBox, FrameResult } from "@/api/monitor";
import { formatScore } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * The latest result's faces, boxed over the video. Boxes are placed as
 * fractions of the frame, so the stage must share the frame's aspect ratio.
 *
 * Match and no match differ in colour, lightness and line style, so they
 * stay apart without colour vision; only a match is ever named.
 */
export function FaceOverlay({ result }: { readonly result: FrameResult }) {
  return (
    <div className="absolute inset-0">
      {result.faces.map((face, index) => (
        <FaceMark
          // Faces have no identity across frames; their order is YuNet's.
          // biome-ignore lint/suspicious/noArrayIndexKey: see above
          key={index}
          face={face}
          style={placement(face.box, result)}
        />
      ))}
    </div>
  );
}

function FaceMark({
  face,
  style,
}: {
  readonly face: Face;
  readonly style: CSSProperties;
}) {
  switch (face.outcome) {
    case "match":
      return (
        <div
          role="img"
          aria-label={`Match: ${face.person.name}, score ${formatScore(face.score)}`}
          className="absolute border-2 border-match-video"
          style={style}
        >
          <Label className="bg-match-video">
            {face.person.name}{" "}
            <span className="tabular-nums">{formatScore(face.score)}</span>
          </Label>
        </div>
      );
    case "no_match":
      return (
        <div
          role="img"
          aria-label={
            face.score === null
              ? "No match"
              : `No match, score ${formatScore(face.score)}`
          }
          className="absolute border-2 border-no-match-video border-dashed"
          style={style}
        >
          {face.score === null ? null : (
            <Label className="bg-no-match-video">
              <span className="tabular-nums">{formatScore(face.score)}</span>
            </Label>
          )}
        </div>
      );
    case "too_small":
      return (
        <div
          role="img"
          aria-label="Face too small to score"
          className="absolute border border-paper/60"
          style={style}
        />
      );
  }
}

function Label({
  className,
  children,
}: {
  readonly className: string;
  readonly children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "-translate-y-full absolute -top-0.5 -left-0.5 whitespace-nowrap rounded-t-sm px-1.5 py-0.5 font-medium text-ink text-xs",
        className,
      )}
    >
      {children}
    </span>
  );
}

function placement(box: FaceBox, frame: FrameResult): CSSProperties {
  const percent = (value: number, of: number) => `${(value / of) * 100}%`;
  return {
    left: percent(box.x, frame.width),
    top: percent(box.y, frame.height),
    width: percent(box.width, frame.width),
    height: percent(box.height, frame.height),
  };
}
