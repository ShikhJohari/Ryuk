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
    // A size container, so a label can be kept inside the stage (`cqh`).
    <div className="absolute inset-0 [container-type:size]">
      {result.faces.map((face, index) => (
        <FaceMark
          // Faces have no identity across frames; their order is YuNet's.
          // biome-ignore lint/suspicious/noArrayIndexKey: see above
          key={index}
          face={face}
          style={placement(face.box, result)}
          labelStyle={labelPlacement(face.box, result)}
        />
      ))}
    </div>
  );
}

function FaceMark({
  face,
  style,
  labelStyle,
}: {
  readonly face: Face;
  readonly style: CSSProperties;
  readonly labelStyle: CSSProperties;
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
          <Label className="bg-match-video" style={labelStyle}>
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
            <Label className="bg-no-match-video" style={labelStyle}>
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

/** A label's height (`h-5`) and the box's border (`border-2`). */
const LABEL_HEIGHT = "1.25rem";
const BOX_BORDER = "0.125rem";

function Label({
  className,
  style,
  children,
}: {
  readonly className: string;
  readonly style: CSSProperties;
  readonly children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "absolute -left-0.5 h-5 whitespace-nowrap rounded-t-sm px-1.5 py-0.5 font-medium text-ink text-xs leading-4",
        className,
      )}
      style={style}
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

/**
 * A label sits on top of its box, unless the box starts too near the top of
 * the stage (or above it), where the stage would clip it: then it moves down
 * into the box, just far enough to be seen whole. `top` is from the box's
 * padding edge; `1cqh` is 1% of the stage's height.
 */
function labelPlacement(box: FaceBox, frame: FrameResult): CSSProperties {
  const boxTop = (box.y / frame.height) * 100;
  return {
    top: `max(calc(-1 * (${LABEL_HEIGHT} + ${BOX_BORDER})), calc(${-boxTop} * 1cqh - ${BOX_BORDER}))`,
  };
}
