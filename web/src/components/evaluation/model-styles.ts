import type { Network } from "@/api/evaluation";

/** How a model's series is drawn: hue, dash and marker together, so no cue rests on hue alone. */
export type SeriesStyle = {
  readonly colour: string;
  /** On and off lengths per unit of line width, as matplotlib scales them; null is solid. */
  readonly dashes: readonly [number, number] | null;
  readonly marker: "circle" | "square" | "diamond";
};

/**
 * #13's per-model style, as `ryuk.plotting.style.MODEL_STYLES` draws it for
 * the report: ArcFace solid navy, FaceNet long-dash violet, SFace dotted
 * ochre. The model hues are #47 Q12's own; the rest of a chart uses the
 * page's tokens.
 */
export const MODEL_STYLES: Readonly<Record<Network, SeriesStyle>> = {
  arcface: { colour: "#1F4E8C", dashes: null, marker: "circle" },
  facenet: { colour: "#8A4FA0", dashes: [7, 2.5], marker: "square" },
  sface: { colour: "#C98A1B", dashes: [1.2, 1.8], marker: "diamond" },
};
