import { Schema } from "effect";
import type { Assert, Equals } from "@/lib/type-equality";
import { API_BASE_URL } from "./base-url";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

/** A face's box in the pixels of the frame it was found in; not clipped. */
export const FaceBox = Schema.Struct({
  x: Schema.Number,
  y: Schema.Number,
  width: Schema.Number,
  height: Schema.Number,
}).annotations({ identifier: "FaceBox" });
export type FaceBox = typeof FaceBox.Type;
export type FaceBoxMatchesContract = Assert<
  Equals<FaceBox, Schemas["FaceBox"]>
>;

export const MatchedPerson = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
}).annotations({ identifier: "MatchedPerson" });
export type MatchedPersonMatchesContract = Assert<
  Equals<typeof MatchedPerson.Type, Schemas["MatchedPerson"]>
>;

/** Only a match names anyone. */
export const MatchFace = Schema.Struct({
  outcome: Schema.Literal("match"),
  box: FaceBox,
  score: Schema.Number,
  person: MatchedPerson,
}).annotations({ identifier: "MatchFace" });
export type MatchFaceMatchesContract = Assert<
  Equals<typeof MatchFace.Type, Schemas["MatchFace"]>
>;

/** `score` is null when nobody is on the watchlist to compare against. */
export const NoMatchFace = Schema.Struct({
  outcome: Schema.Literal("no_match"),
  box: FaceBox,
  score: Schema.NullOr(Schema.Number),
}).annotations({ identifier: "NoMatchFace" });
export type NoMatchFaceMatchesContract = Assert<
  Equals<typeof NoMatchFace.Type, Schemas["NoMatchFace"]>
>;

export const TooSmallFace = Schema.Struct({
  outcome: Schema.Literal("too_small"),
  box: FaceBox,
}).annotations({ identifier: "TooSmallFace" });
export type TooSmallFaceMatchesContract = Assert<
  Equals<typeof TooSmallFace.Type, Schemas["TooSmallFace"]>
>;

export const Face = Schema.Union(MatchFace, NoMatchFace, TooSmallFace);
export type Face = typeof Face.Type;
export type FaceMatchesContract = Assert<Equals<Face, Schemas["Face"]>>;

export const FrameResult = Schema.Struct({
  type: Schema.Literal("result"),
  seq: Schema.Int,
  capturedAt: Schema.Int,
  width: Schema.Int,
  height: Schema.Int,
  modelKey: Schema.String,
  threshold: Schema.Number,
  faces: Schema.Array(Face),
}).annotations({ identifier: "FrameResult" });
export type FrameResult = typeof FrameResult.Type;
export type FrameResultMatchesContract = Assert<
  Equals<FrameResult, Schemas["FrameResult"]>
>;

export const ActiveModelChanged = Schema.Struct({
  type: Schema.Literal("active_model_changed"),
  modelKey: Schema.String,
  threshold: Schema.Number,
}).annotations({ identifier: "ActiveModelChanged" });
export type ActiveModelChangedMatchesContract = Assert<
  Equals<typeof ActiveModelChanged.Type, Schemas["ActiveModelChanged"]>
>;

/** A message the service could not use as a frame; the socket stays open. */
export const MonitorError = Schema.Struct({
  type: Schema.Literal("error"),
  seq: Schema.NullOr(Schema.Int),
  code: Schema.String,
  detail: Schema.String,
}).annotations({ identifier: "MonitorError" });
export type MonitorErrorMatchesContract = Assert<
  Equals<typeof MonitorError.Type, Schemas["MonitorError"]>
>;

export const MonitorMessage = Schema.Union(
  FrameResult,
  ActiveModelChanged,
  MonitorError,
);
export type MonitorMessage = typeof MonitorMessage.Type;
export type MonitorMessageMatchesContract = Assert<
  Equals<MonitorMessage, Schemas["MonitorMessage"]>
>;

/** A text message from the service, decoded; Left if it is not one. */
export const decodeMonitorMessage = Schema.decodeUnknownEither(
  Schema.parseJson(MonitorMessage),
);

/** Why the service closed the socket. */
export const CloseCode = {
  /** The live monitor was opened in another tab, which took over. */
  superseded: 4001,
  /** No evaluated recognition model can be active. */
  noActiveModel: 4002,
} as const;

/** One captured frame, as the service reads it. */
export type Frame = {
  readonly seq: number;
  /** Capture time, ms since the epoch. */
  readonly capturedAt: number;
  readonly width: number;
  readonly height: number;
  readonly jpeg: ArrayBuffer;
};

const FRAME_MESSAGE = 0x01;
const HEADER_BYTES = 17;

/**
 * The binary frame message (#5): a 17-byte big-endian header (message type,
 * sequence number, capture time, width, height), then the JPEG.
 */
export function encodeFrame(frame: Frame): ArrayBuffer {
  const message = new Uint8Array(HEADER_BYTES + frame.jpeg.byteLength);
  const header = new DataView(message.buffer);
  header.setUint8(0, FRAME_MESSAGE);
  header.setUint32(1, frame.seq);
  header.setBigUint64(5, BigInt(frame.capturedAt));
  header.setUint16(13, frame.width);
  header.setUint16(15, frame.height);
  message.set(new Uint8Array(frame.jpeg), HEADER_BYTES);
  return message.buffer;
}

/** The live monitor's WebSocket URL, beside the REST API. */
export function monitorUrl(): string {
  const url = new URL(
    "/api/monitor",
    API_BASE_URL === "" ? window.location.href : API_BASE_URL,
  );
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.href;
}
