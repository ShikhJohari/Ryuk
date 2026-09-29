import { type WebSocketData, ws } from "msw";
import type { SetupServer } from "msw/node";
import type { RecognitionModelInfo } from "@/api/models";
import type { Face, FrameResult, MonitorMessage } from "@/api/monitor";

/** A frame message as the service's parser reads it (`>BIQHH`, then JPEG). */
export type ReceivedFrame = {
  readonly type: number;
  readonly seq: number;
  readonly capturedAt: number;
  readonly width: number;
  readonly height: number;
  readonly jpegBytes: number;
};

/**
 * The service's live monitor socket, answered by the test: the sequence
 * number of every frame received is recorded in `frames`, its whole header
 * in `received`, and `send` and `close` act on the latest connection.
 */
export function mockMonitor(
  server: SetupServer,
  {
    refuseWith,
  }: {
    /** Close every connection with this code before it opens. */
    readonly refuseWith?: number;
  } = {},
) {
  const frames: Array<number> = [];
  const received: Array<ReceivedFrame> = [];
  let connections = 0;
  let send: (message: MonitorMessage) => void = () => undefined;
  let close: (code: number) => void = () => undefined;
  server.use(
    ws.link("*/api/monitor").addEventListener("connection", ({ client }) => {
      connections += 1;
      if (refuseWith !== undefined) {
        client.close(refuseWith, "refused by the test");
        return;
      }
      send = (message) => client.send(JSON.stringify(message));
      close = (code) => client.close(code, "closed by the test");
      client.addEventListener("message", async (event) => {
        const frame = await readFrame(event.data);
        received.push(frame);
        frames.push(frame.seq);
      });
    }),
  );
  return {
    frames,
    received,
    connections: () => connections,
    send: (message: MonitorMessage) => send(message),
    close: (code: number) => close(code),
  };
}

/** A frame message's 17-byte big-endian header, read field by field. */
async function readFrame(data: WebSocketData): Promise<ReceivedFrame> {
  const bytes =
    data instanceof Blob ? await data.arrayBuffer() : (data as ArrayBuffer);
  const header = new DataView(bytes);
  return {
    type: header.getUint8(0),
    seq: header.getUint32(1),
    capturedAt: Number(header.getBigUint64(5)),
    width: header.getUint16(13),
    height: header.getUint16(15),
    jpegBytes: bytes.byteLength - 17,
  };
}

export function frameResult(
  seq: number,
  faces: ReadonlyArray<Face>,
): FrameResult {
  return {
    type: "result",
    seq,
    capturedAt: 1_732_000_000_000,
    width: 640,
    height: 480,
    modelKey: sface.id,
    threshold: 0.498,
    faces,
  };
}

export const box = { x: 200, y: 120, width: 160, height: 200 };

export function recognitionModel(
  network: RecognitionModelInfo["network"],
  name: string,
  state: RecognitionModelInfo["state"],
  threshold: number | null,
): RecognitionModelInfo {
  return {
    id: `${network}-cpu-${"0".repeat(64)}`,
    network,
    provider: "cpu",
    weightsSha256: "0".repeat(64),
    name,
    state,
    threshold,
    dimension: 128,
    msPerFace: threshold === null ? null : 5.4,
  };
}

export const sface = recognitionModel("sface", "SFace", "active", 0.498);
export const facenet = recognitionModel(
  "facenet",
  "FaceNet",
  "available",
  0.709,
);
export const arcface = recognitionModel(
  "arcface",
  "ArcFace (CPU)",
  "not_evaluated",
  null,
);
