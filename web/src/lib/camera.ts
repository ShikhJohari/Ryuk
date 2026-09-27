/** Why the webcam could not be opened. */
export type CameraProblem = "denied" | "missing" | "busy" | "unsupported";

export class CameraError extends Error {
  constructor(readonly problem: CameraProblem) {
    super(`The camera could not be opened: ${problem}`);
  }
}

/** Frames are sent with their long side at most this many pixels. */
const MAX_FRAME_SIDE = 640;
const JPEG_QUALITY = 0.7;

/** The operator's webcam, asked for at 640x480. */
export async function openCamera(): Promise<MediaStream> {
  if (navigator.mediaDevices?.getUserMedia === undefined) {
    throw new CameraError("unsupported");
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
    });
  } catch (error) {
    throw new CameraError(cameraProblem(error));
  }
}

function cameraProblem(error: unknown): CameraProblem {
  // getUserMedia rejects with a DOMException, which is not always an Error.
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? error.name
      : undefined;
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "denied";
    case "NotFoundError":
    case "OverconstrainedError":
      return "missing";
    case "NotReadableError":
    case "AbortError":
      return "busy";
    default:
      return "unsupported";
  }
}

export function stopCamera(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

/**
 * The video's current picture as a JPEG, its long side at most 640 px, or
 * null when there is no picture yet.
 */
export async function captureFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): Promise<{
  readonly jpeg: ArrayBuffer;
  readonly width: number;
  readonly height: number;
} | null> {
  const { videoWidth, videoHeight } = video;
  if (videoWidth === 0 || videoHeight === 0) {
    return null;
  }
  const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(videoWidth, videoHeight));
  const width = Math.round(videoWidth * scale);
  const height = Math.round(videoHeight * scale);
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (context === null) {
    return null;
  }
  context.drawImage(video, 0, 0, width, height);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
  );
  return blob === null
    ? null
    : { jpeg: await blob.arrayBuffer(), width, height };
}
