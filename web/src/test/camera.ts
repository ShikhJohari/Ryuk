import { onTestFinished, vi } from "vitest";

/**
 * A webcam at the browser boundary, which jsdom lacks: `getUserMedia` gives a
 * stream (or fails with `error`), the video plays at `width` by `height`
 * (640x480), and every captured frame encodes to a 4-byte JPEG.
 */
export function fakeCamera({
  error,
  width = 640,
  height = 480,
}: {
  readonly error?: DOMException;
  readonly width?: number;
  readonly height?: number;
} = {}) {
  const stop = vi.fn();
  const stream = { getTracks: () => [{ stop }] } as unknown as MediaStream;
  const getUserMedia = vi.fn(() =>
    error === undefined ? Promise.resolve(stream) : Promise.reject(error),
  );
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLVideoElement.prototype, "videoWidth", "get").mockReturnValue(
    width,
  );
  vi.spyOn(HTMLVideoElement.prototype, "videoHeight", "get").mockReturnValue(
    height,
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    drawImage: () => undefined,
  } as unknown as RenderingContext);
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(
    (callback) => {
      callback(new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])]));
    },
  );
  onTestFinished(() => {
    vi.restoreAllMocks();
    Reflect.deleteProperty(navigator, "mediaDevices");
  });
  return { getUserMedia, stop };
}
