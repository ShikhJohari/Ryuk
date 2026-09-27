import { onTestFinished, vi } from "vitest";

/**
 * A webcam at the browser boundary, which jsdom lacks: `getUserMedia` gives a
 * stream (or fails with `error`), the video plays at 640x480, and every
 * captured frame encodes to a few JPEG bytes.
 */
export function fakeCamera({ error }: { readonly error?: DOMException } = {}) {
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
    640,
  );
  vi.spyOn(HTMLVideoElement.prototype, "videoHeight", "get").mockReturnValue(
    480,
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
