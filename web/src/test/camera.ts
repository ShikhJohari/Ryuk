import { onTestFinished, vi } from "vitest";

/**
 * A webcam at the browser boundary, which jsdom lacks: `getUserMedia` gives a
 * stream (or fails with `error`), the video plays at `width` by `height`
 * (640x480), and every captured frame encodes to a 4-byte JPEG. `track`
 * can end, mute and unmute like a real camera's.
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
  // Each opening gives a new stream with a new track, as a browser does.
  let track = new FakeVideoTrack(stop);
  const getUserMedia = vi.fn(() => {
    if (error !== undefined) {
      return Promise.reject(error);
    }
    track = new FakeVideoTrack(stop);
    const current = track;
    return Promise.resolve({
      getTracks: () => [current],
      getVideoTracks: () => [current],
    } as unknown as MediaStream);
  });
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
  return {
    getUserMedia,
    stop,
    /** The track of the stream opened last. */
    get track() {
      return track;
    },
  };
}

/**
 * The stream's one video track. `end`, `mute` and `unmute` do what the
 * browser does when the camera is unplugged, or stops and resumes sending.
 */
class FakeVideoTrack extends EventTarget {
  readonly kind = "video";
  muted = false;
  readyState: MediaStreamTrackState = "live";
  /** Whether the page let go of this track. */
  stopped = false;

  constructor(private readonly onStop: () => void) {
    super();
  }

  /** As in a browser: ended, with no `ended` event. */
  stop() {
    this.stopped = true;
    this.readyState = "ended";
    this.onStop();
  }

  end() {
    this.readyState = "ended";
    this.dispatchEvent(new Event("ended"));
  }

  mute() {
    this.muted = true;
    this.dispatchEvent(new Event("mute"));
  }

  unmute() {
    this.muted = false;
    this.dispatchEvent(new Event("unmute"));
  }
}
