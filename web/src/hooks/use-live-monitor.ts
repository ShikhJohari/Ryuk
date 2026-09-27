import { Either } from "effect";
import { type RefObject, useEffect, useRef, useState } from "react";
import {
  CloseCode,
  decodeMonitorMessage,
  encodeFrame,
  type FrameResult,
  monitorUrl,
} from "@/api/monitor";
import {
  CameraError,
  type CameraProblem,
  captureFrame,
  openCamera,
  stopCamera,
} from "@/lib/camera";

export type LiveMonitorStatus =
  | { readonly kind: "starting" }
  | { readonly kind: "running" }
  | { readonly kind: "no_camera"; readonly problem: CameraProblem }
  | { readonly kind: "no_active_model" }
  | { readonly kind: "superseded" }
  | { readonly kind: "disconnected" };

/** At most 30 frames a second are sent, however fast results come back. */
const MIN_FRAME_INTERVAL_MS = 1000 / 30;
/** How long to wait for the video's first picture before trying again. */
const NO_PICTURE_RETRY_MS = 100;

/**
 * The live monitor: the webcam into `video`, and its frames to the service.
 *
 * Frames are result-paced: the newest picture is sent as soon as the previous
 * frame's result arrives, at most 30 a second, so the rate follows what the
 * active model can do. Nothing is sent while the tab is hidden.
 */
export function useLiveMonitor({
  video,
  onActiveModelChanged,
}: {
  readonly video: RefObject<HTMLVideoElement | null>;
  readonly onActiveModelChanged: () => void;
}) {
  const [status, setStatus] = useState<LiveMonitorStatus>({ kind: "starting" });
  const [result, setResult] = useState<FrameResult | null>(null);
  const [framesPerSecond, setFramesPerSecond] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);
  // Read when the message arrives, so a new callback never reopens the socket.
  const activeModelChanged = useRef(onActiveModelChanged);
  activeModelChanged.current = onActiveModelChanged;

  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` restarts the monitor
  useEffect(() => {
    const element = video.current;
    if (element === null) {
      return;
    }
    const canvas = document.createElement("canvas");
    const arrivals: Array<number> = [];
    let stopped = false;
    let stream: MediaStream | null = null;
    let socket: WebSocket | null = null;
    let open = false;
    let awaitingResult = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let lastSentAt = Number.NEGATIVE_INFINITY;
    let seq = 0;

    const release = () => {
      if (stream !== null) {
        stopCamera(stream);
        stream = null;
      }
      element.srcObject = null;
    };

    const sendNext = async () => {
      timer = undefined;
      if (
        stopped ||
        !open ||
        awaitingResult ||
        document.visibilityState === "hidden"
      ) {
        return;
      }
      awaitingResult = true;
      const frame = await captureFrame(element, canvas);
      if (stopped || socket === null) {
        return;
      }
      if (frame === null) {
        awaitingResult = false;
        timer = setTimeout(sendNext, NO_PICTURE_RETRY_MS);
        return;
      }
      seq += 1;
      lastSentAt = performance.now();
      socket.send(encodeFrame({ seq, capturedAt: Date.now(), ...frame }));
    };

    const sendAfterAnswer = () => {
      awaitingResult = false;
      const wait = lastSentAt + MIN_FRAME_INTERVAL_MS - performance.now();
      timer = setTimeout(sendNext, Math.max(0, wait));
    };

    const onMessage = (event: MessageEvent) => {
      const decoded = decodeMonitorMessage(event.data);
      if (Either.isLeft(decoded)) {
        console.error("Unreadable live monitor message", decoded.left);
        return;
      }
      const message = decoded.right;
      switch (message.type) {
        case "result": {
          const now = performance.now();
          arrivals.push(now);
          while ((arrivals[0] ?? now) <= now - 1000) {
            arrivals.shift();
          }
          setResult(message);
          setFramesPerSecond(arrivals.length);
          sendAfterAnswer();
          break;
        }
        case "error":
          // The service could not use the frame; the next one may do.
          console.warn("Live monitor frame refused", message.code);
          sendAfterAnswer();
          break;
        case "active_model_changed":
          activeModelChanged.current();
          break;
      }
    };

    const onClose = (event: CloseEvent) => {
      open = false;
      if (stopped) {
        return;
      }
      release();
      setStatus(
        event.code === CloseCode.superseded
          ? { kind: "superseded" }
          : event.code === CloseCode.noActiveModel
            ? { kind: "no_active_model" }
            : { kind: "disconnected" },
      );
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible" && timer === undefined) {
        void sendNext();
      }
    };

    const start = async () => {
      setStatus({ kind: "starting" });
      setResult(null);
      setFramesPerSecond(null);
      try {
        stream = await openCamera();
      } catch (error) {
        if (!stopped) {
          setStatus({
            kind: "no_camera",
            problem:
              error instanceof CameraError ? error.problem : "unsupported",
          });
        }
        return;
      }
      if (stopped) {
        release();
        return;
      }
      element.srcObject = stream;
      try {
        await element.play();
      } catch {
        // Capture reads the picture, not playback; a refused play() changes nothing.
      }
      if (stopped) {
        return;
      }
      socket = new WebSocket(monitorUrl());
      socket.binaryType = "arraybuffer";
      socket.addEventListener("open", () => {
        open = true;
        setStatus({ kind: "running" });
        void sendNext();
      });
      socket.addEventListener("message", onMessage);
      socket.addEventListener("close", onClose);
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    void start();

    return () => {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      socket?.close();
      release();
    };
  }, [video, attempt]);

  return {
    status,
    result,
    framesPerSecond,
    /** Start again, taking the live monitor back from another tab. */
    restart: () => setAttempt((count) => count + 1),
  };
}
