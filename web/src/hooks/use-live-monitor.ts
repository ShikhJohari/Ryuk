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

/** Why the live monitor is not connected to the service. */
export type DisconnectReason =
  /** The socket never opened: the service is down, or refused the handshake. */
  | "unreachable"
  /** The socket opened, then closed. */
  | "lost"
  /** The service's localhost guard refused the socket. */
  | "refused"
  /** The service refused a frame over its size limit. */
  | "frame_too_large"
  /** The service failed while handling the socket. */
  | "service_error"
  /** The service's address is not a valid URL. */
  | "invalid_url";

export type LiveMonitorStatus =
  | { readonly kind: "starting" }
  | { readonly kind: "running" }
  /** Frames are going out, but no result has come back for a while. */
  | { readonly kind: "stalled" }
  | { readonly kind: "no_camera"; readonly problem: CameraProblem }
  /** The camera stopped mid-session: unplugged, muted or access withdrawn. */
  | { readonly kind: "camera_lost" }
  | { readonly kind: "no_active_model" }
  | { readonly kind: "superseded" }
  | { readonly kind: "disconnected"; readonly reason: DisconnectReason };

/** At most 30 frames a second are sent, however fast results come back. */
const MIN_FRAME_INTERVAL_MS = 1000 / 30;
/** How long to wait before trying again when there is no picture to capture. */
const NO_PICTURE_RETRY_MS = 100;
/** With no result for this long after a frame went out, the monitor is stalled. */
export const RESULT_TIMEOUT_MS = 5000;
/** How often the frame rate is recomputed, so it falls when results stop. */
const FRAME_RATE_REFRESH_MS = 250;

/**
 * The live monitor: the webcam into `video`, and its frames to the service.
 *
 * Frames are result-paced: the newest picture is sent as soon as the previous
 * frame's result arrives, at most 30 a second, so the rate follows what the
 * active model can do. Nothing is sent while the tab is hidden or the camera
 * is muted; a camera that ends stops the monitor.
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
    // Unmounted, or restarted: this run is over and must change nothing.
    let unmounted = false;
    // Halted by a close, a lost camera or a bad address: nothing more is sent.
    let halted = false;
    let stream: MediaStream | null = null;
    let track: MediaStreamTrack | null = null;
    let socket: WebSocket | null = null;
    let open = false;
    let everOpened = false;
    let awaitingResult = false;
    let muted = false;
    let stalled = false;
    // Whether a frame rate is being shown; not while nothing is recognised.
    let measuring = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stallTimer: ReturnType<typeof setTimeout> | undefined;
    let rateTimer: ReturnType<typeof setInterval> | undefined;
    let lastSentAt = Number.NEGATIVE_INFINITY;
    let seq = 0;

    const done = () => unmounted || halted;
    const hidden = () => document.visibilityState === "hidden";

    // No frame rate is shown while nothing is being recognised.
    const resetFrameRate = () => {
      measuring = false;
      arrivals.length = 0;
      setFramesPerSecond(null);
    };

    const updateFrameRate = () => {
      if (!measuring) {
        return;
      }
      const now = performance.now();
      while ((arrivals[0] ?? now) <= now - 1000) {
        arrivals.shift();
      }
      setFramesPerSecond(arrivals.length);
    };

    const clearStall = () => {
      clearTimeout(stallTimer);
      stallTimer = undefined;
    };

    /** The status of an open socket, from what is going on. */
    const showRunning = () => {
      if (!done() && open) {
        setStatus(
          muted
            ? { kind: "camera_lost" }
            : stalled
              ? { kind: "stalled" }
              : { kind: "running" },
        );
      }
    };

    const release = () => {
      if (track !== null) {
        track.removeEventListener("ended", onTrackEnded);
        track.removeEventListener("mute", onTrackMuted);
        track.removeEventListener("unmute", onTrackUnmuted);
        track = null;
      }
      if (stream !== null) {
        stopCamera(stream);
        stream = null;
      }
      element.srcObject = null;
    };

    /** Stop sending, let go of the camera and the socket, and say why. */
    const halt = (next: LiveMonitorStatus) => {
      if (done()) {
        return;
      }
      halted = true;
      open = false;
      clearTimeout(timer);
      timer = undefined;
      clearStall();
      clearInterval(rateTimer);
      socket?.close();
      release();
      setResult(null);
      resetFrameRate();
      setStatus(next);
    };

    const sendNext = async () => {
      timer = undefined;
      if (done() || !open || awaitingResult || muted || hidden()) {
        return;
      }
      awaitingResult = true;
      const frame = await captureFrame(element, canvas).catch((error) => {
        console.warn("Could not capture a frame", error);
        return null;
      });
      if (done() || socket === null) {
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
      // From the first frame after the last result, errors included.
      stallTimer ??= setTimeout(() => {
        stallTimer = undefined;
        stalled = true;
        showRunning();
      }, RESULT_TIMEOUT_MS);
    };

    const sendAfterAnswer = () => {
      awaitingResult = false;
      const wait = lastSentAt + MIN_FRAME_INTERVAL_MS - performance.now();
      timer = setTimeout(sendNext, Math.max(0, wait));
    };

    const onMessage = (event: MessageEvent) => {
      if (done()) {
        return;
      }
      const decoded = decodeMonitorMessage(event.data);
      if (Either.isLeft(decoded)) {
        console.error("Unreadable live monitor message", decoded.left);
        return;
      }
      const message = decoded.right;
      switch (message.type) {
        case "result": {
          clearStall();
          if (stalled) {
            stalled = false;
            showRunning();
          }
          if (!hidden() && !muted) {
            measuring = true;
            arrivals.push(performance.now());
            updateFrameRate();
          }
          setResult(message);
          sendAfterAnswer();
          break;
        }
        case "error":
          // The service could not use the frame; the next one may do. Only a
          // result ends a stall.
          console.warn("Live monitor frame refused", message.code);
          sendAfterAnswer();
          break;
        case "active_model_changed":
          activeModelChanged.current();
          break;
      }
    };

    const onClose = (event: CloseEvent) => {
      halt(closedStatus(event.code, everOpened));
    };

    const onTrackEnded = () => {
      halt({ kind: "camera_lost" });
    };

    // A muted camera sends no picture; it may come back.
    const onTrackMuted = () => {
      if (done()) {
        return;
      }
      muted = true;
      clearStall();
      resetFrameRate();
      showRunning();
    };

    const onTrackUnmuted = () => {
      if (done()) {
        return;
      }
      muted = false;
      showRunning();
      if (timer === undefined) {
        void sendNext();
      }
    };

    const onVisibilityChange = () => {
      if (hidden()) {
        clearStall();
        resetFrameRate();
      } else if (timer === undefined) {
        void sendNext();
      }
    };

    const start = async () => {
      setStatus({ kind: "starting" });
      setResult(null);
      resetFrameRate();
      try {
        stream = await openCamera();
      } catch (error) {
        if (!unmounted) {
          setStatus({
            kind: "no_camera",
            problem:
              error instanceof CameraError ? error.problem : "unsupported",
          });
        }
        return;
      }
      if (unmounted) {
        release();
        return;
      }
      track = stream.getVideoTracks()[0] ?? null;
      track?.addEventListener("ended", onTrackEnded);
      track?.addEventListener("mute", onTrackMuted);
      track?.addEventListener("unmute", onTrackUnmuted);
      muted = track?.muted ?? false;
      element.srcObject = stream;
      try {
        await element.play();
      } catch {
        // Capture reads the picture, not playback; a refused play() changes nothing.
      }
      if (done()) {
        return;
      }
      if (track?.readyState === "ended") {
        halt({ kind: "camera_lost" });
        return;
      }
      try {
        socket = new WebSocket(monitorUrl());
      } catch (error) {
        // A malformed VITE_API_BASE_URL: new URL or new WebSocket throws.
        console.error("Could not open the live monitor socket", error);
        halt({ kind: "disconnected", reason: "invalid_url" });
        return;
      }
      socket.binaryType = "arraybuffer";
      socket.addEventListener("open", () => {
        if (done()) {
          return;
        }
        open = true;
        everOpened = true;
        rateTimer = setInterval(updateFrameRate, FRAME_RATE_REFRESH_MS);
        showRunning();
        void sendNext();
      });
      socket.addEventListener("message", onMessage);
      socket.addEventListener("close", onClose);
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    void start();

    return () => {
      unmounted = true;
      clearTimeout(timer);
      clearStall();
      clearInterval(rateTimer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      socket?.close();
      release();
    };
  }, [video, attempt]);

  return {
    status,
    result,
    framesPerSecond,
    /** Start again: reconnect, or take the live monitor back from another tab. */
    restart: () => setAttempt((count) => count + 1),
  };
}

/** What a socket's close code means for the monitor. */
function closedStatus(code: number, everOpened: boolean): LiveMonitorStatus {
  switch (code) {
    case CloseCode.superseded:
      return { kind: "superseded" };
    case CloseCode.noActiveModel:
      return { kind: "no_active_model" };
    case CloseCode.policyViolation:
      return { kind: "disconnected", reason: "refused" };
    case CloseCode.messageTooBig:
      return { kind: "disconnected", reason: "frame_too_large" };
    case CloseCode.internalError:
      return { kind: "disconnected", reason: "service_error" };
    default:
      // A refused handshake and a service that is down both arrive as 1006
      // before the socket opens, and cannot be told apart.
      return {
        kind: "disconnected",
        reason: everOpened ? "lost" : "unreachable",
      };
  }
}
