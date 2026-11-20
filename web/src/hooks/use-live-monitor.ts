import { type RefObject, useEffect, useRef, useState } from "react";
import {
  encodeFrame,
  type FrameResult,
  monitorUrl,
  type SightingMessage,
} from "@/api/monitor";
import {
  CameraError,
  captureFrame,
  openCamera,
  stopCamera,
} from "@/lib/camera";
import {
  type Command,
  createLiveMonitor,
  type Event,
  type LiveMonitorStatus,
} from "./live-monitor-rules";

export {
  type DisconnectReason,
  type LiveMonitorStatus,
  RESULT_TIMEOUT_MS,
} from "./live-monitor-rules";

/**
 * The live monitor: the webcam into `video`, and its frames to the service.
 *
 * Frames are result-paced: the newest picture is sent as soon as the previous
 * frame's result arrives, at most 30 a second, so the rate follows what the
 * active model can do. Nothing is sent while the tab is hidden or the camera
 * is muted; a camera that ends stops the monitor. The rules live in
 * `live-monitor-rules`; this hook carries out what they say.
 */
export function useLiveMonitor({
  video,
  onActiveModelChanged,
  onSighting,
}: {
  readonly video: RefObject<HTMLVideoElement | null>;
  readonly onActiveModelChanged: () => void;
  /** A sighting opened, progressed or ended. */
  readonly onSighting: (message: SightingMessage) => void;
}) {
  const [status, setStatus] = useState<LiveMonitorStatus>({ kind: "starting" });
  const [result, setResult] = useState<FrameResult | null>(null);
  const [framesPerSecond, setFramesPerSecond] = useState<number | null>(null);
  const [attempt, setAttempt] = useState(0);
  // Read when the message arrives, so a new callback never reopens the socket.
  const activeModelChanged = useRef(onActiveModelChanged);
  activeModelChanged.current = onActiveModelChanged;
  const sightingChanged = useRef(onSighting);
  sightingChanged.current = onSighting;

  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` restarts the monitor
  useEffect(() => {
    const element = video.current;
    if (element === null) {
      return;
    }
    const canvas = document.createElement("canvas");
    const step = createLiveMonitor();
    // Unmounted, or restarted: this run is over and must change nothing.
    let unmounted = false;
    let stream: MediaStream | null = null;
    let track: MediaStreamTrack | null = null;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let timerAt: number | null = null;

    /** Tell the rules what happened, then do what they say. */
    const run = (event: Event) => {
      const now = performance.now();
      const { view, commands, wakeAt } = step(event, {
        now,
        hidden: document.visibilityState === "hidden",
      });
      if (!unmounted) {
        setStatus(view.status);
        setResult(view.result);
        setFramesPerSecond(view.framesPerSecond);
      }
      // Before the commands: one may report back at once, and its wake-up
      // time replaces this one.
      wake(wakeAt, now);
      for (const command of commands) {
        carryOut(command);
      }
    };

    const wake = (at: number | null, now: number) => {
      if (at === timerAt) {
        return;
      }
      clearTimeout(timer);
      timerAt = at;
      timer =
        at === null
          ? undefined
          : setTimeout(
              () => {
                timer = undefined;
                timerAt = null;
                run({ type: "tick" });
              },
              Math.max(0, Math.ceil(at - now)),
            );
    };

    const carryOut = (command: Command) => {
      switch (command.do) {
        case "connect":
          connect();
          break;
        case "capture":
          void capture();
          break;
        case "send":
          socket?.send(encodeFrame(command.frame));
          break;
        case "notify":
          if (command.message.type === "active_model_changed") {
            activeModelChanged.current();
          } else {
            sightingChanged.current(command.message);
          }
          break;
        case "stop":
          socket?.close();
          release();
          break;
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

    const capture = async () => {
      const picture = await captureFrame(element, canvas).catch((error) => {
        console.warn("Could not capture a frame", error);
        return null;
      });
      run({ type: "captured", picture, capturedAt: Date.now() });
    };

    const connect = () => {
      try {
        socket = new WebSocket(monitorUrl());
      } catch (error) {
        // A malformed VITE_API_BASE_URL: new URL or new WebSocket throws.
        console.error("Could not open the live monitor socket", error);
        run({ type: "connect_failed" });
        return;
      }
      socket.binaryType = "arraybuffer";
      socket.addEventListener("open", () => run({ type: "opened" }));
      socket.addEventListener("message", (event) =>
        run({ type: "message", data: event.data }),
      );
      socket.addEventListener("close", (event) =>
        run({ type: "closed", code: event.code }),
      );
    };

    const onTrackEnded = () => run({ type: "track", change: "ended" });
    const onTrackMuted = () => run({ type: "track", change: "muted" });
    const onTrackUnmuted = () => run({ type: "track", change: "unmuted" });
    const onVisibilityChange = () => run({ type: "visibility" });

    const start = async () => {
      setStatus({ kind: "starting" });
      setResult(null);
      setFramesPerSecond(null);
      try {
        stream = await openCamera();
      } catch (error) {
        run({
          type: "camera_failed",
          problem: error instanceof CameraError ? error.problem : "unsupported",
        });
        return;
      }
      // A camera that opens after unmount is never played; the rules let go
      // of it.
      if (!unmounted) {
        track = stream.getVideoTracks()[0] ?? null;
        track?.addEventListener("ended", onTrackEnded);
        track?.addEventListener("mute", onTrackMuted);
        track?.addEventListener("unmute", onTrackUnmuted);
        element.srcObject = stream;
        try {
          await element.play();
        } catch {
          // Capture reads the picture, not playback; a refused play() changes nothing.
        }
      }
      run({
        type: "camera_opened",
        muted: track?.muted ?? false,
        ended: track?.readyState === "ended",
      });
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    void start();

    return () => {
      unmounted = true;
      run({ type: "disposed" });
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
