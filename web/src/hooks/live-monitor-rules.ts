import { Either } from "effect";
import {
  type ActiveModelChanged,
  CloseCode,
  decodeMonitorMessage,
  type Frame,
  type FrameResult,
  type SightingMessage,
} from "@/api/monitor";
import type { CameraProblem } from "@/lib/camera";

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

/** A captured picture, ready to go out as a frame. */
export type Picture = {
  readonly jpeg: ArrayBuffer;
  readonly width: number;
  readonly height: number;
};

/** What the platform observed when the event happened. */
export type Moment = {
  /** `performance.now()`; never goes backwards. */
  readonly now: number;
  /** Whether the tab is hidden. */
  readonly hidden: boolean;
};

export type Event =
  /** The camera opened and its `play()` settled. */
  | {
      readonly type: "camera_opened";
      readonly muted: boolean;
      readonly ended: boolean;
    }
  | { readonly type: "camera_failed"; readonly problem: CameraProblem }
  /** Opening the socket threw: the service's address is malformed. */
  | { readonly type: "connect_failed" }
  | { readonly type: "opened" }
  /** A socket message, as received; decoded here. */
  | { readonly type: "message"; readonly data: unknown }
  | { readonly type: "closed"; readonly code: number }
  /** The answer to `capture`: null when there was no picture, or it threw. */
  | {
      readonly type: "captured";
      readonly picture: Picture | null;
      /** Ms since the epoch. */
      readonly capturedAt: number;
    }
  | {
      readonly type: "track";
      readonly change: "muted" | "unmuted" | "ended";
    }
  /** The tab was shown or hidden; `Moment.hidden` says which. */
  | { readonly type: "visibility" }
  /** The wake-up time asked for arrived. */
  | { readonly type: "tick" }
  /** Unmounted, or this attempt replaced by a restart. */
  | { readonly type: "disposed" };

export type Command =
  | { readonly do: "connect" }
  | { readonly do: "capture" }
  | { readonly do: "send"; readonly frame: Frame }
  | {
      readonly do: "notify";
      readonly message: typeof ActiveModelChanged.Type | SightingMessage;
    }
  /** Close the socket, if any, and let go of the camera. */
  | { readonly do: "stop" };

/** What the operator sees. */
export type View = {
  readonly status: LiveMonitorStatus;
  readonly result: FrameResult | null;
  readonly framesPerSecond: number | null;
};

export type Step = {
  readonly view: View;
  /** To carry out in order; `stop` is always last. */
  readonly commands: ReadonlyArray<Command>;
  /** Feed a `tick` at or after this `now`; null means no wake-up is wanted. */
  readonly wakeAt: number | null;
};

type Phase =
  /** Waiting for the camera to open and play. */
  | "camera"
  | "connecting"
  | "open"
  /** Stopped by a close, a lost or refused camera, or a bad address. */
  | "halted"
  /** Unmounted, or restarted: this attempt is over and changes nothing. */
  | "disposed";

/**
 * The live monitor's rules for one attempt: told what happened and when, it
 * says what the operator sees, what to do, and when to wake it next.
 *
 * Frames are result-paced: the newest picture is sent as soon as the previous
 * frame's result arrives, at most 30 a second. Nothing is sent while the tab
 * is hidden or the camera is muted; a camera that ends stops the monitor.
 */
export function createLiveMonitor(): (event: Event, moment: Moment) => Step {
  let phase: Phase = "camera";
  let view: View = {
    status: { kind: "starting" },
    result: null,
    framesPerSecond: null,
  };
  let now = 0;
  let hidden = false;
  let commands: Array<Command> = [];
  let everOpened = false;
  let openedAt = 0;
  let muted = false;
  // A `capture` not yet answered by `captured`.
  let capturing = false;
  // A frame sent and not yet answered by a result or an error.
  let unanswered = false;
  let stalled = false;
  // Whether a frame rate is being shown; not while nothing is recognised.
  let measuring = false;
  const arrivals: Array<number> = [];
  let lastSentAt = Number.NEGATIVE_INFINITY;
  let seq = 0;
  // The three deadlines a `tick` serves: the next send, the stall and the
  // frame rate's refresh.
  let sendAt: number | null = null;
  let stallAt: number | null = null;
  let refreshAt: number | null = null;

  const setFramesPerSecond = (framesPerSecond: number | null) => {
    if (view.framesPerSecond !== framesPerSecond) {
      view = { ...view, framesPerSecond };
    }
  };

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
    while ((arrivals[0] ?? now) <= now - 1000) {
      arrivals.shift();
    }
    setFramesPerSecond(arrivals.length);
  };

  /** The status of an open socket, from what is going on. */
  const showRunning = () => {
    if (phase !== "open") {
      return;
    }
    const kind = muted ? "camera_lost" : stalled ? "stalled" : "running";
    if (view.status.kind !== kind) {
      view = { ...view, status: { kind } };
    }
  };

  /** Stop sending, let go of the camera and the socket, and say why. */
  const halt = (status: LiveMonitorStatus) => {
    phase = "halted";
    sendAt = null;
    stallAt = null;
    refreshAt = null;
    measuring = false;
    arrivals.length = 0;
    view = { status, result: null, framesPerSecond: null };
    commands.push({ do: "stop" });
  };

  const sendNext = () => {
    sendAt = null;
    if (phase !== "open" || capturing || unanswered || muted || hidden) {
      return;
    }
    capturing = true;
    commands.push({ do: "capture" });
  };

  // Timed from the first frame after the last result, errors included; a
  // no-op while one is already running.
  const armStall = () => {
    stallAt ??= now + RESULT_TIMEOUT_MS;
  };

  const stall = () => {
    stallAt = null;
    stalled = true;
    showRunning();
  };

  // Back from hidden or muted. A frame still unanswered is timed again:
  // nothing more is sent until it is answered, so sendNext would not.
  const resume = () => {
    if (capturing || unanswered) {
      armStall();
    } else if (sendAt === null) {
      sendNext();
    }
  };

  const sendAfterAnswer = () => {
    unanswered = false;
    sendAt = Math.max(now, lastSentAt + MIN_FRAME_INTERVAL_MS);
  };

  const refreshFrameRate = () => {
    refreshAt =
      openedAt +
      FRAME_RATE_REFRESH_MS *
        (Math.floor((now - openedAt) / FRAME_RATE_REFRESH_MS) + 1);
    updateFrameRate();
  };

  const onMessage = (data: unknown) => {
    const decoded = decodeMonitorMessage(data);
    if (Either.isLeft(decoded)) {
      console.error("Unreadable live monitor message", decoded.left);
      return;
    }
    const message = decoded.right;
    switch (message.type) {
      case "result": {
        stallAt = null;
        if (stalled) {
          stalled = false;
          showRunning();
        }
        if (!hidden && !muted) {
          measuring = true;
          arrivals.push(now);
          updateFrameRate();
        }
        view = { ...view, result: message };
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
      case "sighting_opened":
      case "sighting_updated":
      case "sighting_ended":
        commands.push({ do: "notify", message });
        break;
    }
  };

  const onCaptured = (picture: Picture | null, capturedAt: number) => {
    capturing = false;
    if (picture === null) {
      sendAt = now + NO_PICTURE_RETRY_MS;
      return;
    }
    seq += 1;
    lastSentAt = now;
    unanswered = true;
    commands.push({ do: "send", frame: { seq, capturedAt, ...picture } });
    armStall();
  };

  // The deadlines that have passed, in the order they fell due.
  const onTick = () => {
    const due: Array<readonly [number, () => void]> = [];
    if (sendAt !== null && sendAt <= now) {
      due.push([sendAt, sendNext]);
    }
    if (stallAt !== null && stallAt <= now) {
      due.push([stallAt, stall]);
    }
    if (refreshAt !== null && refreshAt <= now) {
      due.push([refreshAt, refreshFrameRate]);
    }
    due.sort(([a], [b]) => a - b);
    for (const [, run] of due) {
      run();
    }
  };

  const handle = (event: Event) => {
    if (event.type === "disposed") {
      phase = "disposed";
      return;
    }
    if (phase === "disposed") {
      // A camera that opens after unmount is let go at once.
      if (event.type === "camera_opened") {
        commands.push({ do: "stop" });
      }
      return;
    }
    if (phase === "halted") {
      return;
    }
    switch (event.type) {
      case "camera_failed":
        if (phase === "camera") {
          phase = "halted";
          view = {
            ...view,
            status: { kind: "no_camera", problem: event.problem },
          };
        }
        break;
      case "camera_opened":
        if (phase !== "camera") {
          break;
        }
        if (event.ended) {
          halt({ kind: "camera_lost" });
          break;
        }
        muted = event.muted;
        phase = "connecting";
        commands.push({ do: "connect" });
        break;
      case "connect_failed":
        if (phase === "connecting") {
          halt({ kind: "disconnected", reason: "invalid_url" });
        }
        break;
      case "opened":
        if (phase !== "connecting") {
          break;
        }
        phase = "open";
        everOpened = true;
        openedAt = now;
        refreshAt = now + FRAME_RATE_REFRESH_MS;
        showRunning();
        sendNext();
        break;
      case "closed":
        if (phase === "connecting" || phase === "open") {
          halt(closedStatus(event.code, everOpened));
        }
        break;
      case "message":
        if (phase === "open") {
          onMessage(event.data);
        }
        break;
      case "captured":
        if (phase === "open" && capturing) {
          onCaptured(event.picture, event.capturedAt);
        }
        break;
      case "track":
        switch (event.change) {
          case "ended":
            halt({ kind: "camera_lost" });
            break;
          // A muted camera sends no picture; it may come back.
          case "muted":
            muted = true;
            stallAt = null;
            resetFrameRate();
            showRunning();
            break;
          case "unmuted":
            muted = false;
            showRunning();
            resume();
            break;
        }
        break;
      case "visibility":
        if (hidden) {
          stallAt = null;
          resetFrameRate();
        } else {
          resume();
        }
        break;
      case "tick":
        if (phase === "open") {
          onTick();
        }
        break;
    }
  };

  const wakeAt = (): number | null => {
    if (phase !== "open") {
      return null;
    }
    return Math.min(
      sendAt ?? Number.POSITIVE_INFINITY,
      stallAt ?? Number.POSITIVE_INFINITY,
      refreshAt ?? Number.POSITIVE_INFINITY,
    );
  };

  return (event, moment) => {
    now = moment.now;
    hidden = moment.hidden;
    commands = [];
    handle(event);
    return { view, commands, wakeAt: wakeAt() };
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
