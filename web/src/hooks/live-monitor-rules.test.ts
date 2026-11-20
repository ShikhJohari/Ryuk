import { describe, expect, it, vi } from "vitest";
import type { SightingMessage } from "@/api/monitor";
import {
  type Command,
  createLiveMonitor,
  type Event,
  type Picture,
  RESULT_TIMEOUT_MS,
  type View,
} from "./live-monitor-rules";

/** Capture times are ms since the epoch; a timeline's 0 is this. */
const EPOCH = 1_732_000_000_000;

const picture: Picture = { jpeg: new ArrayBuffer(4), width: 640, height: 480 };

const cameraOpened: Event = {
  type: "camera_opened",
  muted: false,
  ended: false,
};
const opened: Event = { type: "opened" };
/** The camera opens and plays, and the socket opens, all at `ms`. */
const startAt = (ms: number) =>
  [
    [ms, cameraOpened],
    [ms, opened],
  ] as const;

const message = (body: object): Event => ({
  type: "message",
  data: JSON.stringify(body),
});
const result = (seq: number) =>
  message({
    type: "result",
    seq,
    capturedAt: EPOCH,
    width: 640,
    height: 480,
    modelKey: "sface",
    threshold: 0.498,
    faces: [],
  });
const refusal = (seq: number) =>
  message({ type: "error", seq, code: "invalid_frame", detail: "Refused." });
const closed = (code: number): Event => ({ type: "closed", code });
const track = (change: "muted" | "unmuted" | "ended"): Event => ({
  type: "track",
  change,
});
const captured = (shot: Picture | null, ms: number): Event => ({
  type: "captured",
  picture: shot,
  capturedAt: EPOCH + ms,
});

/** What happened at a millisecond: an event, or the tab hidden or shown. */
type Happening = Event | "hide" | "show";

type Seen = {
  readonly view: View;
  /** Every command given so far, with the millisecond it was given at. */
  readonly commands: ReadonlyArray<readonly [number, Command]>;
  readonly wakeAt: number | null;
};

/**
 * One live monitor through `entries`, each `[ms, what happened]`, ticked only
 * at the wake-up times it asks for. By default the camera answers each
 * `capture` at once with a picture; `camera` can give none, or leave every
 * `capture` to the entries. With `answerAfter`, the service answers each
 * frame with a result that many ms after it was sent. At the same
 * millisecond, events come before a tick.
 */
function timeline(
  entries: ReadonlyArray<readonly [number, Happening]>,
  {
    camera = () => picture,
    answerAfter,
  }: {
    readonly camera?: ((ms: number) => Picture | null) | "by_hand";
    readonly answerAfter?: number;
  } = {},
) {
  return {
    /** What the operator sees at `ms`, with everything up to then done. */
    at(ms: number): Seen {
      const step = createLiveMonitor();
      const queue = entries.map(([at, happening], order) => ({
        at,
        happening,
        order,
      }));
      let order = queue.length;
      let hidden = false;
      // Set by `feed`, which the checker cannot follow into.
      let view = null as View | null;
      let wakeAt = null as number | null;
      const commands: Array<readonly [number, Command]> = [];

      const feed = (at: number, happening: Happening) => {
        if (happening === "hide" || happening === "show") {
          hidden = happening === "hide";
        }
        const event: Event =
          typeof happening === "string" ? { type: "visibility" } : happening;
        const next = step(event, { now: at, hidden });
        view = next.view;
        wakeAt = next.wakeAt;
        for (const command of next.commands) {
          commands.push([at, command]);
          if (command.do === "capture" && camera !== "by_hand") {
            queue.push({
              at,
              happening: captured(camera(at), at),
              order: order++,
            });
          }
          if (command.do === "send" && answerAfter !== undefined) {
            queue.push({
              at: at + answerAfter,
              happening: result(command.frame.seq),
              order: order++,
            });
          }
        }
      };

      for (let steps = 0; ; steps++) {
        if (steps > 100_000) {
          throw new Error("The timeline never settles");
        }
        queue.sort((a, b) => a.at - b.at || a.order - b.order);
        const next = queue[0];
        if (
          wakeAt !== null &&
          wakeAt <= ms &&
          (next === undefined || wakeAt < next.at)
        ) {
          feed(wakeAt, { type: "tick" });
        } else if (next !== undefined && next.at <= ms) {
          queue.shift();
          feed(next.at, next.happening);
        } else {
          break;
        }
      }
      if (view === null) {
        throw new Error(`Nothing has happened by ${ms} ms`);
      }
      return { view, commands, wakeAt };
    },
  };
}

/** The frames sent, as `[ms, seq]`. */
const sent = (seen: Seen) =>
  seen.commands.flatMap(([ms, command]) =>
    command.do === "send" ? [[ms, command.frame.seq] as const] : [],
  );

/** When each `capture` was asked for. */
const captures = (seen: Seen) =>
  seen.commands.flatMap(([ms, command]) =>
    command.do === "capture" ? [ms] : [],
  );

/** The commands given at `ms`. */
const commandsAt = (seen: Seen, ms: number) =>
  seen.commands.filter(([at]) => at === ms).map(([, command]) => command);

describe("starting", () => {
  it("connects once the camera has opened and played, and runs once the socket opens", () => {
    const monitor = timeline([
      [0, cameraOpened],
      [40, opened],
    ]);

    expect(monitor.at(0)).toEqual({
      view: {
        status: { kind: "starting" },
        result: null,
        framesPerSecond: null,
      },
      commands: [[0, { do: "connect" }]],
      wakeAt: null,
    });
    const running = monitor.at(40);
    expect(running.view.status).toEqual({ kind: "running" });
    expect(commandsAt(running, 40)).toEqual([
      { do: "capture" },
      {
        do: "send",
        frame: { seq: 1, capturedAt: EPOCH + 40, ...picture },
      },
    ]);
  });

  it("says why the camera could not be opened, and does nothing more", () => {
    const seen = timeline([
      [0, { type: "camera_failed", problem: "denied" }],
      [10, "hide"],
      [20, "show"],
    ]).at(20);

    expect(seen).toEqual({
      view: {
        status: { kind: "no_camera", problem: "denied" },
        result: null,
        framesPerSecond: null,
      },
      commands: [],
      wakeAt: null,
    });
  });

  it("stops at a malformed service address, letting go of the camera", () => {
    const seen = timeline([
      [0, cameraOpened],
      [0, { type: "connect_failed" }],
    ]).at(0);

    expect(seen.view.status).toEqual({
      kind: "disconnected",
      reason: "invalid_url",
    });
    expect(seen.commands).toEqual([
      [0, { do: "connect" }],
      [0, { do: "stop" }],
    ]);
  });

  it("stops without connecting when the camera has ended by the time play() settles", () => {
    const seen = timeline([
      [0, { type: "camera_opened", muted: false, ended: true }],
    ]).at(0);

    expect(seen.view.status).toEqual({ kind: "camera_lost" });
    expect(seen.commands).toEqual([[0, { do: "stop" }]]);
  });

  it("stops when the camera ends while it is starting to play", () => {
    const seen = timeline([
      [0, track("ended")],
      [30, cameraOpened],
    ]).at(30);

    expect(seen.view.status).toEqual({ kind: "camera_lost" });
    expect(seen.commands).toEqual([[0, { do: "stop" }]]);
  });

  it("shows a camera muted from the start as stopped, and sends nothing", () => {
    const seen = timeline([
      [0, { type: "camera_opened", muted: true, ended: false }],
      [0, opened],
    ]).at(1000);

    expect(seen.view.status).toEqual({ kind: "camera_lost" });
    expect(captures(seen)).toEqual([]);
  });
});

describe("sending frames", () => {
  it("sends the next picture as soon as a result comes back", () => {
    const seen = timeline(startAt(0), { answerAfter: 100 }).at(350);

    expect(sent(seen)).toEqual([
      [0, 1],
      [100, 2],
      [200, 3],
      [300, 4],
    ]);
  });

  it("sends at most 30 frames a second, however fast results come back", () => {
    const seen = timeline(startAt(0), { answerAfter: 0 }).at(999);

    const times = sent(seen).map(([ms]) => ms);
    expect(times).toHaveLength(30);
    for (const [index, ms] of times.entries()) {
      expect(ms).toBeCloseTo((index * 1000) / 30);
    }
  });

  it("keeps one frame unanswered at a time", () => {
    const seen = timeline(startAt(0)).at(RESULT_TIMEOUT_MS - 1);

    expect(sent(seen)).toEqual([[0, 1]]);
    expect(captures(seen)).toEqual([0]);
  });

  it("asks for one picture at a time, and ignores a picture it did not ask for", () => {
    const seen = timeline(
      [
        ...startAt(0),
        // Back from hidden while the picture is being taken.
        [10, "hide"],
        [20, "show"],
        [30, captured(picture, 30)],
        [40, captured(picture, 40)],
      ],
      { camera: "by_hand" },
    ).at(1000);

    expect(captures(seen)).toEqual([0]);
    expect(sent(seen)).toEqual([[30, 1]]);
  });

  it("tries again 100 ms after a capture with no picture", () => {
    const seen = timeline(startAt(0), {
      camera: (ms) => (ms < 100 ? null : picture),
    }).at(150);

    expect(captures(seen)).toEqual([0, 100]);
    expect(sent(seen)).toEqual([[100, 1]]);
  });

  it("passes on model switches and sighting events, and changes nothing else", () => {
    const summary = {
      id: "s1",
      person: { id: "ada", name: "Ada Lovelace", status: "on_watchlist" },
      modelKey: "sface",
      threshold: 0.498,
      startedAt: "2026-09-27T10:00:05Z",
      lastSeenAt: "2026-09-27T10:00:05Z",
      endedAt: null,
      bestScore: 0.81,
    } as const;
    const switched = {
      type: "active_model_changed",
      modelKey: "facenet",
      threshold: 0.709,
    } as const;
    const sightings: ReadonlyArray<SightingMessage> = [
      { type: "sighting_opened", sighting: summary },
      { type: "sighting_updated", sighting: summary },
      { type: "sighting_ended", sighting: summary },
    ];
    const monitor = timeline([
      ...startAt(0),
      [100, message(switched)],
      ...sightings.map((body, index) => [200 + index, message(body)] as const),
    ]);

    const seen = monitor.at(300);
    expect(
      seen.commands.filter(([, command]) => command.do === "notify"),
    ).toEqual([
      [100, { do: "notify", message: switched }],
      ...sightings.map((body, index) => [
        200 + index,
        { do: "notify", message: body },
      ]),
    ]);
    expect(seen.view).toEqual(monitor.at(99).view);
    expect(sent(seen)).toEqual([[0, 1]]);
  });

  it("logs and ignores a message it cannot read", () => {
    const error = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const monitor = timeline([
      ...startAt(0),
      [100, { type: "message", data: "not json" }],
    ]);

    const seen = monitor.at(100);
    expect(commandsAt(seen, 100)).toEqual([]);
    expect(seen.view).toEqual(monitor.at(99).view);
    expect(error).toHaveBeenCalledWith(
      "Unreadable live monitor message",
      expect.anything(),
    );
    error.mockRestore();
  });
});

describe("pausing", () => {
  it("pauses while the camera is muted, and carries on when it comes back", () => {
    const monitor = timeline([
      ...startAt(0),
      [100, track("muted")],
      [200, result(1)],
      [1500, track("unmuted")],
    ]);

    const muted = monitor.at(1499);
    expect(muted.view.status).toEqual({ kind: "camera_lost" });
    // The answer is shown, but not counted, and nothing more goes out.
    expect(muted.view.result).toMatchObject({ seq: 1 });
    expect(muted.view.framesPerSecond).toBeNull();
    expect(sent(muted)).toEqual([[0, 1]]);

    const unmuted = monitor.at(1500);
    expect(unmuted.view.status).toEqual({ kind: "running" });
    expect(sent(unmuted)).toEqual([
      [0, 1],
      [1500, 2],
    ]);
  });

  it("sends nothing while the tab is hidden, and carries on when it is shown", () => {
    const monitor = timeline([
      ...startAt(0),
      [100, "hide"],
      [200, result(1)],
      [1000, "show"],
    ]);

    const hidden = monitor.at(999);
    expect(hidden.view.status).toEqual({ kind: "running" });
    expect(hidden.view.result).toMatchObject({ seq: 1 });
    expect(hidden.view.framesPerSecond).toBeNull();
    expect(sent(hidden)).toEqual([[0, 1]]);

    expect(sent(monitor.at(1000))).toEqual([
      [0, 1],
      [1000, 2],
    ]);
  });

  it("sends nothing from a socket that opens while the tab is hidden, until it is shown", () => {
    const monitor = timeline([[0, "hide"], ...startAt(10), [500, "show"]]);

    expect(captures(monitor.at(499))).toEqual([]);
    expect(sent(monitor.at(500))).toEqual([[500, 1]]);
  });

  it("still sends a picture taken as the tab hides, and times it", () => {
    const seen = timeline(
      [...startAt(0), [10, "hide"], [20, captured(picture, 20)]],
      { camera: "by_hand" },
    );

    expect(sent(seen.at(20))).toEqual([[20, 1]]);
    expect(seen.at(20 + RESULT_TIMEOUT_MS - 1).view.status).toEqual({
      kind: "running",
    });
    expect(seen.at(20 + RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "stalled",
    });
  });
});

describe("stalling", () => {
  it("stalls when no result comes back in time, and recovers when one does", () => {
    const monitor = timeline([
      ...startAt(0),
      [100, result(1)],
      [5300, result(2)],
    ]);
    // Frame 2 went out at 100, after the result for frame 1.
    const deadline = 100 + RESULT_TIMEOUT_MS;

    expect(monitor.at(deadline - 1).view.status).toEqual({ kind: "running" });
    const stalled = monitor.at(deadline);
    expect(stalled.view).toEqual({
      status: { kind: "stalled" },
      // Kept: only a running monitor's result is drawn.
      result: expect.objectContaining({ seq: 1 }),
      framesPerSecond: 0,
    });

    const recovered = monitor.at(5300);
    expect(recovered.view.status).toEqual({ kind: "running" });
    expect(recovered.view.result).toMatchObject({ seq: 2 });
    expect(recovered.view.framesPerSecond).toBe(1);
  });

  it("stalls when the service answers only errors, timed from the first unanswered frame", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const monitor = timeline([
      ...startAt(0),
      [100, refusal(1)],
      [200, refusal(2)],
      [RESULT_TIMEOUT_MS + 100, refusal(3)],
    ]);

    expect(sent(monitor.at(200))).toEqual([
      [0, 1],
      [100, 2],
      [200, 3],
    ]);
    expect(monitor.at(RESULT_TIMEOUT_MS - 1).view.status).toEqual({
      kind: "running",
    });
    expect(monitor.at(RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "stalled",
    });
    // An error is not a result: the stall goes on, and frames still go out.
    const after = monitor.at(RESULT_TIMEOUT_MS + 100);
    expect(after.view.status).toEqual({ kind: "stalled" });
    expect(sent(after).at(-1)).toEqual([RESULT_TIMEOUT_MS + 100, 4]);
    warn.mockRestore();
  });

  it("still stalls when the tab was hidden while a frame was unanswered", () => {
    const monitor = timeline([...startAt(0), [1000, "hide"], [2000, "show"]]);

    // Hiding stopped the clock; showing started it again.
    expect(monitor.at(RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "running",
    });
    expect(monitor.at(2000 + RESULT_TIMEOUT_MS - 1).view.status).toEqual({
      kind: "running",
    });
    expect(monitor.at(2000 + RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "stalled",
    });
  });

  it("still stalls when the camera was muted while a frame was unanswered", () => {
    const monitor = timeline([
      ...startAt(0),
      [1000, track("muted")],
      [2000, track("unmuted")],
    ]);

    expect(monitor.at(1999).view.status).toEqual({ kind: "camera_lost" });
    expect(monitor.at(RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "running",
    });
    expect(monitor.at(2000 + RESULT_TIMEOUT_MS - 1).view.status).toEqual({
      kind: "running",
    });
    expect(monitor.at(2000 + RESULT_TIMEOUT_MS).view.status).toEqual({
      kind: "stalled",
    });
  });
});

describe("frame rate", () => {
  it("counts the results of the last second, read every 250 ms from the socket opening", () => {
    const monitor = timeline([...startAt(0), [100, result(1)]]);

    expect(monitor.at(99).view.framesPerSecond).toBeNull();
    expect(monitor.at(100).view.framesPerSecond).toBe(1);
    // The result is a second old at 1100; the next reading is at 1250.
    expect(monitor.at(1249).view.framesPerSecond).toBe(1);
    expect(monitor.at(1250).view.framesPerSecond).toBe(0);
  });

  it("follows the rate results come back at", () => {
    const seen = timeline(startAt(0), { answerAfter: 100 }).at(2000);

    expect(seen.view.framesPerSecond).toBe(10);
  });

  it("shows no frame rate while the tab is hidden or the camera muted", () => {
    const monitor = timeline(
      [...startAt(0), [500, "hide"], [1500, "show"], [2500, track("muted")]],
      { answerAfter: 100 },
    );

    expect(monitor.at(499).view.framesPerSecond).toBe(4);
    expect(monitor.at(500).view.framesPerSecond).toBeNull();
    // Shown again: the first result after it starts a new count.
    expect(monitor.at(1599).view.framesPerSecond).toBeNull();
    expect(monitor.at(1600).view.framesPerSecond).toBe(1);
    expect(monitor.at(2500).view.framesPerSecond).toBeNull();
  });
});

describe("stopping", () => {
  it.each([
    [4001, { kind: "superseded" }],
    [4002, { kind: "no_active_model" }],
    [1008, { kind: "disconnected", reason: "refused" }],
    [1009, { kind: "disconnected", reason: "frame_too_large" }],
    [1011, { kind: "disconnected", reason: "service_error" }],
    [1006, { kind: "disconnected", reason: "lost" }],
    [1000, { kind: "disconnected", reason: "lost" }],
  ])(
    "stops on close code %i after the socket opened, forgetting the last result",
    (code, status) => {
      const seen = timeline([
        ...startAt(0),
        [100, result(1)],
        [200, closed(code)],
      ]).at(200);

      expect(seen.view).toEqual({
        status,
        result: null,
        framesPerSecond: null,
      });
      expect(commandsAt(seen, 200)).toEqual([{ do: "stop" }]);
      expect(seen.wakeAt).toBeNull();
    },
  );

  it("could not connect when the socket closes before it ever opened", () => {
    const seen = timeline([
      [0, cameraOpened],
      [10, closed(1006)],
    ]).at(10);

    expect(seen.view.status).toEqual({
      kind: "disconnected",
      reason: "unreachable",
    });
    expect(commandsAt(seen, 10)).toEqual([{ do: "stop" }]);
  });

  it("stops when the camera ends", () => {
    const seen = timeline([
      ...startAt(0),
      [100, result(1)],
      [200, track("ended")],
    ]).at(200);

    expect(seen.view).toEqual({
      status: { kind: "camera_lost" },
      result: null,
      framesPerSecond: null,
    });
    expect(commandsAt(seen, 200)).toEqual([{ do: "stop" }]);
  });

  it("stays stopped whatever happens next", () => {
    const step = createLiveMonitor();
    const at = (now: number) => ({ now, hidden: false });
    step(cameraOpened, at(0));
    step(opened, at(0));
    const halted = step(closed(4001), at(10));

    for (const [now, event] of [
      [20, opened],
      [30, result(1)],
      [40, captured(picture, 40)],
      [50, track("unmuted")],
      [60, track("ended")],
      [70, { type: "visibility" }],
      [80, { type: "tick" }],
      [90, closed(1006)],
      [100, { type: "connect_failed" }],
      [110, cameraOpened],
    ] as const) {
      const next = step(event, at(now));
      expect(next.view).toBe(halted.view);
      expect(next.commands).toEqual([]);
      expect(next.wakeAt).toBeNull();
    }
  });

  it("changes nothing once disposed, but lets go of a camera that opens afterwards", () => {
    const step = createLiveMonitor();
    const at = (now: number) => ({ now, hidden: false });
    const disposed = step({ type: "disposed" }, at(0));
    expect(disposed).toEqual({
      view: {
        status: { kind: "starting" },
        result: null,
        framesPerSecond: null,
      },
      commands: [],
      wakeAt: null,
    });

    const late = step(cameraOpened, at(10));
    expect(late.view).toBe(disposed.view);
    expect(late.commands).toEqual([{ do: "stop" }]);
    expect(step({ type: "camera_failed", problem: "busy" }, at(20))).toEqual({
      view: disposed.view,
      commands: [],
      wakeAt: null,
    });
  });

  it("changes nothing once disposed mid-session", () => {
    const seen = timeline(
      [...startAt(0), [100, { type: "disposed" }], [200, result(1)]],
      { answerAfter: 50 },
    ).at(1000);

    expect(seen.commands.filter(([ms]) => ms >= 100)).toEqual([]);
    expect(seen.wakeAt).toBeNull();
    expect(seen.view.status).toEqual({ kind: "running" });
  });
});

describe("waking", () => {
  it("does nothing on a tick before the time it asked for", () => {
    const step = createLiveMonitor();
    const at = (now: number) => ({ now, hidden: false });
    step(cameraOpened, at(0));
    const running = step(opened, at(0));

    const early = step({ type: "tick" }, at(100));
    expect(early.view).toBe(running.view);
    expect(early.commands).toEqual([]);
    expect(early.wakeAt).toBe(running.wakeAt);
  });

  it("keeps the view as it was when nothing the operator sees changed", () => {
    const step = createLiveMonitor();
    const at = (now: number) => ({ now, hidden: false });
    step(cameraOpened, at(0));
    step(opened, at(0));
    step(captured(picture, 0), at(0));
    const answered = step(result(1), at(100));

    const reading = step({ type: "tick" }, at(250));
    expect(reading.view).toBe(answered.view);
  });
});
