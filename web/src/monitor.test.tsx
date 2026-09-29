import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it, onTestFinished, vi } from "vitest";
import { RESULT_TIMEOUT_MS } from "./hooks/use-live-monitor";
import { healthy, mockService, problemResponse } from "./test/api-server";
import { fakeCamera } from "./test/camera";
import { gate } from "./test/gate";
import {
  arcface,
  box,
  facenet,
  frameResult,
  mockMonitor,
  sface,
} from "./test/monitor";
import { renderAt } from "./test/render";

const server = mockService(
  healthy,
  http.get("*/api/models", () => HttpResponse.json([sface, arcface, facenet])),
);

describe("live monitor", () => {
  it("draws a match with the person's name and score, and never a name on a no match", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");

    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(
      frameResult(1, [
        {
          outcome: "match",
          box,
          score: 0.874,
          person: { id: "ada", name: "Ada Lovelace" },
        },
      ]),
    );
    const match = await screen.findByRole("img", {
      name: "Match: Ada Lovelace, score 0.874",
    });
    expect(match).toHaveTextContent("Ada Lovelace 0.874");

    // The same face in the next frame, now under the threshold.
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));
    monitor.send(frameResult(2, [{ outcome: "no_match", box, score: 0.412 }]));
    const noMatch = await screen.findByRole("img", {
      name: "No match, score 0.412",
    });
    expect(noMatch).toHaveTextContent("0.412");
    expect(screen.queryByText(/Ada Lovelace/)).not.toBeInTheDocument();
  });

  it("sends each frame with the 17-byte header the service parses", async () => {
    fakeCamera({ width: 1280, height: 720 });
    const monitor = mockMonitor(server);
    const before = Date.now();
    renderAt("/monitor");

    await waitFor(() => expect(monitor.frames).toEqual([1]));
    const [frame] = monitor.received;
    // The long side is scaled to 640 px, keeping the camera's aspect ratio.
    expect(frame).toMatchObject({
      type: 1,
      seq: 1,
      width: 640,
      height: 360,
      jpegBytes: 4,
    });
    expect(frame?.capturedAt).toBeGreaterThanOrEqual(before);
    expect(frame?.capturedAt).toBeLessThanOrEqual(Date.now());
  });

  it("draws a face too small to use as an unlabelled box", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");

    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(frameResult(1, [{ outcome: "too_small", box }]));

    const small = await screen.findByRole("img", {
      name: "Face too small to score",
    });
    expect(small).toBeEmptyDOMElement();
  });

  it("shows the active model, its threshold and the frame rate achieved", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");

    const toolbar = await readings();
    expect(toolbar).toEqual({
      "Active model": "SFace",
      Threshold: "0.498",
      "Frame rate": "—",
    });
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(frameResult(1, []));
    await waitFor(async () =>
      expect((await readings())["Frame rate"]).toBe("1 fps"),
    );

    // Boxes judged by another model, just before a switch reaches the page.
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));
    monitor.send({
      ...frameResult(2, []),
      modelKey: facenet.id,
      threshold: 0.709,
    });
    await waitFor(async () =>
      expect(await readings()).toMatchObject({
        "Active model": "FaceNet",
        Threshold: "0.709",
      }),
    );
  });

  it("sends at most 30 frames a second, however fast results come back", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));

    // Every frame is answered at once, so only the cap paces them.
    const answered = new Set<number>();
    const answer = setInterval(() => {
      for (const seq of monitor.frames) {
        if (!answered.has(seq)) {
          answered.add(seq);
          monitor.send(frameResult(seq, []));
        }
      }
    }, 1);
    onTestFinished(() => clearInterval(answer));
    const started = performance.now();
    await new Promise((resolve) => setTimeout(resolve, 400));
    const elapsed = performance.now() - started;

    expect(monitor.frames.length).toBeGreaterThan(3);
    expect(monitor.frames.length).toBeLessThanOrEqual(
      Math.floor(elapsed / (1000 / 30)) + 2,
    );
  });

  it("stops when another tab takes over, and takes it back on request", async () => {
    const camera = fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));

    monitor.close(4001);

    expect(
      await screen.findByText("The live monitor moved to another tab"),
    ).toBeInTheDocument();
    expect(camera.stop).toHaveBeenCalled();
    expect((await readings())["Frame rate"]).toBe("—");
    fireEvent.click(screen.getByRole("button", { name: "Monitor here" }));
    await waitFor(() => expect(monitor.connections()).toBe(2));
    expect(camera.getUserMedia).toHaveBeenCalledTimes(2);
  });

  it("says so when the service closes because no model can be active", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.connections()).toBe(1));

    monitor.close(4002);

    expect(
      await screen.findByText("No recognition model can be active"),
    ).toBeInTheDocument();
  });

  it("never opens the camera when no model can be active", async () => {
    const camera = fakeCamera();
    server.use(
      http.get("*/api/models", () =>
        HttpResponse.json([{ ...sface, state: "unavailable" }, arcface]),
      ),
    );
    renderAt("/monitor");

    expect(
      await screen.findByText("No recognition model can be active"),
    ).toBeInTheDocument();
    expect(screen.getByText("ryuk weights fetch")).toBeInTheDocument();
    expect(camera.getUserMedia).not.toHaveBeenCalled();
  });

  it("explains a refused camera", async () => {
    fakeCamera({
      error: new DOMException("Permission denied", "NotAllowedError"),
    });
    renderAt("/monitor");

    expect(await screen.findByText("No camera")).toBeInTheDocument();
    expect(screen.getByText(/Camera access was refused/)).toBeInTheDocument();
  });

  it("switches the active model only after a confirm, offering evaluated models only", async () => {
    fakeCamera();
    mockMonitor(server);
    const chosen: Array<unknown> = [];
    server.use(
      http.put("*/api/active-model", async ({ request }) => {
        chosen.push(await request.json());
        return HttpResponse.json({ ...facenet, state: "active" });
      }),
    );
    renderAt("/monitor");

    const select = await screen.findByLabelText("Switch model");
    expect(
      within(select)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["SFace", "FaceNet"]);
    fireEvent.change(select, { target: { value: facenet.id } });
    const dialog = await screen.findByRole("dialog", {
      name: "Switch to FaceNet?",
    });
    expect(chosen).toEqual([]);
    server.use(
      http.get("*/api/models", () =>
        HttpResponse.json([
          { ...sface, state: "available" },
          arcface,
          { ...facenet, state: "active" },
        ]),
      ),
    );
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Switch model" }),
    );

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(chosen).toEqual([{ modelKey: facenet.id }]);
    expect((await readings())["Active model"]).toBe("FaceNet");
  });

  it("says why a switch was refused, and keeps the dialog open", async () => {
    fakeCamera();
    mockMonitor(server);
    server.use(
      http.put("*/api/active-model", () =>
        problemResponse({
          type: "about:blank",
          title: "Conflict",
          status: 409,
          detail:
            "FaceNet cannot be active: its weights are not on this machine.",
          code: "cannot_be_active",
        }),
      ),
    );
    renderAt("/monitor");

    fireEvent.change(await screen.findByLabelText("Switch model"), {
      target: { value: facenet.id },
    });
    const dialog = await screen.findByRole("dialog", {
      name: "Switch to FaceNet?",
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Switch model" }),
    );

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "That model cannot be active",
    );
    expect(
      within(dialog).getByRole("button", { name: "Keep SFace" }),
    ).toBeEnabled();
  });

  it("keeps the switch dialog open while the switch is in flight", async () => {
    fakeCamera();
    mockMonitor(server);
    const held = gate();
    const chosen: Array<unknown> = [];
    server.use(
      http.put("*/api/active-model", async ({ request }) => {
        chosen.push(await request.json());
        await held.opened;
        return HttpResponse.json({ ...facenet, state: "active" });
      }),
    );
    renderAt("/monitor");

    fireEvent.change(await screen.findByLabelText("Switch model"), {
      target: { value: facenet.id },
    });
    const dialog = await screen.findByRole("dialog", {
      name: "Switch to FaceNet?",
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Switch model" }),
    );
    await waitFor(() => expect(chosen).toHaveLength(1));

    expect(
      within(dialog).getByRole("button", { name: "Keep SFace" }),
    ).toBeDisabled();
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(dialog).toHaveAttribute("open");

    held.open();
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(chosen).toHaveLength(1);
  });

  it("refreshes the toolbar when the active model is switched elsewhere", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.connections()).toBe(1));
    server.use(
      http.get("*/api/models", () =>
        HttpResponse.json([
          { ...sface, state: "available" },
          { ...facenet, state: "active" },
        ]),
      ),
    );

    monitor.send({
      type: "active_model_changed",
      modelKey: facenet.id,
      threshold: 0.709,
    });

    await waitFor(async () =>
      expect(await readings()).toMatchObject({
        "Active model": "FaceNet",
        Threshold: "0.709",
      }),
    );
  });

  it("sends nothing while the tab is hidden", async () => {
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));

    setVisibility("hidden");
    monitor.send(frameResult(1, [{ outcome: "too_small", box }]));
    await screen.findByRole("img", { name: "Face too small to score" });
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(monitor.frames).toEqual([1]);
    expect((await readings())["Frame rate"]).toBe("—");

    setVisibility("visible");
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));
  });

  it("says the connection was lost, forgets the last result, and reconnects", async () => {
    const camera = fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send({ ...frameResult(1, []), modelKey: facenet.id });
    await waitFor(async () =>
      expect((await readings())["Active model"]).toBe("FaceNet"),
    );

    monitor.close(1006);

    expect(
      await screen.findByText("Lost the connection to the service"),
    ).toBeInTheDocument();
    expect(camera.stop).toHaveBeenCalled();
    // The model of a result from a closed socket is no longer on screen.
    expect(await readings()).toMatchObject({
      "Active model": "SFace",
      "Frame rate": "—",
    });
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
    await waitFor(() => expect(monitor.connections()).toBe(2));
  });

  it("says it could not connect when the socket never opens", async () => {
    const camera = fakeCamera();
    // A service that is down, or a handshake the localhost guard refused.
    const monitor = mockMonitor(server, { refuseWith: 1006 });
    renderAt("/monitor");

    expect(
      await screen.findByText("Could not connect to the service"),
    ).toBeInTheDocument();
    expect(screen.getByText(/127\.0\.0\.1 or localhost/)).toBeInTheDocument();
    expect(camera.stop).toHaveBeenCalled();
    expect(monitor.frames).toEqual([]);
  });

  it("lets go of the camera when the service's address is malformed", async () => {
    const camera = fakeCamera();
    vi.spyOn(globalThis, "WebSocket").mockImplementation(() => {
      throw new SyntaxError("The URL is invalid.");
    });
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    renderAt("/monitor");

    expect(
      await screen.findByText("The service's address is not valid"),
    ).toBeInTheDocument();
    expect(camera.stop).toHaveBeenCalled();
  });

  it("stops sending and says so when the camera ends", async () => {
    const camera = fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(frameResult(1, [{ outcome: "too_small", box }]));
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));

    act(() => camera.track.end());

    expect(await screen.findByText("The camera stopped")).toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: "Face too small to score" }),
    ).not.toBeInTheDocument();
    expect(camera.stop).toHaveBeenCalled();
    monitor.send(frameResult(2, []));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(monitor.frames).toEqual([1, 2]);

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(monitor.connections()).toBe(2));
    expect(camera.getUserMedia).toHaveBeenCalledTimes(2);
  });

  it("pauses while the camera is muted and carries on when it comes back", async () => {
    const camera = fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));

    act(() => camera.track.mute());
    expect(await screen.findByText("The camera stopped")).toBeInTheDocument();
    monitor.send(frameResult(1, []));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(monitor.frames).toEqual([1]);

    act(() => camera.track.unmute());
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));
    expect(screen.queryByText("The camera stopped")).not.toBeInTheDocument();
  });

  it("shows a stalled state when no result comes back, and recovers when one does", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(frameResult(1, [{ outcome: "too_small", box }]));
    await waitFor(() => expect(monitor.frames).toEqual([1, 2]));
    expect(screen.queryByText("The service stopped answering")).toBeNull();

    // Frame 2 is never answered; an error is not an answer that ends it.
    act(() => {
      vi.advanceTimersByTime(RESULT_TIMEOUT_MS - 100);
    });
    expect(screen.queryByText("The service stopped answering")).toBeNull();
    act(() => {
      vi.advanceTimersByTime(200);
    });

    expect(
      await screen.findByText("The service stopped answering"),
    ).toBeInTheDocument();
    // The boxes of an old frame are not shown over live video.
    expect(
      screen.queryByRole("img", { name: "Face too small to score" }),
    ).not.toBeInTheDocument();
    expect((await readings())["Frame rate"]).toBe("0 fps");

    monitor.send(frameResult(2, [{ outcome: "too_small", box }]));
    expect(
      await screen.findByRole("img", { name: "Face too small to score" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("The service stopped answering")).toBeNull();
  });

  it("stalls when the service answers only errors", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");

    for (const seq of [1, 2, 3]) {
      await waitFor(() => expect(monitor.frames).toHaveLength(seq));
      monitor.send({
        type: "error",
        seq,
        code: "invalid_frame",
        detail: "Refused.",
      });
    }
    act(() => {
      vi.advanceTimersByTime(RESULT_TIMEOUT_MS);
    });

    expect(
      await screen.findByText("The service stopped answering"),
    ).toBeInTheDocument();
  });

  it("lets the frame rate fall when results stop coming", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    fakeCamera();
    const monitor = mockMonitor(server);
    renderAt("/monitor");
    await waitFor(() => expect(monitor.frames).toEqual([1]));
    monitor.send(frameResult(1, []));
    await waitFor(async () =>
      expect((await readings())["Frame rate"]).toBe("1 fps"),
    );

    act(() => {
      vi.advanceTimersByTime(1500);
    });

    await waitFor(async () =>
      expect((await readings())["Frame rate"]).toBe("0 fps"),
    );
  });
});

/** The toolbar's readings, by label. */
async function readings(): Promise<Record<string, string>> {
  const list = await screen.findByText("Active model");
  const terms = within(list.closest("dl") as HTMLElement).getAllByRole("term");
  return Object.fromEntries(
    terms.map((term) => [
      term.textContent,
      term.nextElementSibling?.textContent ?? "",
    ]),
  );
}

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
  onTestFinished(() => {
    Reflect.deleteProperty(document, "visibilityState");
  });
  document.dispatchEvent(new Event("visibilitychange"));
}
