import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it, onTestFinished } from "vitest";
import { healthy, mockService } from "./test/api-server";
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
