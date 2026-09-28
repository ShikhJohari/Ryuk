import "@testing-library/jest-dom/vitest";
import { Blob, File } from "node:buffer";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { polyfillDialog } from "./dialog";

// jsdom does not implement scrolling; the router's scroll restoration calls it.
vi.stubGlobal("scrollTo", () => undefined);

// jsdom has <dialog> without showModal(); see ./dialog.
polyfillDialog();

// jsdom's Blob, File and FormData cannot be sent by Node's fetch, which the
// Effect client and MSW use in tests; a browser has one implementation of
// each. Node's own FormData is only reachable through a parsed body.
const NodeFormData = (
  await new Response("", {
    headers: { "content-type": "application/x-www-form-urlencoded" },
  }).formData()
).constructor;
vi.stubGlobal("Blob", Blob);
vi.stubGlobal("File", File);
vi.stubGlobal("FormData", NodeFormData);

afterEach(() => {
  cleanup();
});
