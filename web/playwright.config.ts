import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

const isCI = Boolean(process.env.CI);

/**
 * What Chrome's fake camera shows, one MJPEG frame played on a loop:
 * tests/fixtures/astronaut.jpg cropped to x 63-383, y 2-242, scaled to
 * 640x480 and saved as a baseline JPEG (quality 90, 4:2:0).
 */
export const CAMERA_PICTURE = fileURLToPath(
  new URL("./e2e/astronaut.mjpeg", import.meta.url),
);

// Runs headless in CI only, against the built client (vite preview) and the
// real service with the fixtures' YuNet and a fake recognition model. The
// preview server proxies /api, the socket included, to the service.
export default defineConfig({
  testDir: "./e2e",
  // One live monitor runs at a time: a second tab takes it over, so tests
  // that open it must not run side by side.
  fullyParallel: false,
  workers: 1,
  forbidOnly: isCI,
  retries: isCI ? 1 : 0,
  // A hung run fails fast instead of holding a CI runner.
  globalTimeout: 5 * 60_000,
  reporter: isCI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    trace: "on-first-retry",
    permissions: ["camera"],
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-video-capture=${CAMERA_PICTURE}`,
      ],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run --project .. python ../tests/e2e_service.py",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: !isCI,
      timeout: 120_000,
    },
    {
      // vite directly, not `pnpm preview`: pnpm 12 starts vite in its own
      // process group, which survives Playwright's shutdown and hangs the run.
      // `pnpm e2e` puts node_modules/.bin on PATH.
      command: "vite preview --host 127.0.0.1 --port 4173 --strictPort",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: !isCI,
    },
  ],
});
