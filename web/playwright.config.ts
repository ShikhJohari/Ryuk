import { defineConfig, devices } from "@playwright/test";

const isCI = Boolean(process.env.CI);

// Runs headless in CI only, against the built client (vite preview) and the
// real service. The preview server proxies /api to the service.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 1 : 0,
  reporter: isCI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run --project .. ryuk serve",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: !isCI,
      timeout: 120_000,
    },
    {
      command: "pnpm preview --host 127.0.0.1 --port 4173 --strictPort",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: !isCI,
    },
  ],
});
