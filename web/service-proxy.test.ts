// @vitest-environment node
import { resolveConfig } from "vite";
import { describe, expect, it } from "vitest";
import { isLoopbackHost, loopbackOnly, servicePort } from "./service-proxy";

describe("servicePort", () => {
  it("is 8000 unless RYUK_PORT says otherwise", () => {
    expect(servicePort({})).toBe(8000);
    expect(servicePort({ RYUK_PORT: "" })).toBe(8000);
    expect(servicePort({ RYUK_PORT: "8123" })).toBe(8123);
  });

  it.each(["0", "65536", "80a", "-1", "8000.5", "http://127.0.0.1:8000"])(
    "refuses RYUK_PORT=%s",
    (value) => {
      expect(() => servicePort({ RYUK_PORT: value })).toThrow(/RYUK_PORT/);
    },
  );
});

describe("isLoopbackHost", () => {
  it.each([
    undefined,
    false,
    "localhost",
    "127.0.0.1",
    "127.1.2.3",
    "::1",
    "[::1]",
  ])("accepts %s", (host) => {
    expect(isLoopbackHost(host)).toBe(true);
  });

  it.each([true, "0.0.0.0", "::", "100.64.0.7", "ohmahgahpc", "127.0.0.256"])(
    "refuses %s",
    (host) => {
      expect(isLoopbackHost(host)).toBe(false);
    },
  );
});

describe("loopbackOnly", () => {
  const resolve = (config: Parameters<typeof resolveConfig>[0]) =>
    resolveConfig(
      {
        configFile: false,
        logLevel: "silent",
        plugins: [loopbackOnly()],
        ...config,
      },
      "serve",
    );

  it("refuses a dev server or preview on a non-loopback host", async () => {
    await expect(resolve({ server: { host: "0.0.0.0" } })).rejects.toThrow(
      /Refusing to serve/,
    );
    await expect(resolve({ server: { host: true } })).rejects.toThrow(
      /every address/,
    );
    await expect(resolve({ preview: { host: "0.0.0.0" } })).rejects.toThrow(
      /Refusing to preview/,
    );
  });

  it("allows loopback, as Playwright's preview uses", async () => {
    await expect(
      resolve({ preview: { host: "127.0.0.1", port: 4173 } }),
    ).resolves.toBeDefined();
    await expect(resolve({})).resolves.toBeDefined();
  });
});
