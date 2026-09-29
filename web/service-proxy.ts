import type { Plugin } from "vite";

/** The service's port when `RYUK_PORT` is unset, as in `ryuk serve`. */
export const DEFAULT_SERVICE_PORT = 8000;

/**
 * The port the service listens on: `RYUK_PORT`, which `ryuk serve` reads
 * too, else 8000. Anything but a TCP port number is refused, rather than
 * proxying uploads to wherever a typo points.
 */
export function servicePort(env: Readonly<Record<string, string | undefined>>) {
  const raw = env.RYUK_PORT?.trim();
  if (raw === undefined || raw === "") {
    return DEFAULT_SERVICE_PORT;
  }
  const port = /^\d+$/.test(raw) ? Number(raw) : Number.NaN;
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(
      `RYUK_PORT must be a port number from 1 to 65535, not "${raw}".`,
    );
  }
  return port;
}

/**
 * Whether a Vite `host` option binds loopback only. Unset is Vite's default,
 * localhost; `true` (a bare `--host`) is every address.
 */
export function isLoopbackHost(host: string | boolean | undefined): boolean {
  if (host === undefined || host === false) {
    return true;
  }
  if (host === true) {
    return false;
  }
  const name = host
    .trim()
    .toLowerCase()
    .replace(/^\[(.*)\]$/, "$1");
  return (
    name === "localhost" ||
    name === "::1" ||
    /^127(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$/.test(name)
  );
}

/**
 * Refuses to serve the client, dev or preview, on anything but loopback.
 * Its `/api` proxy reaches a service with no authentication; on another
 * address, every device that can reach this machine could enroll, rename and
 * fetch photos through it. This holds whatever a machine's own rules say
 * about binding dev servers to 0.0.0.0.
 */
export function loopbackOnly(): Plugin {
  return {
    name: "ryuk:loopback-only",
    configResolved(config) {
      const hosts = {
        server: config.server.host,
        preview: config.preview.host,
      };
      for (const [mode, host] of Object.entries(hosts)) {
        if (!isLoopbackHost(host)) {
          const where = host === true ? "every address" : `"${host}"`;
          throw new Error(
            `Refusing to ${mode === "server" ? "serve" : "preview"} Ryuk's client on ${where}: ` +
              "its /api proxy would expose the service's unauthenticated API beyond this machine. " +
              "Use --host 127.0.0.1 (or ::1, or localhost), or leave --host out.",
          );
        }
      }
    },
  };
}
