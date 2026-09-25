import { HttpResponse, http, type RequestHandler } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll } from "vitest";
import type { Problem } from "@/api/problem";

/**
 * MSW standing in for the service for one test file. `handlers` answer every
 * test; `server.use(...)` overrides them for one test only.
 */
export function mockService(...handlers: ReadonlyArray<RequestHandler>) {
  const server = setupServer(...handlers);
  beforeAll(() => {
    server.listen({ onUnhandledRequest: "error" });
  });
  afterEach(() => {
    server.resetHandlers();
  });
  afterAll(() => {
    server.close();
  });
  return server;
}

export const healthy = http.get("*/api/health", () =>
  HttpResponse.json({ status: "ok", version: "0.1.0" }),
);

export const unavailable = http.get("*/api/health", () =>
  problemResponse({
    type: "about:blank",
    title: "Service Unavailable",
    status: 503,
    detail: "The service is starting up.",
    code: "unavailable",
  }),
);

/** An RFC 9457 problem response, as the service sends it. */
export function problemResponse(problem: Problem) {
  return HttpResponse.json(problem, {
    status: problem.status,
    headers: { "content-type": "application/problem+json" },
  });
}
