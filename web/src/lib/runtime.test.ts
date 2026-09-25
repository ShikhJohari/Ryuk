import { HttpResponse, http } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { ApiProblem } from "@/api/api-client";
import { getHealth } from "@/api/health";
import { runQuery } from "./runtime";

const server = setupServer();

beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => {
  server.resetHandlers();
});
afterAll(() => {
  server.close();
});

describe("runQuery", () => {
  it("resolves with the decoded body", async () => {
    server.use(
      http.get("*/api/health", () =>
        HttpResponse.json({ status: "ok", version: "0.1.0" }),
      ),
    );

    await expect(runQuery(getHealth)).resolves.toEqual({
      status: "ok",
      version: "0.1.0",
    });
  });

  it("rejects with the tagged error itself, so callers can branch on it", async () => {
    server.use(
      http.get("*/api/health", () =>
        HttpResponse.json(
          {
            type: "about:blank",
            title: "Service Unavailable",
            status: 503,
            detail: "The service is starting up.",
            code: "unavailable",
          },
          {
            status: 503,
            headers: { "content-type": "application/problem+json" },
          },
        ),
      ),
    );

    const failure = await runQuery(getHealth).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ApiProblem);
    expect(failure).toMatchObject({ _tag: "ApiProblem", code: "unavailable" });
  });
});
