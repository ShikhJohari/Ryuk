import { describe, expect, it } from "vitest";
import { ApiProblem } from "@/api/api-client";
import { getHealth } from "@/api/health";
import { healthy, mockService, unavailable } from "@/test/api-server";
import { runQuery } from "./runtime";

const server = mockService();

describe("runQuery", () => {
  it("resolves with the decoded body", async () => {
    server.use(healthy);

    await expect(runQuery(getHealth)).resolves.toEqual({
      status: "ok",
      version: "0.1.0",
    });
  });

  it("rejects with the tagged error itself, so callers can branch on it", async () => {
    server.use(unavailable);

    const failure = await runQuery(getHealth).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ApiProblem);
    expect(failure).toMatchObject({ _tag: "ApiProblem", code: "unavailable" });
  });
});
