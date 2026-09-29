import { isNotFound as isRouterNotFound } from "@tanstack/react-router";
import { describe, expect, it } from "vitest";
import { ApiProblem } from "@/api/api-client";
import { orNotFound } from "./route-loading";

function problem(code: string, status: number) {
  return new ApiProblem({
    type: "about:blank",
    title: "Problem",
    status,
    detail: "Something the service said.",
    code,
  });
}

describe("orNotFound", () => {
  it("gives what loaded", async () => {
    await expect(orNotFound(Promise.resolve(42))).resolves.toBe(42);
  });

  it("turns the service's not_found into the router's not-found page", async () => {
    const thrown = await orNotFound(
      Promise.reject(problem("not_found", 404)),
    ).catch((error: unknown) => error);

    expect(isRouterNotFound(thrown)).toBe(true);
  });

  it("lets any other failure through, for the route's error page", async () => {
    const failure = problem("internal_error", 500);

    await expect(orNotFound(Promise.reject(failure))).rejects.toBe(failure);
  });
});
