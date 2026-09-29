import {
  HttpClientError,
  HttpClientRequest,
  HttpClientResponse,
} from "@effect/platform";
import { Either, Schema } from "effect";
import { describe, expect, it } from "vitest";
import { ApiProblem } from "@/api/api-client";
import { problemMessage } from "./problems";

const request = HttpClientRequest.get("/api/persons");

function problem(code: string, status: number) {
  return new ApiProblem({
    type: "about:blank",
    title: "Title",
    status,
    detail: "The service's own words, which the operator never sees.",
    code,
  });
}

function responseError(
  reason: HttpClientError.ResponseError["reason"],
  status: number,
) {
  return new HttpClientError.ResponseError({
    request,
    response: HttpClientResponse.fromWeb(
      request,
      new Response(null, { status }),
    ),
    reason,
  });
}

describe("problemMessage", () => {
  it("maps a problem's code to copy, never showing its detail", () => {
    const message = problemMessage(problem("no_face", 422));

    expect(message).toMatch(/^No face was found in the photo/);
    expect(message).not.toMatch(/never sees/);
  });

  it("names the status of a problem whose code it does not know", () => {
    expect(problemMessage(problem("teapot", 418))).toMatch(/HTTP 418/);
  });

  it("says the service could not be reached when the request never got an answer", () => {
    expect(
      problemMessage(
        new HttpClientError.RequestError({ request, reason: "Transport" }),
      ),
    ).toMatch(/could not be reached/);
  });

  it("names the status of an error answered without a problem body", () => {
    expect(problemMessage(responseError("StatusCode", 502))).toMatch(
      /HTTP 502/,
    );
  });

  it("tells an unreadable answer apart from an unreachable service", () => {
    const parseError = Either.flip(
      Schema.decodeUnknownEither(Schema.Struct({ id: Schema.String }))({}),
    ).pipe(Either.getOrThrow);

    for (const error of [responseError("Decode", 200), parseError]) {
      const message = problemMessage(error);
      expect(message).toMatch(/could not be read/);
      expect(message).not.toMatch(/could not be reached/);
    }
  });

  it("falls back to a plain message for anything else", () => {
    expect(problemMessage(new Error("boom"))).toBe(
      "Something went wrong. Reload the page, then try again.",
    );
  });
});
