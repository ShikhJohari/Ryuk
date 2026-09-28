import {
  HttpClient,
  HttpClientError,
  type HttpClientRequest,
  HttpClientResponse,
} from "@effect/platform";
import { assert, describe, expect, it } from "@effect/vitest";
import { ConfigProvider, Effect, Layer, Ref, Schema } from "effect";
import { ApiClient, ApiClientLive, ApiProblem } from "./api-client";
import { getHealth } from "./health";

/** An HttpClient that answers every request with `respond`, never touching the network. */
const fakeHttpClient = (respond: () => Response) =>
  Layer.succeed(
    HttpClient.HttpClient,
    HttpClient.make((request) =>
      Effect.succeed(HttpClientResponse.fromWeb(request, respond())),
    ),
  );

const withConfig = (env: Record<string, string>) =>
  Layer.setConfigProvider(ConfigProvider.fromMap(new Map(Object.entries(env))));

const apiClientResponding = (respond: () => Response) =>
  ApiClientLive.pipe(
    Layer.provide(fakeHttpClient(respond)),
    Layer.provide(withConfig({})),
  );

const json = (body: unknown, status: number, contentType: string) => () =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": contentType },
  });

const problem = {
  type: "about:blank",
  title: "Internal Server Error",
  status: 500,
  detail: "The service failed to handle the request.",
  code: "internal_error",
};

describe("ApiClient", () => {
  it.effect("decodes a healthy response", () =>
    Effect.gen(function* () {
      const health = yield* getHealth;
      expect(health).toEqual({ status: "ok", version: "0.1.0" });
    }).pipe(
      Effect.provide(
        apiClientResponding(
          json({ status: "ok", version: "0.1.0" }, 200, "application/json"),
        ),
      ),
    ),
  );

  it.effect("fails with ApiProblem carrying the code on a problem 5xx", () =>
    Effect.gen(function* () {
      const error = yield* Effect.flip(getHealth);
      assert(error._tag === "ApiProblem");
      expect(error.code).toBe("internal_error");
      expect(error.status).toBe(500);
      expect(error.title).toBe(problem.title);
      expect(error.detail).toBe(problem.detail);
    }).pipe(
      Effect.provide(
        apiClientResponding(
          json(problem, 500, "application/problem+json; charset=utf-8"),
        ),
      ),
    ),
  );

  it.effect("fails with ParseError on a malformed 200 body", () =>
    Effect.gen(function* () {
      const error = yield* Effect.flip(getHealth);
      expect(error._tag).toBe("ParseError");
    }).pipe(
      Effect.provide(
        apiClientResponding(
          json({ status: "degraded" }, 200, "application/json"),
        ),
      ),
    ),
  );

  it.effect(
    "fails with a StatusCode ResponseError on a non-problem error body",
    () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(getHealth);
        assert(error._tag === "ResponseError");
        expect(error.reason).toBe("StatusCode");
        expect(error.response.status).toBe(502);
      }).pipe(
        Effect.provide(
          apiClientResponding(
            () =>
              new Response("<h1>Bad Gateway</h1>", {
                status: 502,
                headers: { "content-type": "text/html" },
              }),
          ),
        ),
      ),
  );

  it.effect(
    "fails with a StatusCode ResponseError when a problem body does not decode",
    () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(getHealth);
        assert(error._tag === "ResponseError");
        expect(error.reason).toBe("StatusCode");
        expect(error.cause).toBeDefined();
      }).pipe(
        Effect.provide(
          apiClientResponding(
            json({ title: "Oops" }, 500, "application/problem+json"),
          ),
        ),
      ),
  );

  it.effect(
    "resolves paths against VITE_API_BASE_URL and accepts problems",
    () =>
      Effect.gen(function* () {
        const seen =
          yield* Ref.make<HttpClientRequest.HttpClientRequest | null>(null);
        const recordingClient = Layer.succeed(
          HttpClient.HttpClient,
          HttpClient.make((request) =>
            Ref.set(seen, request).pipe(
              Effect.as(
                HttpClientResponse.fromWeb(
                  request,
                  json({ ok: true }, 200, "application/json")(),
                ),
              ),
            ),
          ),
        );
        const layer = ApiClientLive.pipe(
          Layer.provide(recordingClient),
          Layer.provide(
            withConfig({ VITE_API_BASE_URL: "http://service.test" }),
          ),
        );

        yield* Effect.flatMap(ApiClient, (api) =>
          api.get("/api/anything", Schema.Struct({ ok: Schema.Boolean })),
        ).pipe(Effect.provide(layer));

        const request = yield* Ref.get(seen);
        assert(request !== null);
        expect(request.url).toBe("http://service.test/api/anything");
        expect(request.headers.accept).toContain("application/problem+json");
      }),
  );

  const conflict = {
    type: "about:blank",
    title: "Conflict",
    status: 409,
    detail: "Refused.",
    code: "last_photo",
  };
  const Echo = Schema.Struct({ ok: Schema.Boolean });

  const changes: ReadonlyArray<{
    readonly method: string;
    readonly call: (
      api: typeof ApiClient.Service,
    ) => Effect.Effect<unknown, unknown>;
  }> = [
    {
      method: "POST",
      call: (api) => api.postForm("/api/x", new FormData(), Echo),
    },
    { method: "PATCH", call: (api) => api.patch("/api/x", { a: 1 }, Echo) },
    { method: "PUT", call: (api) => api.put("/api/x", { a: 1 }, Echo) },
    { method: "DELETE", call: (api) => api.delete("/api/x") },
  ];

  for (const { method, call } of changes) {
    it.effect(`fails a ${method} with ApiProblem on a problem response`, () =>
      Effect.gen(function* () {
        const error = yield* Effect.flip(Effect.flatMap(ApiClient, call));
        assert(error instanceof ApiProblem);
        expect(error.code).toBe("last_photo");
        expect(error.status).toBe(409);
      }).pipe(
        Effect.provide(
          apiClientResponding(json(conflict, 409, "application/problem+json")),
        ),
      ),
    );

    it.effect(
      `fails a ${method} with a StatusCode ResponseError without a problem body`,
      () =>
        Effect.gen(function* () {
          const error = yield* Effect.flip(Effect.flatMap(ApiClient, call));
          assert(error instanceof HttpClientError.ResponseError);
          expect(error.reason).toBe("StatusCode");
        }).pipe(
          Effect.provide(
            apiClientResponding(() => new Response(null, { status: 500 })),
          ),
        ),
    );
  }

  it.effect("sends a PATCH and a PUT body as JSON", () =>
    Effect.gen(function* () {
      const sent: Array<HttpClientRequest.HttpClientRequest> = [];
      const layer = ApiClientLive.pipe(
        Layer.provide(
          Layer.succeed(
            HttpClient.HttpClient,
            HttpClient.make((request) =>
              Effect.sync(() => {
                sent.push(request);
                return HttpClientResponse.fromWeb(
                  request,
                  json({ ok: true }, 200, "application/json")(),
                );
              }),
            ),
          ),
        ),
        Layer.provide(withConfig({})),
      );

      yield* Effect.flatMap(ApiClient, (api) =>
        Effect.all([
          api.patch("/api/persons/ada", { name: "Ada" }, Echo),
          api.put("/api/active-model", { modelKey: "sface" }, Echo),
        ]),
      ).pipe(Effect.provide(layer));

      expect(
        sent.map((request) => [
          request.method,
          request.url,
          request.body._tag === "Uint8Array"
            ? new TextDecoder().decode(request.body.body)
            : request.body._tag,
          request.body.contentType,
        ]),
      ).toEqual([
        ["PATCH", "/api/persons/ada", '{"name":"Ada"}', "application/json"],
        [
          "PUT",
          "/api/active-model",
          '{"modelKey":"sface"}',
          "application/json",
        ],
      ]);
    }),
  );

  it.effect("accepts a 204 with no body for a DELETE", () =>
    Effect.flatMap(ApiClient, (api) => api.delete("/api/x")).pipe(
      Effect.provide(
        apiClientResponding(() => new Response(null, { status: 204 })),
      ),
    ),
  );
});
