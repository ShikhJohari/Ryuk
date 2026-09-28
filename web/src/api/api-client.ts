import {
  Headers,
  HttpClient,
  HttpClientError,
  HttpClientRequest,
  HttpClientResponse,
} from "@effect/platform";
import {
  Config,
  Context,
  Effect,
  Layer,
  Option,
  type ParseResult,
  Schema,
} from "effect";
import type { Assert, Equals, Simplify } from "@/lib/type-equality";
import { WarningsProblem } from "./problem";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

/**
 * Any problem body: a `Problem`, with the one extension the service sends,
 * `WarningsProblem`'s `warnings`, on `409 warnings` only.
 */
const ProblemBody = Schema.Struct({
  ...WarningsProblem.fields,
  warnings: Schema.optional(WarningsProblem.fields.warnings),
});

export type ProblemBodyMatchesContract = Assert<
  Equals<
    typeof ProblemBody.Type,
    Simplify<
      Omit<Schemas["WarningsProblem"], "warnings"> &
        Partial<Pick<Schemas["WarningsProblem"], "warnings">>
    >
  >
>;

/**
 * A non-2xx response from the service whose body is a decodable problem.
 * `warnings` is set only on `409 warnings` (a `WarningsProblem`).
 */
export class ApiProblem extends Schema.TaggedError<ApiProblem>()(
  "ApiProblem",
  ProblemBody.fields,
) {}

/**
 * Every way a call to the service can fail:
 * - `ApiProblem`: the service answered non-2xx with a problem body.
 * - `RequestError`: the request never got a response (network, bad URL).
 * - `ResponseError`: a non-2xx without a decodable problem body, or a 2xx
 *   body that is not JSON.
 * - `ParseError`: a 2xx JSON body that does not match the expected schema.
 */
export type ApiError =
  | ApiProblem
  | HttpClientError.HttpClientError
  | ParseResult.ParseError;

export class ApiClient extends Context.Tag("ryuk/ApiClient")<
  ApiClient,
  {
    /** GET `path` and decode a 2xx JSON body with `schema`. */
    readonly get: <A, I>(
      path: string,
      schema: Schema.Schema<A, I>,
    ) => Effect.Effect<A, ApiError>;
    /** POST `body` as multipart/form-data and decode a 2xx JSON body. */
    readonly postForm: <A, I>(
      path: string,
      body: FormData,
      schema: Schema.Schema<A, I>,
    ) => Effect.Effect<A, ApiError>;
    /** PATCH `body` as JSON and decode a 2xx JSON body. */
    readonly patch: <A, I>(
      path: string,
      body: unknown,
      schema: Schema.Schema<A, I>,
    ) => Effect.Effect<A, ApiError>;
    /** PUT `body` as JSON and decode a 2xx JSON body. */
    readonly put: <A, I>(
      path: string,
      body: unknown,
      schema: Schema.Schema<A, I>,
    ) => Effect.Effect<A, ApiError>;
    /** DELETE `path`, expecting a 2xx with no body. */
    readonly delete: (path: string) => Effect.Effect<void, ApiError>;
  }
>() {}

const PROBLEM_MEDIA_TYPE = "application/problem+json";

const isProblemResponse = (
  response: HttpClientResponse.HttpClientResponse,
): boolean =>
  Headers.get(response.headers, "content-type").pipe(
    Option.exists(
      (value) =>
        (value.split(";")[0] ?? "").trim().toLowerCase() === PROBLEM_MEDIA_TYPE,
    ),
  );

const failNonSuccess = (
  response: HttpClientResponse.HttpClientResponse,
): Effect.Effect<never, ApiProblem | HttpClientError.ResponseError> => {
  const statusError = (cause?: unknown) =>
    new HttpClientError.ResponseError({
      request: response.request,
      response,
      reason: "StatusCode",
      description: `non-2xx status ${response.status} without a problem body`,
      cause,
    });

  if (!isProblemResponse(response)) {
    return Effect.fail(statusError());
  }
  return HttpClientResponse.schemaBodyJson(ProblemBody)(response).pipe(
    Effect.mapError(statusError),
    Effect.flatMap((problem) => Effect.fail(new ApiProblem(problem))),
  );
};

const succeedEmpty = (
  response: HttpClientResponse.HttpClientResponse,
): Effect.Effect<void, ApiError> =>
  response.status >= 200 && response.status < 300
    ? Effect.void
    : failNonSuccess(response);

const decodeResponse =
  <A, I>(schema: Schema.Schema<A, I>) =>
  (
    response: HttpClientResponse.HttpClientResponse,
  ): Effect.Effect<A, ApiError> =>
    response.status >= 200 && response.status < 300
      ? HttpClientResponse.schemaBodyJson(schema)(response)
      : failNonSuccess(response);

/**
 * The live client. Paths are resolved against `VITE_API_BASE_URL`, which
 * defaults to "" so requests stay relative and go through the Vite proxy.
 */
export const ApiClientLive = Layer.effect(
  ApiClient,
  Effect.gen(function* () {
    const baseUrl = yield* Config.string("VITE_API_BASE_URL").pipe(
      Config.withDefault(""),
    );
    const client = (yield* HttpClient.HttpClient).pipe(
      HttpClient.mapRequest(HttpClientRequest.prependUrl(baseUrl)),
      HttpClient.mapRequest(
        HttpClientRequest.setHeader(
          "accept",
          `application/json, ${PROBLEM_MEDIA_TYPE}`,
        ),
      ),
    );

    return ApiClient.of({
      get: (path, schema) =>
        client
          .execute(HttpClientRequest.get(path))
          .pipe(Effect.flatMap(decodeResponse(schema))),
      postForm: (path, body, schema) =>
        client
          .execute(
            HttpClientRequest.post(path).pipe(
              HttpClientRequest.bodyFormData(body),
            ),
          )
          .pipe(Effect.flatMap(decodeResponse(schema))),
      patch: (path, body, schema) =>
        client
          .execute(
            HttpClientRequest.patch(path).pipe(
              HttpClientRequest.bodyUnsafeJson(body),
            ),
          )
          .pipe(Effect.flatMap(decodeResponse(schema))),
      put: (path, body, schema) =>
        client
          .execute(
            HttpClientRequest.put(path).pipe(
              HttpClientRequest.bodyUnsafeJson(body),
            ),
          )
          .pipe(Effect.flatMap(decodeResponse(schema))),
      delete: (path) =>
        client
          .execute(HttpClientRequest.del(path))
          .pipe(Effect.flatMap(succeedEmpty)),
    });
  }),
);
