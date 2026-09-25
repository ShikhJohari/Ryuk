import { FetchHttpClient } from "@effect/platform";
import {
  Cause,
  ConfigProvider,
  type Effect,
  Exit,
  Layer,
  ManagedRuntime,
} from "effect";
import { type ApiClient, ApiClientLive } from "@/api/api-client";

/** Effect `Config` reads come from Vite's env (`VITE_*` variables). */
const ViteEnvConfig = Layer.setConfigProvider(
  ConfigProvider.fromJson(import.meta.env),
);

/** The client's single Layer composition root. */
export const AppLayer = ApiClientLive.pipe(
  Layer.provide(FetchHttpClient.layer),
  Layer.provide(ViteEnvConfig),
);

/** Built once, shared by every TanStack Query function and route loader. */
export const runtime = ManagedRuntime.make(AppLayer);

/**
 * The only way to leave Effect: call it from a TanStack Query `queryFn` or
 * `mutationFn`, or a route loader, and nowhere else.
 *
 * `runtime.runPromise` would reject with Effect's `FiberFailure` wrapper;
 * this rejects with the typed failure itself (an `ApiProblem`, say), so
 * `query.error` can be matched on its `_tag`.
 */
export const runQuery = <A, E>(
  effect: Effect.Effect<A, E, ApiClient>,
  options?: { readonly signal?: AbortSignal },
): Promise<A> =>
  runtime
    .runPromiseExit(effect, options)
    .then((exit) =>
      Exit.isSuccess(exit)
        ? exit.value
        : Promise.reject(Cause.squash(exit.cause)),
    );
