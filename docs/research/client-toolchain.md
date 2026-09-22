# Client toolchain: TanStack Router, Effect, shadcn/ui, Tailwind

Research for issue #6. Versions below are what `npm view <pkg> version` returned against the npm registry on 2026-09-22; pin these (or newer patch/minor) at scaffold time rather than trusting this document a year from now.

## Scaffold recipe

The client lives at `apps/web` per `~/.claude/docs/stack.md`. Run everything with `pnpm`.

1. Scaffold with the TanStack CLI, not `create-tsrouter-app`. That package now prints a deprecation warning on every run and points to its replacement:

   ```
   npx create-tsrouter-app@latest --help
   > Warning: create-tsrouter-app is deprecated. Use "tanstack create --router-only"
   > or "npx @tanstack/cli create --router-only" instead.
   ```

   So scaffold with:

   ```bash
   pnpm dlx @tanstack/cli@latest create apps/web --router-only \
     --package-manager pnpm --toolchain eslint --add-ons tailwind
   ```

   `--router-only` gives file-based routing without the TanStack Start server runtime, which matches the ADR: the client renders, the FastAPI service owns logic. Use `--blank` if you want to drop the demo routes it seeds; otherwise `pnpm dlx create-tsrouter-app@latest clean-demos` strips them after the fact.

2. Confirm the generated `vite.config.ts` has the router plugin ahead of the React plugin. This ordering is load-bearing, not stylistic: the router plugin generates `src/routeTree.gen.ts` from `src/routes/**` before the React plugin transforms JSX, and code-splitting only works in that order.

   ```ts
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'
   import { tanstackRouter } from '@tanstack/router-plugin/vite'

   export default defineConfig({
     plugins: [
       tanstackRouter({ target: 'react', autoCodeSplitting: true }),
       react(),
     ],
   })
   ```
   (TanStack Router, installation with Vite: https://tanstack.com/router/latest/docs/framework/react/installation/with-vite)

3. Add Tailwind v4 and shadcn/ui. Tailwind v4 has no `tailwind.config.js` by default; configuration is CSS-first.

   ```bash
   pnpm add tailwindcss @tailwindcss/vite
   pnpm add -D @types/node
   ```

   `src/index.css`:
   ```css
   @import "tailwindcss";
   ```

   `vite.config.ts` adds the Tailwind plugin and a `@` path alias:
   ```ts
   import path from "path"
   import tailwindcss from "@tailwindcss/vite"
   // ...
   export default defineConfig({
     plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss()],
     resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
   })
   ```
   `tsconfig.json` (and `tsconfig.app.json`) need the matching `paths` entry:
   ```json
   { "compilerOptions": { "baseUrl": ".", "paths": { "@/*": ["./src/*"] } } }
   ```
   Then init shadcn and add components as needed:
   ```bash
   pnpm dlx shadcn@latest init
   pnpm dlx shadcn@latest add button
   ```
   (shadcn/ui, Vite install guide: https://ui.shadcn.com/docs/installation/vite)

4. Add Effect, TanStack Query, and their router glue:

   ```bash
   pnpm add effect @effect/platform @effect/platform-browser @tanstack/react-query
   pnpm add -D @effect/vitest vitest @vitejs/plugin-react @testing-library/react jsdom \
     msw eslint typescript-eslint eslint-plugin-react-hooks @tanstack/eslint-plugin-router \
     @tanstack/eslint-plugin-query
   ```

Pinned versions as of this research (npm registry, 2026-09-22):

| package | version | note |
|---|---|---|
| `@tanstack/react-router` | 1.170.38 | |
| `@tanstack/router-plugin` | 1.168.40 | |
| `@tanstack/react-router-devtools` | 1.167.2 | |
| `@tanstack/react-query` | 5.103.2 | |
| `@tanstack/react-query-devtools` | 5.103.2 | |
| `@tanstack/eslint-plugin-router` | 1.162.0 | |
| `@tanstack/eslint-plugin-query` | (published under `@tanstack/eslint-plugin-query`, track alongside react-query) | |
| `effect` | 3.22.2 | `latest` dist-tag. `4.0.0-rc.117` exists but is not GA yet; do not adopt v4 until it ships `latest`. |
| `@effect/platform` | 0.97.2 | |
| `@effect/platform-browser` | 0.77.1 | |
| `@effect/vitest` | 0.30.0 | |
| `vite` | 8.3.0 | |
| `@vitejs/plugin-react` | 6.1.1 | |
| `tailwindcss` | 4.3.3 | |
| `@tailwindcss/vite` | 4.3.3 | |
| `shadcn` | 4.21.0 | CLI package name is `shadcn`, not `shadcn-ui` (renamed a while back) |
| `typescript` | 7.0.2 | |
| `eslint` | 10.11.0 | |
| `typescript-eslint` | 8.70.1 | |
| `vitest` | 5.0.1 | |
| `msw` | 2.15.0 | for mocking the FastAPI backend in component/unit tests |
| `pnpm` | 12.5.1 | |

Do not add `@effect/schema` as a dependency. It is deprecated; npm shows `this package has been merged into the main effect package`. `Schema` now ships from `effect` itself (`import { Schema } from "effect"`, or `effect/Schema`). (Effect docs, Schema getting started: https://effect.website/docs/schema/getting-started/)

## Effect integration pattern

The ADR is explicit that "any logic that appears in the client is a smell." Effect's job in this client is narrow: own the boundary with the FastAPI service (HTTP calls, WebSocket frame streaming, response decoding, typed failures) and hand already-validated data to React. No business logic, no re-implementing anything the service already does.

### Layers: one runtime, built once

Per `effect.md` rule 4, services come from `Context.Tag` and get wired with `Layer` at a single composition root, not grabbed from module scope. In a React SPA there is no long-lived server process to be that root, so the accepted pattern is a `ManagedRuntime` built once at module scope in a dedicated `runtime.ts` and reused everywhere. `ManagedRuntime` exists for exactly this case: "environments where you have limited control over the main application entry point," such as a frontend framework (Effect docs, Introduction to Runtime).

```ts
// src/lib/effect-runtime.ts
import { ManagedRuntime, Layer } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { ApiConfigLive } from "./api-config"

const AppLayer = Layer.mergeAll(FetchHttpClient.layer, ApiConfigLive)

export const runtime = ManagedRuntime.make(AppLayer)
```

### Schema at the boundary, typed errors, no promises in domain code

Define the FastAPI response shape as a `Schema`, and use `@effect/platform`'s `HttpClient` (not raw `fetch`) so decoding and HTTP failures both land in the effect's typed error channel instead of a thrown exception.

```ts
// src/api/watchlist.ts
import { Effect, Schema } from "effect"
import { HttpClient, HttpClientRequest, HttpClientResponse } from "@effect/platform"

export class PersonOfInterest extends Schema.Class<PersonOfInterest>("PersonOfInterest")({
  id: Schema.String,
  name: Schema.String,
  enrolledAt: Schema.DateFromString,
}) {}

const Watchlist = Schema.Array(PersonOfInterest)

// Effect<Watchlist, HttpClientError.HttpClientError | ParseResult.ParseError, HttpClient.HttpClient>
export const fetchWatchlist = Effect.gen(function* () {
  const client = yield* HttpClient.HttpClient
  const response = yield* client.execute(HttpClientRequest.get("/api/watchlist"))
  return yield* HttpClientResponse.schemaBodyJson(Watchlist)(response)
})
```

`HttpClient.execute` fails with tagged `RequestError` / `ResponseError` from `@effect/platform`'s `HttpClientError` module (network failure, non-2xx, etc.), and `schemaBodyJson` fails with `ParseError` from `effect/ParseResult` when the body does not match `Watchlist`. Both are typed in the effect's error channel, so a caller can `Effect.catchTags` on `RequestError | ResponseError | ParseError` and never needs a `catch (e: any)`. This satisfies `effect.md` rules 1, 2, and 5 in one shape: no naked promise, tagged errors, schema decoding at the trust boundary.

### Exposing effects to TanStack Query and router loaders

`Effect.runPromise` is reserved for "the app entry point / adapter edge" (`effect.md` rule 9). TanStack Query's `queryFn` and TanStack Router's `loader` are exactly that edge: they are the seam where the outside world (React, the router) asks for a `Promise`. Run the effect there, once, through the shared runtime:

```ts
// src/api/watchlist.queries.ts
import { queryOptions } from "@tanstack/react-query"
import { runtime } from "../lib/effect-runtime"
import { fetchWatchlist } from "./watchlist"

export const watchlistQueryOptions = queryOptions({
  queryKey: ["watchlist"],
  queryFn: () => runtime.runPromise(fetchWatchlist),
})
```

```ts
// src/routes/watchlist.tsx
import { createFileRoute } from "@tanstack/react-router"
import { useSuspenseQuery } from "@tanstack/react-query"
import { watchlistQueryOptions } from "../api/watchlist.queries"

export const Route = createFileRoute("/watchlist")({
  loader: ({ context }) => context.queryClient.ensureQueryData(watchlistQueryOptions),
  component: () => {
    const { data } = useSuspenseQuery(watchlistQueryOptions)
    return <WatchlistTable people={data} />
  },
})
```

The router's root route wires a single `QueryClient` into `context` and wraps the tree in `QueryClientProvider`, following the router's own external-data-loading guide (`context: { queryClient }`, `defaultPreloadStaleTime: 0` so the router defers caching decisions to Query). TanStack Query surfaces a rejected `queryFn` promise as `query.error`; a rejected loader promise renders the route's `errorComponent`. Either way, the failure that reaches React is whatever tagged error the effect failed with, since `runtime.runPromise` rejects the promise with that same value rather than an opaque `Error`. If you want to render distinct UI per failure kind (network vs. validation), catch tags before the promise boundary:

```ts
queryFn: () =>
  runtime.runPromise(
    fetchWatchlist.pipe(
      Effect.catchTag("ResponseError", () => Effect.succeed([] as const)),
    ),
  ),
```

This is the extent of the pattern: `Layer` composes the `HttpClient`, `Schema` decodes the boundary, tagged errors flow through, and exactly two call sites (`queryFn`, `loader`) touch `runtime.runPromise`. Everything upstream of those two call sites stays inside Effect's world with no `async/await`.

## CI job for the client

Following `~/.claude/docs/stack.md` ("CI running typecheck + tests + lint on PR") and the ADR's split into two toolchains, the client gets its own GitHub Actions job, separate from the Python service's `uv`-based job, gated on changes under `apps/web/`.

```yaml
# .github/workflows/client.yml
name: client
on:
  pull_request:
    paths: ["apps/web/**"]
  push:
    branches: [main]
    paths: ["apps/web/**"]

jobs:
  client:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: apps/web
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
          cache-dependency-path: apps/web/pnpm-lock.yaml
      - run: pnpm install --frozen-lockfile
      - run: pnpm typecheck   # tsc --noEmit (or tsc -b for project refs)
      - run: pnpm lint        # eslint .
      - run: pnpm test        # vitest run
      - run: pnpm build       # vite build; catches routeTree.gen.ts drift too
```

Order matters: typecheck first (cheapest signal), then lint, then test, then build last since `vite build` also re-runs the TanStack Router codegen and will fail if a route file is malformed in a way tests didn't catch. Each script should be a plain `package.json` script (`"typecheck": "tsc -b --noEmit"`, `"lint": "eslint ."`, `"test": "vitest run"`, `"build": "vite build"`) so CI and local runs use the same command.

## Evidence and citations

- TanStack Router installation (Vite plugin order, `tanstackRouter` from `@tanstack/router-plugin/vite`): https://tanstack.com/router/latest/docs/framework/react/installation/with-vite
- `create-tsrouter-app` deprecation notice, observed directly by running `npx create-tsrouter-app@latest --help` against the published CLI (v0.54.43): the tool prints the warning shown above and lists `clean-demos`, `pin-versions`, and `--router-only` as current subcommands/flags.
- TanStack Router + TanStack Query wiring (`context: { queryClient }`, `dehydrate`/`hydrate`, `queryClient.ensureQueryData` in `loader`, `useSuspenseQuery` in component, `defaultPreloadStaleTime: 0`): https://tanstack.com/router/latest/docs/framework/react/guide/external-data-loading
- Effect Schema module location (`effect/Schema`, `@effect/schema` deprecated): https://effect.website/docs/schema/getting-started/ and `npm view @effect/schema deprecated` → "this package has been merged into the main effect package"
- Effect `ManagedRuntime` for framework integration ("environments where you have limited control over the main application entry point"; create once, reuse, `runtime.runPromise`): Effect docs, Introduction to Runtime, and corroborated via web search against effect-ts.github.io/effect API docs for `ManagedRuntime`
- `@effect/platform` `HttpClient` / `HttpClientRequest` / `HttpClientResponse.schemaBodyJson` pattern and `FetchHttpClient.layer` for browser/fetch-based clients: verified code sample cross-checked against Effect-TS/effect's `packages/platform` usage patterns (gist by an Effect maintainer-adjacent source, cross-referenced with the package's public API surface: `HttpClient`, `HttpClientRequest`, `HttpClientResponse`, `FetchHttpClient` are all exported from `@effect/platform`, confirmed via `npm view @effect/platform` and its listed dependents)
- shadcn/ui Vite + Tailwind v4 install steps (`@tailwindcss/vite`, CSS-first config, `pnpm dlx shadcn@latest init`): https://ui.shadcn.com/docs/installation/vite
- Package versions: `npm view <package> version` / `npm view <package> dist-tags --json` against the public npm registry, run 2026-09-22
- ADR context (client renders only, no client-side logic, two CI jobs): `docs/adr/0001-python-inference-service-with-react-client.md`

### Gaps and things I could not confirm from primary sources

- I could not load the current canonical `@effect/platform` HttpClient doc page directly (the `effect.website/docs/platform/http-client` and `/docs/v3/platform/http-client` URLs both 404 at time of writing, likely a docs restructure in flight). The code pattern above is reconstructed from the package's public exports and a maintainer-adjacent example, not copy-pasted from an official page. Re-check `https://effect.website/docs/platform/` before treating the exact API names (`HttpClientRequest.get` vs `client.get`) as final; both spellings show up across sources and the package has changed shape before ("HttpClient.fetch client implementation was removed").
- `@tanstack/eslint-plugin-query`'s exact current version was not independently isolated in this pass; install it alongside `@tanstack/react-query` and let pnpm resolve to whatever is current rather than hand-pinning a possibly stale number.
- Effect v4 is at release-candidate (`4.0.0-rc.117`) as of this research, not yet the `latest` dist-tag. Worth a follow-up ticket once v4 ships stable, since `@effect/platform` and `@effect/platform-browser` both publish `beta`/`rc` tags in lockstep and the HttpClient API has shifted across major versions before.

## Constraints from effect.md that this setup must honour

- **No naked promises in domain code** (rule 1): the only two `Effect.runPromise` call sites in the whole client are the `queryFn` in query option factories and route `loader`s. Nothing else should call `runtime.runPromise` or `Effect.runPromise` directly, including inside components; a raw `await` in a component body is a review finding.
- **Errors are typed and tagged** (rule 2): every function exported from `src/api/*` must have an error channel that is a union of tagged errors (`RequestError | ResponseError | ParseError`, or a mapped domain error), never `unknown`. If you wrap a third-party SDK that throws, map its exception in `Effect.tryPromise`'s `catch` to a `Data.TaggedError`, don't let it surface as-is.
- **Services via `Context.Tag`, wiring via `Layer`** (rule 4): the `HttpClient` service, any API base URL config, and any WebSocket client for the frame-streaming feature (issue elsewhere in this epic) are all `Layer`s composed once in `effect-runtime.ts`. Don't import a configured client as a module-level singleton from `api/client.ts`.
- **`Schema` at every boundary** (rule 5): every FastAPI response gets a `Schema.Class` or `Schema.Struct`, decoded with `HttpClientResponse.schemaBodyJson` or `Schema.decodeUnknown`. No `as PersonOfInterest[]` casts on `response.json()`.
- **Config via `Config`** (rule 6): the API base URL (`http://localhost:8000` in dev, whatever the deployed FastAPI origin is in prod) is read through Effect's `Config` module inside a `Layer`, not `import.meta.env.VITE_API_URL` sprinkled through call sites. Fail fast with a readable error if it's missing.
- **`Effect.runPromise` only at the entry point / adapter edge** (rule 9, restated in the review checklist): this is the rule most likely to get violated in a React app, because every `onClick` handler and `useEffect` looks like a legitimate "edge." Route loaders and `queryFn` are the sanctioned edges; a button's `onClick` that needs to fire a mutation should call `mutate()` from a `useMutation` whose `mutationFn` is the one place that runs the effect, not a fresh `runtime.runPromise` scattered into the handler itself.
- **Testing** (Testing section): FastAPI calls are tested by providing a test `Layer` for `HttpClient` (e.g. via `@effect/platform`'s test HTTP client or an `msw`-backed one) rather than mocking `fetch` globally, and error paths get their own `it.effect` test asserting on the tagged failure with `Effect.flip`.
