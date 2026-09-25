# Scaffold session, 25 September 2026

The record of the sixth working session, the first build session. It implemented ticket #23 (Scaffold, contract pipeline and CI) with `/implement` and opened PR #33. Read the closing session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- PR https://github.com/ShikhJohari/Ryuk/pull/33 on `ShikharJohari/23-scaffold`, `Closes #23`. All four CI jobs green. **Not merged**: waiting on Shikhar.
- Every other build ticket (#24 to #32) is blocked, directly or transitively, by #23. The next is #24 (Data fetch, weights and YuNet detection), unblocked once PR #33 merges.

## Patterns later tickets should copy

- **Service.** `create_app()` in `src/ryuk/api/__init__.py`; one router module per area under `src/ryuk/api/`; models extend `ApiModel` (camelCase on the wire). A response model sets every field explicitly: with `separate_input_output_schemas=False` a field with a default is optional in the contract. Operation IDs are the camelCased route function names. Errors: raise `ProblemError(status=..., code=..., detail=...)`; every error is RFC 9457 `application/problem+json` with a stable `code`.
- **Contract.** After changing any API model: `uv run ryuk openapi && pnpm --dir web gen:api`, commit both files. `pytest` and the CI contract job both fail on a stale `openapi.json`. WebSocket message models go in `WEBSOCKET_MODELS` in `src/ryuk/api/contract.py`.
- **Client.** One Effect schema per response in `web/src/api/`, each followed by `Assert<Equals<Schema.Type, components["schemas"][...]>>`. Calls go through the `ApiClient` service; effects leave Effect only through `runQuery` (from a queryFn, mutationFn or loader), which rejects with the tagged error, e.g. `ApiProblem` with its `code`. Tests: API boundary through a test `HttpClient` Layer; routes against MSW via `web/src/test/api-server.ts`.
- **Settings.** `RYUK_*` env vars via `src/ryuk/settings.py`; the bind host must be loopback. A Host-header guard also refuses non-localhost requests (DNS rebinding).
- **Logging.** JSON lines through `ryuk.logs.configure_logging`, uvicorn included.

## Things that bit

- Starlette 1.7's test client wants `httpx2`, not `httpx`.
- TypeScript 7 ships no compiler API; `web/.pnpmfile.cjs` gives openapi-typescript a private TypeScript 5.9.3.
- pnpm 12 starts child processes in their own process group, so Playwright's `webServer` must run `vite preview` directly, not `pnpm preview`: the first CI run hung for 16 minutes on it.
- GitHub Action floating tags: `astral-sh/setup-uv` has none since v7 (pinned `v10.2.0`); `pnpm/action-setup@v6` is stale (pnpm/action-setup#289; pinned `v6.1.0`).
- Corepack can't launch pnpm 11+ (nodejs/corepack#775). pnpm 12.6.0 is now installed globally with npm in `~/.npm-global`.
- pnpm 12 enforces a minimum release age, so a package published today may refuse to install; pin the previous patch.

## Decisions and open items

- `TO-BE-REVIEWED.md`: the client's module-scope `ManagedRuntime` was kept (spec and #6) against a review reading of `effect.md` rule 4.
- Blacksmith runners were researched and not adopted: Blacksmith only serves GitHub organizations and this repo is under a personal account, and the CI is ~2.5 billed minutes per run. Revisit when the weights smoke job lands, where sticky disks could cut cache restores.

## Resuming

After Shikhar merges PR #33, start a fresh session with `/implement` on #24, per the closing session's working rules: one ticket per session, a `ShikharJohari/<n>-<slug>` branch, a PR that says `Closes #N`, nothing merged without Shikhar, role agents only, never Haiku, never Playwright locally.
