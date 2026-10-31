## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for ShikhJohari/Ryuk, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Commands

- Service: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest` (85% coverage floor).
- Client, from `web/`: `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`.
- Contract: after changing an API model run `uv run ryuk openapi && pnpm --dir web gen:api` and commit `openapi.json` and `web/src/api/schema.gen.ts`. Each hand-written Effect schema in `web/src/api/` asserts type equality with its generated type, so drift fails `tsc`.
- Never run Playwright locally; `pnpm e2e` is a headless CI-only job.
- Ryuk's client stays on `127.0.0.1` whatever a machine's own rules say about binding dev servers to `0.0.0.0`: the API has no authentication. Reach a remote box through an SSH local forward to `localhost` (README, "On a remote machine").

