# Wave 3 session, 4 October 2026

What happened in the session that resolved #69's conflicts and started wave 3. That wave is #32, the Evaluation page and the final report: the last ticket between Ryuk and a finished project.

Read the same-person session record (`2026-10-03-same-person-session.md`) and the LFW gate session record (`2026-10-04-lfw-gate-session.md`) first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- **`main` is at `e79f377`.** Every ticket up to and including #49 has merged: #28, #61, #51, #52 and #69 (#49).
- **Open issues:** #32, #50, #62, and the spec #22, which closes last.
- **`ShikharJohari/32-evaluation-page`** (worktree `~/.t3/worktrees/Ryuk/32-evaluation-page`) is pushed. There is no PR yet. It holds:
  - `6f61467`: the service half of #32, `GET /api/evaluation`.
  - `f5c48b3`: `main` merged in (no file changes).
- **Checks on `6f61467`:**
  - `ruff`, `ruff format --check` and `mypy` are clean.
  - `pytest` passes: 1101 tests, 96.55% coverage.
  - `pnpm --dir web typecheck` and `lint` are clean.
  - After the merge, the evaluation API, contract and startup tests still pass.
- **The Evaluation page** (the client half of #32) was briefed to an implementer agent. That agent was stopped before it wrote anything. Nothing is uncommitted.

## How #32 is split

Two PRs that touch disjoint files, so two threads can run in parallel:

| Part | Owns | Branch |
|---|---|---|
| **#32a** Evaluation page | `src/ryuk/api/`, `src/ryuk/settings.py`, `src/ryuk/watchlist/load.py`, `src/ryuk/cli.py`, `web/`, `openapi.json` | `ShikharJohari/32-evaluation-page` (service done; client next) |
| **#32b** Report and README | `report/`, `README.md` | new, from `main` |

Each PR says `Part of #32`. The last one to merge says `Closes #32`.

## #32a: what the service already does

- **`GET /api/evaluation`** (operation `getEvaluation`, tag `evaluation`) is in `src/ryuk/api/evaluation.py`. Its models and the mapper are in `src/ryuk/api/evaluation_models.py`, whose docstrings define every field.
  - It serves a trimmed camelCase view of `evaluation/results.json` and `eda/summary.json`: 113 KB from the real files, against about 1 MB of results. A test holds it under 128 KiB.
  - Left out of the view: identity lists, the validation draw, method curves, gap histograms, the top-gap samples and provenance. A test checks those keys never appear.
- **Startup.**
  - New setting `eda: Path = Path("eda")` (`RYUK_EDA`), closing trap T14.
  - `ryuk serve` reads results and the summary once, maps them, and passes the view to `create_app(..., evaluation=...)`. The watchlist opens with the same `Results` object.
  - A missing or invalid summary is a `StartupError` raised before the database is touched.
  - `create_app()` with no view answers `503 evaluation_unavailable`.
- **Response shape** (`EvaluationReport`):
  - Always present: `dataset` and `verification`, which includes `sfaceInt8`.
  - Nullable: `identification` (test-draw curve only), `firstActiveModel`, `learning` (methods, gains, the live rule), `bias` (every model and rule; the client filters to best-photo) and `live` (`samePerson`, `smallGalleries`).
  - Shared shapes: `Rate {value, ci{low,high}, adjustedWilson}` and `ModelRef`. Its `id` equals `GET /api/models`' `id`.
- **Choices the service agent made:**
  - FaceNet's "± 0.25" exists only as text in `published.note`. There is no structured spread field, so the page shows the note.
  - `targetFpir` is per model, as results.json has it.
  - `verification.tolerancePoints` is the code constant. The recorded gate is `firstActiveModel.lfwGate`.
  - The dataset section gives counts, not rates; the client divides.
  - `tests/e2e_service.py` now passes the committed evaluation in. CI's browser smoke exercises it; it wasn't run locally.

## #32a: the client brief (not started)

Build `web/src/routes/evaluation.tsx`. It is a PageHeader placeholder today.

**Rulings that bind the page:**
- **#47 Q12:**
  - Charts are hand-rolled SVG, with no chart dependency.
  - #13's style: light grid (hairline `#D8D2C4`), ink axes, colour plus dash per model, direct labels, no legend.
  - Model styles: ArcFace `#1F4E8C` solid; FaceNet `#8A4FA0` long dash (7, 2.5); SFace `#C98A1B` dotted (1.2, 1.8).
  - Mirror `src/ryuk/plotting/style.py` and `src/ryuk/evaluation/figures.py`:
    - `lfw_roc`: log FAR from 1e-4, step-post.
    - `openset_curves`: log FPIR from `10^floor(log10(1/nonMatedProbes))`, the frozen-threshold point marked.
    - `group_fpir`: best-photo rule only, a dashed line at overall FPIR, indicative rows shaded.
- **#13:**
  - Numbered figures and tables with serif captions, using the existing `TableCaption` pattern.
  - Ruled tables, no cards.
  - Only the tokens in `web/src/styles.css`.
- **#20:**
  - Tables and figures, plus the models table, read-only.
  - Switching models happens only in the monitor toolbar.
- **The #32 audit's labelling traps:**
  - "ms per face" here is end to end, detection included. SFace int8's footnote figure is embedding only.
  - FaceNet's ± is a fold standard deviation, not a standard error.
  - TAR at FAR 0.1%, and anything flagged `indicative`, is marked indicative.

**Page order, mirroring report Section 5.** Every nullable section shows a short "not measured yet" line when it's null.

1. Models, read-only: state, threshold, ms per face. Then the first active model, its reason and the eligibility table.
2. Dataset.
3. LFW:
   - Table 1, both accuracy figures.
   - Figure 1, the ROC.
   - The int8 footnote.
4. CelebA test draw:
   - Table 2.
   - Figure 2, TPIR against FPIR.
5. Learning: a methods table per model, plus the live rule and its reason.
6. Bias:
   - Group FPIR bars.
   - A per-group table, with each attribute's basis labelled.
7. Live:
   - Small galleries: 500/100/20/5 identities × 5 or 1 photos.
   - The same-person threshold table.

**Engineering:**
- Follow `web/src/api/models.ts` and `models.queries.ts`: an Effect `Schema` per shape, each asserting type equality with `schema.gen.ts`.
- Route loader: `ensureQueryData` for evaluation and models; the page uses `useSuspenseQuery`.
- Components go in `web/src/components/evaluation/`. Add one small typed SVG primitive set: linear and log scales, axes and grid, a dashed series, direct labels, bars.
- Each `<svg>` gets `role="img"` and `aria-labelledby` pointing at its caption.
- Tests are vitest with MSW, modelled on `web/src/sightings.test.tsx`. Fixtures go in `web/src/test/evaluation.ts`. Cover:
  - every table and figure rendered from the API;
  - null sections;
  - a problem response;
  - no switch control on the page;
  - unit tests for scales and log ticks.

  This is the acceptance criterion "renders every table and chart from the API and is covered by route tests".

## #32b: the report and README (not started; start from `main`)

**Report sections:**
- 17 `::: stub` blocks remain across sections 01, 03, 04, 05, 06, 07, 08 and 09 (`grep -c "::: stub" report/sections/*.qmd`).
- Write each one from committed outputs only: no hand-typed result numbers.
- #47 Q11: 5.3, 5.4 and 6.1 belong to #28 (already written). Figures stay where they are written; no floats.
- #47 Q5: 5.1's gate paragraph is already in place (#52). The 5.1 stub still needs Table 1, the ROC and footnotes, including FaceNet's View 2 being scored twice (PR #35).

**Render check (T17):**
- Add a check that fails the render on any `::: stub` left.
- `report/style/stub.lua` never fails a render today.

**Limitations:**
- 8.x must cover no skin-tone label, possible training-set overlap, no liveness check, the unmeasured confirmation rule and no authentication.
- 8.9 (what purge can't erase) exists. #51 merged with its "Flash/SSD" bullet as written. Shikhar asked what "SSD flash" meant (it's the Mac's internal SSD keeping stale cells), so a one-line rewording is optional if he wants it.

**README:**
- Setup, fetching data and weights, reproducing every result, running the app.
- T16: "a clean checkout reproduces the committed outputs" holds only on Apple Silicon (#47 Q2). Linux runs SFace only; ArcFace on CPU is `not_evaluated`. Say so.

**Other audit notes:**
- Section 2 cites 3.3, which must exist before render.
- T18: check nothing from #46's doc fixes regressed.

## The other open tickets

- **#50 (ready-for-human):** measure 1280 px before switching (#47 Q4).
  - Gate 1: ≥ 6 fps through the SSH tunnel.
  - Gate 2: YuNet ≥ 0.9 on 35 px faces.
  - If both pass, change only `MAX_FRAME_SIDE` and the camera constraint in `web/src/lib/camera.ts`, plus `detector.py`'s 640 px comment. Otherwise stay at 640 px and record the ~1 m range.
  - This needs Shikhar, the Mac camera and the tunnel. Run it from a Mac thread.
- **#62 (needs-triage):** recognise faces in an uploaded photo.
  - Recommended answers, waiting on Shikhar:
    1. Reuse the live path unchanged.
    2. Process in memory and discard (ADR 0003/0004).
    3. Stateless: no sightings.
    4. The same limits and validation as enrollment uploads.
  - Don't start it before he rules.
- **#22** closes when everything above has merged.

## What happened this session

- **#69's conflicts.**
  - `main` (#51, #52) was merged into #49's branch.
  - In `active.py`, `assemble` now passes both `live_rules` and `lfw_gate(verification)`. Both tests keep both sides.
  - The auto-merged `results.json` was checked by re-assembling it from its own sections: identical, schema unchanged.
  - The full suite, notebooks and report render all passed, and CI went green on all 7 checks. Shikhar merged it.
- **Lost work, and the T3 server.**
  - The first try at #32a's service was killed at 02:46 when the Mac reopened its lid. The Mac's T3 app ran the server through an SSH environment, which it kills and relaunches when the tunnel goes stale (upstream T3 issues #10934 and #5749).
  - The Mac is now paired with the always-on `t3code.service` (127.0.0.1:3773, reached through Tailscale Serve, in `agents.slice`), so closing the lid no longer stops agents.
  - The Linux desktop T3 app had run a third server on the same `~/.t3` database and started a duplicate Claude on this thread. It's closed now.
  - `~/.t3/userdata/server-runtime.json` was deleted by that app on exit (upstream #14115). The service rewrites it on its next restart.
  - **Updating:** update the Mac app any time. Update the box (`Update server`, or `ssh ohmahgahpc 't3 update && t3 service restart'`) only when no thread is mid-turn, because the restart interrupts them.

## Working rules

- Nothing merges without Shikhar's go-ahead (#47 Q16). Merge with a merge commit.
- No `Co-Authored-By` or other Claude credit in commits (#47 Q17). CI runs `.github/scripts/check-attribution.sh`. PR bodies end with "🤖 Generated with [Claude Code](https://claude.com/claude-code)".
- Role agents only (`explorer`, `researcher`, `implementer`, `reviewer`), never Haiku. Run long agents in the background so the thread stays responsive.
- Never run Playwright locally; `pnpm e2e` is CI only. Verify the service against a real `ryuk serve`, with and without weights. Ryuk's client stays on 127.0.0.1.
- `results.json` is written only by `ryuk evaluate`. To refresh derived blocks, re-assemble the file from its own sections; never hand-edit it.
- Stay in your own worktree. Write the handoff before merge.
