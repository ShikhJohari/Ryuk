# Wave 3 close session, 4 October 2026

The session that finished #32: the Evaluation page (#32a, PR #71) and the report and README (#32b, PR #70). Read the wave 3 session record (`2026-10-04-wave-3-session.md`, on #71's branch) first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- **PR #71**: `ShikharJohari/32-evaluation-page`, "Part of #32". It holds the service half, unchanged from the wave 3 session, and the client half: the Evaluation page.
- **PR #70**: `ShikharJohari/32-report`, "Closes #32". It holds the report's remaining sections, the stub check and the README. It merges after #71, because the README and Section 7 describe the page.
- **Merging.** Shikhar ruled that both PRs merge with no further go-ahead once CI is green and they don't conflict (merge commits).
- **Open after both merge:**
  - #50 needs the Mac camera and the tunnel.
  - #62 waits on Shikhar's four answers.
  - #22 closes last.

## #32a: the client

- **Where it lives.** The page is `web/src/routes/evaluation.tsx`, and its sections and SVG primitives are in `web/src/components/evaluation/`. The schemas are in `web/src/api/evaluation.ts`, the tests in `web/src/evaluation.test.tsx` plus unit tests beside the primitives, and the fixtures in `web/src/test/evaluation.ts`.
- **Numbering.** Tables and figures are numbered in page order, computed in `numbering.ts`, so an unmeasured section leaves no gap. Page numbers don't match the report's.
- **Readouts.** Hover readouts are HTML overlays, each focusable, positioned over a point or bar. Anything focusable inside an SVG with `role="img"` is hidden from screen readers.
- **Model hues.** They are `styles.css` tokens (`--color-model-*`) in an `@theme static` block, so Tailwind emits them even though only SVG attributes use them.
- **Review fixes, `98fc5a9`:**
  - indicative ROC points are drawn hollow;
  - the learning table marks an indicative headline TPIR instead of dropping it;
  - the group FPIR caption says why the overall line is missing when CelebA isn't measured;
  - an unscored LFW gap reads "not scored" alone;
  - `falseAlarmFloor` and `linearTicks` throw on input they can't honour;
  - a missing test draw is an error.
- **Left as they are, with the reasons:**
  - **FaceNet's ± caveat.** It keys on the "±" in `published.note`, because the service has no structured spread field. A `published.spread {value, kind}` field would be the honest fix if the note's wording ever changes.
  - **Tab stops.** There are about 45 readout tab stops, one per bar and point. A visually hidden data table per figure would be the alternative.
  - **Bias tables.** They show best photo for every model. The report's per-group table shows the first active model under its live rule. The brief was ambiguous, and the page follows "the client filters to best-photo".
  - **Duplication.** `roc-chart.tsx` and `open-set-chart.tsx` repeat their layout constants.
- **Unverified.** The readouts' placement and focus rings have not been seen in a real browser; the jsdom tests and an `rsvg-convert` render of the SVGs have. CI's e2e smoke step loads the page.

## #32b: the report and README

- **Stubs.** All 17 are written, and every result number is read from committed outputs.
  - FaceNet's superseded first View 2 run is described qualitatively: it fell more than the tolerance short, and PR #35 records it. Its figure isn't in `results.json`, so it isn't typed.
  - FaceNet's "± 0.25", the fold count and the draws' seed are read from the results, and their cells assert them.
- **The stub check (T17).** `report.qmd` sets `stubs: forbidden`, and `style/stub.lua` exits 1 naming every stub left.
  - Quarto only logs a plain `error()` from a filter, so the filter has to exit the process itself.
  - An unrecognised `stubs:` value also fails the render.
  - `progress.qmd` doesn't set it. It now includes all nine sections, so it is the full report plus a status note.
- **Section 7** is prose only. Screenshots need the Mac's camera and a consenting face.
- **Section 8.** The purge section is now 8.10, because #49 inserted "Random impostors" before it. The "Flash/SSD" bullet is untouched; the optional rewording is still open.
- **External claims worth Shikhar's check:**
  - the NIST demographic-differences sentence in 8.1;
  - the training sets named in 8.7: WebFace600K for ArcFace, VGGFace2 for FaceNet.
- **The README** says only an Apple Silicon Mac reproduces the committed outputs (T16). On Linux, ArcFace on CPU is `not_evaluated`, SFace runs, and the evaluate commands refuse to run.

## Working notes

- **T3 renamed this thread's branch** from `t3code/6cb30b8a` to `t3code/wave-3-32a-eval` mid-session. #71 was pushed from it to `ShikharJohari/32-evaluation-page`, so the worktree `~/.t3/worktrees/Ryuk/32-evaluation-page` still has that branch at `20f2daa` locally. Fetch before using it.
- The #32b implementer couldn't be resumed for its review fixes, so the main thread applied them.
