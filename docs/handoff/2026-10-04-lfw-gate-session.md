# LFW gate session, 4 October 2026

The record of a small build session. A lead session implemented #52 (#47 Q5: the LFW gate is judged on the 5,917 scored pairs, both figures reported) in its own worktree, had it reviewed by a reviewer agent, applied what it raised and checked the result against a real `ryuk serve`. It ran beside #51, in a separate worktree and PR. Read the photo delete and rename session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #52 is implemented on `ShikharJohari/52-gate-provenance` (worktree `~/.t3/worktrees/Ryuk/52-gate-provenance`, from `main` at `d7d160d`). Nothing merges without Shikhar's go-ahead (#47 Q16).
- Checks on the branch head: `ruff check`, `ruff format --check` and `mypy` clean; `pytest` 1035 passed, 17 deselected, 96.33% coverage; notebooks 3 passed under nbmake; the report renders; `ryuk evaluate schema` and `ryuk openapi` leave their files unchanged; every commit passes the attribution check.

## What changed

- **`LfwGate`** (`evaluation/results.py`) on `FirstActiveModel.lfw_gate`: `accuracy: "scored-pairs"`, `scored_pairs`, `pairs` and `tolerance_points`. The committed file now reads `scored-pairs`, 5,917 of 6,000, 0.5 points.
- **The record is what is applied.** `first_active_model(contenders, gate)` reads its tolerance from the gate (`active.lfw_gate(verification)` builds it in `assemble`). `FirstActiveModel` refuses eligibility its gate would not have judged, `Results` refuses a gate over other pairs than verification scored, and the registry's on-device fallback (`Evaluation.first_active_for`) applies the recorded gate; an `Evaluation` with contenders and no gate is refused.
- **One set of scored pairs.** `Verification` refuses a model whose scored and excluded pairs miss View 2's, or models that score different numbers of pairs (one detector finds every face); `Verification.scored_pairs` is that number, and Table 1's notes use it.
- **Report.** Section 5's setup chunk moved to the top of the section so 5.1 can use it. 5.1 has its gate paragraph above #32's stub: both figures for every model, computed from the results, why the scored pairs decide, and the results key that records it. 3.3's rule bullet, 5.2's first-active sentence and notebook 02's rule prose say "on the scored pairs".

## Interpretations

- **"Record in `results.json` provenance"** was read as "recorded beside the decision it governs". `Provenance` is the commit, dirty, time and machine stamp every section shares, the wrong type for it. `Verification` would have been natural too, but only `ryuk evaluate lfw` on the Mac writes that section; `FirstActiveModel` is derived by `assemble` from the committed sections, which is where the gate is applied.
- **The committed file was re-assembled, not edited.** Its own sections were loaded and passed through `assemble`, the step every evaluate command ends with (as the open-set session did for derived blocks). The diff is the new six-line block only, and `assemble(sections) == file` holds. The next Mac run of any evaluate command reproduces it.

## Verification

The lead ran the branch against a real `ryuk serve` on Linux. With the weights, SFace was active, ArcFace `not_evaluated` (CPU, a different model from the CoreML one evaluated) and FaceNet available: the fallback rule ran with the recorded gate. Without weights, every model was unavailable and the service started. No errors in either log.

## Review

One reviewer agent (spec and standards): nothing blocking. Applied: the 5.1 paragraph's last sentence (it named a measure Section 2.4 does not use), the consequence of the excluded-as-errors reading (no first active model), "every model would still pass" losing "still", the contenders-without-a-gate trap, the eligibility-against-tolerance check, the missing `scored_pairs` docstring, notebook 02's rule prose.

## Resuming

Shikhar reviews the PR (`Closes #52`) and merges it with a merge commit. Working rules as before: one ticket per session; nothing merged without Shikhar; role agents only, never Haiku; never Playwright locally; verify against `ryuk serve` with and without weights; write the handoff before merge.
