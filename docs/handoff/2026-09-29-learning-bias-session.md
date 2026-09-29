# Learning and bias session, 29 September 2026

The record of the session that implemented #28 (learning on embeddings and the bias breakdown) as PR #65, branch `ShikharJohari/28-learning-bias`. It ran in parallel with #31 (sightings, PR #64, session ryuk-1b) and #47 (audit decisions, merged as PR #63, session ryuk-aa). Read the open-set session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- **PR #65 is open, not merged.** It waits for Shikhar.
  - **Merge it with a merge commit, never squash or rebase.** `results.json` provenance names `dbc08c4` (learning) and `96d37c4` (bias), and both must stay reachable from main.
  - Main moved on after the branch was cut (#63, the #47 fix pack), but no file overlaps, so the merge is clean.
- **#64 (#31) and #65 both contain #31's ranking change** (`4f32150` there, cherry-picked here as `3ef57f4` with a conflict resolution). Whichever merges second resolves a small `src/ryuk/watchlist/live.py` conflict:
  - keep #31's `ranking`/`_scores` shape;
  - the `mean` case lives inside `_scores`;
  - `learned` is dispatched in `ranking` to `_learned`;
  - `recognise` passes `evaluated.learned_rule`.
- **Blocked on #65:** #49 (the 1:1 threshold and the small-gallery operating point, Q6) and #52 (the LFW gate provenance). Both touch `results.json` and its schema. #32 is blocked on #28 and #31.
- **#47's rulings that shaped this ticket:**
  - Q11: #28 writes 5.3, 5.4 and 6.1, and figures stay in place.
  - Q20: cap the Wilson N\* at N, disclosed.
  - Q6: small-gallery work goes to #49, not here.
  - Q19: model state stays out of `results.json`, and `schema_version` stays 1.
  - Q2: no per-provider results.
  - Q17: CI fails any commit that credits Claude. Every commit on this branch is Shikhar's alone, and `.github/scripts/check-attribution.sh origin/main..HEAD` passes.

## Headline results (CelebA test draw)

TPIR gain at FPIR 1% over best photo, read off each method's curve, with the paired identity-level 95% interval:

| | SFace | ArcFace (CoreML) | FaceNet |
|---|---|---|---|
| Mean rule | +1.36 [+0.61, +1.96] ★ | −0.19 [−0.39, +0.01] | +3.95 [+1.60, +5.80] ★ |
| kNN (retrains) | −3.93 | −4.35 | −8.04 |
| Logistic regression (retrains) | +1.27 [+0.67, +2.00] | +0.04 | +4.28 [+1.53, +7.09] |
| Linear SVM (retrains) | +0.48 | +0.01 | −0.76 |
| Learned rule (c) | +0.13 | −0.01 | +1.33 |

- **Live rules** (the `thresholds` block):
  - SFace uses mean at 0.494.
  - ArcFace uses best-photo at 0.342 and is still the first active model.
  - FaceNet uses mean at 0.691.
- **At the frozen thresholds, mean still wins:**
  - SFace goes from 95.03 to 96.19 TPIR, at FPIR 0.74% against 0.91%.
  - FaceNet goes from 85.13 to 88.36 TPIR, at FPIR 0.96% against 1.20%.
- **Chosen hyperparameters:**
  - kNN: k = 5 for SFace, and k = 15 (the edge of its grid) for ArcFace and FaceNet.
  - Logistic regression: C = 10 for every model.
  - Linear SVM: C = 1 for every model.
- **The learned rule measurably improves nothing.** 6.1 says so, and explains why: it was fitted on 500-identity galleries, while live watchlists are small.
- **Bias: FPIR is higher for identities labelled not male under every model and rule.**
  - ArcFace is 1.05% against 0.26%, a ratio of 4.1.
  - Under the mean rule, SFace's male group has no false alarm in 1,956 probes. Its ratio is undefined; the adjusted Wilson upper bound is 1.58%.
  - The Male×Young cell "not male, not young" has 29 gallery identities, so its TPIR is too few to estimate.

## What exists now

- **`ryuk.evaluation.learning`**:
  - `compare(validation, test, model, *, seed) -> Comparison`, holding `.result: LearningModel`, `.scores[draw][method]` and `.runner_up[draw]`.
  - `score_rule(rule, draw, learned)` scores a draw under a live rule exactly as `compare` did; it is used by bias.
  - Sealed tokens: `choose_hyperparameter` returns `ChosenHyperparameter`, and `fit_learned_rule` returns `FittedRule`. Both refuse the test draw.
  - `train_classifier` takes only a gallery's enrolled mapping.
  - A `ConvergenceWarning` becomes an error.
  - kNN never tries a k larger than the gallery's enrolled photos.
  - Per-fold learned-rule fits are built with `LearnedRule.model_construct`, which skips validation: they are never kept or run live.
- **`ryuk.evaluation.bias`**:
  - `GroupLabels.from_table` / `read_group_labels`: majority over all of an identity's images, using the EDA's `RULES.majority_agreement`.
  - `LabelledProbes` and `bias_model(...)`: the test draw only, with a plain-float threshold.
  - `MIN_IDENTITIES = 30`.
  - Each group is seeded by `(seed, attribute, group, side)`, so its intervals don't depend on other groups.
- **`ryuk.evaluation.openset`**:
  - `Gallery.averaged()` (the mean rule) and `Gallery.top_two()` (returning `TopTwo`).
  - `paired_gain(baseline, method, target, *, seed) -> Gain`.
  - `EmbeddedDraw` (with `.score()`), `Probes.images`, and a public `probe_rate`.
- **`ryuk.evaluation.results`**:
  - `MatchRule = best-photo | mean | learned`.
  - `LearnedRule`, with `.probability(top, gap)`, the one formula evaluation and the live monitor share. Its validator refuses `top_score + 2·gap < 0`, which would rank the runner-up above the top candidate.
  - `Learning` and `LearningModel`, with `.method`, `.find` and `.bias_rules`. `live_rule` is re-derived by `winning_rule(methods)` in the validator.
  - `MethodResult`, `Gain`, `GapHistogram`, `TopGapSample`, `Bias`, `BiasModel`, `AttributeBreakdown` and `GroupRates`.
  - `learning_mismatch` and `bias_mismatch`.
  - `ModelThreshold.learned_rule`, present if and only if the rule is `learned`.
- **`ryuk.evaluation.active`**:
  - `assemble(verification, identification, learning=None, bias=None)` derives the thresholds and the first active model under each live rule.
  - `carried(...)` keeps learning and bias across LFW or CelebA reruns only while they still apply.
- **`ryuk.evaluation.celeba`**:
  - `CelebaEvaluation.learn(...)` / `.bias(...)`.
  - `embed` and `embed_draw` are public, as the #27 handoff asked.
- **`ryuk.evaluation.scores`**: per-probe Parquet under `data/cache/scores/<model id>/<selection_sha256>.parquet` (T11).
- **CLI:**
  - `ryuk evaluate learn` needs `evaluate celeba`, and drops a previous bias section with a warning.
  - `ryuk evaluate bias` needs `evaluate learn`.
  - The order is lfw → celeba → learn → bias. The README says so.
- **Service (T8):**
  - `registry.Evaluated.learned_rule`.
  - `LIVE_RULES` is gone: the service computes every `MatchRule`.
  - `WatchlistEmbeddings.ranking(probe, rule, learned=None)`. Under `learned`, the top candidate and the runner-up both get the rule's match score from their cosine and their margin over each other.
  - With one person on the watchlist the margin is 0.
- **Outputs:**
  - `tables`: `learning_markdown`, `bias_table(..., side="gallery" | "held-out")` (Table 4 is split in two to fit A4) and `fpir_ratio_markdown`.
  - `figures`: `learning_gains`, `learned_rules`, `gap_distributions` and `group_fpir`.
  - Notebook 03 (`notebooks/03-learning-and-bias.ipynb`).
  - Report 5.3, 5.4, 6.1, Section 3.3.1–3.3.2 and Section 8.2–8.5.
- **Tests:**
  - `tests/results_files.py` holds the shared synthetic builders: `synthetic_verification/identification/results`, which are cached, plus `synthetic_learning` and `synthetic_bias`.
  - `test_assemble`, `test_learning`, `test_bias`, `test_scores`, `test_live_rules` and `test_learning_outputs` are new.

## Loose ends (do these before or at merge)

1. **Re-execute notebook 03.** The review fixes changed Figures 4 and 5 (the "wrong top candidate" class is now INK with an × marker, not the No match colour), and the committed outputs predate that. Run `cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 03-learning-and-bias.ipynb`, then commit. Notebook 02 is unaffected.
2. **Test and render the final commit.** Per Shikhar's instruction, the final state was not re-run: the last full suite was 849 passed at 96.04% on the code-fix commit, and the report-side commit touched only qmd, README and TO-BE-REVIEWED. CI runs everything, including the report render.
3. **Replace the typed "99%" in the Figure 12 (gap distributions) caption.** It mirrors a literal `0.99` in `figures.gap_distributions`; expose it as a constant (e.g. `GAP_AXIS_COVERAGE`) and import it in the caption.
4. **`live_reason` wording.** The strings stored in `results.json` say "improves … on best-photo" and use a hyphen-minus for negative numbers. Changing them means re-running `ryuk evaluate learn` and `ryuk evaluate bias`, since they are stored.
5. **`TO-BE-REVIEWED.md` gained two entries for Shikhar**, both easily reversible:
   - the Section 3 and Section 8 additions stay in #28;
   - an undefined FPIR ratio stays "—", with its Wilson bound given, rather than ∞.
6. **The Typst template now justifies only table body cells** (`show table.cell.where(y: 0): set par(justify: false)`). The broader rule broke Table 3 and Table 4 cells.
7. **The progress report now includes Sections 3, 6 and 8**, following its own rule that a section is included once it is being written. Revert that if only Section 5 is wanted.

## Things that bit

- **#44's cache key orphaned the old embeddings.** The pipeline id now includes `v<PIPELINE_VERSION>-<runtime>`, so every CelebA image was re-embedded (16.5 min on the Mac). The old, unsuffixed Parquet files in `data/cache/embeddings/*/` are dead: a cleanup candidate, being a runtime resource.
- **Symlinks make a worktree dirty.** `data` and `models/weights`, linked into a worktree, show as untracked, and provenance then says `dirty: true`. `/data` and `/models/weights` are now in the repo's shared `.git/info/exclude`.
- **Heavy runs from a clean commit.** Provenance has to come from a clean commit, but implementers were editing the main worktree at the time. So the runs went into a detached worktree at the branch head (`../ryuk-28-run`), and `results.json` was copied back and committed after each run. Bias ran from the commit holding learn's results, so each provenance commit is on the branch.
- **A two-class softmax is not distance-aware.** On the tiny CelebA rehearsal fixture (14 non-mated probes), logistic regression can't freeze FPIR 1%, because a face far from both classes scores higher than a genuine one. The orchestration tests therefore leave out the classifiers, and the classifiers keep their own tests. On the real draws every classifier froze.
- **`git stash` in a shared worktree stashes everyone's work.** A fixer did it for two seconds while another agent had uncommitted edits. Nothing was lost, but don't do it.
- **`dataclasses.replace` and sealed tokens** are the same trap as in #27. `ChosenHyperparameter` and `FittedRule` are plain `__slots__` classes.

## Worktrees and runtime state on this machine

- `../ryuk-28` is the working worktree for #65.
- `../ryuk-28-run` is a clean, detached worktree used only for the runs. Remove it with `git worktree remove ../ryuk-28-run` when #65 is merged.
- `data/cache/embeddings` holds the new runtime-keyed embeddings for both CelebA draws and all three models, next to the dead unsuffixed files.
- `data/cache/scores` holds the per-probe Parquet for both draws and all three models.

## Resuming

- #65 needs Shikhar's review and a merge commit, then loose end 1 (or re-execute it before the merge).
- After #65 and #64 merge, #32 (report assembly and charts) is unblocked, and so are #49 and #52.
- Working rules as before:
  - one ticket per session;
  - `Closes #N`;
  - nothing merged without Shikhar;
  - role agents only, never Haiku;
  - no Co-Authored-By lines (CI enforces it now);
  - never Playwright locally;
  - heavy runs from a clean commit in a separate worktree.
