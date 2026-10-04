# Open-set session, 25 September 2026

The record of the ninth working session, the fourth build session. It implemented #27 (CelebA open set and frozen thresholds) and merged it as PR #38 (merge commit `509afb0`). Read the EDA and recognition session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #23 to #27 are closed.
- **#28 and #29 are now unblocked.** They are independent and can run as two parallel sessions:
  - #28 (learning on embeddings and bias breakdown) is evaluation work.
  - #29 (watchlist management) is app work.
  - They share one seam, the thresholds block of `evaluation/results.json`; see "Shared seam" below.
- Still blocked: #30 on #29, #31 on #30, #32 on #28 and #31.
- On this machine:
  - `data/cache/embeddings` holds every model's embeddings for both CelebA draws (about 29,000 images per model).
  - A re-run of `ryuk evaluate celeba` therefore only rescans (about 7 s per split) and rescores. The first run took 12.5 min.

## Headline numbers

The CelebA test draw, at each model's threshold frozen at FPIR 1% on the validation draw:

| | SFace | ArcFace (CoreML) ★ | FaceNet |
|---|---:|---:|---:|
| Rank-1 | 98.0 [97.6–98.5] | 98.6 [98.3–99.0] | 96.6 [95.9–97.2] |
| Frozen threshold | 0.498 | 0.342 | 0.709 |
| TPIR | 95.0 [94.2–95.8] | 98.5 [98.1–98.8] | 85.1 [83.5–86.7] |
| FPIR | 0.91 [0.62–1.24] | 0.66 [0.40–0.94] | 1.20 [0.74–1.73] |
| Misidentification | 0.16 [0.05–0.31] | 0.13 [0.04–0.25] | 0.39 [0.24–0.55] |
| TPIR @ FPIR 0.1% (indicative) | 90.0 | 98.4 | 72.9 |
| ms per face | 5.4 | 8.4 | 11.7 |

- **First active model: ArcFace (CoreML).** All three models are eligible under #9's rule, and no other model's TPIR interval overlaps ArcFace's.
- **Draws** (seed 27):
  - validation: 19,070 of 19,867 images usable; 500 gallery and 485 held-out identities; 7,500 mated and 3,929 non-mated probes;
  - test: 19,228 of 19,962 images usable; 500 gallery and 500 held-out identities; 7,500 mated and 4,072 non-mated probes.
  - #10 expected about 4,700 non-mated probes. Many held-out identities have fewer than 10 usable images.
- **FaceNet has a large gap between rank-1 and TPIR** (11.4 points): its right identity often comes first but scores under the threshold. That is where #28's alternative rules have the most room to show a gain.
- **The adjusted Wilson check** agrees with the bootstrap on every rate in the 1% tails.

## What exists now

- **`ryuk.evaluation.draws`**:
  - `make_draw(draw, usable, seed, gallery_size=500) -> OpenSetDraw`.
  - `OpenSetDraw.images()`, and `.selection_sha256`, which is committed per draw in `results.json` so a rebuilt draw can be checked against it.
  - Constants: `ENROLLED_PER_IDENTITY`, `MATED_PROBES_PER_IDENTITY`, `NON_MATED_PROBES_PER_IDENTITY`, `MIN_GALLERY_IMAGES`, `DRAW_SEED = 27`.
  - Held-out means every non-gallery identity with at least one usable image, which is #10's correction.
- **`ryuk.evaluation.openset`**:
  - `Gallery.enrol({identity: [embeddings]})`. `Gallery.top_candidates(probes)` implements the best-photo rule with `np.maximum.reduceat` and returns the top identity and score only, with no runner-up yet.
  - `Probes`, `ScoredProbes` (draw, mated identity/score/correct, non-mated identity/score), `score_probes`, `rank_1`, `open_set_curve`, `tpir_at_fpir`.
  - `freeze(validation_scores, target_fpir=0.01) -> FrozenThreshold` refuses any draw but validation.
    - `FrozenThreshold` is a plain slotted class with read-only properties. It is deliberately not a dataclass: `dataclasses.replace` copied the seal and forged thresholds.
  - `draw_result(scored, frozen, seed=...) -> DrawResult`: every rate with its interval, the operating points (re-thresholded per resample), and the downsampled curve.
- **`ryuk.evaluation.bootstrap`**:
  - `identity_weights(groups, resamples, rng)` gives multinomial counts, resamples × identities.
  - `ratio_interval`, `percentile_interval`.
  - `adjusted_wilson(errors, trials)`: Fogliato's N* = max(p(1−p)/Var, G/2) with the cluster-robust variance.
  - `disagree(a, b)`: an end moves by more than 25% of the wider width.
  - Constants: `RESAMPLES = 2000`, `BOOTSTRAP_SEED = 2000`.
  - Gallery and held-out identities are resampled independently, each with its own weight matrix from one seeded rng (`openset._Groups`).
  - **Paired comparison for #28:** two methods scored on the same draw with the same seed get identical weight matrices, so a per-resample TPIR difference is a paired bootstrap for free.
- **`ryuk.evaluation.celeba.CelebaEvaluation`**:
  - `prepare(draw)`: scan the split, exclude unusable images, make the draw. This is the one place a draw label is tied to its CelebA split.
  - `run(models, crops, provenance)`: per model, validation is scored, the threshold frozen, then the test draw scored once.
  - `_embed` and `_score` are private. #28 needs them; make them public rather than copying them.
- **`ryuk.evaluation.active`**:
  - `first_active_model(contenders)` and `failures(eligibility)`.
  - `assemble(verification, identification) -> Results` derives `thresholds` and `first_active_model`. Always write results through it.
- **`ryuk.evaluation.names.model_name`** is a leaf module, so `tables` can import `active` without a cycle.
- **`ryuk.evaluation.results`** gains:
  - `Interval`, `Rate` (value, ci, adjusted_wilson);
  - `OpenSetPoint`, `AtThreshold`, `OpenSetCurve`;
  - `DrawResult` (with `.indicative_point`) and `OpenSetModel` (with `.on(draw)`);
  - `DrawSelection` and `Identification` (with `.selection(draw)`);
  - `MatchRule = Literal["best-photo"]`, `ModelThreshold`, `Eligibility`, `FirstActiveModel` (`.eligibility`, not `.candidates`: a candidate is a person of interest in `GLOSSARY.md`).
  - Validators:
    - thresholds list every identification model in order;
    - the first active model exists iff identification does;
    - identification's models and crops must match verification's (`identification_matches`).
- **CLI:**
  - `ryuk evaluate celeba` needs `evaluate lfw` first, for the crops and LFW eligibility.
  - `ryuk evaluate lfw` now keeps identification, and drops it with a warning if the models or crops changed.
- **Outputs:** Table 2 (`tables.openset_table`, `openset_notes`, `openset_markdown`, `wilson_notes`) and the figure (`figures.openset_curves`).
  - Table 2 is transposed: models are columns and measures are rows.
  - The figure labels each curve at FPIR 0.1% and marks each frozen threshold.
- **Notebook 02 is complete.** Report Section 5.2 is written, and the progress report includes Section 5 (5.1, 5.3 and 5.4 are still stubs).

## Shared seam between #28 and #29

- #29's model registry reads `results.thresholds` to decide which models are evaluated, and `results.first_active_model` for the first active model.
- #28 may change a model's live rule. #10 says the rule is judged live, and the threshold becomes a cut-off on that rule's output.
- Agree on this before either session writes code:
  - `MatchRule` gains the winning rule's name, for example `"mean"` or `"learned"`.
  - `ModelThreshold.rule` and `.threshold` then describe that rule.
  - `assemble` judges eligibility and the first active model on the live rule's test TPIR and FPIR.
- **#29 must read `rule` and refuse, or treat as not evaluated, any rule it cannot compute.** It must not assume best-photo.
- **Provider mismatch** (found while surveying #29):
  - The thresholds cover `sface/cpu`, `arcface/coreml` and `facenet/cpu`.
  - On Linux, including CI, ArcFace runs on CPU, which is a different recognition model with no threshold, so it is "not evaluated" there.
  - The e2e job runs `ryuk serve` with no weights at all, so every model is unavailable.
  - Both are correct under #12. #29's registry and tests have to expect them.

## Patterns later tickets should copy

- **Everything in the EDA session's list still holds.** In particular: committed JSON through frozen pydantic with generated schemas, every number in report prose an inline `{python}` expression, `MODEL_STYLES`, and direct labels.
- **Fitting only on validation, enforced in code:**
  - a function that fits accepts only validation data and raises on anything else;
  - the token it returns (like `FrozenThreshold`) cannot be built or copied another way;
  - tests assert both.
  - #28's acceptance criterion "no test-draw data reaches any fitting step (asserted in tests)" wants exactly this, for hyperparameters (k, C), the learned rule and its cut-off.
- **Derived blocks are derived, never stored independently.** The thresholds and the first active model come out of `assemble`. When a key is renamed, re-derive the file through `assemble` and check that nothing but the key changed; that is how `candidates` became `eligibility` without a re-run.
- **Report rule numbers:** import constants (`draws.*`, `active.MAX_*`, `verification.TOLERANCE_POINTS`) or read them from the JSON. Never type them into prose. Look points up by meaning (`indicative_point`, `selection("test")`), never by list position.
- **Wide tables in the Typst PDF:**
  - Pandoc sizes a wide pipe table's columns from the dash counts in the separator row.
  - With seven interval columns nothing fits A4, so transpose.
  - Render and look at the page image (`pdftoppm -f N -l N -r 70 -png`) before calling a table done.
- **Heavy runs go to a subagent** with a self-contained brief. Don't edit tracked files until the run has started: provenance is stamped at start, and the subagent will flag edits it didn't make.

## Things that bit

- **`dataclasses.replace` defeats a sealed dataclass.** It copies the private seal field. Use a plain class with `__slots__` and read-only properties.
- **Ruff lints notebooks.** `assert` (S101) and literal en dashes (RUF001) fail `ruff check .`. Use `raise` and `"\N{EN DASH}"`.
- **Executing a notebook:** `cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 02-evaluation.ipynb`. New cells need an `id` (nbformat 4.5). Execution timestamps in the diff are normal.
- **mypy on a single file** reports `import-untyped` for the package. Run `uv run mypy` with no arguments; the config covers `src` and `tests`.
- **BSD `sed` has no `\b`.** Use Python for word-boundary edits.
- **Figure labels:** each open-set curve is labelled at FPIR 0.1%, where the curves are furthest apart. A label at the frozen-threshold marker collided with the next model's marker.
- **Table 2's first draft** had seven columns with intervals. It wrapped every cell, and word joiners made cells overflow into each other.

## Decisions and open items

- **Interpretations made in #27 and accepted at merge:**
  - #9's "highest threshold with FPIR ≤ 1%" means the lowest threshold within the budget.
  - ms per face is timed end to end: detection, crop, embedding and the gallery search, not decoding.
  - Table 2 is transposed.
- **Declined in review** (easily reversible):
  - a bootstrap band on the TPIR-vs-FPIR curve;
  - `caption()` for notebook 02's Figure 2;
  - reusing Section 2's format helpers in Section 5;
  - binding `ScoredProbes` to the draw object.
- **Deferred to #28:**
  - #10's per-probe raw-score parquet (gitignored, under `data/cache`);
  - the runner-up score, which rule (c) and the gap analysis need: `Gallery.top_candidates` must return the top two.
- **Report sections:** the section-5 stubs say "To be written in #32", but #27 wrote 5.2 because the previous handoff said to. Recommend that #28 write 5.3 and 5.4 the same way; Shikhar has not ruled on it.
- **Bias groups:**
  - The EDA takes an identity's majority label over all its images, usable or not (`ryuk.eda.aggregate.majority_label`, 80% agreement). #28 should use the same basis so its group counts match Section 2.
  - Photo conditions group probes by their own image label.
  - #10 says to report groups under 30 identities as too few to estimate, never to drop them.
  - The detector skew in Section 2 (majority-male and not-young identities are detected less often) is a caveat the bias section must state.
- **The follow-up tickets from the EDA session are still open:**
  - `write_into_place` lives in `ryuk.fetch`;
  - there are two provenance modules;
  - `GroupStats` has no per-group gallery candidates;
  - report figures are not floated.
- **`TO-BE-REVIEWED.md`** gained nothing this session. No two subagents disagreed; declined review items are in PR #38.

## Resuming

Start one fresh session per ticket with `/implement`, running in parallel if you want both:
- #28 on `ShikharJohari/28-learning-bias`;
- #29 on `ShikharJohari/29-watchlist`.

Working rules as before:
- one ticket per session;
- a PR that says `Closes #N`;
- nothing merged without Shikhar;
- role agents only, never Haiku;
- never Playwright locally;
- heavy runs from a long session go to a subagent.

**#28 builds on:**
- `CelebaEvaluation.prepare`, with `_embed` and `_score` made public;
- the cached embeddings;
- `Gallery`, which gains top-two and a mean rule;
- `freeze` and `draw_result`;
- `bootstrap`, where the same seed gives a paired comparison;
- `assemble`, which must learn the live rule;
- `read_labels(root, split, attributes=[...])` for the bias attributes;
- `majority_label`.

It needs scikit-learn as a new dependency. Pin the previous patch release. It adds notebook 03 and fills report Sections 5.3 and 5.4.

**#29 builds on:**
- `create_app()`, which has no lifespan or dependency injection yet;
- `ApiModel` (every response field explicit);
- `ProblemError`;
- the contract pipeline;
- `Detector` and `usable_faces`;
- `load_model` and `weights.*`;
- `ModelKey.id`;
- the thresholds block.

On the client:
- `ApiClient` has only `get`;
- the watchlist routes are placeholders;
- MSW route tests;
- the palette tokens in `web/src/styles.css`.

New dependencies: SQLAlchemy, Alembic, python-multipart and Pillow (currently only transitive). Also a settings field for the database path, which must not be `cache_dir`. Startup must survive missing weights, as the e2e job has none.
