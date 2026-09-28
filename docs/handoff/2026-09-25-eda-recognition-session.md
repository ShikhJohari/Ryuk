# EDA and recognition sessions, 25 September 2026

The record of the eighth working session, the third build session. It ran as two parallel sessions:
- one implemented #25 (EDA, figures and report scaffold) and merged it as PR #36;
- one implemented #26 (Recognition models and LFW verification) and merged it as PR #35, rebased onto #25.

They agreed shared seams by message as they went. Read the data and detection session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #23 to #26 are closed; main holds all four (merge commits `e421101` for #25, `da551b6` for #26).
- #27 (CelebA open set and frozen thresholds) is unblocked and is the only ticket that is. Everything after it chains: #28 and #29 wait on #27, #30 on #29, #31 on #30, #32 on #28 and #31.
- On this machine:
  - `data/raw` and `models/weights` are populated and verified.
  - `data/cache` holds #26's benchmark embeddings at the 70 px pipeline.
  - `uv run ryuk eda` takes about 35 s.
  - `uv run ryuk evaluate lfw` takes a few minutes and re-embeds only if the pipeline changes.

## Headline numbers

- **Minimum usable face size: 70 px** (#17's rule: the largest multiple of 10 px keeping at least 99% of CelebA detections). It keeps 99.50% of 38,489; 80 px would keep 84.7%. `MIN_USABLE_FACE_SIZE` is 70 in `ryuk.detector`, and a test holds it equal to `eda/summary.json`. It governs enrollment and the live monitor too.
- **LFW:**
  - 13,233 images: 13,161 detected, 13,144 usable.
  - View 2 drops 83 of 6,000 pairs (58 images with no usable face).
  - Both sessions counted 83 independently, so the exclusion rule agrees across `ryuk.eda` and `ryuk.evaluation`.

| LFW View 2 | Ours (± SE) | Every dropped pair as an error | Published |
|---|---:|---:|---:|
| SFace fp32 | 99.38 ± 0.12 | 98.00 | 99.40 |
| ArcFace w600k_r50 (CoreML) | 99.78 ± 0.08 | 98.40 | 99.83 |
| FaceNet VGGFace2 (box margin 32, chosen on View 1) | 99.36 ± 0.11 | 97.98 | 99.65 |

  All three models are within 0.5 points of their published figures, which is #9's first eligibility test. SFace int8 is the footnote: 99.12 at 11.4 ms per face against fp32's 4.1 ms.

- **CelebA:**

| | Validation draw | Test draw |
|---|---:|---:|
| Images / identities | 19,867 / 985 | 19,962 / 1,000 |
| Detected / usable | 96.52% / 95.99% | 96.75% / 96.32% |
| Gallery candidates (≥ 20 images) | 629 | 616 |
| Eligible after exclusion (≥ 20 usable) | 584 | 572 |

- **Detector skew before any recognition model runs:**
  - Majority-male identities' faces are detected less often: 95.4% against 97.5% on the validation draw.
  - Not-young identities: 94.8% against 97.1%.
  - Photo conditions: Blurry 87.8%, Wearing_Hat 89.6%, Eyeglasses 90.5%.
  - The report states this as a caveat for #28's bias breakdown.
- **Eligible Male identities (with / without):**
  - validation 233 / 351, test 200 / 372.
  - #10's 259 / 370 was counted before exclusion.
- **Eligible Young identities (with / without):**
  - validation 425 / 99, test 438 / 91.
  - 60 validation and 43 test identities are mixed under the 80% majority rule, so they get no Young group.

## What exists now

- **`ryuk.datasets`** is the shared reader for both benchmarks. #26 deleted its own parser and uses it; #27 should too.
  - `lfw`:
    - `list_identities(root)`: identities by folder, ignoring the `pairs_*.txt` files.
    - `LfwImage(identity, number).path`.
    - `read_pairs(root, name) -> PairsFile(name, folds, pairs)`: strict about headers and counts. `Pair(first, second, fold).matched`.
  - `celeba`:
    - `read_labels(root, split=None, attributes=None)`.
    - `iter_images(root, labels)`: reads one row group at a time, only the groups its labels need, and checks each path. Pass a subset of label rows to read only those images.
    - `CelebaImage.decode()`.
- **`ryuk.eda`**:
  - `summary.EdaSummary`: the committed summary's frozen pydantic model. It has lookup methods: `summary.draw("validation")`, `draw.group("Male", True)`, `draw.prevalence("Blurry")`, `images_per_identity.at_least(20)`.
  - `DRAW_SPLITS` maps draw to CelebA split (`validation` → `valid`).
  - `aggregate`: pure functions tested with hand-checked answers.
  - `pose`: a rough head pose, a generic 3D face fitted to the five landmarks with SQPnP.
  - `scan.Scanner`: one YuNet per thread, `cv2.setNumThreads(1)` during the scan, about 500 images/s per worker.
  - `build`: `build_summary`, `provenance`.
  - `files`: summary, schema and figures, all through `write_into_place`.
- **`ryuk.recognition`**:
  - The interface is an aligned BGR face in, and an L2-normalised float32 embedding out.
  - `ModelKey` is (network, sha256 of the loaded weights, provider).
  - Networks are exactly `"sface"`, `"arcface"`, `"facenet"`:
    - SFace runs on OpenCV.
    - ArcFace runs on onnxruntime, on CoreML MLProgram on Apple Silicon and on CPU elsewhere; each is a distinct key.
    - FaceNet runs on facenet-pytorch 2.5.3, with torch from the CPU index on Linux.
  - Each call embeds one face, so an embedding never depends on its batch neighbours. On CoreML they would.
  - `fake.py` is a test model whose key always says `fake`, so it can never reach `results.json`.
- **`ryuk.evaluation`**:
  - `verification`: the 10-fold OpenCV zoo / InsightFace recipe, with SE as ddof=1 over √10.
  - `metrics`: AUC, and TAR at FAR 1e-2 / 1e-3 pooled over scored pairs.
  - `embeddings`: the benchmark cache under `RYUK_CACHE_DIR` (default `data/cache`), keyed by image bytes, model key and pipeline (detector hash, minimum face size, crop). It is never the app database.
  - `results`: the `results.json` pydantic model.
  - `tables`, `figures` (the ROC), `provenance`.
- **`ryuk.plotting`**:
  - #13's palette.
  - `MODEL_STYLES[network]` sets colour and dash: ArcFace solid, FaceNet long dash, SFace dotted.
  - `DATASET_STYLES` for LFW and the two draws.
  - `use_lab_style()`, `direct_label`, opt-in `caption(fig, n, text)`, and `save_figure`, which writes deterministic SVG.
  - Fonts are vendored in `src/ryuk/plotting/fonts` (Newsreader 16pt, Public Sans, OFL). The report's Typst build uses the same folder.
  - `EDA_FIGURES` holds the eight EDA figures by slug.
- **CLI:**
  - `ryuk eda [--workers N] [--figures-only]`
  - `ryuk evaluate lfw`
  - `ryuk evaluate schema`
  - `ryuk data fetch --dataset lfw`: LFW only.
- **Committed outputs:**
  - `eda/summary.json`, `eda/summary.schema.json`, `eda/figures/*.svg`.
  - `evaluation/results.json`, `evaluation/results.schema.json`.
  - Both summaries record the commit they came from and a dirty flag. Generate them from a clean, committed tree.
- **Notebooks:**
  - `notebooks/01-eda.ipynb` and `02-evaluation.ipynb` read committed JSON only and are committed with outputs.
  - CI runs them with `uv run pytest --nbmake notebooks --no-cov`.
  - Notebook 02 is begun; #27 completes it.
- **Report:** `report/` is Quarto rendering to PDF through Typst. `quarto-cli` is a dev dependency, so nothing is installed system-wide.
  - `uv run quarto render report` builds `report/_output/ryuk-report.pdf` and `ryuk-progress-report.pdf` in about 5 s.
  - Sections live in `report/sections/0N-*.qmd`. Section 2 (Data and EDA) is written; the others are `::: stub` blocks naming the ticket that fills them.
  - `progress.qmd` includes only the written sections, with a hand-kept status table.
- **CI jobs:**
  - `python`: ruff, mypy, pytest and nbmake.
  - `report`: renders both PDFs and uploads them as an artifact.
  - `client`, `contract` and `e2e` as before.
  - `weights-smoke` (last): the real models on Linux CPU, with weights and LFW fetched and cached by the hash of `weights.py`.

## Patterns later tickets should copy

- **Committed JSON:**
  - a frozen pydantic model with `extra="forbid"` and `schema_version: Literal[1]`;
  - a schema generated from it and committed beside it;
  - a staleness test, and a test that validates the committed file against the committed schema with `jsonschema`, then parses it with the model;
  - written as indent 2 plus a trailing newline.
  - #27 extends `evaluation/results.json` this way and bumps nothing unless the shape breaks.
- **Numbers in the report:**
  - Every number in prose is an inline `{python}` expression over committed JSON, and every figure comes from `ryuk.plotting` in a cell with `#| label: fig-…` and `#| fig-cap:`.
  - Report prose names rules and sections, not issue numbers. Stubs may keep "To be written in #N".
- **Figures:**
  - Style by network through `MODEL_STYLES`.
  - Label lines directly, with no legend.
  - Never bake in "Figure N": the report captions them, and notebooks use `caption()`.
  - Palette Match / No match colours mean live outcomes only; don't use them for ground-truth pairs.
- **Evaluation runs:**
  - Keep derived data in `data/cache`.
  - Key cached embeddings by the whole pipeline.
  - Drop and report pairs or probes with no usable face, with a sensitivity bound, rather than hiding them.
- **Heavy re-runs from a long session:** hand them to a subagent with a self-contained brief (Shikhar's rule from this session).

## Things that bit

- **tnum misplaces figure text.** matplotlib 3.11 measures text without OpenType features but draws it with them, so `tnum` shifts labels off their anchors. Figures use proportional digits; the report's Typst body keeps tabular numerals.
- **Quarto:**
  - A `.qmd` rendered from inside `report/` that isn't in the project's render list skips `_quarto.yml` and falls back to HTML.
  - Inline `{python}` runs in document order, so it must come after the cell that loads the summary.
  - Quarto binds the figure formatter at kernel start, so `%config InlineBackend` does nothing. Use `set_matplotlib_formats("svg", bbox_inches=None)` to keep figures at the 6.3 in column width.
  - Typst prints "#9's" and "View 1's" with a prime; `report/style/digit-apostrophe.lua` fixes it.
- **Head pose.** The ArcFace template read every dataset as pitched about −17°: YuNet puts the mouth 0.99 interpupillary distances below the eyes, against the template's 1.16. The model now uses adult anthropometric means that match YuNet's median layout, so zero pitch means the typical portrait, not a measured level head. `SOLVEPNP_ITERATIVE` needs 6 points; SQPnP works with 5.
- **Git status and untracked files.** The `git_dirty` flag counts only tracked changes (`--untracked-files=no`). Otherwise the uncommitted output folder itself marked every first run dirty.
- **FaceNet's crop.** With YuNet's five-point alignment, FaceNet read 99.12, flagged against its published figure. The box with margin 32, which is what the published figure used, reaches 99.36, and was chosen on View 1 only.
  - **Correction, 28 September 2026 (#46):** the 99.12 was the box with margin 14, View 1's choice at the time, not the five-point alignment, which scored 98.17 on View 1 and never ran on View 2. Margin 32 was added as a View 1 candidate after that View 2 run flagged FaceNet, so FaceNet's View 2 was scored twice (PR #35).
- **Tooling:**
  - InsightFace's per-fold `calculate_val` crashes on current SciPy, so TAR@FAR is pooled over all scored pairs.
  - OpenCV prints two `setPreferableTarget ... not supported` lines per `Detector` created. It is noise, and about 66 lines appear during `ryuk eda`.

## Decisions and open items

- **For #27, settle first:** "every other eligible identity held out" vs #10's "about 485–500 held-out identities per draw". Only 84 validation and 72 test identities have at least 20 usable images beyond the 500 gallery identities. #10's count needs held-out identities to be any identity with a usable image, capped at 10 probes. Read #9 and #10 and confirm with Shikhar if it is still ambiguous.
- **Declined in review** (easily reversible):
  - no eligible-identity count for photo-condition groups, which group images, not identities (#10);
  - no tabular numerals in figures;
  - the CI `report` job stays although a reviewer called it scope creep;
  - #26 has no per-network descriptor and no detector hash on `Detector`.
  - Listed in PRs #35 and #36.
- **`TO-BE-REVIEWED.md`** gained one entry: the fake recognition model no longer poses as a real network (#26).
- **Follow-ups worth a small ticket:**
  - `write_into_place` lives in `ryuk.fetch` and raises `FetchError`, so a failed figure or summary write reports as a fetch error. It should move to a neutral module with a general error.
  - `ryuk.eda.build` and `ryuk.evaluation.provenance` each have provenance and a `ProvenanceError`; the CLI imports both under different names. Merge them.
  - `GroupStats` has no per-group gallery candidates before exclusion, so the report can't say how many Male identities exclusion removed, only how many remain.
  - Report figures are not floated, which leaves white space before tall figures. Shikhar hasn't chosen yet.
- **Notebook 01's builder script** was a scratch file and is gone. Edit the notebook directly in Jupyter, then re-execute it.
- **The progress report's status table** is edited by hand. Update it when a section is written.
- `docs/handoff/2026-09-25-data-detection-session.md` still calls the minimum face size a provisional 40. It is a historical record; this file supersedes it.

## Resuming

Start a fresh session with `/implement` on #27. Working rules as before:
- one ticket per session, on a `ShikharJohari/<n>-<slug>` branch;
- a PR that says `Closes #N`;
- nothing merged without Shikhar;
- role agents only, never Haiku;
- never Playwright locally;
- heavy runs from a long session go to a subagent.

#27 builds on:
- `ryuk.datasets.celeba`, reading only the drawn images;
- `ryuk.detector.benchmark_face` at 70 px;
- `ryuk.recognition` and the `data/cache` embedding cache;
- `evaluation/results.json`, extended with the thresholds block, the open-set tables and the first active model;
- `MODEL_STYLES` for the TPIR-vs-FPIR figure.

It also completes notebook 02 and fills report Section 5.2.
