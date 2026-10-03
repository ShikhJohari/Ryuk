# Same-person threshold session, 3 October 2026

The record of the session that implemented #49 (#47 Q6): a 1:1 threshold for the `may_not_be_same_person` warning, and the small-gallery / one-photo operating point. It ran on the Mac, the machine of record (#47 Q2), while #51 and #52 ran on the Linux box. Read the learning and bias session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- Branch `ShikharJohari/49-same-person-threshold`, from `main` at `d7d160d`, in the T3 checkout `~/Ryuk`. **Pushed as PR #69 (`Closes #49`), not merged**: nothing merges without Shikhar's go-ahead (#47 Q16).
- **Shikhar's ruling (3 Oct):** the project is presented on the Mac, where ArcFace (CoreML) is the active model, and its one-photo drop (TPIR from 98.5% to 94.7%) is acceptable. Nothing more is done about one-photo recognition for now. Measuring 2 to 4 photos, an enrollment nudge, and a threshold frozen at a live-sized gallery were offered and not taken up.
- **Merge with a merge commit, never squash or rebase.** `results.json` provenance names `e983186`, which must stay reachable from main.
- Two acceptance items are open, both Shikhar's:
  - **The agreed figure: settled.** Shikhar agreed it on 3 Oct: under **10%** of a person's own photos warned with one photo enrolled, for every model that can be active. ArcFace (3.8%) and SFace (7.6%) pass. FaceNet (10.4%) narrowly misses, which is recorded, not tuned away.
  - **The #12 amendment.** Not posted yet; the draft is below, to post when the branch merges.
- Checks on the branch head:
  - `ruff check`, `ruff format --check` and `mypy` are clean.
  - `pytest`: 1,075 passed, 17 deselected, 96.39% coverage.
  - `openapi.json` regenerates unchanged.
  - Every commit passes the attribution check.
  - Quarto is not installed on this Mac, so the report was not rendered. Every Python chunk and inline expression of `report.qmd` and `progress.qmd` was executed in one namespace, as Quarto's kernel runs them, and all pass. CI renders it.
- **Merge order with #52.** #52 also touches `results.json` and its schema. Whichever lands second regenerates `evaluation/results.schema.json` (`uv run ryuk evaluate schema`) and checks that the committed results still validate.

## Live check

`ryuk serve` was run on the Mac with the real weights (ArcFace (CoreML) active) and a scratch database.

- A person was enrolled from one CelebA test image of identity 4930.
- Three more images of 4930 were added with 201 and no warning.
- An image of identity 4931 answered 409 `may_not_be_same_person`: "its best score against their photos is 0.022, under the same-person threshold 0.216".
- Without weights, the service started with every model unavailable, and adding a photo answered 503 (no detector).
- No errors in either log. The scratch database was deleted.

## Results (CelebA test draw)

Same-person thresholds, frozen at FAR 0.1% on the validation draw's 5,707,000 one-photo impostor pairs:

| | SFace | ArcFace (CoreML) | FaceNet |
|---|---|---|---|
| Same-person threshold | 0.377 | 0.216 | 0.521 |
| 1:N threshold (live rule) | 0.494 (mean) | 0.342 (best photo) | 0.691 (mean) |
| Own photos warned, 1 photo | 7.6% [6.1, 9.3] | 3.8% [2.6, 5.1] | 10.4% [8.6, 12.1] |
| Own photos warned, 5 photos | 1.7% | 1.2% | 1.8% |
| Same, at the 1:N threshold, 1 photo (the old warning) | 21.5% | 5.3% | 36.8% |
| FAR, 1 photo / 5 photos | 0.09% / 0.36% | 0.10% / 0.35% | 0.11% / 0.36% |

The 1:N numbers reproduce the audit's M4 (22%, 5%, 42%; FaceNet's 42% was at its old best-photo threshold).

Small galleries, under each live rule at its frozen threshold. TPIR barely moves with gallery size; with one photo it falls from 96.2% to 78.5% (SFace), 98.5% to 94.7% (ArcFace) and 88.4% to 63.2% (FaceNet). FPIR at 20 identities with one photo is 0.011%, 0.006% and 0.028%. That reproduces M3, and no model misidentified a single mated probe there.

## What changed

- **`ryuk.evaluation.same_person`**:
  - `pairs(draw, photos)` scores every mated and impostor pair by the probe's best cosine to a gallery identity's first `photos` enrolled photos.
  - `freeze_same_person` returns the sealed `FrozenSamePerson`. It refuses the test draw and anything but one photo, and reads the threshold off the open-set curve, with impostor pairs in the place of non-mated probes.
  - `same_person(validation, test, live_threshold=..., seed=...)` builds the `SamePerson` record.
  - The warning rate is grouped by gallery identity. The FAR interval resamples both identities of each pair (`bootstrap.pair_ratio_interval`, the pigeonhole bootstrap). Its adjusted Wilson check groups by the probe's identity, as for FPIR.
- **`ryuk.evaluation.small_galleries`**:
  - `partition(identities, size, seed=49)` splits the test gallery into disjoint galleries of one size.
  - `small_galleries(draw, frozen: ModelThreshold, seed=..., sizes=(100, 20, 5))` scores the rehearsal's own gallery and then each size, with 5 photos and with 1.
  - Mated probes are scored against their own gallery, non-mated probes against every gallery. Rates are pooled and grouped by identity through `openset.IdentityGroups` (formerly the private `_Groups`).
- **`results.json`**:
  - A `live` section (`Live`, `LiveModel`, `SamePerson`, `SamePersonRates`, `SmallGallery`).
  - `ModelThreshold.same_person: SamePersonThreshold | None` holds the threshold, target FAR, validation FAR, impostor pairs, commit and date. `assemble` derives it from `live`.
  - `live_mismatch`: `live` must be on identification's draws, cover its models in order, and have been scored at each model's current live rule and threshold.
  - `schema_version` stays 1. Q19 holds: no model state.
- **`active`**:
  - `assemble(..., live=None)`, and `frozen_thresholds(identification, learning)`, the thresholds block before any same-person threshold.
  - `carried(identification, learning, bias, live)` drops `live` with a reason when a live rule or threshold moves.
- **CLI**:
  - `ryuk evaluate live` needs `celeba`, and belongs after `learn`.
  - `lfw`, `celeba`, `learn` and `bias` carry `live` while it still applies. `learn` keeps it unless a live rule changed.
- **Service**:
  - `Evaluated.same_person_threshold` is read from the thresholds block.
  - `Watchlist._not_same_person` warns when the best cosine is under it. With no same-person threshold it raises no warning (TO-BE-REVIEWED), where it used to fall back to the 1:N threshold. The warning's detail names "the same-person threshold".
- **Report**:
  - New Section 5.5, "Away from the rehearsal", with two tables (`tables.small_gallery_table` and `same_person_table`, each with notes). FPIR under 0.1% gets a third decimal there.
  - Section 5.2's "exactly as in the live monitor" is qualified.
  - New Section 3 paragraph "Away from the rehearsal".
  - New Section 8 limitation "Random impostors": the FAR is a floor for look-alike mistakes.
- **Docs**:
  - CONTEXT.md defines Same-person threshold, Impostor pair and Mated pair.
  - The README lists `ryuk evaluate live`.
  - TO-BE-REVIEWED gains two entries: the FAR 0.1% pick, and no warning without a threshold.
- **Tests**:
  - New: `test_same_person`, `test_small_galleries`, `test_live_outputs`.
  - Extended: `test_assemble`, `test_celeba_evaluation`, `test_bootstrap`, `test_watchlist_photos` and `test_watchlist_models`.
  - `tests/embedded_draws.py` holds the synthetic draw builder `test_learning` had privately.
  - `results_files.synthetic_live` builds a synthetic `live` section.
  - `watchlist_service.evaluated()` gives fakes a same-person threshold of 0.8, between two looks and two shots of one.

## Interpretations

- **One photo means each gallery identity's first enrolled photo** in draw order, for both the threshold and the small galleries. It is not an average over every single-photo enrollment.
- **Impostor pairs include other gallery identities' mated probes**, not only held-out ones. Any photo of someone else is what the warning should catch.
- **The thresholds stay frozen at the rehearsal's operating point.** The small-gallery table measures where the live monitor operates; it does not re-freeze anything.
- **The "1:N" rows** apply the live rule's threshold to the best cosine, exactly as the old warning did. For a learned rule the cell is a dash: its threshold is a probability.

## Review

Spec and Standards ran as reviewer agents.

- **Spec** found every requirement met except the two open items above, and checked the statistics. It raised that the FAR interval resampled only the probe's identity; this is fixed with the pigeonhole bootstrap and the results were rerun.
- **Standards** raised several points, all applied:
  - the glossary's avoided term "1:1 threshold" in a docstring;
  - four test names mangled by the builder extraction;
  - the duplicated bootstrap grouping;
  - positional readers of `SamePerson.test`, now pinned by a validator;
  - a CLI middle-man helper;
  - glossary gaps.
- **Not changed:**
  - The name `live` for a section that also holds enrollment's threshold: it is defensible ("where enrollment and the live monitor actually operate"), and renaming it means rerunning.
  - The field names that differ between `SamePerson` (`validation_far`) and `SamePersonThreshold` (`far`): each reads right where it sits.

## Found, not fixed

`Watchlist._looks_like_other` compares the photo's **best-photo** cosine with the active model's threshold under its **live rule**. For SFace and FaceNet (mean) the two scales are close. Under a learned rule, though, the threshold is a probability and the comparison is meaningless. No model runs a learned rule today. This is a candidate needs-triage ticket.

## Draft amendment for #12

> **Amendment (#49, #47 Q6).** `may_not_be_same_person` no longer compares a new photo with the 1:N match threshold. It warns when the photo's best cosine to the person's enrolled photos is under the active model's **same-person threshold**, a one-to-one cut-off measured per model on CelebA impostor pairs at FAR 0.1% (`thresholds[].same_person` in `evaluation/results.json`). When the active model has no same-person threshold, the warning is not raised. With one photo enrolled it now fires on 7.6% (SFace), 3.8% (ArcFace) and 10.4% (FaceNet) of a person's own photos, against 21.5%, 5.3% and 36.8% before.

## Things that bit

- **A blanket find-and-replace in tests.** Turning `_draw(` into `embedded_draw(` also rewrote `test_..._on_the_test_draw(` names. Replace whole identifiers.
- **`ruff format` between scripted edits** changes the text that a later scripted replacement expects. Re-read the file before replacing again.
- **The pigeonhole bootstrap can draw no valid pair.** No identity is its own impostor, so on a two-identity gallery a resample can hold only empty pairs; such resamples are left out.

## Worktrees and runtime state on this machine

- `~/Ryuk` is a fresh clone. `data` and `models/weights` are symlinks into `~/Projects/ryuk`, which holds the datasets, weights and embedding cache; both are in `.git/info/exclude`.
- `~/ryuk-49-run` is a clean, detached worktree at `e983186`, used only for the runs. Remove it with `git worktree remove ../ryuk-49-run` once the branch merges.
- `ryuk evaluate live` takes about three minutes here from the cached embeddings: two YuNet scans of about 8 s each, then the pairs and galleries.

## Resuming

- Shikhar rules on the two TO-BE-REVIEWED entries and reviews PR #69, which he merges himself.
- Then:
  1. merge with a merge commit on his go-ahead;
  2. post the #12 amendment above;
  3. remove `~/ryuk-49-run`.
- Working rules as before:
  - one ticket per session;
  - nothing merged without Shikhar;
  - role agents only, never Haiku;
  - no Co-Authored-By lines;
  - never Playwright locally;
  - heavy runs from a clean commit in a separate worktree.
