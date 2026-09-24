# Frontier session, 23 September 2026

The record of the second working session. It cleared the three frontier tickets that did not need a grilling: the toolchain spike, the dataset fetch, and the visual direction. Read `2026-09-22-charting-session.md` first. GitHub issues are the source of truth for state; this file is the narrative and the reasoning.

The map: https://github.com/ShikhJohari/Ryuk/issues/1

## What happened, in order

1. Shikhar asked for a walkthrough of the whole project, its purpose, and how it relates to crimdet. That was explanation only; nothing was decided.
2. `/setup-matt-pocock-skills` ran. It added `CLAUDE.md` with the agent skills block and `docs/agents/triage-labels.md`. `docs/agents/issue-tracker.md` and `docs/agents/domain.md` were already there. This landed as [PR 15](https://github.com/ShikhJohari/Ryuk/pull/15), with no linked issue because none covered it.
3. Shikhar asked to "start implementing the project into life". The map's standing preference is decisions before code, so this was read as working the frontier. Three tickets were claimed: the toolchain spike and the dataset fetch went to background agents, and the visual direction was drawn in Claude Design in the foreground while they ran.
4. The visual direction canvas went up with two directions. Shikhar chose B.
5. Both agents reported back. Each report was spot-checked against the files on disk before its resolution was posted and its ticket closed.

## Visual direction ([issue 13](https://github.com/ShikhJohari/Ryuk/issues/13))

The canvas: https://claude.ai/artifact/Dk7gsWYrJnMn5YrTSk7NEB. Three screens (live monitor, watchlist, evaluation), each drawn in two directions. Nav links work in Play mode.

- **A, ops dark.** Built from crimdet's Nord palette. Left sidebar, tight panels, IBM Plex Sans for labels and IBM Plex Mono for every number, frost cyan for a match, orange for no match. This was the direction the charting session expected to win.
- **B, lab notebook.** The contrast. Light, like a paper report: Newsreader serif headings, Public Sans body, ruled tables instead of cards, numbered figure and table captions. The idea is that the app reads like the report being graded. The video feed stays dark.

Content choices in the mockup: watchlist entries are CelebA numeric identities, not real names. On the evaluation screen, the published LFW figures are real numbers from the research docs, every Ryuk column says `[pending]`, and the ROC and score-distribution charts are labelled illustrative.

Three questions went to Shikhar: A, B, or a mix (A's dark shell for monitor and watchlist with B for evaluation was offered); density, typefaces and match colours; anything missing for the viva demo.

What Shikhar said: "when it comes to the ui and design i like B. the slighty off white tones with the tasteful buttons are my kinda jam. so the dark is off the counter." The density question did not land; it was explained as how tightly screens are packed (text size, spacing, how much fits), and B's spacing as drawn became the baseline, adjustable during the build. On missing screens: nothing yet, and Shikhar will say if something comes up.

So the charting session's dark preference is reversed, and the mix is not taken: every page, evaluation included, is light. The palette, type, components and chart styling are in the [resolution comment](https://github.com/ShikhJohari/Ryuk/issues/13#issuecomment-5800560019). No separate design system was made; the B artboards are the reference.

Things the mockup drew that it does not decide:

- Both int8 and fp32 SFace appear as rows. That belongs to the evaluation protocol ticket.
- "Remove from watchlist" appears on the detail screen. Whether history is kept belongs to the domain model ticket.
- Sightings are listed one per match, with no grouping across consecutive frames. That belongs to live monitor behaviour, not yet ticketed.

## Toolchain spike ([issue 7](https://github.com/ShikhJohari/Ryuk/issues/7))

All three models and YuNet run on the M4 (16 GB, Python 3.12.12 via uv). Full versions, weight sha256s and latencies are in the [resolution comment](https://github.com/ShikhJohari/Ryuk/issues/7#issuecomment-5800457712). The similarity figures come from 12 LFW images, so they are a sanity check, not a benchmark.

End to end per face, warm: SFace fp32 about 5.5 ms, ArcFace on CoreML about 3.9 ms (about 28 ms on CPU), facenet on CPU about 12 ms. At 10 fps every model uses a small part of the 100 ms frame budget. MPS was slower than CPU for facenet at batch size 1.

Findings that contradict `docs/research/models.md`:

- facenet-pytorch must be pinned to 2.5.3. 2.6.0 forces numpy below 2 and drags OpenCV back to 4.x.
- The YuNet weights are now `face_detection_yunet_2026may.onnx`, output identical to 2023mar.
- SFace int8 is about 3x slower than fp32 on OpenCV 5 (11.1 vs 3.9 ms), its embeddings drift (cosine 0.95 to 0.97), and int8bq will not load. fp32 looks like the baseline. Posted to the evaluation protocol ticket as input; the decision stays there.
- `buffalo_l.zip` is 288.6 MB, not 326, and only `w600k_r50.onnx` is needed.

Both alignment checks came out favourable. OpenCV `alignCrop` and InsightFace `norm_crop` fed YuNet landmarks produce the same crop, and a YuNet 5-point crop resized to 160 works for facenet with `fixed_image_standardization`. So YuNet is the only detector; InsightFace's detector and MTCNN are dropped.

Raised for the spec, not decided: InsightFace 2.0 has grown a GUI, video pipeline and licensing module, and Ryuk only needs one ONNX file. A direct onnxruntime session would drop the dependency.

Score ranges differ by model, so thresholds are per model, and embeddings from different model variants or execution providers must never share a watchlist. That second point feeds the domain model's model-swap scenario.

## Dataset fetch ([issue 8](https://github.com/ShikhJohari/Ryuk/issues/8))

Everything is in `data/raw/`, 2.9 GB, gitignored, every checksum verified. Full shape in the [resolution comment](https://github.com/ShikhJohari/Ryuk/issues/8#issuecomment-5800625352).

- LFW: 13,233 images of 5,749 identities, 10 folds of 300 matched and 300 mismatched pairs, every pair resolvable on disk. The deep-funneled variant was also fetched from the Wayback Machine and its MD5 matches, contrary to `datasets.md`.
- CelebA at pinned revision `2d738f5`: labels (identity and 40 attributes) for all 202,599 images, plus images for the official valid and test splits only: 39,829 images, 1,985 identities, 629 and 616 identities with at least 20 images. The two splits share no identities, so they serve as the validation and test draws. Train images were not fetched.

Findings carried forward:

- Use 300 to 400 held-out identities per draw instead of about 100. With 100, an FPIR of 0.1 percent rests on roughly one false alarm. Posted to the evaluation protocol ticket.
- CelebA images are pre-cropped PNGs at 178×218, not JPEG, so YuNet has to re-detect them. EDA should measure how often it succeeds.
- Report caveat: CelebA celebrities may be in the models' training sets, which would flatter identification numbers.

## Loose ends to carry

- `docs/research/models.md` and `datasets.md` still contain claims the agents disproved: the facenet version, the int8 recommendation, file sizes, PNG versus JPEG, the deep-funneled MD5. Offered as a small fix; not yet done.
- Whether to drop the InsightFace package for a direct onnxruntime session. Spec time.
- EDA is ready to become a ticket now that the data is on disk.
- The spike and fetch scripts live in local scratch projects and were not committed. The real download script should reuse the fetch approach: pinned revision, rename into place only after the checksum passes, range requests for CelebA labels, images read straight from the parquet shards.
- Still carried from the charting session: confirm with the instructor that a TypeScript client over a Python service counts as a Python project, and settle `web/` versus `apps/web`.

## Resuming

The frontier is now all grilling: the domain model ([issue 11](https://github.com/ShikhJohari/Ryuk/issues/11)) and the evaluation protocol ([issue 9](https://github.com/ShikhJohari/Ryuk/issues/9)). Take the domain model first, since the API contract ticket waits on it and through that the live monitor and client tickets. Then the evaluation protocol, which unblocks learning on embeddings and the bias breakdown.
