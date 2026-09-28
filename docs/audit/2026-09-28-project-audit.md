# Project audit, 28 September 2026

An audit of the whole repository at `7d9038c` (main, after #41), written so that another agent can act on the fixes. Four reviewer agents read the service, the machine learning and evaluation code, the client, and the decision trail (every issue, PR, handoff, ADR and research write-up). The lead then checked the sharpest claims by running the real service with real weights on OhmahgahPC, the Linux box Shikhar now develops on.

## How to use this document

- **IDs.** Every finding has an ID: **B** for bugs, **E** for the Linux box and remote access, **M** for measurement and methodology, **T** for traps waiting in the open tickets, **L** for low-priority hygiene, and **N** for findings the audit missed, added in the [second pass](#verification-and-errata-28-september-second-pass).
- **Fields.** Each finding lists:
  - severity;
  - **cause**: *Environment* means it comes from moving off the Mac to the Linux box and from reaching it remotely; *Everywhere* means it would bite on the Mac too;
  - **confidence**: *Confirmed* means reproduced, or traced end to end in the code; *Plausible* means the code path is real but the impact is inferred;
  - location, evidence and a fix direction.
- **Decisions first.** A finding tagged **Decision: Qn** needs Shikhar's ruling on that question in [Open questions for Shikhar](#open-questions-for-shikhar) before anyone acts on it. Everything else can be fixed directly.
- **Re-check before fixing.** Line numbers are as of `7d9038c`, so re-find them before editing. Re-check every *Plausible* finding before fixing it.
- **The working rules still hold:**
  - one ticket per PR, with `Closes #N`;
  - test first, through the public seams in spec #22's Testing Decisions;
  - nothing is merged without Shikhar;
  - role agents only;
  - never run Playwright locally.

## Verification and errata (28 September, second pass)

Four reviewer agents re-checked every finding against a detached worktree at `7d9038c`, with the real weights and the CelebA embedding cache from the Mac run. No finding was outright wrong. Some were overstated or cited the wrong place; the corrections are below, and each affected finding carries a "(see errata)" pointer. The original text is left as it was.

The second pass also found what the audit missed. Those are new findings **N1** to **N25**, in four "Missed by the audit" subsections after L5, starting at [service (N)](#missed-by-the-audit-service-n). The decision-free fixes are tracked in #43 (service), #44 (evaluation), #45 (client) and #46 (docs); the decisions are tracked in #47.

### Corrections

- **B6.** The label is not cut off all at once. It clips progressively once the box starts less than 20 CSS px below the top edge, and disappears only when `box.y <= 0`. "Boxes that start above the edge are not clipped" is wrong: the stage's `overflow-hidden` clips them.
- **B9.** Three of its claims overstate the gap.
  - `ProblemBody` is pinned indirectly: it spreads the pinned `Problem.fields` and `EnrollmentWarning`.
  - The PATCH body is pinned through `PersonOfInterestChangesMatchContract`. Only the call site at `web/src/api/persons.ts:132` is untyped.
  - The frame header does match the service's byte for byte (`">BIQHH"`); only a test is missing.

  What remains: the multipart field names, the call site, a test of the whole header, and `formatDate`.
- **B10.** The enroll dialog does focus its Name field (`autoFocus`). The warnings and switch dialogs take no initial focus. The audit also missed that no dialog returns focus to the control that opened it (N16).
- **B11.** The detector's hash is not in a top-level `detector` block. It is at `verification.detector` and `identification.detector`.
- **T3.** The watchlist's clock is injectable, through `start_watchlist(clock=)`. Only `LiveMonitor` and `_Connection` lack a seam.
- **E8.** This is not an environment finding. SQLite's rollback journal, in its default `DELETE` mode, copies a deleted photo's pages to `ryuk.sqlite3-journal`, which is then unlinked, not overwritten (N1). ADR 0004's promise fails on the Mac too. The snapshots on this box add a second copy.
- **M1.** Table 1's notes already print `accuracy_if_excluded_were_errors` (`src/ryuk/evaluation/tables.py:90-99`). Only the gate choice and the Section 5.1 disclosure remain.
- **M6.** The reviewer was right: Fogliato et al. floor N* at G for per-identity (FRR-type) rates, not G/2. The real defect is the zero-variance branch. For 20 identities × 5 trials with no errors, the upper bound is 3.7% today, against 16.1% with the G floor and 27.8% with G/2.
- **L1.** PRs #40 and #41 made regenerating `uv.lock` optional; they did not ask for it.
- **Minimum usable face size row.** The cite `src/ryuk/eda/build.py:273-280` does not exist; the file has 233 lines. The code is at `build.py:148-155`.
- **Q7 is already decided.** #9 holds out "every remaining identity", and #10 corrects the count to about 485–500 a draw; `CONTEXT.md:109` defines the term to match. The same goes for the "Held-out identities" row, whose approval is on record. Only the "remaining eligible identities" wording in #22 and #27 is stale.
- **Q14 is already decided** in #10: a winning rule that needs no retraining goes live. Only who owns the service side is open (T8).
- **Q11 is partly decided** by #19, which chose Quarto rendered through Typst. Whether the report is solo, whether figures float, and who writes Sections 5.3 and 5.4 are still open.
- **Q2's option (c) conflicts with two approved decisions.** #9 says CoreML MLProgram "matches CPU embeddings exactly"; #12 makes the execution provider part of a recognition model's identity. The question is which of the two gives way, and it is now framed that way.
- **Q3 is essentially forced** by spec story 59 and ADR 0001. **Q5 has a signal in #9**: "reproduce the recipe exactly" means scoring all 6,000 pairs.
- **"Approval: None" on the UI rows is overstated.** The closing handoff (line 46) records that Shikhar delegated UI taste to the implementer, and checks between chunks. The rows for monitor and watchlist client behaviour were within that delegation.
- **`looks_like_other` against removed persons** is latent until #31 adds removal. #12's "a different person of interest" arguably includes removed ones.
- **`duplicate_name`.** "Space-insensitive" in #12 can be read literally, so "Ann Lee" equalling "AnnLee" may be what was asked. The name key is also a stored, indexed column (`name_key`), so any change to it needs a data migration.
- **Fast merges** were all on green CI, 21–60 s after the last job finished. The point about no review from Shikhar stands.

### New measurements

From the CelebA embedding cache:

- **M3.** FPIR at the frozen threshold on the test draw, going from the rehearsal's gallery to a live-sized one:
  - SFace: 0.91% (500 identities × 5 photos), 0.25% (500 × 1), about 0 (20 × 1);
  - ArcFace: 0.66%, 0.15%, 0.01%.
- **M4.** The share of a person's own photos that raise `may_not_be_same_person` with one photo enrolled: SFace 22%, FaceNet 42%, ArcFace 5%.
- **M7.** About 10 of ArcFace's 39 budgeted validation false alarms come from 4 probe images and one held-out identity (2594). Three of the pairs are adjacent CelebA IDs (1532 and 1529, 2490 and 2491, 2789 and 2790), which fits duplicates.

## Summary

### State of the build

| Ticket | Scope | State |
|---|---|---|
| #23 | Scaffold, contract pipeline, CI | Done |
| #24 | Data fetch, weights, YuNet detection | Done |
| #25 | EDA, figures, report scaffold | Done |
| #26 | Recognition models, LFW verification | Done |
| #27 | CelebA open set, frozen thresholds | Done |
| #29 | Watchlist: persistence, registry, enrollment, client | Done |
| #30 | Live monitor: socket, boxes, toolbar, model switch | Done |
| #31 | Sightings, removal, restore, purge | Not started |
| #28 | Learning on embeddings, bias breakdown | Not started |
| #32 | Evaluation page, final report, README | Not started |

In the client, these routes are real:
- `/monitor`, which has no sightings rail yet;
- `/watchlist`;
- `/watchlist/$personId`, which shows photos, add, delete and rename, but no sightings, remove, restore or purge.

`/sightings`, `/sightings/$sightingId` and `/evaluation` are header-only placeholders.

In the report, only Section 2 and Section 5.2 are written; every other section is a stub. The remaining work is about 40% of the code by volume, plus most of the report.

### Verified on OhmahgahPC

- **Checks.**
  - `uv run pytest`: 660 passed, coverage 96.3%.
  - `pnpm typecheck`: clean. `pnpm test`: 40 passed.
  - CI on main is green, including the browser smoke test (run 36328341508).
- **Startup.** `uv run ryuk serve` with the fetched weights and the committed `results.json` makes **SFace** the active model (threshold 0.498). ArcFace is keyed `arcface-cpu-…` and is not evaluated (E1).
- **Enrollment** through `POST /api/persons`:
  - a 512 px photo enrolls;
  - the same face as an 800 px close-up is detected, and correctly draws the `looks_like_other` warning;
  - the same close-up at 2048 px is `422 no_face` (B1).
- **Live monitor.** A script drove the socket with 640×480 JPEG frames at quality 0.7. The enrolled face came back as a named match at 0.922, at about 48 results a second on loopback with SFace.
- **Origin guard.** A change sent with `Origin: http://ohmahgahpc:5173` is `403 cross_origin`; with `Origin: http://localhost:5173` it passes (E3).

### Was it the Linux box?

**Partly.** The move from the M4 Mac to the Linux box explains two things: why the app can't be reached the usual way, and why ArcFace isn't the active model. Those are E1 to E9. It explains none of the bugs (B), the measurement findings (M), the traps (T) or the agents' unapproved decisions, all of which would bite on the Mac too.

**None of the models is an Apple model.** SFace (OpenCV), ArcFace (InsightFace `w600k_r50`) and FaceNet (facenet-pytorch) are all open source. The Apple-specific part is CoreML, the onnxruntime execution provider that ran ArcFace on the M4. #12 made the provider part of a recognition model's identity (`CONTEXT.md`, "Recognition model"). So the ArcFace threshold measured under CoreML does not apply to the same weights running on CPU.

## Running it on OhmahgahPC today

What works today is the watchlist and the live monitor. The browser must reach the client as `localhost`, so use an SSH tunnel. The Tailscale URL cannot work (E3).

On OhmahgahPC, from the checkout you want to run:

```sh
uv sync && pnpm --dir web install
uv run ryuk weights fetch                  # ~335 MB into models/weights; without them nothing can be active
ss -ltn 'sport = :8000'                    # must print nothing: the client only proxies to 127.0.0.1:8000 (E4)
uv run ryuk serve                          # 127.0.0.1:8000; run it from the repository root (B7)
# second shell; keep the client on loopback, never 0.0.0.0 for this app (E5)
PORT=$(freeport)
pnpm --dir web dev --host 127.0.0.1 --port "$PORT" --strictPort
```

On the Mac:

```sh
ssh -N -L 5173:127.0.0.1:<PORT> blaze@ohmahgahpc
```

Then open `http://localhost:5173` on the Mac.

**Why the tunnel works.**
- `localhost` is a secure context, so the browser opens the Mac's own webcam.
- Vite accepts the `localhost` host name.
- The page's `Origin` is loopback, so the service's origin guard lets changes and the socket through.
- An editor's port forwarding (VS Code or Cursor Remote-SSH) works the same way, as long as the Mac's URL is `localhost:<n>`.

**What to expect until the fixes land.**
- SFace is the active model (E1).
- Faces are only scored within about 1 m of the webcam (M11).
- Close-up photos larger than about 1,000 px fail to enroll (B1).
- "May not be the same person" warns often (M4).

## Findings

### Bugs (B)

#### B1. Close-up photos cannot be enrolled

- **High** · Everywhere · Confirmed (reproduced) · no decision needed
- **Where:** `src/ryuk/watchlist/service.py:288`, in `_enrollable`.
- **What:**
  - Enrollment runs detection on the full-resolution stored photo, which can be up to 2048 px. YuNet misses faces much larger than about 400 px, so an ordinary head-and-shoulders phone portrait is refused as `422 no_face`.
  - Commit `c1f17fc` ("fix: detect enrollment photos within a bounded long side") says this was fixed, and both the watchlist and live-monitor handoffs repeat the claim. But only the commit's import change to `service.py` landed; `_enrollable` never passes `max_side`.
  - The commit's test exercises `Detector.detect` on its own, so CI stayed green.
  - Live frames are fine: `src/ryuk/watchlist/live.py:145` passes `max_side=MAX_DETECTION_SIDE`.
- **Evidence:**
  - Against the real service, the astronaut fixture cropped to head and shoulders enrolls at 800 px and is `no_face` at 2048 px.
  - A reviewer reproduced it with a 3024×4032 portrait. Faces 20–25% of the photo's width (323–395 px) enroll; faces 30% or wider (477 px and up) are `no_face`. The bounded copy finds the face every time.
- **Fix:**
  - Call `self._detector.detect(pixels, max_side=MAX_DETECTION_SIDE)`.
  - Add a test through the HTTP seam with a face of at least 500 px in a 2048 px photo. The existing `test_a_large_photo_is_stored_with_its_long_side_bounded` passes only because its face is about 360 px.
  - Look at M5 at the same time.
  - Correct the two handoffs' claim.

#### B2. JPEGs that Pillow reads as MPO are refused

- **Medium** · Everywhere · Confirmed (reproduced by a reviewer) · no decision needed
- **Where:** `src/ryuk/watchlist/photos.py:89` and `:98`.
- **What:** Pillow opens a JPEG that carries a multi-picture (MPF) segment as `MpoImageFile`, with `format == "MPO"`. `MEDIA_TYPES` has no `MPO` entry, so the upload is `422 unsupported_image`. The same pixels saved as a plain JPEG enroll. Some cameras and phones write MPF segments into ordinary JPEGs.
- **Fix:** treat `MPO` as JPEG: take the first frame and store it as `image/jpeg`. Test with a Pillow-written MPO.

#### B3. Cancel or Escape during an enrollment neither cancels it nor stays put

- **Medium** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:**
  - `web/src/components/enroll-dialog.tsx:102`: Cancel is never disabled.
  - `web/src/components/ui/dialog.tsx:18`: Escape always closes the dialog.
  - `web/src/hooks/use-acknowledged-mutation.ts:35-38`: the success callback lives on `useMutation`.
  - `enroll-dialog.tsx:31`: the success callback navigates.
- **What:**
  - Closing the dialog doesn't stop the request, and TanStack Query v5 still runs a `useMutation`-level `onSuccess` after the component unmounts. So the person is enrolled anyway.
  - The app then navigates to that person's page. If the operator has meanwhile gone to the monitor, this unmounts it and stops the camera.
  - The same happens when Escape is pressed during "Enroll anyway". Go back is disabled while pending; Escape isn't.
- **Fix:**
  - Either disable Cancel and ignore Escape while the request is pending, or move the navigation into the per-call `mutate(…, { onSuccess })`, which does not run after unmount.
  - Add a route test: close the dialog mid-upload and assert there is no navigation.

#### B4. A lost camera is never noticed

- **Medium** · Everywhere · Plausible: the missing handling is confirmed; the browser behaviour was not reproduced · no decision needed
- **Where:** `web/src/hooks/use-live-monitor.ts`. Nothing listens for the video track's `ended` or `mute` events.
- **What:** if a USB webcam is unplugged or camera permission is revoked mid-session, the `<video>` element keeps its last frame and size. The frozen picture keeps being sent at up to 30 fps with the status still "running". Once #31 lands, a frozen face would keep a sighting open.
- **Fix:** listen for `ended` and `mute` on the track, then move to a camera-lost state and stop sending.

#### B5. A stalled service is invisible on the monitor

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `web/src/hooks/use-live-monitor.ts`:
  - `:88` waits for a result with no timeout;
  - `:120-131` recomputes the frame rate only when a result arrives.
- **What:** if the service hangs or only answers errors:
  - the boxes freeze over live video;
  - the toolbar keeps showing the last frame rate;
  - the status stays "running".

  `docs/research/frame-streaming.md` recommended "the result or a timeout".
- **Fix:** add a result timeout that shows a stalled state, and let the frame rate decay on a timer.

#### B6. Match names disappear for faces near the top of the frame (see errata)

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `web/src/components/face-overlay.tsx:92`, where the label sits above the box, and `web/src/routes/monitor.tsx:70`, where the stage has `overflow-hidden`.
- **What:** a match whose box starts within about 13 px of the frame's top edge is drawn as a solid box with no name. Boxes that start above the edge are not clipped either. This is common with a close laptop webcam, and M11 makes close the normal case.
- **Fix:** put the label inside or below the box when there is no room above it.

#### B7. Face data can land in git

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `src/ryuk/settings.py:18-26`, whose paths resolve against the working directory, and `.gitignore:29`, whose `data/ryuk.sqlite3*` pattern only matches at the repository root.
- **What:** `uv run ryuk serve` started from `web/` creates `web/data/ryuk.sqlite3`. `git check-ignore` confirms git does not ignore it, and it holds face photos in a public repository. That same run also finds no weights and no `results.json`, so nothing can be active.
- **Fix:** ignore `**/data/ryuk.sqlite3*`. Also either refuse to start (or warn loudly) when `results.json` is missing, or resolve the default paths against the repository root.

#### B8. Error messages misdiagnose failures

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `web/src/lib/problems.ts:5-9`, and the monitor's handling of socket close codes.
- **What:**
  - Any failure that isn't a problem response reads "The service could not be reached". That includes a 2xx body that no longer matches its schema, and a body that isn't JSON.
  - When the origin guard closes the socket (1008), the browser sees 1006, and the UI says "Lost the connection to the service".
  - The client shows the service's `detail` text verbatim, against the handoff pattern that "the client maps `code` to copy".
- **Fix:** tell decode failures apart from network failures, and give each failure its own accurate message.

#### B9. Contract seams that `tsc` doesn't guard (see errata)

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:**
  - `web/src/api/api-client.ts:20-23`: `ProblemBody` is a local schema, not pinned to the generated type. If the service renamed `warnings`, the warnings dialog would silently show an empty list.
  - `web/src/api/persons.ts`: the multipart field names and the PATCH body are not pinned.
  - `web/src/api/monitor.ts` and `src/ryuk/api/frames.py`: the 17-byte frame header is mirrored by hand, and only the e2e test checks that the two agree.
  - `web/src/lib/format.ts:4-6`: `formatDate` throws `RangeError` on an unparseable timestamp, which takes the whole table down.
- **Fix:**
  - Pin `ProblemBody` to the generated problem type.
  - Assert the request shapes against the contract.
  - Add a test that the client's frame header matches the service's parser.
  - Make `formatDate` total.

#### B10. Keyboard and focus (see errata)

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:**
  - `web/src/components/ui/dialog.tsx`: dialogs take no initial focus and have no focus trap. When the warnings dialog replaces the enroll form, focus falls back to the page, and a keyboard user must tab through the whole watchlist table to reach "Enroll anyway".
  - `web/src/components/ui/button.tsx:6` and `web/src/styles.css:50`: focus rings are ink-coloured, which is invisible on the ink monitor stage. "Reconnect" and "Monitor here" show no visible focus.
  - `web/src/routes/watchlist/$personId.tsx:155-171`: "Add photo" is a label wrapping a visually hidden input, so focus lands on a clipped 1 px element.
- **Fix:**
  - Focus the dialog's first control and trap focus inside the dialog.
  - Use a paper-coloured focus ring on the stage.
  - Make the Add photo label show focus when its input has it.

#### B11. The detector's weights are never verified at startup (see errata)

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `src/ryuk/watchlist/load.py:31`.
- **What:** recognition models are keyed by their real sha256, but YuNet is loaded unchecked. Nothing compares its hash, or `MIN_USABLE_FACE_SIZE`, with the `detector` block of `results.json`. The spec's startup step says "verify weights hashes". A swapped detector would silently change enrollment and live detection, both of which were calibrated against that block.
- **Fix:** check the loaded hash against `detector.weights_sha256` in the committed results. On a mismatch, refuse to start, or mark every model not evaluated.

### Environment: the Linux box and remote access (E)

#### E1. ArcFace, evaluation's first active model, cannot be active on Linux

- **High for the demo** · Environment · Confirmed (ran it) · **Decision: Q2**
- **Where:**
  - `src/ryuk/recognition/arcface.py:30-33`: `default_provider`, which is `cpu` off Apple Silicon.
  - `src/ryuk/watchlist/registry.py:46-57`: `first_active_for`.
- **What:**
  - The committed thresholds key ArcFace as `arcface-coreml-…`. On Linux the same weights run on CPU as `arcface-cpu-…`, which has no threshold, so that model is `not_evaluated`.
  - The registry re-applies #9's rule to the models that can run here and picks SFace: test TPIR 95.0%, against ArcFace's 98.5%.
  - That fallback was an implementer's decision, and it is never persisted (see [Decisions agents made without Shikhar](#decisions-agents-made-without-shikhar)).
  - #7 found that CoreML (MLProgram) and CPU embeddings match, but #12's identity rule keeps them distinct recognition models.
- **Fix directions**, to choose between after Q2:
  - evaluate ArcFace on CPU too, and keep both thresholds in `results.json` (see E2);
  - or rule that CoreML MLProgram and CPU are the same recognition model and share one threshold, backed by an embedding parity test.

  Either way, measure ArcFace's CPU time per face on this box first. It took 27.8 ms on the M4's CPU (#7), against a 30 ms eligibility limit, and this x86 box is slower.

#### E2. The committed results can only be reproduced on Apple Silicon

- **High:** blocks #32's acceptance on Linux · Environment · Confirmed · **Decision: Q2**
- **Where:**
  - `src/ryuk/cli.py:134`: models load with the machine's default provider.
  - `src/ryuk/cli.py:150-157`: the CelebA results are dropped on a mismatch.
  - `src/ryuk/evaluation/results.py:361-376`: `identification_matches` and `_mismatch`.
- **What:**
  - On Linux, `ryuk evaluate lfw` measures `arcface-cpu` and finds that the committed identification results no longer match.
  - It prints a warning, drops every CelebA result, every threshold and the first active model, and overwrites `evaluation/results.json`.
  - #32 requires that "following the README on a clean checkout reproduces the committed outputs"; today that holds only on an Apple Silicon Mac.
  - `results.json` can hold only one ArcFace provider.
- **Fix:**
  - Let the results carry an entry per provider (evaluate and merge rather than replace), or name one machine of record and say so in the README.
  - Either way, don't let a run overwrite committed thresholds on the strength of a warning alone. Require an explicit flag.

#### E3. The UI can't be used through `http://ohmahgahpc:<port>`

- **High for Shikhar's workflow** · Environment · Confirmed (the service side was run) · **Decision: Q3**
- **Three independent blocks:**
  1. **Vite refuses the host.** Vite binds loopback by default. Even with `--host 0.0.0.0`, its host check rejects the name `ohmahgahpc`, because `server.allowedHosts` is unset. Raw IP addresses get through.
  2. **No camera.** A plain-http page that isn't localhost is not a secure context. `navigator.mediaDevices` is undefined, `openCamera` throws `unsupported` (`web/src/lib/camera.ts:16-17`), and the monitor says "No camera".
  3. **The origin guard refuses changes.** `src/ryuk/api/localhost.py:68-90` refuses every change, and the socket, from a non-loopback `Origin`. Vite's proxy option `changeOrigin` rewrites `Host` only, not `Origin`.
     - Measured: `403 cross_origin` for a POST, and the socket closed with 1008.
     - GETs still pass, so the header keeps saying "Service connected".
- **What works:** an SSH local forward, so the browser's origin is `http://localhost` (see [Running it on OhmahgahPC today](#running-it-on-ohmahgahpc-today)).
- **Fix:** document the tunnel in the README; no code change is needed. Don't loosen the origin guard without an answer to Q3.

#### E4. The client can only reach a service on port 8000

- **Medium** · Environment (shared box) · Confirmed · no decision needed
- **Where:** `web/vite.config.ts:12-14`, plus Playwright's fixed ports 8000 and 4173.
- **What:** `RYUK_PORT` moves the service, but the proxy can't follow it. On a box where other agents run servers, a Ryuk client pointed at 8000 would send photo uploads to whatever holds that port.
- **Fix:** read the proxy target from an environment variable (for example `RYUK_PORT`), defaulting to 8000, and document using `PORT=$(freeport)`.

#### E5. Binding the client to 0.0.0.0 exposes the unauthenticated API to the tailnet

- **Medium** · Environment · Confirmed by reading (not exercised) · no decision needed
- **Where:** the same proxy, and `src/ryuk/api/localhost.py:85-90`, where a request with no `Origin` passes.
- **What:**
  - This machine's agent rules say to bind dev servers to `0.0.0.0`. For Ryuk that would expose the API to every tailnet device.
  - Any device could `curl http://100.x:<port>/api/persons`, and fetch the photo URLs too.
  - A POST with no `Origin` could enroll, rename or delete through the proxy.
  - Spec story 59 says the unauthenticated API is never exposed.
- **Fix:** state in the README and the repo's `CLAUDE.md` that Ryuk's client stays on `127.0.0.1` whatever the machine rules say. Optionally, make the Vite config refuse a non-loopback host.

#### E6. The data, the weights and the CelebA embedding cache live only on the Mac

- **Medium** · Environment · Confirmed · **Decision: Q2**
- **What:**
  - `~/Work/Ryuk` and the worktrees here have no `data/` and no `models/weights/`.
  - The app needs only the weights (`ryuk weights fetch`, about 335 MB).
  - #28 needs the 2.6 GB of data, and ideally the embedding cache from the Mac's CelebA run (12.5 minutes there).
  - Rebuilding that cache here re-keys ArcFace (E1, E2).

#### E7. Mac measurements are trusted on Linux without a check

- **Low** · Environment · Plausible · no decision needed
- **What:**
  - SFace and FaceNet keep their model keys across machines, so the Mac's thresholds apply here.
  - Nothing checks that Linux embeddings match the Mac's. `tests/smoke/test_real_models.py` checks only stability on one machine.
  - torch comes from different builds on the two platforms: PyPI on the Mac, the CPU index on Linux.
  - `GET /api/models` shows the M4's ms per face on any machine.
  - The expected drift is tiny, but nobody has measured it.
- **Fix:** commit a handful of reference embeddings from the Mac, and assert in the smoke job that each Linux embedding has a cosine above 0.9999 to its reference.

#### E8. Purge cannot truly erase on this box (see errata)

- **Medium** · Environment (btrfs snapshots here; the Mac has the APFS and Time Machine equivalent) · Confirmed · **Decision: Q13**
- **What:**
  - ADR 0004 and spec story 20 promise that purge erases a face.
  - The database sits under `~/Work` or `~/.t3`, which have hourly read-only btrfs snapshots. btrfs copy-on-write also defeats SQLite's `secure_delete`.
  - The #29 review flagged the Mac version of this problem, and ADR 0004 was never amended.
- **Fix:** put the default database outside the snapshotted paths (for example `~/.local/share/ryuk`), or state the limitation in ADR 0004 and in the report.

#### E9. Mac-only wording

- **Low** · Environment · Confirmed · no decision needed
- **Where:**
  - `README.md:58` ("a few minutes on Apple Silicon"), and the same line in the `cli.py` docstring;
  - ADR 0002 ("one Apple M4 laptop");
  - ADR 0003 ("20 ms per photo on the M4");
  - issue #1 and the handoffs ("Local clone at `~/Projects/ryuk`").

### Measurement and methodology (M)

#### M1. The "within 0.5 points of published" gate is judged on the easier pairs (see errata)

- **Medium** · Everywhere · Confirmed · **Decision: Q5**
- **Where:** `src/ryuk/evaluation/verification.py:199-223`.
- **What:**
  - Accuracy is measured over the 5,917 pairs YuNet could score, but the published figures cover all 6,000.
  - The 83 dropped pairs are the ones where YuNet found no usable face, so they are likely the hardest.
  - Counting them as errors gives SFace 98.00, ArcFace 98.40 and FaceNet 97.98. All three are 1.4 points or more below published (99.40, 99.83 and 99.65).
  - `results.json` already carries `accuracy_if_excluded_were_errors`, but the gate ignores it.
  - FaceNet stays within 0.5 points only if at most 13 of the 83 pairs would be scored wrong.
- **Fix:** report both figures in Table 1, decide which one the gate uses, and disclose the choice.

#### M2. FaceNet's LFW View 2 was scored twice

- **Medium** · Everywhere · Confirmed (PR #35) · **Decision: Q5**
- **What:**
  - PR #35 added the margin-32 crop as a View 1 candidate after View 2 had flagged FaceNet at 99.12 with a margin of 14.
  - #9 says View 2 runs once per model, but it also calls a gap over 0.5 points "a pipeline bug to fix".
  - The EDA and recognition handoff (line 152) attributes the 99.12 to the five-point crop, which contradicts the PR.
  - Notebook 02 says the crop was "chosen on View 1 only".
- **Fix:** disclose the second run in report Section 5.1 and notebook 02, and correct the handoff.

#### M3. The live operating point is not the one that was measured (see errata)

- **Medium** · Everywhere · Plausible (the design is confirmed; the size of the effect is unmeasured) · **Decision: Q6**
- **What:**
  - The thresholds were frozen on galleries of 500 identities with 5 photos each (2,500 embeddings). A live watchlist starts with one photo per person and a handful of people.
  - Under best-photo scoring, FPIR at a fixed threshold grows with the number of enrolled embeddings, so live FPIR will sit well under 1%.
  - TPIR with one enrolled photo will sit under Table 2's figures: 98.5% for ArcFace, 95.0% for SFace.
  - `report/sections/05-evaluation.qmd:87` says the scoring is "exactly as in the live monitor".
- **Fix:** using the cached embeddings, measure TPIR and FPIR with one enrolled photo per identity and for small galleries, then qualify the report's sentence. This fits in #28 or a new ticket.

#### M4. The 1:N threshold is reused for a 1:1 warning (see errata)

- **Medium** · Everywhere · Confirmed · **Decision: Q6**
- **Where:** `src/ryuk/watchlist/service.py:343` (`looks_like_other`) and `:374` (`may_not_be_same_person`).
- **What:**
  - "May not be the same person" fires when a new photo scores below the 1:N threshold against the person's own photos.
  - That threshold is much stricter than a 1:1 one. SFace's is 0.498, against LFW fold thresholds around 0.335; FaceNet's is 0.709, against about 0.445.
  - So legitimate extra photos, from another angle or in other lighting, will often warn, especially with SFace active on Linux.
- **Fix:** give this warning a 1:1 threshold (from LFW, or from a false-accept rate on CelebA mated pairs), chosen under Q6.

#### M5. Enrolled faces are cut from large photos without anti-aliasing

- **Medium** · Everywhere · Plausible · no decision needed, but measure first
- **Where:**
  - `src/ryuk/recognition/faces.py`: the five-point crop is OpenCV's `alignCrop`, a bilinear warp with no prefilter.
  - `src/ryuk/watchlist/photos.py:65`: the ICC profile is dropped without converting to sRGB.
- **What:**
  - A 400–800 px enrolled face is warped straight down to 112 px, a 4–7× shrink.
  - CelebA gallery faces (median 86 px) were cropped at close to 1:1.
  - Display P3 uploads are read as if they were sRGB.
  - So enrolled embeddings may differ systematically from the ones the threshold was set on. InsightFace's own pipeline shares the shrink, so the effect may be small.
- **Fix:**
  - Measure first: embed CelebA faces at 1×, then upscaled 5× and cropped, and compare the embeddings.
  - If it matters, crop from a copy downscaled so that the face is about 112–160 px.
  - Convert ICC profiles to sRGB on upload.

#### M6. The adjusted Wilson interval collapses to the naive one when there are no errors (see errata)

- **Medium:** latent now, hits #28's bias cells · Everywhere · Confirmed (code) · no decision needed
- **Where:** `src/ryuk/evaluation/bootstrap.py:79`: `effective = total if variance == 0 else max(…, wrong.size / 2)`.
- **What:**
  - With no errors the variance is 0, so N* = N and the interval is the textbook Wilson interval that Fogliato et al. warn against.
  - A reviewer also reads the paper as flooring per-identity (FRR-type) rates at G, not G/2. Verify that against arXiv:2306.01198 before changing it.
  - No current result is affected, but #28's zero-error group cells will be.
- **Fix:** apply the floor in the zero-variance case too, and test with 20 identities × 5 trials and no errors.

#### M7. CelebA labels look noisy in the validation draw (see errata)

- **Medium to low** · Everywhere · Plausible · no decision needed
- **What:**
  - For ArcFace on validation, the top-scoring non-mated probe outscores about half of the correct mated matches, and 7 non-mated probes score in the genuine range.
  - TPIR at FPIR 0.1% is 80.7% on validation, against 98.4% on test.
  - SFace and FaceNet show the same pattern.
  - The likely cause is the same person appearing under two CelebA IDs.
  - This spends about 7 of ArcFace's roughly 39 allowed false alarms at FPIR 1%, which pushes its threshold up.
- **Fix:** list the top-scoring non-mated pairs and look at them. If they are duplicates, report that as a limitation; don't silently relabel.

#### M8. Embedding keys omit the pipeline version

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:**
  - `src/ryuk/evaluation/verification.py:94`: the cache key is the detector hash, the minimum face size and the crop.
  - `src/ryuk/watchlist/service.py:514-521`: the database key is the model key and the crop.
- **What:** a change to any of these would silently reuse stale embeddings, in both places. Nothing has changed yet.
  - `embed()` normalisation or the alignment;
  - YuNet's score or NMS thresholds;
  - the CoreML model format (#7 found NeuralNetwork shifts embeddings);
  - a library version.
- **Fix:** add a `PIPELINE_VERSION` constant to both keys.

#### M9. Smaller evaluation gaps

- **Low** · Everywhere · Confirmed · no decision needed
- **TAR at FAR is in-sample.** LFW TAR at FAR is pooled over the folds (`verification.py:227`), with the threshold chosen on the same data. The 1e-3 point rests on 2 false accepts. This is disclosed.
- **Rebuilt draws are never checked.** `selection_sha256` is recorded but not verified when a draw is rebuilt (`src/ryuk/evaluation/celeba.py:162`). #28 must rebuild the same draws, and if one identity crosses the 20-image line, `rng.choice` reshuffles the whole gallery.
- **"Test scored once" holds only within a run.** Nothing stops a rerun of `ryuk evaluate celeba` from scoring the test draw again.
- **ms per face is timed on the wrong images.** It is measured on 178×218 CelebA images, not 640×480 frames, and on the M4.

#### M10. Report and notebook inconsistencies

- **Low** · Everywhere · Confirmed · no decision needed
- `report/sections/03-methodology.qmd:32` says "the highest" threshold; the code and Section 5.2 use the lowest.
- `gallery_candidates` counts before exclusion in `eda/summary.json` (629) and after it in `results.json` (584).
- "ms per face" means embedding only in the int8 footnote (4.1 ms), but end to end in Table 2 (5.4 ms).
- FaceNet's published "± 0.25" is a standard deviation across folds, printed next to our standard error.
- `05-evaluation.qmd:97` makes qualitative claims that nothing computes from the results ("about as intended", "every interval including or lying below").
- Section 2 cites Section 3.3, which is a stub, and nothing makes a render fail while stubs remain.
- `report/progress.qmd:16`: the status table stops at #27.

#### M11. Faces are only usable within about 1 m of the webcam

- **Medium (product)** · Everywhere · Confirmed (code), with arithmetic · **Decision: Q4**
- **Where:**
  - `web/src/lib/camera.ts:11-12, 21-22`: the camera is asked for 640×480, and frames are sent with a 640 px long side at JPEG quality 0.7.
  - `src/ryuk/detector.py:19`: `MIN_USABLE_FACE_SIZE = 70`.
- **What:**
  - A 15 cm-wide face reaches 70 px in a 640 px frame only within about 0.85–1.2 m, for a webcam with a 60–78° field of view. Further away, every face is `too_small`.
  - The server already accepts frames up to 1920 px. It detects on a 640 px copy and scales the boxes back, so 1280 px frames would roughly double the range at little detection cost, paying instead in bandwidth and decoding.
  - Nobody decided on 640 px; #5 used it only as a size for counting bytes.
  - The 70 px minimum came from CelebA's detection statistics (#17), not from recognition quality at distance.
- **Fix:** after Q4. If the answer is 1280 px, measure the frame rate through the SSH tunnel first.

### Traps waiting in the open tickets (T)

**#31, sightings, removal and purge:**

- **T1. `sighting.model_key` must not cascade from `recognition_model`.**
  - When weights or a crop change, `_sync_embeddings` (`src/ryuk/watchlist/service.py:453-472`) deletes the model's row to cascade its embeddings.
  - An `ON DELETE CASCADE` foreign key on sightings would therefore silently erase that model's sighting history at startup. `RESTRICT` would crash startup instead.
  - #12 draws no foreign key there, so keep it a plain column.
- **T2. PATCH must change shape.** `PersonOfInterestChanges` (`src/ryuk/api/persons.py:60-67`) requires `name` and refuses `status`. Make both optional, keep `extra="forbid"`, and regenerate the contract.
- **T3. No clock reaches the monitor.** (see errata)
  - `create_app` (`src/ryuk/api/__init__.py:20`) takes only the watchlist factory.
  - `Watchlist._clock` is private.
  - `LiveMonitor` has no clock at all.

  The 500 ms confirmation and the 3 s gap need an injected clock (spec seam 1).
- **T4. End sightings in a `finally`.**
  - `_Connection._recognise` (`src/ryuk/api/monitor.py:214-218`) catches only `FrameError`.
  - Any other exception escapes the task group, and uvicorn closes the socket with 1011.
  - `_Connection.run` has no `finally`, so "open sightings end on close" must be written there.
- **T5. The runner-up needs a second candidate.** `WatchlistEmbeddings.top_candidate` (`src/ryuk/watchlist/live.py:122`) returns only one.
- **T6. The switch confirm must say that open sightings will end** (story 28). Today it says "Nobody needs enrolling again" (`web/src/components/monitor-toolbar.tsx:130`). Only the live-monitor handoff assigns this to #31; the ticket itself doesn't.
- **T7. Removal needs no new columns.** `person_of_interest` already has `status`, with a CHECK constraint, and `status_changed_at`.

**#28, learning on embeddings and bias breakdown:**

- **T8. Only best-photo can run live.**
  - `LIVE_RULES = {"best-photo"}` (`src/ryuk/watchlist/registry.py:21`), and `Evaluation.from_results` drops thresholds recorded for any other rule (`:64-73`).
  - So a model whose recorded live rule is `mean` or a learned rule silently becomes `not_evaluated`.
  - Its crop then falls back to five-point (`:118-120`), which rebuilds its embeddings.
  - If every model's winner is something other than best-photo, the monitor refuses to start (4002).
  - `MatchRule`, `LIVE_RULES` and `top_candidate` must change together, and no ticket owns the service side (Q14).
- **T9. The data and the CelebA embedding cache are on the Mac** (E6). Running #28 on Linux re-keys ArcFace (E1, E2).
- **T10. Two measurement findings will bite #28.** M6's Wilson collapse will hit the per-group cells. M9's unchecked `selection_sha256` matters when #28 rebuilds the draws.
- **T11. Deferred to #28, but missing from its ticket:** the per-probe score parquet (#10), and top-two candidates in `Gallery` (the open-set handoff).
- **T12. Bias groups use each identity's majority label** over all its images, usable or not (open-set handoff, line 143; `ryuk.eda.aggregate.majority_label`). #28 is told to follow this; it was an agent's choice.
- **T13. There is no scikit-learn dependency yet.**

**#32, evaluation page and final report:**

- **T14. No settings path for `eda/summary.json`.** `Settings` has none, and `GET /api/evaluation` must serve that file.
- **T15. No chart library is chosen** (`web/package.json`). #17 only says the client draws its own interactive charts (Q12).
- **T16. Reproduction works only on Apple Silicon.** "A clean checkout reproduces the committed outputs" holds only there (E2).
- **T17. Add a render check that fails on any `::: stub` block** left in the report (`report/style/stub.lua`).
- **T18. Fix stale docs on the way:**
  - `README.md:3` says the client shows evaluation results.
  - `docs/research/frame-streaming.md:9` says 10 fps.
  - `docs/research/client-toolchain.md:7,20-21` says `apps/web` and ESLint.
  - The live-monitor handoff still says "Not merged" and warns that the browser smoke test is untried. It passed: run 36326993888, and main run 36328341508.

### Low-priority hygiene (L)

- **L1. `uv.lock` was spliced by hand** in `f1b7064` and `f9b8b80`. Regenerate it with `uv lock` where the PyTorch index is reachable (it is from OhmahgahPC), as PRs #40 and #41 asked (see errata).
- **L2. `write_into_place` lives in `ryuk.fetch`,** yet `eda/files.py:13` and `evaluation/results.py:17` import it. There are also two provenance modules, `eda/build.py:56` and `evaluation/provenance.py:12`. Both were carried across handoffs.
- **L3. #17's "exclusions reported per group" is only partly met,** because `GroupStats` has no per-group gallery candidates.
- **L4. Still open from the #29 review:**
  - no `hide_parameters` on the database engine;
  - no file mode set on the database;
  - no `no-store` on JSON responses (only images get it, `src/ryuk/api/persons.py:135`);
  - control and bidi characters accepted in names;
  - no ApiClient seam tests for `postForm`, `patch`, `put` and `delete`.
- **L5. Live frames wait during enrollment.** Enrollment holds the watchlist lock for its whole transaction, including embedding the photo under every loaded model; on Linux that includes ArcFace on CPU, even though it isn't evaluated there.

### Missed by the audit: service (N)

#### N1. Deleted photo bytes survive in the SQLite rollback journal

- **Medium** · Everywhere · Confirmed · no decision needed for the journal; where the database lives is **Decision: Q13**
- **Where:** `src/ryuk/watchlist/database.py:112-119`. `_on_connect` turns on `secure_delete` and foreign keys, and leaves `journal_mode` at SQLite's default, `DELETE`.
- **What:**
  - Before a transaction changes a page, SQLite copies the page to `ryuk.sqlite3-journal`. Deleting a photo therefore copies its blob there.
  - `secure_delete` zeroes the page in the main file, but at commit the journal is unlinked, not overwritten. The bytes stay in free disk blocks, and in any snapshot taken meanwhile.
  - This is why E8 holds on the Mac too. ADR 0004's promise fails on any disk.
- **Fix:**
  - Choose a journal mode that leaves no freed copy of a deleted page behind: `TRUNCATE` or `PERSIST`, or WAL with a checkpoint after each delete. Let the test pick the mode.
  - Run `VACUUM` after a purge.
  - Test that after a delete no journal or WAL file holds the photo's bytes.
  - Amend ADR 0004.

#### N2. A socket with no `Origin` passes the guard and takes over the monitor

- **Medium** · Everywhere · Confirmed · no decision needed
- **Where:**
  - `src/ryuk/api/localhost.py:68-69`: `all(…)` over the handshake's `Origin` headers is true when there are none.
  - `src/ryuk/api/monitor.py:120-125`: every new connection supersedes the current one.
- **What:**
  - Browsers always send `Origin` on a WebSocket handshake; other clients need not. A handshake with no `Origin` is accepted.
  - It then closes the operator's monitor with 4001, "opened in another tab", and the monitor stays down until someone presses "Monitor here".
  - The REST rule, which lets a missing `Origin` through for curl, was carried over to the socket, where nothing needs it. With the client on `0.0.0.0` (E5), any tailnet device could do this.
- **Fix:** require an `Origin` on the socket handshake and keep the current rule for REST. Test that a handshake without one is refused and a loopback one still passes.

#### N3. `ryuk evaluate` ignores `RYUK_RESULTS`

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `src/ryuk/cli.py:55`: `RESULTS = Path("evaluation/results.json")`, used by both `evaluate` commands.
- **What:** `ryuk serve` reads the results path from `Settings.results`, so `RYUK_RESULTS` moves it. `ryuk evaluate` reads and writes the hard-coded path. A per-machine results file, which E2 and Q2's option (b) need, can be served but not produced.
- **Fix:** make `evaluate` use the same setting, and test it through the CLI seam.

#### N4. Any unexpected error in recognition closes the socket

- **Low** · Everywhere · Plausible (the path is real; no frame was found that triggers it) · no decision needed
- **Where:** `src/ryuk/api/monitor.py:214-218`, where `_recognise` catches only `FrameError`; for example, `src/ryuk/recognition/faces.py:50-51` raises `ValueError` for a box outside the image.
- **What:** T4 described this as a trap for #31, but it is a bug today. Any other exception escapes the task group, uvicorn closes the socket with 1011, and the operator sees "Lost the connection to the service".
- **Fix:** catch per frame, log it, send a `MonitorError` with a new `internal_error` code, and keep the socket open. Test with a recogniser that raises `ValueError`.

### Missed by the audit: evaluation (N)

#### N5. `ryuk evaluate celeba` on Linux runs for 12 minutes, then crashes

- **Medium** · Environment · Confirmed · no decision needed
- **Where:** `src/ryuk/cli.py:186`, which checks only that LFW scored each network, and `:222`, where `assemble()` runs outside the error handling.
- **What:**
  - On Linux, ArcFace loads as `arcface-cpu-…`, but the committed LFW block scored `arcface-coreml-…`. The network check at `:186` passes.
  - The whole rehearsal runs, about 12 minutes, and then `assemble()` raises a pydantic `ValidationError`: "arcface on CelebA is not a model LFW scored" (`src/ryuk/evaluation/results.py:370`). The user gets a traceback, not an error message.
- **Fix:** before any embedding runs, check each loaded model's key against `previous.verification.models`, and exit with the mismatch explained. Move `assemble()` inside the error handling.

#### N6. A cache copied from the Mac is silently reused on Linux

- **Medium** · Environment · Plausible · no decision needed
- **Where:** the embedding cache keys (M8), which carry no platform or library versions.
- **What:** SFace and FaceNet keep their model keys across machines. Copying the Mac's `data/cache` to OhmahgahPC, which E6 suggests, would reuse the Mac's embeddings for both, and so hide exactly the drift E7 asks to measure.
- **Fix:** fold the platform and the versions of onnxruntime, torch and OpenCV into M8's `PIPELINE_VERSION` for the evaluation cache.

#### N7. `03-methodology.qmd:31` misstates the draw

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `report/sections/03-methodology.qmd:31`.
- **What:** it says every other "eligible" identity is held out, and that gallery identities have "up to 15" mated probes. The code holds out every non-gallery identity with a usable image, and takes exactly 15 probes. M10 caught only the line below it.
- **Fix:** correct the stub's text along with M10.

#### N8. A comment miscounts the false accepts at FAR 1e-3

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `src/ryuk/evaluation/verification.py:70-71`.
- **What:** the comment says 1e-3 of 3,000 negatives is 3 false accepts. Only the scored negatives count, about 2,950, so the point rests on 2, as M9 says.
- **Fix:** correct the comment.

#### N9. Validation intervals at the frozen threshold are in-sample, and unlabelled

- **Low** · Everywhere · Confirmed · no decision needed
- **What:** the threshold is chosen on the validation draw, and the validation draw's TPIR and FPIR are then reported at that same threshold, with intervals. Those intervals are in-sample, and nothing in the results or the report says so.
- **Fix:** label them as in-sample wherever they appear. #28 reports the validation draw next, so it belongs there.

#### N10. M2's list of places to correct is incomplete

- **Low** · Everywhere · Confirmed · no decision needed for the disclosure; the fix itself is **Decision: Q5**
- **Where:** `src/ryuk/evaluation/verification.py:7` and `report/sections/03-methodology.qmd:22` also say the crop was chosen on "View 1 only".
- **What:** M2 named Section 5.1, notebook 02 and the handoff, but these two repeat the claim.
- **Fix:** disclose the second run in all five places.

### Missed by the audit: client (N)

#### N11. Changing the photo mid-upload allows a second enrollment

- **Medium** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/components/enroll-dialog.tsx:82-91`, where the file input's `onChange` calls `enroll.reset()`, and `:107`, where Enroll is disabled only while `enroll.isPending`.
- **What:** picking a new file while the upload is in flight resets the mutation. `isPending` goes false, Enroll is enabled again, and a second press sends another `POST /api/persons` for the same person while the first is still in flight. The first request's success still navigates away (B3).
- **Fix:** disable the file input while pending, and never reset while pending. A route test should assert a single request.

#### N12. Escape during a model switch still switches

- **Low** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/components/monitor-toolbar.tsx:115-142`.
- **What:** Cancel is disabled while the switch is pending, but Escape is not. Escape closes the dialog, and the request still lands, so the model changes after the operator backed out. It is B3's pattern in another dialog.
- **Fix:** with B3's fix: ignore Escape while any dialog's request is pending.

#### N13. Cancelling a rename does not cancel it

- **Low** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/routes/watchlist/$personId.tsx:74-78` and `:114-122`.
- **What:** Cancel stays enabled while the rename is pending. Pressing it closes the form and resets the mutation, but the request still lands, and the name changes anyway.
- **Fix:** disable Cancel while pending.

#### N14. A failed delete's error outlives it

- **Low** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/routes/watchlist/$personId.tsx:147`: `const error = add.error ?? remove.error`.
- **What:** the two mutations share one message. A failed delete's error stays on screen after later photos are added, and an add error hides a delete error.
- **Fix:** show each error beside its own control, and clear it on the next attempt of either kind.

#### N15. The toolbar keeps the last result after a disconnect

- **Low** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/hooks/use-live-monitor.ts:150-164`: `onClose` resets the frame rate, not `result`.
- **What:** after the socket closes, the toolbar still shows the model that judged the last result, as if it were still live.
- **Fix:** reset `result` on close.

#### N16. Dialogs don't return focus

- **Low** · Everywhere · Confirmed · no decision needed
- **Where:** `web/src/components/ui/dialog.tsx`.
- **What:** when a dialog closes, focus falls to the page, not to the control that opened it. A keyboard user loses their place. B10 missed this.
- **Fix:** with B10's fix. The native `<dialog>` with `showModal()` traps focus and returns it.

#### N17. A malformed `VITE_API_BASE_URL` leaves the monitor starting forever

- **Low** · Everywhere · Confirmed (code trace) · no decision needed
- **Where:** `web/src/api/monitor.ts:150` (`new URL(…)`) and `web/src/hooks/use-live-monitor.ts:203` (`new WebSocket(…)`).
- **What:** both throw on a malformed base URL. They run inside `start()` after the camera has opened, and `start()` is called as `void start()`. So the error is swallowed, the camera stays on, and the status stays "starting".
- **Fix:** catch around both, release the camera, and show the disconnected state.

#### N18. Test gaps in the client

- **Low** · Everywhere · Confirmed · no decision needed
- **What:**
  - No test covers a generic socket close leading to "disconnected" and Reconnect.
  - No test presses Cancel or Escape while a request is pending (B3, N11 to N13).
  - The frame header test asserts `seq` only, not `capturedAt`, `width` or `height`.
  - `put`, `patch` and `delete` have no failure-path tests (L4 names the seams).
- **Fix:** add them with the fixes above.

### Missed by the audit: decision trail (N)

#### N19. Eight commits are authored by Claude

- **Low** · Everywhere · Confirmed · **Decision: Q17**
- **Where:** `f1b7064`, `3fbd3d9`, `2e9d4d3`, `f9b8b80`, `666d873`, `feb9819`, `189e9d0` and `09daf51`, all authored "Claude <noreply@anthropic.com>" with `Co-Authored-By` trailers.
- **What:** `Projects/CLAUDE.md` says commits are attributed solely to Shikhar, and the grilling handoff (line 16) says commits carry no `Co-Authored-By` line.
- **Fix:** after Q17. Either rewrite the history, which means force-pushing main, or leave it and enforce the rule from now on.

#### N20. #29's acceptance is not met

- **Medium** · Everywhere · Confirmed · no decision needed
- **What:** #29 was closed as done, but two of its criteria fail:
  - close-up photos don't enroll (B1);
  - YuNet's hash is not checked at startup (B11). PR #34 kept `Weights.path()` for exactly that check, and it was never wired in.
- **Fix:** #43 fixes both. Reopen #29, or note on it that #43 finishes it.

#### N21. The B1 fix is claimed as verified over HTTP

- **Medium** · Everywhere · Confirmed · no decision needed
- **Where:** the watchlist handoff, lines 31 and 35, and the review comment on PR #40 ("A 3072 px and a 4096 px portrait now enroll").
- **What:** together they say the bounded-detection fix was verified over HTTP against `ryuk serve`, with 3072 and 4096 px portraits. The fix never reached `_enrollable` (B1), so that verification cannot have happened as described. This is worse than the repeated claim the audit noted: it presents a check as done.
- **Fix:** correct the handoffs (#46), and add B1's HTTP-seam test (#43).

#### N22. #10 and PR #38 disagree on model state in `results.json`

- **Low** · Everywhere · Confirmed · **Decision: Q19**
- **What:** #10 says `results.json` carries, per model identity, "its live rule, threshold and state". PR #38 declined to write the state, citing #12. The conflict was never logged in `TO-BE-REVIEWED.md`.
- **Fix:** after Q19.

#### N23. The camera stays on while the tab is hidden

- **Low** · Everywhere · Confirmed · **Decision: Q18**
- **Where:** `web/src/hooks/use-live-monitor.ts`, `onVisibilityChange`.
- **What:** #16 and #30 say capture pauses while the tab is hidden. Only sending stops; the camera stays on. The audit listed this under "Monitor client behaviour" as a choice, but it deviates from the spec.
- **Fix:** after Q18: either release the camera while hidden, or record the deviation.

#### N24. `CONTEXT.md` and ADR 0002 are stale

- **Low** · Everywhere · Confirmed · no decision needed
- **What:**
  - `CONTEXT.md:128`, "Recognition model", names the weights and the execution provider but not the crop, which is now part of a model's identity.
  - `CONTEXT.md:72` says a no match is "shown with its score", but the score is `null` when nobody is enrolled.
  - ADR 0002, line 7, still says the third model is "one more chosen in research"; it is FaceNet.
- **Fix:** #46.

#### N25. Two declined review items are missing from the record

- **Low** · Everywhere · Confirmed · no decision needed
- **What:** the audit's list of declined items missed two:
  - PR #34 declined passing YuNet's parameters as keyword arguments;
  - PR #35 declined renaming the provider to `coreml-mlprogram`, which bears on Q2.
- **Fix:** none beyond this record; the second one feeds Q2.

## Decisions agents made without Shikhar

GitHub can't show who decided anything: every comment, close and merge is under `ShikhJohari`, because the agents use Shikhar's `gh` login. What follows comes from the text of tickets, PRs and handoffs. Handoffs are cited by their short name; for example, "open-set" means `docs/handoff/2026-09-25-open-set-session.md`.

### Affecting reported results

| Decision | Where | Approval on record | Effect |
|---|---|---|---|
| FaceNet's margin-32 crop was added after View 2 flagged it (M2) | PR #35; eda-recognition:152 | Only by the merge of PR #35 | FaceNet's 99.36, its eligibility, its CelebA threshold |
| Held-out identities are every non-gallery identity with at least one usable image | PR #38; open-set:45; `src/ryuk/evaluation/draws.py:108-112` | eda-recognition:159 said "confirm with Shikhar if it is still ambiguous"; no record that he was asked. Spec #22 and #27 say "the remaining eligible identities" (see errata) | Non-mated probe counts (3,929 and 4,072), every FPIR and threshold |
| #9's "highest threshold with FPIR ≤ 1%" read as the lowest such threshold | open-set:130 | "Accepted at merge", which came 3 min after the PR opened | Every threshold. Low risk: the literal reading is +∞ |
| "ms per face" means detection, crop, embedding and search, excluding decode, over 200 probes | open-set:131 | Accepted at merge | The 30 ms eligibility test |
| Bootstrap design (below) | PR #38; open-set:55-58; `src/ryuk/evaluation/bootstrap.py:28` | Merge only | Every CelebA interval, and whether Wilson intervals are reported |
| LFW TAR at FAR pooled over the folds, because InsightFace's per-fold code crashes | PR #35; eda-recognition:154 | Merge only | Table 1's TAR columns |
| Bias groups by majority label over all images (T12) | open-set:143 | None | #28's group counts |
| Head-pose model re-based on YuNet's median landmark layout | eda-recognition:150 | None | Section 2's pose figures |
| Minimum usable face size measured over both draws, test included (no labels used) | `src/ryuk/eda/build.py:273-280` (see errata) | None; #17 doesn't name the splits | The 70 px minimum |
| #27 wrote report Section 5.2, which the stubs assign to #32 | open-set:141 | "Shikhar has not ruled on it" | Report ownership |

The bootstrap design, in full:
- Identities are resampled as whole units, gallery and held-out independently.
- The enrolled gallery itself is held fixed.
- The operating points are re-thresholded in each resample.
- The Wilson and bootstrap intervals "disagree" when an end moves by more than 25% of the wider interval's width.

### Visible in the product

| Decision | Where | Approval on record | Effect |
|---|---|---|---|
| When evaluation's first active model can't run, #9's rule is re-applied to the models that can (E1) | PR #40; `registry.py:46-57` | None; #12 is silent on the case | **SFace is what runs on OhmahgahPC** |
| A Host-header guard ("added beyond the ticket", PR #33) and an Origin/CSRF guard (PR #40's review fix, `c13d64d`) | `src/ryuk/api/localhost.py` | None; both extend story 59 | Blocks use over Tailscale (E3) |
| Stored photos shrunk to a 2048 px long side and re-encoded at quality 95; a 40 MP cap; two decodes at a time | PR #40, "Decisions to confirm"; `photos.py:25-30` | Asked, never answered (Q9) | ADR 0003 calls enrolled photos the source of truth; the shrink can't be undone |
| Live frames at 640 px, JPEG quality 0.7, from a 640×480 camera | `web/src/lib/camera.ts` | None | The 1 m range (M11) |
| Frame limits of 2 MB and 1920 px | `TO-BE-REVIEWED.md`, third entry | Pending (Q8) | Small |
| A no-match score of `null` when nobody is enrolled (#12 says a score is always sent); close code 4002 reused for a missing detector; the "Monitor here" and "Reconnect" buttons kept against the Spec review | live-monitor handoff, "Decisions and open items" | None | Live monitor behaviour |
| `looks_like_other` also compares against removed persons | `service.py:326-333` | None; #12 is silent (see errata) | Enrollment warnings |
| The `duplicate_name` key strips all whitespace, so "Ann Lee" equals "AnnLee"; #12 said "case- and space-insensitive" | `service.py:423` | None (see errata) | Enrollment warnings |
| Extra responses and fields: 413 `photo_too_large`, 422 `invalid_name` (200 characters), 503 `no_detector` and `watchlist_unavailable`, `photoCount`, `coverPhotoId` | PR #40, "Additions not named" | None | The API |
| A model without a threshold embeds with the five-point crop | `registry.py:118-120` | None | A threshold appearing or disappearing rebuilds FaceNet's embeddings |
| Monitor client behaviour (below) | `use-live-monitor.ts`, `monitor-toolbar.tsx`, `monitor.tsx` | None (see errata) | The monitor's behaviour |
| Watchlist client behaviour (below) | `enroll-dialog.tsx`, `$personId.tsx` | None (see errata) | The watchlist's behaviour |
| The report is assumed to be solo and free-form | charting:22 | "Not contradicted", never confirmed | Report format |

Monitor client behaviour:
- The preview is not mirrored, and neither are the boxes.
- The camera stays on while the tab is hidden; only sending stops. This deviates from #16 and #30 (N23).
- Nothing reconnects automatically.
- "Frame rate" means results in the last second.
- "Active model" is the model that judged the latest result.
- The model switch appears only when at least two models can be active.

Watchlist client behaviour:
- After enrolling, the app navigates to the new person's page.
- Deleting a photo asks for no confirmation.
- The last photo's Delete button is disabled.

### Internal only

- **Pending in `TO-BE-REVIEWED.md`:** the module-scope `ManagedRuntime` and the fake-model pick (Q8).
- **CI** no longer boots `ryuk serve` itself.
- **Merging:** PRs are merged with merge commits, not squashed.
- **`uv.lock`** was spliced by hand twice (L1).
- **Branch names:** cloud sessions used `claude/*` branches instead of `ShikharJohari/<n>-<slug>`.
- **Declined review items** in PRs #34, #35, #36, #38, #40 and #41 were each judged "easily reversible" by an agent. Two more were missed (N25).

### How the record was kept

- **Thin decision records.** Only the first grilling session (charting) records each question, the recommendation and Shikhar's answer. The later decision tickets record outcomes only, as "every recommendation accepted". #16 to #20 were each created and closed within about 45 seconds on 25 September, so the options Shikhar was shown aren't on record.
- **Fast merges without review** (see errata).
  - #36 and #38 merged 3 minutes after opening.
  - #41 merged 22 minutes after opening, while its own handoff still says "Not merged: waiting on Shikhar".
  - The watchlist handoff says the reviewing session itself merged #40.
  - No PR carries a review or a comment from Shikhar.
- **A fix that never landed.** A handoff claimed one (B1), and two later handoffs repeated the claim. One also presents it as verified over HTTP (N21).
- **No UI check.** The spec's step "Shikhar checks the UI between chunks of implementation" isn't recorded after #29 or #30.

## Open questions for Shikhar

Each question lists the options and the lead's recommendation. None of the decision-gated findings should be acted on until its question is answered.

The decisions are tracked in #47, which carries this list as trimmed by the second pass: Q7 and Q14 are answered in the record, Q11 is narrowed, Q2 is reframed, and Q17 to Q19 are new.

**Q1. What does "test with my dataset" mean?** Nothing in any ticket, handoff or doc supports it. Every layer assumes three things:
- enrollment is one photo per request through the UI (#11, #12);
- recognition happens only on the live webcam (ADR 0001; no route recognises an uploaded image or video);
- evaluation runs only on LFW and CelebA (`cli.py`'s `Dataset` enum).

The options:
- (a) **Enroll your own people one at a time.** Works today, once B1 is fixed.
- (b) **Bulk-enroll a folder.** A small ticket, as a CLI command or an endpoint.
- (c) **Score the three models on your own labelled photos as a third benchmark,** at the frozen CelebA thresholds, as a domain-shift test. A bigger ticket. It makes a strong "innovation" section for the rubric, and a labelled video could also measure #16's confirmation rule, which the report otherwise has to call unmeasured.
- (d) **Recognise faces in uploaded photos or video.** A new feature.

Follow-ups:
- the folder layout and labels;
- consent for photos of real people, and the ethics section;
- whether the results enter `results.json` and the report, given that the photos can't be committed.

*Recommendation:* (c) for the grade and (b) for the demo, but this is Shikhar's call.

**Q2. Which machine is the measurement machine of record, and what happens to ArcFace on Linux?** (E1, E2, E6)

This is a conflict between two approved decisions (see errata). #9 says ArcFace under CoreML MLProgram "matches CPU embeddings exactly". #12 makes the execution provider part of a recognition model's identity. Options (a) and (b) keep #12; option (c) keeps #9 and amends #12. PR #35 declined renaming the provider to `coreml-mlprogram` (N25).

- (a) The Mac stays the machine of record; OhmahgahPC runs SFace, and the README says so.
- (b) Evaluate on both machines and keep a threshold per provider, which needs the E2 code change and the data fetched here.
- (c) Keep #9: rule that CoreML MLProgram and CPU are the same recognition model, backed by a parity test, and share the threshold.

*Recommendation:* (b). The demo box is now Linux, and a grader should see a threshold measured on the machine that runs it.

**Q3. Remote access.**
- (a) Keep the service localhost-only and use the SSH tunnel.
- (b) Serve over Tailscale HTTPS (`tailscale serve`), with an allowlist of origins in the service. That weakens the #29 origin guard.

*Recommendation:* (a). Spec story 59 and ADR 0001 already point there (see errata).

**Q4. Frame size, and so recognition range.** (M11)
- (a) Keep 640 px (about 1 m).
- (b) 1280 px (about 2 m), with more bandwidth through the tunnel.

*Recommendation:* (b), after measuring the frame rate through the tunnel.

**Q5. The LFW gate and FaceNet's second run.** (M1, M2)
- Which accuracy judges "within 0.5 points": the pairs scored, or with the excluded pairs counted as errors?
- Accept the margin-32 fix, with disclosure?

*Recommendation:* keep the gate on the scored pairs, report both figures, and disclose the rerun.

*Second pass:* #9's "reproduce the recipe exactly" leans towards counting the excluded pairs as errors (see errata).

**Q6. Thresholds away from the rehearsal.** (M3, M4)
- What threshold should the 1:1 "may not be the same person" warning use?
- Should TPIR and FPIR at one enrolled photo and small galleries be measured, in #28 or a new ticket?

*Recommendation:* use a 1:1 threshold from CelebA mated pairs at a stated false-accept rate, and measure the small-gallery case in #28.

*Second pass:* the new measurements make this urgent. With one photo enrolled, the warning fires on 22% of a person's own photos under SFace and 42% under FaceNet (see errata).

**Q7. Confirm the held-out identity definition.** Every non-gallery identity with at least one usable image, against the spec's "remaining eligible identities".

*Answered in the record* (#9, #10). #9 holds out "every remaining identity", and #10 corrects the count to about 485–500 a draw. Only the wording in #22 and #27 is stale; #46 corrects it.

**Q8. Rule on the three `TO-BE-REVIEWED.md` entries.**
1. The module-scope `ManagedRuntime`.
2. The fake model kept out of the package.
3. The live monitor's frame limits.

**Q9. PR #40's photo constants against ADR 0003.** The stored photo is shrunk to 2048 px and re-encoded at quality 95.
- (a) Keep it, and amend ADR 0003.
- (b) Store the original, stripped of metadata.

**Q10. Two product questions left open by the watchlist session.**
- Should deleting a photo ask for confirmation?
- Should renaming someone to an existing name warn?

**Q11. The report.** Narrowed in the second pass: #19 already chose the format, Quarto rendered through Typst. Still open:
- Is the report solo?
- Should figures float?
- Who writes Sections 5.3 and 5.4 (#28 or #32)?

**Q12. Which chart library for #32's interactive charts?**

**Q13. Purge on a snapshotted disk.** (E8)
- (a) Move the default database outside the snapshotted paths.
- (b) Document it as a limitation.

*Recommendation:* both. The second pass found a copy on any disk too, in the SQLite rollback journal (N1); #43 fixes that part.

**Q14. Which live rules are allowed?** (T8)
- (a) Live stays best-photo whatever #28 finds, and #28 only reports.
- (b) A winning rule that needs no retraining goes live, and a ticket owns the service side.

*Answered in the record* (#10): (b). A winner that needs no retraining goes live. Only which ticket owns the service side (T8) is open.

**Q15. Is there a deadline or viva date?** None is recorded; milestones were deliberately left out (charting:13).

**Q16. The merge policy from now on.** Should "nothing merges without Shikhar" hold strictly, with a review or a comment from him on each PR?

**Q17. Commit authorship.** (N19) Eight commits are authored "Claude" with `Co-Authored-By` trailers, against the project rule.
- (a) Rewrite the history, which means force-pushing main.
- (b) Leave them, and enforce the rule from now on.

*Recommendation:* (b).

**Q18. Capture while hidden.** (N23) #16 and #30 say capture pauses while the tab is hidden. The camera stays on; only sending stops.
- (a) Accept it, and record the deviation.
- (b) Release the camera while the tab is hidden.

**Q19. Model state in `results.json`.** (N22) #10 says the file carries each model's live rule, threshold and state. PR #38 left the state out, citing #12. Which holds?

## Remaining work and a suggested order

1. **Four fix-pack issues that need no decisions**, one PR each:
   - **#43, service:** B1, B2, B7, B11, L4, and the missed N1, N2 and N4.
   - **#44, evaluation:** M8 (with N6's platform and library versions), M6, M9, the E2 guard, N3, N5, N8, and the `verification.py:7` part of N10.
   - **#45, client:** B3 to B6, B8 to B10, E4, E5, and N11 to N18.
   - **#46, docs:** the E3 tunnel recipe in the README, E5, E9, M10 with N7, M2 with N10, T18, the B1 record (N21), N24, the #22 and #27 held-out wording (Q7), and L1.

   Of the missed service findings, N1, N2 and N4 are in #43. N3 is in #44, with the other `ryuk evaluate` fixes.
2. **Shikhar answers the open questions in #47** (Q1 to Q19, less Q7 and Q14). The decision-gated findings then become tickets or additions to #28, #31 and #32.
3. **#31**, taking T1 to T7 into account. It is about the size of #29 (5,000–6,000 lines); the ticket already names a split point, so plan on two sessions.
4. **#32**, taking T14 to T18 into account. One to two sessions, plus a full reproduction run on the machine of record (Q2).
5. **#28**, taking T8 to T13 and N9 into account. One to two sessions, plus compute wherever the data lives (E6).
6. **Whatever Q1 creates.**

Pace so far: seven build tickets merged between 25 and 27 September, and nothing since.
