# Live monitor session, 27 September 2026

The record of the eleventh working session, the sixth build session. A cloud session implemented #30 (live monitor) with `/implement`, reviewed it with `/code-review` in two sub-agents (Standards and Spec), fixed what they found and opened the PR. Read the watchlist session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #30 is implemented on `claude/pr-review-launch-t54x7v` (the cloud session's branch, not `ShikharJohari/30-live-monitor`), in a PR that says `Closes #30`. **Not merged**: waiting on Shikhar.
- The Playwright smoke has not run yet: it only runs in CI. It is the first CI job to use Chrome's fake camera; if it fails, look there first (see "Things to watch").
- **#31 (sightings) is unblocked once this merges.** #28 is still open and independent.
- No schema change: the live monitor needs no new table. `0001_watchlist` is still the only migration.

## What exists now

- **Live recognition** (`ryuk.watchlist.live`). `Gallery` holds the active model's embeddings of every person on the watchlist as one float32 matrix, a row per enrolled photo with each person's rows together. `Gallery.top_candidate(probe, rule)` dispatches on the threshold's `MatchRule`: `best-photo` is `np.maximum.reduceat` over the person's rows. `recognise(detector, active, loaded, gallery, frame)` boxes every detection: too small (`is_usable` is false) is `TooSmall` and never embedded; a usable face is `Match` at or above the threshold, else `NoMatch`, whose score is None when nobody is on the watchlist.
- **Watchlist** (`ryuk.watchlist.service`). `Evaluated` gained `rule`. The gallery loads at startup and after every change: `enroll`, `add_photo`, `delete_photo` and `rename` run inside `Watchlist._change()`, which takes the lock, commits, then reloads. `recognise(frame)` runs under the same lock as enrollment, since the detector and models are not thread-safe. `activate(model_id)` writes the `setting` row, switches the registry and reloads the gallery; `404 not_found` for an unknown key, `409 cannot_be_active` for a model that is unavailable or not evaluated. `can_monitor` is false without the detector or an active model.
- **Frames** (`ryuk.api.frames`). `parse_frame` checks #5's 17-byte header on the event loop (type, size, at most 2 MB and 1920 px a side); `decode_frame` decodes the JPEG on the worker thread and refuses one whose size differs from the header. Codes: `invalid_frame`, `frame_too_large`.
- **The socket** (`ryuk.api.monitor`). `/api/monitor` accepts, then closes with 4002 when `can_monitor` is false. `LiveMonitor` (on `app.state.monitor`) keeps the one connection; a new one closes the old with 4001. Each `_Connection` runs a receive loop that parses frames into a one-slot buffer (the latest wins) and one worker that recognises the waiting frame with `run_in_threadpool`; all sends go through one lock. Messages: `result` (`seq`, `capturedAt`, `width`, `height`, `modelKey`, `threshold`, `faces` by `outcome`), `active_model_changed`, `error` (the socket stays open). `MonitorMessage` is a discriminated `RootModel`, merged into `openapi.json`.
- **API.** `PUT /api/active-model {modelKey}` answers the model's `RecognitionModelInfo` and announces `active_model_changed` to the live monitor.
- **Client.** `/monitor` loads `GET /api/models`; with no active model it shows why and never opens the camera. Otherwise `useLiveMonitor` opens the camera (640x480 ideal), then the socket, and sends the newest frame (long side at most 640 px, JPEG 0.7) as soon as the previous result or error arrives, at most 30 a second, and nothing while the tab is hidden. `FaceOverlay` places boxes as fractions of the frame on a stage with the frame's aspect ratio: match solid in `match-video` with the name and score, no match dashed in `no-match-video` with its score only, too small thin and unlabelled. `MonitorToolbar` shows the active model, threshold and results in the last second, and a select of evaluated models behind a confirm dialog. States: starting, no camera (denied, missing, busy, unsupported), no active model (4002), taken over (4001, "Monitor here" takes it back), connection lost. `ApiClient` gained `put`. The Vite proxy carries the socket (`ws: true`).
- **Tests.** `tests/test_live_monitor.py` drives the socket in-process with the real YuNet and fakes: every outcome, the best-photo rule, the gallery reload, stale frames dropped (a gated fake holds inference while frames 2 to 4 arrive; an invalid message is the ordering barrier), 4001, 4002 (no evaluated model, no detector, no watchlist), every frame error, and a switch taking effect on the next frame. `test_watchlist_models.py` covers the switch being kept across restarts and refused for unknown, unavailable and not-evaluated models. `web/src/monitor.test.tsx` renders the route against MSW's `ws.link` and a fake camera at the browser boundary (`src/test/camera.ts`). The e2e smoke enrolls `web/e2e/astronaut.mjpeg`, which Chrome's fake camera plays, and expects a named match; it runs against `tests/e2e_service.py`, the real service with the fixtures' YuNet and a fake standing in for SFace, on a temporary database.

## Patterns later tickets should copy

- **Everything in the earlier lists still holds.**
- **Change the watchlist inside `_change()`.** Anything that alters who is on the watchlist, their photos or their names must go through it, or the live monitor keeps matching against the old gallery. #31's removal, restore and purge belong there.
- **Parse on the loop, decode off it.** The receive loop only reads the header and answers errors in order; pixels are decoded on the worker thread with the frame.
- **Test concurrency with a gate and a barrier.** A fake whose `embed` waits on a `threading.Event` holds inference; an invalid message sent afterwards is answered by the receive loop in order, which proves every earlier frame has arrived. No sleeps.
- **Let the test answer the socket.** `mockMonitor(server)` records each frame's sequence number and lets the test `send` and `close` when it chooses; an auto-replying mock races the assertions.
- **Stub browser APIs, not modules.** jsdom has no camera, playback or canvas encoding; `fakeCamera()` stubs `getUserMedia`, `play`, the video size and `toBlob`, and restores them when the test ends.

## Things that bit

- **uvicorn serves no WebSocket without a library.** Neither `websockets` nor `wsproto` was installed; `websockets` is now a dependency. `uv.lock` was spliced by hand again (the PyTorch index is unreachable from the cloud container) and checked with `uv lock --check`.
- **Starlette's TestClient opens a socket as `testserver`**, which the localhost guard closes with 1008: connect to `ws://127.0.0.1/api/monitor`.
- **TestClient's `receive_json` has no timeout**: a test waiting for a message the service never sends hangs the run.
- **The fake recognition model is scale-sensitive.** A face detected on the 640 px bounded copy of an 800 px frame scored 0.89 against its own enrolled photo. Keep test frames at most 640 px, and enroll at the size the frame shows.
- **jsdom's `DOMException` is not an `Error`** there, so camera failures are told apart by `name`.
- **A `pkill -f` pattern matches its own shell**; bracket one letter (`[e]2e_service.py`).

## Things to watch

- **Chrome's fake camera with an MJPEG file** (`--use-file-for-fake-video-capture=…/astronaut.mjpeg`) is untried here. If the smoke fails at the match, check the trace for the camera state first; a `.y4m` file is the fallback format.
- **Enrollment holds the lock for its whole transaction**, so live frames wait while a photo is enrolled (tens of ms with the fake, more with FaceNet).

## Decisions and open items

- **`NoMatch.score` is None with nobody on the watchlist**, rather than inventing a score. The client shows the dashed box without a label.
- **4002 also covers a missing detector.** The client's copy names both remedies (`ryuk weights fetch`).
- **The e2e service is test code.** The fake never enters `ryuk serve` (the #26 pick in `TO-BE-REVIEWED.md`).
- **Playwright runs one worker**, since a second live monitor takes over the first.
- **Open**, not ticketed: the handoff's earlier client items still stand; the model switch's confirm does not mention sightings, which do not exist yet (#31 should add "open sightings will end").

## Resuming

Start a fresh session with `/implement` on #31 once this PR merges. Working rules as before: one ticket per session; a `ShikharJohari/<n>-<slug>` branch; a PR that says `Closes #N`; nothing merged without Shikhar; role agents only, never Haiku; never Playwright locally; verify against `ryuk serve` with and without weights; write the handoff before merge.

**#31 builds on:**
- `_Connection._recognise` is where a frame's `Recognition` becomes a `result`; confirmation and sightings hook in after it, with the frame's image still at hand for the best crop.
- `live.Match` carries the person and score; the runner-up needs `Gallery.top_candidate` to return the second-ranked person too.
- `LiveMonitor.announce` sends to the running connection; `sighting_*` messages go the same way, and `ActiveModelChanged` is where switching ends open sightings.
- A closed socket ends `_Connection.run`; that is where open sightings end at their last-seen time.
