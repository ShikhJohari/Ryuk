# Watchlist session, 26 September 2026

The record of the tenth working session, the fifth build session. A cloud session implemented #29 (watchlist management) and opened PR #40; a second, local session reviewed that PR against the #29 brief, fixed the seven defects the review found, wrote this record and merged it. Read the open-set session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #29 is closed by PR #40 (branch `claude/pipeline-issue-impl-1kpx7e`, not the `ShikharJohari/29-watchlist` the brief asked for; the cloud session named it).
- **#30 (live monitor) is unblocked.** #28 (learning on embeddings and bias breakdown) is still open and independent; the two can run in parallel.
- Still blocked: #31 on #30, #32 on #28 and #31.
- The schema has now shipped: `0001_watchlist` is the first migration and may no longer be edited in place. A `data/ryuk.sqlite3` made from a commit of PR #40 before `f79d24b` lacks the `crop` column and must be deleted.

## What exists now

- **Service wiring.** `create_app(start_watchlist)` runs the factory in a lifespan and keeps the result on `app.state.watchlist`; routes take `WatchlistDep`, which answers `503 watchlist_unavailable` when there is none (`ryuk openapi`, the shell tests). `ryuk serve` passes `open_watchlist(settings)`. Settings gained `database` (`data/ryuk.sqlite3`, never `cache_dir`) and `results` (`evaluation/results.json`).
- **Persistence** (`ryuk.watchlist.database`, `tables`, `ryuk.migrations`). SQLite through SQLAlchemy 2.1 with Alembic as the only schema path; the service runs `upgrade head` at startup. Every pooled connection has `foreign_keys` and `secure_delete` on. The engine follows SQLAlchemy's pysqlite recipe (no driver-side transactions, an explicit `BEGIN`), so DDL rolls back with its transaction. `migration_transaction(engine)` runs a migration with foreign keys off and a `foreign_key_check` before commit; `env.py` renders batch operations. Tables: `person_of_interest`, `enrolled_photo` (image BLOB, media type, size, face box, landmarks and score), `embedding` (photo, model key) with a float32 BLOB, `recognition_model` (key, network, sha256, provider, dim, **crop**), `setting`.
- **Model registry** (`ryuk.watchlist.registry`). `Evaluation.from_results` keeps a threshold only for a rule in `LIVE_RULES` (`best-photo` today); `register` keys every model by (network, sha256, provider), the unavailable ones under their pinned hash; `ModelRegistry.state` gives `active`, `available`, `not_evaluated` or `unavailable`. `Evaluation.first_active_for(runnable)` re-applies #9's rule to the models this machine can run when evaluation's choice cannot (ArcFace on CoreML is absent on Linux, so SFace is active in CI).
- **Startup** (`ryuk.watchlist.service.start_watchlist`). For every loaded model, embeddings recorded under other weights or another crop are dropped and rebuilt from the stored photos, reusing the stored detection, and missing ones are filled in; all before the first request. Only evaluation's own first active model is ever persisted; a fallback is active for one run. A persisted choice that cannot run now is kept.
- **Enrollment** (`ryuk.watchlist.photos`, `service`). `prepare_photo`: JPEG, PNG or WebP by decoded content, at most 10 MB and 40 megapixels, orientation applied, stored with the long side at most 2048 px, re-encoded from a fresh image so no EXIF, GPS, ICC, XMP or comment survives; two decodes at a time. Detection runs on a copy bounded to `PHOTO_DETECTION_SIDE` (640 px) with the box and landmarks scaled back. `enrollable_face`: no detection is `no_face`, none usable is `face_too_small`, more than one usable is `multiple_faces`; smaller faces are ignored. Every model with weights present embeds the photo, under the crop its threshold was measured with. Warnings (`duplicate_name` on an NFKC, casefolded, whitespace-free name key that counts removed persons; `looks_like_other` and `may_not_be_same_person` under the active model only, skipped when none can be active) are computed before any write and confirmed by resending with every raised code in `acknowledgedWarnings`; unraised codes are ignored. Everything that reads then writes runs under one `threading.Lock`, which also covers the detector and the models, none of which are thread-safe.
  - **Correction, 28 September 2026 (#46):** the bounded detection never reached enrollment. Commit `c1f17fc` gave `Detector.detect` a `max_side` (the bound is now `MAX_DETECTION_SIDE` in `ryuk.detector`), but only its import reflow landed in `service.py`: `_enrollable` still detects on the full stored photo, so a face much over 400 px is `no_face`. The fix is in #43.
- **API.** `GET /api/models`; `GET /api/persons?status=` (default `on_watchlist`), `POST /api/persons` (multipart `name`, `photo`, `acknowledgedWarnings`), `GET`/`PATCH /api/persons/{id}` (name only; extra fields are 422), `POST .../photos`, `GET .../photos/{pid}/image` (`cache-control: no-store`), `DELETE .../photos/{pid}` (`409 last_photo`). Problems: 422 `no_face`, `multiple_faces`, `face_too_small`, `unsupported_image`, `invalid_name`; 409 `warnings` with a `warnings` list, and `last_photo`; 413 `photo_too_large` (bytes or megapixels); 503 `no_detector`, `watchlist_unavailable`; 403 `cross_origin`.
- **Middleware** (`ryuk.api.localhost`, `uploads`). The Host guard now also refuses any change or socket handshake whose Origin is not a loopback host (`403 cross_origin`; `Origin: null` included). `PhotoUploadLimitMiddleware` refuses a POST to the two upload routes from its Content-Length, or cuts a chunked body off once past the ceiling, before the multipart parser spools it to disk.
- **Client.** `ApiClient` has `get`, `postForm`, `patch` and `delete`; `ApiProblem` carries `warnings`. `/watchlist?status=` (the filter is validated in the URL, the default left out), the enroll dialog, the warnings dialog (`useAcknowledgedMutation` resends the same `File` with every code), `/watchlist/$personId` with photos as figures, add, delete (disabled for the last one) and rename. Route tests with MSW in `watchlist.test.tsx` and `person-of-interest.test.tsx`; the test setup swaps jsdom's `Blob`, `File` and `FormData` for Node's, since Node's `fetch` cannot send jsdom's. The e2e smoke also renders `/watchlist` against a service with no weights.
- **Tests.** Every rejection and warning through the HTTP seam with the real YuNet fixture and the fake model wrapped under a real network name in `tests/synthetic.py` (the registry refuses a `fake` network); the weights-change and crop-change rebuilds; the EXIF and GPS test on the stored bytes with a sideways upload; `compare_metadata` after `upgrade head` and a downgrade; the pragmas on every connection; the batch-rebuild cascade trap; cross-origin refusals; the body ceiling. 635 service tests at 96% coverage.

## Patterns later tickets should copy

- **Everything in the scaffold and open-set lists still holds.**
- **Startup does the work, requests stay simple.** Anything derived from stored data (embeddings, the active model) is brought up to date in `start_watchlist` before the lifespan yields, in one transaction, so no route has to cope with a stale row.
- **Identity is the whole key.** Whatever makes a stored derivative incomparable (weights, provider, crop) is part of the row that records it, and a mismatch means rebuild, never update in place. When a delete and an insert share a key, `flush()` between them, or SQLAlchemy turns the pair into an UPDATE and the database cascade never fires.
- **Refuse before you read.** Limits that protect memory or disk go in a pure-ASGI middleware: FastAPI parses the form before it resolves dependencies, so a route or dependency check is too late.
- **The service's own words in problems.** Codes are stable snake_case, details are one sentence in the glossary's terms, and the client maps `code` to copy, never `detail`.
- **Verify a fix over HTTP against `ryuk serve`, with and without weights**, not only in the test client: the review reproduced every defect that way before and after fixing it.
  - **Correction, 28 September 2026 (#46):** that cannot have held for the close-up enrollment fix, which never reached `service.py` (see Enrollment above; #43). A test through the HTTP seam with a large face would have caught it.

## Things that bit

- **YuNet misses faces over about 400 px on a side.** A close-up phone portrait at full resolution had no detections at all; every test image was under 700 px. Detection now runs on a bounded copy.
  - **Correction, 28 September 2026 (#46):** not for enrollment: the change never reached `service.py` (see Enrollment above; #43). Only live frames, added in #30, are detected on a bounded copy.
- **Python's sqlite3 runs DDL outside the transaction**, so a failed migration left a partial schema. SQLAlchemy's pysqlite recipe (`isolation_level = None` plus `BEGIN` on the `begin` event) fixes it; do not use `autocommit=False`, which opens a transaction at connect and turns the pragmas into no-ops.
- **A batch table rebuild with foreign keys on cascade-deletes everything beneath the table**, and `secure_delete` makes it unrecoverable. Foreign keys go off around every migration.
- **`warnings.catch_warnings` is not thread-safe**; used as a decompression-bomb guard it let a bomb through under two threads. An explicit pixel check after `Image.open` replaced it.
- **Pillow raises `SyntaxError` for a corrupt PNG chunk**, past the usual `OSError`/`ValueError` tuple; every decode failure is caught now.
- **Starlette spools any multipart part over 1 MB to a temp file before the route runs**, so a size check in the route cannot bound disk use.
- **Pydantic ignores unknown fields unless `extra="forbid"`**, so a PATCH with a field the model lacks succeeded silently.
- **`uv lock` needed the PyTorch index**, which the cloud container could not reach; the lock was spliced by hand and checked with `uv lock --check`. Run `uv lock` locally if it drifts.
- **Node's `fetch` cannot send jsdom's `FormData`**; the client test setup swaps the globals.

## Decisions and open items

- **Constants set in review, each one line to change:** stored long side 2048 px (`MAX_PHOTO_SIDE`), detection side 640 px (`PHOTO_DETECTION_SIDE`), 40 megapixels (`MAX_PHOTO_PIXELS`), two concurrent decodes, a 64 KiB multipart allowance over 10 MB.
  - **Correction, 28 September 2026 (#46):** the detection side is `MAX_DETECTION_SIDE` in `ryuk.detector`, and only live frames use it; enrollment does not yet (#43).
- **Declined in PR #40 and still standing:** the `Watchlist` class name; client-side restatements of service rules as affordances; `looks_like_other` comparing against removed persons too; the shared shape of the two face-warning checks.
- **Open from the review, not ticketed yet** (worth one issue each, or one client and one service issue): Escape and Cancel work while an upload is in flight, so a cancelled enrollment still completes; the dialog claims `aria-modal` without a focus trap; no ApiClient seam tests for `postForm`, `patch` and `delete`; `photoImageUrl` reads `import.meta.env` untyped beside the Config base URL; multipart and PATCH bodies are not pinned to the contract types; the e2e run's `web/data/ryuk.sqlite3` is not gitignored; names accept control and bidi characters; JSON responses carrying names have no `no-store`; `hide_parameters=True` on the engine, else a locked-database error logs a name and image bytes; the database file is created 0644; photo deletion has no confirm and renaming to a duplicate name gives no warning (product questions); ADR 0004 should say erasure is logical (the rollback journal holds a deleted photo until unlinked; APFS and Time Machine copies are out of purge's reach).
- **`TO-BE-REVIEWED.md` gained nothing.** The three review agents and the four fix agents did not conflict.

## Resuming

Start a fresh session per ticket with `/implement`, in parallel if you want both:
- #30 on `ShikharJohari/30-live-monitor`;
- #28 on `ShikharJohari/28-learning-bias`.

Working rules as before: one ticket per session; a `ShikharJohari/<n>-<slug>` branch; a PR that says `Closes #N`; nothing merged without Shikhar; role agents only, never Haiku; never Playwright locally; verify against `ryuk serve` with and without weights; write the handoff before merge.

**#30 builds on:**
- `Watchlist` and `ModelRegistry`: `registry.active` gives the active model with its `evaluated.threshold` and `crop`; `loaded()` gives every model with weights. `PUT /api/active-model` must write the `setting` row and refuse a model that `can_be_active` is false for.
- The private `Watchlist._lock` guards the detector and the models; the live monitor needs its own `Detector` and model instances or the same lock. `Detector.detect(max_side=...)` exists if frames are large.
- Embeddings live in `embedding` as float32 BLOBs under `model_key`; #12 wants the active model's watchlist embeddings in an in-memory matrix, reloaded after every watchlist change. Nothing loads that yet: the warning checks query per request.
- `LIVE_RULES` and `MatchRule`: the live score must dispatch on the threshold's `rule`; today only `best-photo` is computable.
- `LocalhostOnlyMiddleware` already closes a cross-origin WebSocket handshake with 1008; `WEBSOCKET_MODELS` in `api/contract.py` merges socket messages into `openapi.json`.
- The client's `runtime.ts`, `ApiClient` and MSW harness; the palette tokens; `useAcknowledgedMutation` as the pattern for a confirm-then-resend flow.
