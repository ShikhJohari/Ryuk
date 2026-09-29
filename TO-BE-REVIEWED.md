# To be reviewed

Conflicts between agents' recommendations, with the pick made and why. Shikhar has final say; an entry with a ruling is resolved.

## Module-scope ManagedRuntime in the client (#23, 2026-09-25)

- **Option A (kept):** one `ManagedRuntime` built at module scope in `web/src/lib/runtime.ts`, run through `runQuery` from TanStack Query functions and route loaders. This is the pattern the client toolchain research recommended (#6, `docs/research/client-toolchain.md`) and what #23 asks for ("Effect 3.22 behind one managed runtime run from TanStack Query").
- **Option B (code review):** treat the module-level runtime as a singleton that `~/.claude/docs/effect.md` rule 4 forbids; build it in `main.tsx` and pass it through router context, so tests could provide a test Layer instead of mocking HTTP with MSW.
- **Pick: A.** The spec asks for it, and `runtime.ts` is the single composition root rule 4 asks for. Route tests go through MSW by design (spec #22, testing seam 4), and API-boundary tests already provide test Layers. Easily reversible: moving the runtime into router context touches `runtime.ts`, `main.tsx` and the query factories.
- **Ruling (Shikhar, 2026-09-29, #47 Q8):** A accepted.

## The fake recognition model posing as a real network (#26, 2026-09-25)

- **Option A (Spec review):** keep `FakeRecognitionModel(network=...)`, so the harness tests can feed fakes through code that only takes real networks (`sface`, `arcface`, `facenet`). Small and justified.
- **Option B (Standards review, kept):** a test-only hook in the shipped package that defeats `model_id`'s guard against the fake reaching `results.json`. The fake's key always says `fake`; a wrapper in `tests/test_verification.py` presents it under a real network.
- **Pick: B.** The guard only means something if the package's fake can never pass it; the tests that need a stand-in own the stand-in. Easily reversible: one keyword argument on the fake.
- **Ruling (Shikhar, 2026-09-29, #47 Q8):** B accepted.

## Live monitor frame limits (#30, 2026-09-27)

- **Option A (Spec review):** the 2 MB and 1920 px frame limits and the header-against-JPEG size check are behaviour #30 did not ask for; defensible under #12's `error` message, but scope creep.
- **Option B (Standards review, kept):** keep them, and enforce the byte limit before the message is read: uvicorn buffers up to 16 MiB by default, so `ryuk serve` now passes `ws_max_size` and a larger message is closed with 1009 ("Refuse before you read", watchlist handoff).
- **Pick: B.** An unauthenticated socket that decodes whatever it is sent needs a ceiling, and the watchlist session set the pattern of refusing oversized bodies before they are buffered. Easily reversible: two constants in `ryuk.api.frames` and one argument to `uvicorn.run`.
- **Ruling (Shikhar, 2026-09-29, #47 Q8):** B accepted, with the limits raised to 4 MB and 2560 px a side. The two constants now hold those values.
