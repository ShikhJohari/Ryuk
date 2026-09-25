# To be reviewed

Conflicts between agents' recommendations, with the pick made and why. Shikhar has final say.

## Module-scope ManagedRuntime in the client (#23, 2026-09-25)

- **Option A (kept):** one `ManagedRuntime` built at module scope in `web/src/lib/runtime.ts`, run through `runQuery` from TanStack Query functions and route loaders. This is the pattern the client toolchain research recommended (#6, `docs/research/client-toolchain.md`) and what #23 asks for ("Effect 3.22 behind one managed runtime run from TanStack Query").
- **Option B (code review):** treat the module-level runtime as a singleton that `~/.claude/docs/effect.md` rule 4 forbids; build it in `main.tsx` and pass it through router context, so tests could provide a test Layer instead of mocking HTTP with MSW.
- **Pick: A.** The spec asks for it, and `runtime.ts` is the single composition root rule 4 asks for. Route tests go through MSW by design (spec #22, testing seam 4), and API-boundary tests already provide test Layers. Easily reversible: moving the runtime into router context touches `runtime.ts`, `main.tsx` and the query factories.

## The fake recognition model posing as a real network (#26, 2026-09-25)

- **Option A (Spec review):** keep `FakeRecognitionModel(network=...)`, so the harness tests can feed fakes through code that only takes real networks (`sface`, `arcface`, `facenet`). Small and justified.
- **Option B (Standards review, kept):** a test-only hook in the shipped package that defeats `model_id`'s guard against the fake reaching `results.json`. The fake's key always says `fake`; a wrapper in `tests/test_verification.py` presents it under a real network.
- **Pick: B.** The guard only means something if the package's fake can never pass it; the tests that need a stand-in own the stand-in. Easily reversible: one keyword argument on the fake.
