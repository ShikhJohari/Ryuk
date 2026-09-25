# To be reviewed

Conflicts between agents' recommendations, with the pick made and why. Shikhar has final say.

## Module-scope ManagedRuntime in the client (#23, 2026-09-25)

- **Option A (kept):** one `ManagedRuntime` built at module scope in `web/src/lib/runtime.ts`, run through `runQuery` from TanStack Query functions and route loaders. This is the pattern the client toolchain research recommended (#6, `docs/research/client-toolchain.md`) and what #23 asks for ("Effect 3.22 behind one managed runtime run from TanStack Query").
- **Option B (code review):** treat the module-level runtime as a singleton that `~/.claude/docs/effect.md` rule 4 forbids; build it in `main.tsx` and pass it through router context, so tests could provide a test Layer instead of mocking HTTP with MSW.
- **Pick: A.** The spec asks for it, and `runtime.ts` is the single composition root rule 4 asks for. Route tests go through MSW by design (spec #22, testing seam 4), and API-boundary tests already provide test Layers. Easily reversible: moving the runtime into router context touches `runtime.ts`, `main.tsx` and the query factories.
