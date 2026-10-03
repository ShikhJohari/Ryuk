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

## `Promise.all` in route loaders (#31, 2026-09-29)

- **Option A (kept):** the sightings history, sighting detail and person routes' loaders wait on several TanStack Query `ensureQueryData` promises with `Promise.all`, at the router edge where the values are already promises, not effects.
- **Option B (client Standards review):** `~/.claude/docs/effect.md` rule 7 says never `Promise.all`; build each loader as one Effect (`Effect.all` over the queries) run through `runQuery`.
- **Pick: A.** Rule 7 is about composing effects; these loaders compose TanStack Query promises that `runQuery` already produced, and wrapping them back into an Effect only to unwrap them for the router adds a layer with no error channel to gain. Easily reversible: three loaders.

## "In view" marker on the sightings rail (#31, 2026-09-29)

- **Option A (kept, client implementer):** a rail item whose sighting is matched in the current frame reads "· in view", from the match face's `sightingId`.
- **Option B (Spec review):** neither #16 nor #31 asks for it; scope creep, however small.
- **Pick: A.** It is the only visible use of `sightingId` on a match, which #12 put in the contract so the monitor can tie a face to its sighting, and it costs one line of state. Easily reversible: one prop on `SightingsRail` and its test.

## Report scope for the learning and bias sections (#28, 2026-09-29)

- **Option A (Spec review):** Q11 names only Sections 5.3, 5.4 and 6.1, so the new "Learning and bias" paragraph in Section 3 and the four Section 8 subsections (annotator labels, the edge of the kNN grid, wide intervals for small groups, the learned rule and gallery size) belong to #32.
- **Option B (implementer and lead, kept):** keep them. Q20 requires the adjusted Wilson N* disclosure in Section 3 anyway, and each Section 8 paragraph is a caveat on a #28 number that should ship with the number.
- **Pick: B.** A number without its caveat reads stronger than it is, and #32 can still rewrite the prose. Easily reversible: delete the paragraphs.

## The undefined worst-to-best FPIR ratio (#28, 2026-09-29)

- **Option A (Spec review):** when the best group raised no false alarm, report the ratio as "∞ (0 of N)" or as a bounded ratio, so that SFace under the mean rule keeps a headline Male disparity: 1.38% for held-out identities labelled not male against 0 false alarms in 1,956 non-mated probes of 240 male identities.
- **Option B (kept):** leave the ratio undefined ("—") and give the zero cell's dependence-adjusted Wilson upper bound in the prose and table notes, 1.58% for that group. An infinite ratio from about 2,000 probes with no false alarm overstates what is known: the upper bound is above the not-male group's own rate.
- **Pick: B.** Easily reversible: one branch in `bias._fpir_ratio` and Table 6's cell format.

## The same-person threshold's false-accept rate (#49, 2026-10-03)

- **Option A (kept):** freeze each same-person threshold at FAR 0.1% on the validation draw's one-photo impostor pairs (about 5.7 million, so some 5,700 false accepts pin it down). With one photo enrolled the warning then fires on 7.6% of a person's own test photos under SFace, 3.8% under ArcFace and 10.4% under FaceNet; with five, under 2% for all three.
- **Option B:** FAR 1%, the target the 1:N threshold uses, which would warn less often but let one stranger's photo in a hundred join a person unwarned.
- **Pick: A.** 0.1% is the conventional verification operating point (the LFW results read TAR there too), and a wrong photo silently added to a person costs more than an extra dialog. Q6 asked for "a stated false-accept rate" without a number, and #49's acceptance needs an agreed warning-rate figure, which is Shikhar's to set. Easily reversible: `same_person.TARGET_FAR`, then `ryuk evaluate live` (about three minutes on the Mac from the cached embeddings).

## No same-person threshold, no warning (#49, 2026-10-03)

- **Option A (kept):** when the active model has no same-person threshold (results from before `ryuk evaluate live`, or a section dropped as stale), enrollment raises no `may_not_be_same_person` warning.
- **Option B:** fall back to the model's 1:N threshold, the old behaviour.
- **Pick: A.** The fallback is the cut-off #49 measured as wrong for this question (it warned on 21.5% and 36.8% of SFace's and FaceNet's own photos), and the committed results carry a same-person threshold for every model. Easily reversible: one condition in `Watchlist._not_same_person`.
