# Contract session, 24 September 2026

The record of the fourth working session. It closed the last two open grilling tickets: the API contract and data model ([issue 12](https://github.com/ShikhJohari/Ryuk/issues/12)) and learning on embeddings with the bias breakdown ([issue 10](https://github.com/ShikhJohari/Ryuk/issues/10)). Read the three earlier records first, oldest first. GitHub issues are the source of truth for state; this file is the narrative and the reasoning.

The map: https://github.com/ShikhJohari/Ryuk/issues/1

## How it went

1. Shikhar asked for the three handoff records and `CONTEXT.md` to be read, then issue 12 with `/grilling` and `/domain-modeling`, then issue 10.
2. Issue 12 in two rounds. One background researcher agent checked how the client can get typed Effect models from FastAPI's OpenAPI output. Shikhar accepted every recommendation in both rounds.
3. Issue 10 in two rounds. One background explorer agent measured CelebA attribute group sizes and label consistency from `data/raw/celeba/metadata/celeba_meta.parquet`. Shikhar accepted every recommendation in both rounds.
4. Pushing to `main` from the session was refused by the auto-mode permission classifier, so Shikhar pushed the docs commits from the terminal.

## API contract and data model (issue 12)

The decisions, in short:

- **Face data lives inside SQLite.** Enrolled photos, embeddings and sighting crops are BLOBs, with `secure_delete` and foreign keys on, so a purge is one cascading transaction that also overwrites the freed pages. Recorded as ADR 0004. Files on disk were rejected because an interrupted purge would leave behind face photos that nothing references.
- **A recognition model is network plus weights hash plus execution provider.** ArcFace on CoreML and ArcFace on CPU are different models. A weights mismatch at startup rebuilds that model's embeddings from the enrolled photos (cheap because of ADR 0003).
- **Model states:** active, available, not evaluated, unavailable. Only an evaluated model can be active. One global active model, saved across restarts. With none usable, watchlist management still works and the live monitor refuses to start.
- **Enrollment in the API:** a person of interest is created with a name and one photo; more photos arrive one per request. An enrolled photo now needs exactly one face *large enough to use*, so a bystander in the background no longer causes a rejection. Three warnings (duplicate name, looks like another person of interest, may not be the same person) are confirmed by resending the request.
- **Removal, restore, purge:** status field for the first two, `DELETE` for purge. A purged runner-up is cleared from other people's sightings, score kept.
- **Live monitor:** one connection at a time. Results carry a score for every face, a name only on a match, and never the runner-up. Sighting events travel on the same WebSocket.
- **Client types:** no maintained generator produces Effect schemas plus a client for Effect 3.22 (the good ones went v4-only). So `openapi.json` is committed with the WebSocket models merged in, `openapi-typescript` generates types, and hand-written Effect schemas carry a type-equality check each, so drift fails `tsc`. Revisit when Effect v4 ships.
- No login; the service binds to localhost only, and the report says a real deployment needs authentication and an audit trail.

Glossary: new term **Runner-up**; **Enrolled photo**, **Enrollment**, **Purge**, **Live monitor**, **Active model** and **Recognition model** sharpened.

The full REST surface, WebSocket messages and table list are in the [resolution comment](https://github.com/ShikhJohari/Ryuk/issues/12#issuecomment-5823093597).

## Learning on embeddings and bias (issue 10)

The framing given to Shikhar, following the previous session's advice to lead with what a thing is for: the recognition models are frozen, so Ryuk's own learning is everything after the embedding, meaning how several enrolled photos become one score and how that score becomes match or no match.

- **Three families compared on all three models:** scoring rules (best photo, which is the baseline, and mean), classic classifiers trained on the gallery (kNN, logistic regression, linear SVM), and a learned decision rule on two inputs, the top score and its gap to the runner-up. The classifiers are there because the course and rubric expect them; the decision rule is the one expected to have a chance.
- **Honesty:** tuning and fitting on the validation draw only, with identity-grouped cross-fitting for the decision rule; the test draw scored once. A method wins only if the paired bootstrap interval of its TPIR gain excludes zero.
- **A winner goes live** if it needs no retraining when the watchlist changes, so classifiers never go live. This is what made the glossary grow **Match score** and widen **Threshold** to a cut-off on it.
- **No "ambiguous" state.** This answers the question #11 handed on: the gap only matters through the learned rule.
- **Misidentification** is now a named evaluation term and a reported rate: the wrong identity above the threshold, the error that matters most on a watchlist.
- **Bias breakdown:** single frozen threshold; per-group TPIR, FPIR and misidentification with CIs, and the worst-to-best FPIR ratio. Groups are Male and Young (and their four combinations, indicative) plus the photo conditions Eyeglasses, Wearing_Hat and Blurry.
- **Outputs:** CLI experiment commands, one committed `evaluation/results.json` that the service, the client's evaluation page and the report figures all read.

### What the data allowed

The explorer's counts decided the bias groups, so they are worth keeping here. Among identities eligible for the gallery (at least 20 images), Male splits 259 male / 370 not in validation and 228 / 388 in test; Young splits 482 young / 147 not and 489 / 127. Pale_Skin has 0 eligible identities in validation and 1 in test, and CelebA has no race, ethnicity or skin-tone label, so that breakdown cannot be done and the ethics section says so instead of using a proxy. Heavy_Makeup is nearly a stand-in for "not Male" in CelebA and was left out. Bald, Chubby and Gray_Hair have 11 to 29 eligible identities.

A correction to #9: each draw leaves about 485 to 500 held-out identities, not about 400, so about 4,700 non-mated probes.

## Where the map stands

Every grilling ticket on the map is closed. Still under "Not yet specified", none ticketed yet:

- EDA scope and the dataset report. Must include the YuNet re-detection rate on CelebA and the minimum usable face size from #12.
- Live monitor behaviour: the gap that ends a sighting, when a better match replaces the crop, alerts and cooldowns. Unblocked by #12.
- Client information architecture: routes and state. Unblocked by #12; evaluation outputs come from the API.
- CI specifics for two toolchains, model weights in CI, and the stale-`openapi.json` check from #12.
- Report structure and viva demo script. Last.
- Spec synthesis and slicing into build issues, the final ticket.

## Loose ends to carry

- The map's Notes still say "Compare SFace int8"; #9 made int8 a footnote.
- Doc fixes, not yet done: `evaluation-protocol.md` quotes 99.77 for ArcFace; `models.md` and `datasets.md` carry the stale claims listed in the 23 September record.
- Whether to drop the InsightFace package for a direct onnxruntime session. Spec time.
- Confirm with the instructor that a TypeScript client over a Python service counts as a Python project. Settle `web/` versus `apps/web`.

## Resuming

Open `~/Projects/ryuk`, read the four handoff records and `CONTEXT.md`, then ticket the remaining fog items. Live monitor behaviour and client information architecture are grilling tickets; EDA scope can be ticketed and grilled or handed to an agent. The spec-and-slice ticket comes last. Implementation starts only after it closes. Role agents only, never bare agents, never Haiku.
