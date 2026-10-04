# Grilling session, 24 September 2026

The record of the third working session. It closed two grilling tickets: the domain model ([issue 11](https://github.com/ShikhJohari/Ryuk/issues/11)) and the evaluation protocol ([issue 9](https://github.com/ShikhJohari/Ryuk/issues/9)). Read `2026-09-22-charting-session.md` and `2026-09-23-frontier-session.md` first. GitHub issues are the source of truth for state; this file is the narrative and the reasoning.

The map: https://github.com/ShikhJohari/Ryuk/issues/1

## How it went

1. Shikhar asked for the project's state and whether the visual direction from the previous session had been tracked. It had: issue 13's resolution and the map both record direction B. The gap was that no session record existed for 23 September, so one was written from that session's saved transcript plus the issue comments (`2026-09-23-frontier-session.md`).
2. Ordering. The status answer first suggested the evaluation protocol, contradicting the previous session's advice to take the domain model first. On reflection the domain model unblocks more (the API contract, and through it the live monitor and client tickets), so the order went back to 11, then 9. Shikhar agreed.
3. Issue 11 in two rounds with `/grilling` and `/domain-modeling`, updating `GLOSSARY.md` as terms resolved and writing ADR 0003.
4. Issue 9 in two rounds, with one pause to explain int8 and what CelebA is for, and one background researcher agent to check published LFW numbers.

## Standing preference, new this session

Docs-only changes (handoff records, glossary, ADRs) go straight to `main` without a PR. Shikhar: "no need to open a pr just to update docs." Code still goes through a branch and PR. Commits carry no Co-Authored-By line.

## Domain model (issue 11)

Shikhar accepted every recommendation in round one. In round two Shikhar accepted three and changed one: a person of interest carries a name and enrolled photos, "and nothing else". The recommendation had been to allow an optional free-text note, which was dropped.

The decisions, in the glossary's words:

- One person of interest per person. More photos go into the same record. Names need not be unique; a duplicate name gets a warning.
- An enrolled photo must contain exactly one face, or it is rejected with the reason. Single photos can be deleted, except the last one.
- Removal takes a person of interest off the watchlist and keeps their photos and sightings; it is reversible. Purge erases the person, photos, embeddings, sightings and sighting crops permanently.
- Enrolled photos are the source of truth. Enrollment computes an embedding under every recognition model up front, so switching the active model is instant. Recorded as ADR 0003 because it means storing face photos, which is a privacy trade-off that is hard to undo.
- A detection whose top candidate is below the threshold is a no match: shown with its score, never a name, never logged. Naming someone next to a face that did not match invites the misidentification the threshold exists to prevent.
- Look-alikes: the top candidate wins and the sighting records the runner-up.
- A sighting is one person of interest seen continuously under one active model. The mirror case (same person twice in a frame) is one sighting. It keeps the face crop of the best match, never the whole frame, so bystanders are not stored. Switching the active model closes open sightings.
- The threshold is fixed per model from evaluation and the operator cannot change it. A threshold tunable live defeats the point of measuring one.
- Gallery, probe and identity are evaluation-only terms; watchlist and person of interest are live-system terms.

Resolution: https://github.com/ShikhJohari/Ryuk/issues/11#issuecomment-5822327174

## Evaluation protocol (issue 9)

### Two explanations Shikhar asked for

These are worth knowing because they show where the project's vocabulary was not landing.

**Why int8 exists.** Shikhar asked what "3x slower" meant and then why int8 exists at all if fp32 is better. The explanation given: int8 is the same network with weights rounded to 8-bit integers, four times smaller, and built for phones, small boards and chips with fast integer maths. On the M4 with OpenCV 5 the fast path is fp32, so int8 keeps its costs (drifted embeddings needing their own threshold) and loses its benefit. It is in the conversation only because crimdet shipped `face_recognition_sface_2021dec_int8.onnx`. It stays as a footnote because that answers the viva question "why not crimdet's model?" with a measurement.

**CelebA is a dataset, not a model.** Shikhar referred to CelebA as "that model". It was explained as the second of two exams: LFW asks "same person or not?" and proves the pipeline reproduces published numbers; CelebA builds a pretend watchlist (enrolled celebrities, their other photos as faces that should match, never-enrolled celebrities as strangers) to rehearse what the live monitor does, choose the threshold, and, through its 40 attribute labels, make the bias breakdown possible. The validation and test draws were explained as a practice exam and a real exam. When explaining evaluation to Shikhar, lead with what a thing is for before the metric names.

### Decisions

All recommendations accepted in both rounds. In short:

- Rows: SFace fp32, ArcFace (CoreML MLProgram), FaceNet. int8 a footnote.
- Pipeline choices tuned on LFW View 1; View 2 run once per model with the OpenCV zoo / InsightFace 10-fold recipe.
- CelebA valid and test splits are the validation and test draws: 500 gallery identities with 5 enrolled photos, about 400 held-out identities.
- Threshold per model at FPIR 1% on the validation draw, frozen before the test draw.
- Identity-level bootstrap CIs; adjusted Wilson check for error rates under 1%.
- Two report tables: verification on LFW, watchlist search on CelebA.
- The model the live monitor starts with: eligible if it reproduces its published LFW accuracy within 0.5 points, keeps test FPIR at or under 2%, and runs at or under 30 ms per face; highest TPIR wins, overlapping CIs go to the faster model. The glossary lists "default model" under Avoid, so this is phrased as the first active model.

### Published numbers checked

A researcher agent checked the published LFW accuracy for the exact checkpoints: SFace 99.40, ArcFace `buffalo_l` 99.83, FaceNet 99.65 ± 0.25. The 99.77 in `evaluation-protocol.md` belongs to a different ArcFace checkpoint (R100 on MS1MV2). Two footnotes are needed: InsightFace's table also has a 99.80 row for the R50 WebFace600K backbone alone, and FaceNet's published number used MTCNN crops and flipped-image averaging, so a small shortfall under Ryuk's YuNet crop is expected.

Resolution: https://github.com/ShikhJohari/Ryuk/issues/9#issuecomment-5822916457

## Where the map stands

Not done with decisions yet. The map's standing preference is decisions before code, and its destination is a spec plus sliced build issues with no open decisions.

Open grilling tickets, both unblocked:

- [API contract and data model](https://github.com/ShikhJohari/Ryuk/issues/12). Must represent removal vs purge, per-model embeddings, and sightings with crop, runner-up, model and threshold. Unblocks live monitor behaviour and client information architecture.
- [Learning on embeddings and bias breakdown](https://github.com/ShikhJohari/Ryuk/issues/10). Takes #9's draws and baseline scoring rule. Also owns how several enrolled embeddings combine into one score, and whether a small top-1 vs runner-up gap is ambiguous (both handed on from #11).

Still under "Not yet specified" on the map, with no ticket:

- EDA scope and the dataset report. Ready to ticket now. Must include the YuNet re-detection rate on CelebA.
- Live monitor behaviour: the gap that ends a sighting, cooldowns, alerts. After #12.
- Client information architecture: routes, state, how evaluation outputs reach the client. After #12.
- CI specifics for two toolchains, model weights in CI.
- Report structure and viva demo script. Last.
- Spec synthesis and slicing into build issues, the final ticket.

## Loose ends to carry

- Doc fixes, not yet done: `evaluation-protocol.md` quotes 99.77 for ArcFace; `models.md` and `datasets.md` carry the stale claims listed in the 23 September record.
- Whether to drop the InsightFace package for a direct onnxruntime session. Spec time.
- Confirm with the instructor that a TypeScript client over a Python service counts as a Python project. Settle `web/` versus `apps/web`.

## Resuming

Open `~/Projects/ryuk`, read the three handoff records and `GLOSSARY.md`, and take [issue 12](https://github.com/ShikhJohari/Ryuk/issues/12) first with `/grilling` and `/domain-modeling`, since it unblocks two of the unticketed items. Then [issue 10](https://github.com/ShikhJohari/Ryuk/issues/10). Ticket the EDA scope at any point. Implementation starts only after the spec-and-slice ticket closes. Role agents only, never bare agents, never Haiku.
