# Charting session, 22 September 2026

The record of the session that started Ryuk. Read this before picking up a wayfinder ticket. GitHub issues are the source of truth for state; this file is the narrative and the reasoning behind the decisions, which the issues only gist.

The map: https://github.com/ShikhJohari/Ryuk/issues/1

## How Ryuk came about

Shikhar has a graded Java project, [crimdet](https://github.com/ShikhJohari/crimdet), that wraps YuNet and SFace from OpenCV behind a JavaFX app. The new semester's course, Statistical Machine Learning, requires a Python project graded on this rubric (20 marks): problem identification 2, dataset and exploratory analysis 1.5, methodology and model selection 1.5, progress report 1, mid-term viva 4, completeness of implementation 2, model performance and evaluation 1, innovation 1, working prototype 1, report 1, final viva 4.

The first question was whether to port crimdet to Python. The answer was no. Crimdet has no dataset, no training, no evaluation, and its match threshold of 0.363 was copied from the SFace author's script. The ML in it is two API calls; the other five thousand lines are UI and persistence, which the rubric does not credit. A line-by-line port would also look like resubmission of graded work. So Ryuk reuses the problem, the pipeline shape, and what was learned, and adds the parts an ML project needs.

Shikhar's standing instruction for this effort: ignore the milestone dates. Finish the project end to end in phases.

## Grilling round one

Each question, the recommendation, and what Shikhar decided.

1. Where does the project live? Recommended a new repo. Decided: new repo, named Ryuk.
2. What does the map deliver? Recommended decisions and sliced issues, then the normal build lifecycle. Decided: yes.
3. Milestone 1 status. Assumed nothing submitted and topic open. Shikhar said not to plan around milestones at all.
4. Solo or group, report format. Assumed solo, free-form report. Not contradicted.
5. Compute. Recommended laptop plus a free GPU for one fine-tuning experiment. Overridden: no GPU work, use open-source pretrained models for anything heavy. Recorded as ADR 0002.
6. Keep the "criminal" framing? Recommended reframing as watchlist recognition with an ethics and bias section. Decided: yes. The glossary avoids "criminal", "suspect", "target".
7. Innovation angle. Recommended rigorous benchmarking plus learning on top of frozen embeddings plus fine-tuning. Fine-tuning dropped per question 5. The story is now "which open model, and what to learn on top of it".
8. Prototype UI. Recommended Streamlit. Overridden: React with a designed front end.
9. What "highly durable" means. Recommended uv, Python 3.12, ruff, mypy strict, pytest with coverage in CI, pydantic, SQLAlchemy over SQLite, scripted dataset download. No Docker, no DVC. Decided: yes, and Docker is explicitly out.
10. Dataset. Recommended LFW for verification plus a CelebA subset for identification and attribute-based bias analysis. Decided: yes.

## Grilling round two

1. Repo name. Recommended facewatch. Decided: Ryuk.
2. Open-source model for the heavy workflow. Confirmed: compare SFace, InsightFace ArcFace, and one more, no fine-tuning, training limited to classifiers and calibration on frozen embeddings.
3. Front-end stack. Recommended TanStack Router without Effect. Overridden: follow stack.md, so TanStack Router with Effect, plus shadcn/ui and Tailwind.
4. Backend and live frames. Recommended FastAPI with browser-captured frames over a WebSocket. Decided: yes. Recorded as ADR 0001.
5. Layout. Single repo with a Python package, notebooks, experiments, and a web client. Decided: single repo. Whether the client sits at `web/` or `apps/web` is still open; the toolchain research used `apps/web` to match stack.md.
6. Visual direction. Recommended a dark ops dashboard like crimdet's Nord Dark theme. Shikhar likes that direction but wants to see it rendered before deciding, through Claude Design. That is the prototype ticket.
7. Phase order. Decided as listed on the map: scaffold, data, core pipeline, evaluation harness, learning on embeddings, API and persistence, React client, live monitor, report.
8. Embedding search. NumPy brute-force cosine. FAISS out of scope.

## Research outcomes

Five research tickets ran as background agents. Each has a full write-up in `docs/research/` and a resolution comment on its issue.

Datasets ([issue 2](https://github.com/ShikhJohari/Ryuk/issues/2), `docs/research/datasets.md`). LFW from scikit-learn's figshare mirror with md5 checksums, because the official UMass host no longer resolves in DNS. CelebA from the ungated Hugging Face dataset `flwrlabs/celeba`, which has identity and attribute labels. The whole thing is 11.7 GB, so the fetch task should take only the shards the subset needs. Suggested subset: 500 enrolled identities at up to 20 images each, plus about 100 never-enrolled identities as unknown probes.

Models ([issue 3](https://github.com/ShikhJohari/Ryuk/issues/3), `docs/research/models.md`). SFace from the OpenCV zoo as the baseline, InsightFace ArcFace `buffalo_l` as the accuracy ceiling, facenet-pytorch's VGGFace2 model as the licence-clean third. YuNet as the one shared detector. Every wheel verified by download for Python 3.12 on arm64. Two claims still need an empirical check in the toolchain spike: that YuNet landmarks aligned through OpenCV and InsightFace land on the same template, and what alignment facenet-pytorch needs.

Evaluation protocol ([issue 4](https://github.com/ShikhJohari/Ryuk/issues/4), `docs/research/evaluation-protocol.md`). LFW 10-fold pairs exactly as the reference implementations do it. An IJB-C style open-set split on CelebA with held-out identities. Threshold frozen on a disjoint validation split. Confidence intervals by identity-level bootstrap. Expect about 99.5 percent LFW accuracy for SFace and 99.8 for ArcFace. The DIR formula came from secondary sources because the IJB-A paper PDF would not parse.

Frame streaming ([issue 5](https://github.com/ShikhJohari/Ryuk/issues/5), `docs/research/frame-streaming.md`). Binary WebSocket frames with a small header. Server drains into a one-slot latest-frame buffer, the same rule crimdet's FrameProcessor used, and runs inference in a threadpool. Results return as small JSON. 10 fps capture default. WebRTC rejected.

Client toolchain ([issue 6](https://github.com/ShikhJohari/Ryuk/issues/6), `docs/research/client-toolchain.md`). TanStack CLI in router-only mode, Tailwind v4 via the Vite plugin, shadcn init, Effect pinned at 3.22. Effect owns only the API boundary behind one managed runtime. The Effect platform HTTP client docs page was unreachable, so exact API names need confirming when scaffolding.

## Open decisions, system

These are live tickets. The frontier is what can be picked up now.

Frontier:

- [Toolchain spike](https://github.com/ShikhJohari/Ryuk/issues/7). Agent-driven. Install the three models and YuNet on this Mac, run a few images, record latencies and versions, do the two alignment checks above.
- [Fetch the datasets](https://github.com/ShikhJohari/Ryuk/issues/8). Agent-driven. Partial CelebA fetch only.
- [Evaluation protocol and the comparison table](https://github.com/ShikhJohari/Ryuk/issues/9). Grilling. Which metrics and columns go in the report table, how the operating threshold is chosen, whether int8 versus fp32 SFace is a row or its own experiment.
- [Domain model walked through scenarios](https://github.com/ShikhJohari/Ryuk/issues/11). Grilling. Enrolled twice, removed with history, model swap, top-1 below threshold, look-alikes, mirror in frame.

Blocked:

- [Learning on embeddings and bias breakdown](https://github.com/ShikhJohari/Ryuk/issues/10). Waits on the evaluation decision and the datasets on disk.
- [API contract and data model](https://github.com/ShikhJohari/Ryuk/issues/12). Waits on the domain model.

Still in the fog on the map, not yet sharp enough to ticket: EDA scope, bias analysis design, live monitor behaviour (alert policy, deduplication across frames, cooldowns), client information architecture, CI specifics, report structure and viva script, and the final spec-and-slice ticket.

## Open decisions, visual

One ticket: [Visual direction via Claude Design](https://github.com/ShikhJohari/Ryuk/issues/13). What Shikhar said: the dark theme sounds right, but decide only after seeing it. What to produce: a Claude Design mockup (Artifact quickstart, intent design) of three screens.

- Live monitor: video feed, bounding boxes with name and similarity, a sightings rail.
- Watchlist: list plus a person of interest detail with enrolled photos.
- Evaluation: the model comparison table and ROC curves.

Start from a dark ops-dashboard direction and offer one contrasting variant to react against. Unresolved within that: palette, type, density, how charts sit on dark, whether the evaluation pages go lighter. The resolution records the chosen direction and the design-system link for the client build tickets.

## Loose ends to carry

- Check with the instructor that a TypeScript client over a Python service satisfies "Python project". Everything the rubric marks is Python.
- Client directory: `web/` on the map versus `apps/web` in the toolchain research. Settle at spec time.
- Do not commit to main directly. This handoff itself went in as a PR.

## Resuming

Open `~/Projects/ryuk` and invoke `/wayfinder` with the map link. Name a ticket or let it take the first on the frontier. A sensible next session: start the two agent-driven tasks, then do the visual direction prototype together while they run. Role agents only, never bare agents, never Haiku.
