# Ryuk

Face recognition against an enrolled watchlist, with every recognition model measured on public benchmarks before the live monitor is allowed to trust it.

A Python service detects faces with YuNet, embeds them with frozen open-source models (SFace, ArcFace, FaceNet), and matches them against the people you enroll. A React client runs the live monitor on your webcam, manages the watchlist, logs sightings, and shows the evaluation that decided which model is active.

## Motivation

Ryuk succeeds [crimdet](https://github.com/ShikhJohari/crimdet), a Java desktop app that ran the same detect, embed and match pipeline without ever measuring it. Its threshold was a guess, and nothing said how often it named the wrong person.

Ryuk keeps the pipeline and adds what a machine learning project owes its users:

- **Measured models.** Each model is scored on LFW for verification and on a CelebA watchlist rehearsal for identification, and is checked against its published accuracy.
- **Thresholds that come from data.** The match threshold for each model is frozen on a validation split at a 1% false-positive identification rate, then scored once on a held-out test split.
- **A rule, not a preference, for the active model.** The service only lets a model with a measured threshold run live, and picks the first one by test performance, speed and confidence intervals.
- **Reproducibility.** Every number lives in a committed `evaluation/results.json` that records the commit and machine that produced it. The notebooks, the report and the service all read that file and nothing else.

## Results

From `evaluation/results.json`, measured on an Apple Silicon Mac:

| Model | Embedding | LFW accuracy | CelebA test TPIR at 1% FPIR | ms per face |
| --- | --- | --- | --- | --- |
| ArcFace R50 (CoreML) | 512 | 99.78% | **98.5%** [98.1, 98.8] | 8.4 |
| SFace | 128 | 99.38% | 95.0% [94.2, 95.8] | 5.4 |
| FaceNet VGGFace2 | 512 | 99.36% | 85.1% [83.5, 86.7] | 11.7 |

ArcFace is the first active model. The brackets are 95% bootstrap intervals. ROC curves and the full method are in `notebooks/02-evaluation.ipynb` and the report. The bias breakdown is still in progress (#32).

## Quick Start

You need [uv](https://docs.astral.sh/uv/), Python 3.12 (uv fetches it), and the pnpm version pinned in `web/package.json`.

```sh
git clone git@github.com:ShikhJohari/Ryuk.git && cd Ryuk
uv sync
uv run ryuk weights fetch         # ~335 MB; without weights no model can be active
uv run ryuk serve                 # service on http://127.0.0.1:8000
```

In a second shell:

```sh
pnpm --dir web install
pnpm --dir web dev                # client on http://localhost:5173, proxies /api to the service
```

Open `http://localhost:5173`. Start the service from the repository root, because its database, `data/ryuk.sqlite3`, is relative to where it starts.

> [!WARNING]
> The API has no authentication. Keep both the service and the client on loopback. Binding the client to `0.0.0.0` would hand the API, and every enrolled photo, to anything on the network. To use a remote machine, forward a port over SSH (see below).

## Usage

### The app

- **Watchlist.** Add a person of interest with a name and one or more photos. A photo is accepted only if it holds exactly one usable face. Removing someone takes them off the watchlist and is reversible. Purging erases them, their photos, embeddings and sightings for good.
- **Live monitor.** Runs recognition on your webcam. A face is named only when its top candidate clears the active model's threshold, and a sighting is logged only once the match holds steady across several frames.
- **Sightings.** Each sighting keeps the best face crop, the runner-up candidate, and the model and threshold that produced it.
- **Evaluation.** Shows the measured results and why the active model was chosen. You can switch models without re-enrolling anyone.

`CONTEXT.md` defines these terms precisely.

### Data and weights

Both come from pinned sources and are checked against pinned checksums. A rerun skips files that are already verified. Neither is ever committed.

```sh
uv run ryuk weights fetch                # YuNet, SFace (fp32, int8), ArcFace w600k_r50, FaceNet into models/weights
uv run ryuk data fetch                   # LFW and CelebA into data/raw (~2.6 GB)
uv run ryuk data fetch --dataset lfw     # LFW only
```

Set `RYUK_WEIGHTS_DIR` or `RYUK_DATA_DIR` to put them elsewhere.

### Evaluation

```sh
uv run ryuk eda                   # dataset summary and figures into eda/
uv run ryuk evaluate lfw          # every model on LFW View 2
uv run ryuk evaluate celeba       # the watchlist rehearsal: thresholds frozen on validation, test scored once
uv run ryuk evaluate learn        # learning on the frozen embeddings against best photo, and each model's live rule
uv run ryuk evaluate bias         # per-group rates on the test draw at each model's single frozen threshold
uv run ryuk evaluate live         # the same-person warning's thresholds, and the live rules on smaller watchlists
uv run quarto render report       # the PDF report, into report/_output/
```

Run the evaluations in that order, each needing the one before (`live` needs only `celeba`, but scores each model's live rule, so it belongs after `learn`). `learn` drops any previous bias section, since the bias breakdown depends on each model's live rule, so run `bias` after it; it keeps the `live` section only while every live rule and threshold it was measured at still holds.

Run evaluations from a clean, committed tree on the machine of record, an Apple Silicon Mac with ArcFace on CoreML. On any other machine ArcFace runs on CPU, which counts as a different model with no threshold. There, SFace becomes the active model, and the commands refuse to overwrite the committed CelebA results. Benchmark embeddings are cached under `data/cache` (`RYUK_CACHE_DIR`).

### On a remote machine

The browser has to reach the client as `localhost`, both for the camera (a secure context) and for the service's origin guard. Run everything on loopback on the remote box:

```sh
uv sync && pnpm --dir web install
uv run ryuk weights fetch
uv run ryuk serve
pnpm --dir web dev --host 127.0.0.1 --port 5173 --strictPort   # second shell
```

Then forward the port from your laptop and open `http://localhost:5173`:

```sh
ssh -N -L 5173:127.0.0.1:5173 <user>@<remote-host>
```

Editor port forwarding (VS Code or Cursor Remote-SSH) works the same way. Opening the box by hostname, a Tailscale URL for example, will not work: Vite rejects the host, the browser withholds the camera, and the service refuses every write with `403 cross_origin`.

## Project layout

```
src/ryuk/        the service: FastAPI app (api/), YuNet detector, recognition models,
                 evaluation harness, data and weights fetch, CLI
web/             React client (TanStack Router, Effect, shadcn, Tailwind)
tests/           pytest against the HTTP surface; smoke/ runs the real models
evaluation/      results.json, every measured number, and its JSON Schema
eda/             dataset summary and figures
notebooks/       one notebook per measured phase, reading committed JSON only
report/          the Quarto report
openapi.json     the API contract, generated and committed
docs/adr/        architecture decision records
CONTEXT.md       the domain glossary
```

## Contributing

Work is tracked in [GitHub Issues](https://github.com/ShikhJohari/Ryuk/issues). The spec is [#22](https://github.com/ShikhJohari/Ryuk/issues/22), and its sub-issues are the build tickets. Read `CONTEXT.md` and `docs/adr/` before changing behaviour, and use the glossary's terms in code and prose.

CI runs these on every pull request. Run them before you push:

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest   # 85% coverage floor
uv run pytest --nbmake notebooks --no-cov
uv run pytest -m smoke --no-cov   # needs `ryuk weights fetch` and `ryuk data fetch --dataset lfw`

cd web && pnpm typecheck && pnpm lint && pnpm test && pnpm build
```

If you change an API model, regenerate the contract and the client types, and commit both files:

```sh
uv run ryuk openapi && pnpm --dir web gen:api
```

The hand-written Effect schemas in `web/src/api/` assert equality with the generated types, so drift fails `tsc`. The browser smoke test (`pnpm e2e`) runs headless in CI only.

Commits are attributed to their human author alone. CI rejects a pull request with any commit that credits Claude as author, committer or co-author. Check a branch with `.github/scripts/check-attribution.sh origin/main..HEAD`.

## Responsible use

Ryuk is a study of how well face recognition works and where it fails. It is not a surveillance product. LFW is research-only by convention. CelebA is under the CUHK non-commercial agreement, which forbids redistribution. The InsightFace weights are licensed for non-commercial research only. Enroll only people who have agreed to it.

## License

The code is under the [MIT License](LICENSE). Datasets and model weights keep their own licences, listed above.
