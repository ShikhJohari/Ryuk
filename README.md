# Ryuk

Watchlist face recognition, built as a machine learning project. A Python service detects faces, turns them into embeddings with open-source pretrained models, matches them against an enrolled watchlist, and reports how well each model does on public benchmarks. A React client shows the watchlist, the live monitor, and the evaluation results.

This is the successor to [crimdet](https://github.com/ShikhJohari/crimdet), a Java desktop app that wrapped the same pipeline without measuring it. Ryuk keeps the pipeline design and adds the parts a machine learning project needs: datasets, exploratory analysis, an evaluation harness, model comparison, and a bias breakdown.

## Status

Planning. The route is charted on the wayfinder map in this repo's issues. Nothing runs yet.

## Layout (planned)

```
src/ryuk/        Python package: data, models, pipeline, eval, api
notebooks/       exploratory analysis
experiments/     scripted runs with committed outputs
web/             React client (TanStack Router, Effect, shadcn, Tailwind)
docs/adr/        architecture decision records
CONTEXT.md       glossary
```
