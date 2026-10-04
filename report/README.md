# The Ryuk report

The written report for Ryuk, structured by #19: nine sections from problem and ethics to a reproduction appendix. It is a Quarto project rendered to PDF through Typst, so it needs no TeX install; Quarto, Typst and the Jupyter kernel all come from the project's dev dependencies (`uv sync`).

The report reads only committed files, `eda/summary.json`, `evaluation/results.json` and the API contract `openapi.json`, and draws its figures with `ryuk.plotting` and `ryuk.evaluation.figures`. Design constants, such as the confirmation window or the stored photo's size, are imported from the code that applies them. It never touches raw data or weights, so it renders on any machine with the repo checked out, and its numbers cannot disagree with the code that measured them.

## Rendering

From the repo root:

```sh
uv run quarto render report                   # both targets
uv run quarto render report/report.qmd        # the full report only
uv run quarto render report/progress.qmd      # the progress report only
```

The PDFs land in `report/_output/` (gitignored): `ryuk-report.pdf` and `ryuk-progress-report.pdf`.

## Two targets, one source

Each section is one file in `sections/`. `report.qmd` includes all nine; `progress.qmd` includes the sections being written, after a short status note and table, and now that every section is written it includes all nine too. A section joins the progress report by adding its include there and updating the table. If the progress report leaves a section out, the sections after it keep their full-report numbers through a heading-counter line before their include, and a section in the progress report must not cross-reference one that is not, or the reference will not resolve.

A passage not written yet is a `::: stub` div naming the ticket that writes it. In the progress report stubs render muted behind a rule, so they cannot be mistaken for finished prose. The full report allows none: `report.qmd` sets `stubs: forbidden`, and `style/stub.lua` then fails the render, naming every stub left. Since `report.qmd` includes every section, `quarto render report`, which CI runs, fails on a stub anywhere.

## Layout

```
_quarto.yml              project and format settings shared by both targets
report.qmd, progress.qmd the two targets
sections/                one file per section
style/typst-template.typ the look: #13's lab-notebook direction, tuned for print
style/stub.lua           renders `::: stub` divs, and fails a target that sets `stubs: forbidden`
style/digit-apostrophe.lua  keeps "View 1's" from printing as "View 1′s" (Typst reads ' after a digit as a prime)
```

Fonts are the ones vendored with the plotting module (`src/ryuk/plotting/fonts`: Newsreader for titles, headings and captions, Public Sans for body text), passed to Typst through `font-paths`, so nothing has to be installed.

## Code in the report

Python cells run in the project venv's `python3` kernel with `echo: false`. Numbers in prose are inline expressions, `` `{python} expr` ``, over the loaded summary or results; figures are `ryuk.plotting` and `ryuk.evaluation.figures` calls in cells and are embedded as SVG. A cell that raises fails the render, so a cell asserts every qualitative claim its prose makes about the results, and a rerun that changes one stops the render instead of leaving the prose wrong.

To inspect the Typst source Quarto generates, add `-M keep-typ:true`; the `.typ` files are written next to the `.qmd` files.
