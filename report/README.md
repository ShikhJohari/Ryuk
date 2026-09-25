# The Ryuk report

The written report for Ryuk, structured by #19: nine sections from problem and ethics to a reproduction appendix. It is a Quarto project rendered to PDF through Typst, so it needs no TeX install; Quarto, Typst and the Jupyter kernel all come from the project's dev dependencies (`uv sync`).

The report reads only committed JSON, `eda/summary.json` now and `evaluation/results.json` later, and draws its figures with `ryuk.plotting`. It never touches raw data or weights, so it renders on any machine with the repo checked out, and its numbers cannot disagree with the code that measured them.

## Rendering

From the repo root:

```sh
uv run quarto render report                   # both targets
uv run quarto render report/report.qmd        # the full report only
uv run quarto render report/progress.qmd      # the progress report only
```

The PDFs land in `report/_output/` (gitignored): `ryuk-report.pdf` and `ryuk-progress-report.pdf`.

## Two targets, one source

Each section is one file in `sections/`. `report.qmd` includes all nine; `progress.qmd` includes only the sections written so far, after a short status note and table. A section joins the progress report by adding its include there and updating the table. Sections keep their full-report numbers in the progress report through a heading-counter line before each include, and a section in the progress report must not cross-reference one that is not, or the reference will not resolve.

A passage not written yet is a `::: stub` div naming the ticket that writes it; stubs render muted behind a rule so they cannot be mistaken for finished prose.

## Layout

```
_quarto.yml              project and format settings shared by both targets
report.qmd, progress.qmd the two targets
sections/                one file per section
style/typst-template.typ the look: #13's lab-notebook direction, tuned for print
style/stub.lua           renders `::: stub` divs
style/digit-apostrophe.lua  keeps "View 1's" from printing as "View 1′s" (Typst reads ' after a digit as a prime)
```

Fonts are the ones vendored with the plotting module (`src/ryuk/plotting/fonts`: Newsreader for titles, headings and captions, Public Sans for body text), passed to Typst through `font-paths`, so nothing has to be installed.

## Code in the report

Python cells run in the project venv's `python3` kernel with `echo: false`. Numbers in prose are inline expressions, `` `{python} expr` ``, over the loaded summary or results; figures are `ryuk.plotting` calls in cells and are embedded as SVG. A cell that raises fails the render.

To inspect the Typst source Quarto generates, add `-M keep-typ:true`; the `.typ` files are written next to the `.qmd` files.
