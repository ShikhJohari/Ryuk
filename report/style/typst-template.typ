// The report's look: #13's lab-notebook direction, tuned for print.
//
// Replaces Quarto's `typst-template.typ` partial. `article` keeps the name and parameters
// Quarto's `typst-show.typ` calls it with; the parameters this style fixes itself (fonts, colours,
// heading styles) are absorbed by `..fixed` and ignored.
//
// Print tuning: a white page rather than the client's #F4F1E8 paper, so nothing is spent on
// background ink and the hairlines stay visible on any printer; the client's ink, muted and
// hairline colours are kept.

#let ink = rgb("#1C1B19")
#let muted = rgb("#5E5A52")
#let hairline = rgb("#D8D2C4")

// Vendored in src/ryuk/plotting/fonts and found through `font-paths` in _quarto.yml.
#let serif = "Newsreader 16pt"
#let sans = "Public Sans"

// A passage not written yet (`::: stub` in the source, see stub.lua).
#let stub(body) = block(
  width: 100%,
  inset: (left: 10pt, y: 2pt),
  stroke: (left: 1.5pt + hairline),
  {
    set text(fill: muted)
    set par(justify: false)
    body
  },
)

#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  keywords: (),
  lang: "en",
  region: none,
  fontsize: 10.5pt,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  ..fixed,
  doc,
) = {
  // Quarto passes `lang: en-GB` whole; Typst wants the language and region apart.
  let (language, ..rest) = lang.split("-")
  let region = if region != none { region } else { rest.at(0, default: none) }

  set document(title: title, keywords: keywords)
  set document(author: authors.map(a => content-to-string(a.name))) if authors not in (none, ())

  set page(
    footer: context {
      set text(font: sans, size: 8pt, fill: muted, number-width: "tabular")
      title
      h(1fr)
      counter(page).display()
    },
  )

  // Body: Public Sans with tabular numerals, so figures line up in prose and tables alike.
  set text(
    font: sans,
    size: fontsize,
    fill: ink,
    lang: language,
    region: region,
    number-width: "tabular",
  )
  set par(justify: true, leading: 0.65em, spacing: 1.1em)
  // The vendored families stop at SemiBold.
  show strong: set text(weight: "semibold")
  show link: underline.with(stroke: 0.5pt + hairline, offset: 2pt)
  show raw: set text(fill: ink)
  show raw.where(block: true): set block(
    width: 100%,
    inset: 8pt,
    fill: none,
    stroke: 0.5pt + hairline,
  )
  set list(marker: text(fill: muted)[–])

  // Headings: Newsreader, with the section number muted.
  set heading(numbering: sectionnumbering)
  show heading: it => block(
    above: if it.level == 1 { 2.2em } else { 1.6em },
    below: 0.9em,
    sticky: true,
    {
      set text(
        font: if it.level <= 2 { serif } else { sans },
        weight: "semibold",
        size: (1.6em, 1.22em, 1em).at(calc.min(it.level, 3) - 1),
      )
      set par(justify: false)
      if it.numbering != none {
        text(fill: muted, counter(heading).display(it.numbering))
        h(0.6em)
      }
      it.body
    },
  )

  // Figures and tables: numbered serif captions, "Figure 1." / "Table 1.", set to the left.
  set figure(gap: 0.9em)
  show figure: set block(above: 1.8em, below: 1.8em)
  show figure.caption: it => align(left, {
    set text(font: serif, size: 1em)
    set par(justify: false)
    text(weight: "semibold")[#it.supplement #context it.counter.display(it.numbering).]
    h(0.35em)
    it.body
  })

  // Booktabs: heavy rules above and below, a hairline under the header (Pandoc's `table.hline`),
  // no vertical rules.
  set table(stroke: none, inset: (x: 6pt, y: 4.5pt))
  set table.hline(stroke: 0.6pt + hairline)
  show table.cell.where(y: 0): set text(weight: "semibold")
  show table: it => box(stroke: (top: 1pt + ink, bottom: 1pt + ink), it)

  // Title block.
  // Display lines measured to their descenders, so the spacing below them is true.
  if title != none {
    block(below: 0.7em, text(font: serif, size: 2.6em, bottom-edge: "descender", title))
  }
  if subtitle != none {
    block(below: 1em, text(
      font: serif,
      size: 1.5em,
      fill: muted,
      bottom-edge: "descender",
      subtitle,
    ))
  }
  let byline = ()
  if authors not in (none, ()) {
    byline.push(authors.map(a => a.name).join(", ", last: " and "))
  }
  if date != none {
    byline.push(date)
  }
  if byline != () {
    block(below: 1.2em, text(fill: muted, byline.join(h(0.5em) + [·] + h(0.5em))))
  }
  line(length: 100%, stroke: 0.6pt + hairline)
  if abstract != none {
    block(above: 1.4em, below: 1.4em, {
      text(font: serif, weight: "semibold", abstract-title)
      h(0.6em)
      abstract
    })
  }

  if toc {
    v(1.2em)
    show outline.entry.where(level: 1): set block(above: 0.9em)
    show outline.entry.where(level: 1): set text(weight: "semibold")
    outline(title: toc_title, depth: toc_depth, indent: 1.4em)
    pagebreak(weak: true)
  }

  doc
}
