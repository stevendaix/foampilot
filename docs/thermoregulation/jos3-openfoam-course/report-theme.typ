// Shared visual theme for the skill-owned report entries.
// Owns page defaults, typography, paragraph rhythm, headings, code blocks,
// links, figures, and the optional running header.

#let report-accent = rgb("#1a5fb4")

// Relative rhythm values use B = body-size.
// report = user-selected R4 chapter-emphasis profile.
// longform = user-selected T1 no-indent longform profile.
#let report-rhythms = (
  report: (
    paragraph-gap: 0.84,
    leading: 0.87,
    first-line-indent: none,
    levels: (
      h1: (size: 1.55, before: 2.38, after: 1.04, weight: 700),
      h2: (size: 1.30, before: 1.64, after: 0.92, weight: 700),
      h3: (size: 1.13, before: 1.13, after: 0.86, weight: 650),
      h4: (size: 1.00, before: 0.96, after: 0.84, weight: 650),
    ),
  ),
  longform: (
    paragraph-gap: 0.65,
    leading: 0.82,
    first-line-indent: none,
    levels: (
      h1: (size: 1.55, before: 2.15, after: 0.88, weight: 700),
      h2: (size: 1.30, before: 1.48, after: 0.68, weight: 700),
      h3: (size: 1.13, before: 1.10, after: 0.58, weight: 650),
      h4: (size: 1.00, before: 0.88, after: 0.52, weight: 650),
    ),
  ),
)

#let report-theme(
  title: none,
  author: none,
  rhythm: "report",
  body-size: 11pt,
  first-line-indent: none,
  paragraph-spacing: none,
  running-header: false,
  body,
) = {
  if title != none or author != none {
    set document(title: title, author: author)
  }

  let p = report-rhythms.at(rhythm)
  let paragraph-gap = if paragraph-spacing == none {
    p.paragraph-gap * body-size
  } else {
    paragraph-spacing
  }
  let indent = if first-line-indent == none {
    p.first-line-indent
  } else {
    first-line-indent
  }
  let h1 = p.levels.h1
  let h2 = p.levels.h2
  let h3 = p.levels.h3
  let h4 = p.levels.h4

  set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, x: 2.2cm),
    numbering: "1",
    header: if running-header {
      context {
        if counter(page).get().first() > 0 {
          set text(size: 9pt, fill: luma(100))
          grid(
            columns: (1fr, 1fr),
            align(left)[#title],
            align(right)[#author],
          )
          v(-0.6em)
          line(length: 100%, stroke: 0.4pt + luma(180))
        }
      }
    } else {
      none
    },
  )

  set text(
    font: ("Libertinus Serif", "Noto Serif CJK SC"),
    size: body-size,
    lang: "zh",
    region: "cn",
  )

  set par(
    justify: true,
    leading: p.leading * body-size,
    spacing: paragraph-gap,
    first-line-indent: if indent == none {
      0pt
    } else {
      (amount: indent, all: false)
    },
  )

  set heading(numbering: "1.1")
  show heading: set text(
    font: ("Noto Sans", "Noto Sans CJK SC"),
    fill: report-accent,
  )

  show heading.where(level: 1): set text(size: h1.size * body-size, weight: h1.weight)
  show heading.where(level: 1): set block(
    above: h1.before * body-size,
    below: h1.after * body-size,
    sticky: true,
    breakable: false,
  )

  show heading.where(level: 2): set text(size: h2.size * body-size, weight: h2.weight)
  show heading.where(level: 2): set block(
    above: h2.before * body-size,
    below: h2.after * body-size,
    sticky: true,
    breakable: false,
  )

  show heading.where(level: 3): set text(size: h3.size * body-size, weight: h3.weight)
  show heading.where(level: 3): set block(
    above: h3.before * body-size,
    below: h3.after * body-size,
    sticky: true,
    breakable: false,
  )

  show heading.where(level: 4): set text(size: h4.size * body-size, weight: h4.weight)
  show heading.where(level: 4): set block(
    above: h4.before * body-size,
    below: h4.after * body-size,
    sticky: true,
    breakable: false,
  )

  show raw.where(block: true): it => block(
    fill: luma(246),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    text(size: 9pt, it),
  )
  show link: set text(fill: report-accent)
  show figure: set block(breakable: true)

  body
}
