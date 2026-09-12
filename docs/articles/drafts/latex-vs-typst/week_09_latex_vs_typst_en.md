# Generating Professional CFD Reports: LaTeX vs Typst with foampilot

*How to choose between LaTeX (via PyLaTeX) and Typst for generating professional PDF calculation notes from Python, with foampilot.*

---

## Introduction: The Report as a Finished Product

You've completed a simulation. You now need to produce a calculation note for your manager or a client. You could take Paraview screenshots and paste them into Word… but that would forget that CFD deserves better.

**Promise**: With foampilot, you generate professional PDF reports directly from Python. Two engines are available: LaTeX (via PyLaTeX) and Typst. Here's how to choose.

---

## Part 1: Needs of a CFD Report

A CFD report contains:
- **Meta-information**: title, author, date, version number
- **Statistics**: Reynolds numbers, dimensionless numbers, wall values
- **Tables**: parameter summary, convergence
- **Figures**: streamlines, slices, isosurfaces, mesh
- **Equations**: Reynolds number, wall shear stress, etc.
- **References**: literature, standards

---

## Part 2: LaTeX Solution with PyLaTeX

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Plane Poiseuille Flow — Validation",
    author="CFD Engineer",
    filename="poiseuille_report"
)

doc.add_section("Summary", "Validation of the simulation...")
doc.add_table(
    data=[["Re", "65800", "-", "Reynolds number"]],
    headers=["Parameter", "Value", "Unit", "Description"],
    caption="Physical parameters"
)
doc.add_figure("streamlines.png", "Velocity streamlines")
doc.add_math(r"\tau_w = \mu \frac{du}{dy}\bigg|_{y=0}")
doc.generate_pdf()
```

### LaTeX Advantages

- **Maturity**: 40 years old, massive community
- **Specialized packages**: `siunitx`, `booktabs`, `pgfplots`
- **Universal compatibility**: accepted by all journals
- **Full control** over typography and layout

### LaTeX Disadvantages

- **Steep learning curve**: sometimes cryptic syntax
- **Slow compilation**: especially with many figures
- **Error management**: difficult-to-interpret error messages
- **No dynamic reflow**: layout is fixed

---

## Part 3: Typst Solution

```python
from foampilot.report.typst_pdf import ScientificDocument

doc = ScientificDocument(
    title="Plane Poiseuille Flow — Validation",
    author="CFD Engineer"
)

doc.add_section("Summary", "Validation of the simulation...", level=1)
doc.add_table(
    [["Parameter", "Value", "Unit", "Description"],
     ["Re", "65800", "-", "Reynolds number"]],
    caption="Physical parameters",
    label="tab:params"
)
doc.add_figure("streamlines.png", caption="Velocity streamlines")
doc.render(doc)
```

### Typst Advantages

- **Modern**: compiled in Rust, very fast
- **Simple syntax**: HTML-like, more intuitive than LaTeX
- **Dynamic reflow**: layout adapts to content
- **Modern fonts**: native integration of modern fonts
- **Better SVG support**: high-quality vector rendering

### Typst Disadvantages

- **Young ecosystem**: fewer packages than LaTeX
- **Smaller community**: fewer online resources
- **Complex equations**: support still maturing
- **Python integration**: API still nascent

---

## Part 4: Detailed Comparison

| Criterion | LaTeX | Typst |
|-----------|-------|-------|
| Learning curve | Steep | Gentle |
| Compilation speed | Medium (2-5s) | Fast (<1s) |
| Error handling | Cryptic | Clear |
| Equations | ✅ Excellent | ✅ Good |
| Tables | ✅ Excellent | ✅ Good |
| Figures | ✅ Excellent | ✅ Good |
| Bibliography | Native BibTeX | Maturing |
| Community | Massive | Growing |
| Python integration | Mature PyLaTeX | Nascent API |
| PDF size | ~200 KB | ~150 KB |

---

## Part 5: Recommended Use Cases

### Choose LaTeX if:

- You already have a LaTeX team
- You need specialized packages (`siunitx`, `chemfig`, `tikz`)
- You publish in journals requiring LaTeX
- You already master LaTeX

### Choose Typst if:

- You're starting a new project
- You want a gentle learning curve
- You prioritize compilation speed
- You like modern syntax
- You want dynamic reflow

---

## Part 6: Integration in foampilot Workflow

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(
    case_path="./poiseuille",
    title="Plane Poiseuille Flow — Validation"
)

# Add statistics
report.add_statistic("Re", 65800, "-", "Reynolds number")
report.add_statistic("nu", 1.52e-5, "m²/s", "Kinematic viscosity")
report.add_statistic("y+_max", 0.8, "-", "Maximum y+")

# Generate both versions
report.save_latex_report(compile_pdf=True)      # LaTeX
report.save_typst_report()                       # Typst
```

### Side-by-side comparison

```python
import time

# LaTeX
start = time.time()
report.save_latex_report(compile_pdf=True)
latex_time = time.time() - start

# Typst
start = time.time()
report.save_typst_report()
typst_time = time.time() - start

print(f"LaTeX: {latex_time:.2f}s")
print(f"Typst: {typst_time:.2f}s")
```

---

## Conclusion: Best of Both Worlds

LaTeX and Typst are not in competition — they serve different needs. foampilot lets you choose based on your context.

**My personal advice**: if you're starting out, begin with Typst. If you already have a LaTeX background, stick with LaTeX. In both cases, foampilot generates the report for you — you just choose the engine.

In the next article, I'll show you how to automate figure generation and integration into your reports.

**Resources:**
- PyLaTeX: https://pylatex.readthedocs.io
- Typst: https://typst.app
- foampilot report module: https://stevendaix.github.io/foampilot/

---

*Article under improvement — version 1.0*
