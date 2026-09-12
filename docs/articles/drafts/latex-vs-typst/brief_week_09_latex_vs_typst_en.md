# Brief — Week 9 : "LaTeX vs Typst: Generating Professional CFD Reports"

**Goal**: Compare the two report engines in foampilot (LaTeX via PyLaTeX and Typst) for CFD calculation notes.
**Angle**: Technical comparison, tutorial, tool selection.

---

## Proposed Structure

### Introduction — The report as a finished product

Storytelling: You've completed a simulation. You now need to produce a calculation note for your manager or a client. You could take Paraview screenshots and paste them into Word… but that would forget that CFD deserves better.

**Promise**: With foampilot, you generate professional PDF reports directly from Python. Two engines are available: LaTeX (via PyLaTeX) and Typst. Here's how to choose.

### Section 1 — Needs of a CFD report

A CFD report contains:
- **Meta-information**: title, author, date, version number
- **Statistics**: Reynolds numbers, dimensionless numbers, wall values
- **Tables**: parameter summary, convergence
- **Figures**: streamlines, slices, isosurfaces, mesh
- **Equations**: Reynolds number, wall shear stress, etc.
- **References**: literature, standards

### Section 2 — LaTeX solution with PyLaTeX

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

**LaTeX advantages**:
- Maturity: 40 years old, massive community
- Specialized packages: `siunitx`, `booktabs`, `pgfplots`
- Universal compatibility
- Full typography control

**LaTeX disadvantages**:
- Steep learning curve
- Slow compilation (especially with figures)
- Difficult compilation error management
- No dynamic reflow

### Section 3 — Typst solution

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
# Typst supports HTML-like for equations
doc.render(doc)
```

**Typst advantages**:
- Modern: compiled in Rust, fast
- Simpler syntax (HTML-like)
- Dynamic reflow
- Native modern font integration
- Better SVG support

**Typst disadvantages**:
- Younger ecosystem
- Fewer specialized packages
- Smaller community
- Complex equation support still maturing

### Section 4 — Detailed comparison

| Criterion | LaTeX | Typst |
|-----------|-------|-------|
| Learning curve | Steep | Gentle |
| Compilation speed | Medium | Fast |
| Error handling | Cryptic | Clear |
| Equations | ✅ Excellent | ✅ Good |
| Tables | ✅ Excellent | ✅ Good |
| Figures | ✅ Excellent | ✅ Good |
| Bibliography | Native BibTeX | Maturing |
| Community | Massive | Growing |
| Python integration | Mature PyLaTeX | Nascent API |

### Section 5 — Recommended use cases

**Choose LaTeX if**:
- You already have a LaTeX team
- You need specialized packages (`siunitx`, `chemfig`, etc.)
- You publish in journals requiring LaTeX
- You already master LaTeX

**Choose Typst if**:
- You're starting a new project
- You want a gentle learning curve
- You prioritize compilation speed
- You like modern syntax
- You want dynamic reflow

### Section 6 — Integration in foampilot workflow

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(
    case_path="./poiseuille",
    title="Plane Poiseuille Flow — Validation"
)

# Add statistics
report.add_statistic("Re", 65800, "-", "Reynolds number")
report.add_statistic("nu", 1.52e-5, "m²/s", "Kinematic viscosity")

# Generate both versions
report.save_latex_report(compile_pdf=True)      # LaTeX
report.save_typst_report()                       # Typst
```

### Conclusion — Best of both worlds

CTA: *"In the next article, I'll show you how to automate figure generation and integration into your reports."*

---

## Code to prepare

- [ ] Complete LaTeX report script
- [ ] Complete Typst report script
- [ ] Side-by-side comparison script
- [ ] Compilation benchmark (time, PDF size)

## Images to prepare

- [ ] LaTeX PDF screenshot
- [ ] Typst PDF screenshot
- [ ] Compilation time graph
- [ ] Side-by-side visual comparison
