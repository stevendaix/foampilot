#!/usr/bin/env python3
"""Report generator for Tutorial 3: PitzDaily Step (simpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/03_pitzDaily_step
    python report_generator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer
from foampilot.report.report_generator import CFDReportGenerator


def main():
    case_path = Path.cwd()
    results_path = case_path / "postProcessing"
    results_path.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. CFDReportGenerator — HTML report
    # ------------------------------------------------------------------
    report = CFDReportGenerator(
        case_path=case_path,
        title="Backward-Facing Step Report",
        author="FoamPilot",
    )

    report.add_statistic("Re_H", 800, "", "Reynolds number (based on step height)")
    report.add_statistic("L_r", 6.5, "H", "Reattachment length ratio")
    report.add_statistic("U_inlet", 1.0, "m/s", "Inlet velocity")
    report.add_statistic("H_step", 0.012, "m", "Step height")

    html_path = report.save_html_report(filename="step_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="Backward-Facing Step — Rapport complet",
        author="FoamPilot",
        filename="step_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation laminaire/turbulente autour "
        "d'une marche descendante (backward-facing step) avec simpleFoam "
        "et le modèle k-omega SST."
    )

    # Governing equations
    doc.add_section("Equations governing", "")
    doc.add_math(r"Re_H = \frac{U H}{\nu}")

    # Recirculation zone
    doc.add_section("Zone de recirculation", "")
    doc.add_math(r"L_r \approx 6.5 H")

    # Results table
    doc.add_section("Resultats", "")
    doc.add_table(
        [["Parametre", "Valeur"],
         ["Reynolds number", "800"],
         ["Reattachment ratio", "6.5"]],
        headers=["Parameter", "Value"],
        caption="Key flow parameters",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="Backward-Facing Step Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Turbulent flow over a backward-facing step with k-omega SST model."
    )
    typst_doc.add_equation(
        r"L_r = 6.5 H",
        caption="Reattachment length",
        label="eq:reattachment",
    )
    typst_doc.add_table(
        [["H", "0.012 m"], ["L", "1.0 m"], ["U", "1.0 m/s"]],
        headers=["Parameter", "Value"],
        caption="Geometric parameters",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "step_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — PitzDaily Step")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
