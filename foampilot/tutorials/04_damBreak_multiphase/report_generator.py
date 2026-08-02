#!/usr/bin/env python3
"""Report generator for Tutorial 4: DamBreak VOF (interFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/04_damBreak_multiphase
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
        title="DamBreak VOF Simulation Report",
        author="FoamPilot",
    )

    report.add_statistic("Re", 2800, "", "Reynolds number (water-air interface)")
    report.add_statistic("U_freefall", 4.4, "m/s", "Free-fall velocity (sqrt(2gh))")
    report.add_statistic("g", 9.81, "m/s²", "Gravity acceleration")

    html_path = report.save_html_report(filename="damBreak_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="DamBreak VOF — Rapport complet",
        author="FoamPilot",
        filename="damBreak_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation d'un écoulement à deux phases "
        "(eau/air) avec le modèle VOF et le solveur interFoam."
    )

    # VOF equation
    doc.add_section("Equation VOF", "")
    doc.add_math(r"\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \, \alpha) = 0")

    # Momentum equation
    doc.add_section("Equation d'impulsion", "")
    doc.add_math(
        r"\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = "
        r"-\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} + \sigma \kappa \nabla \alpha"
    )

    # Results
    doc.add_section("Resultats", "")
    doc.add_table(
        [["Parametre", "Valeur"],
         ["Vitesse libre-fin", "4.4", "m/s"],
         ["Temps impact mur", "~3", "s"]],
        headers=["Paramètre", "Valeur", "Unité"],
        caption="DamBreak simulation parameters",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="DamBreak VOF Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Two-phase flow simulation using VOF model with interFoam solver."
    )
    typst_doc.add_equation(
        r"\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \alpha) = 0",
        caption="VOF transport equation",
        label="eq:vof",
    )
    typst_doc.add_table(
        [["Phase", "α"], ["Water", "1"], ["Air", "0"]],
        headers=["Phase", "Alpha"],
        caption="Phase fractions",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "damBreak_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — DamBreak VOF")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
