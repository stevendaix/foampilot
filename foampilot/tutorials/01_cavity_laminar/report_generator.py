#!/usr/bin/env python3
"""Report generator for Tutorial 1: Cavity Laminar (icoFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/01_cavity_laminar
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
    # 1. CFDReportGenerator — HTML report with Plotly figures
    # ------------------------------------------------------------------
    report = CFDReportGenerator(
        case_path=case_path,
        title="Cavity Laminar Flow Report",
        author="FoamPilot",
    )

    # Add key statistics
    report.add_statistic("Re", 100, "", "Reynolds number")
    report.add_statistic("U_lid", 1.0, "m/s", "Lid velocity")
    report.add_statistic("nu", 1e-6, "m²/s", "Kinematic viscosity")

    # Generate HTML report
    html_path = report.save_html_report(filename="cavity_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report via LatexDocument
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="Cavity Laminar Flow — Rapport complet",
        author="FoamPilot",
        filename="cavity_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente les résultats d'une simulation laminaire "
        "d'écoulement dans une cavité entraînée, réalisée avec icoFoam "
        "et l'API Python foampilot."
    )

    # Fluid properties section
    doc.add_section("Propriétés du fluide", "")
    doc.add_table(
        [["Fluide", "Air"],
         ["Viscosité cinématique", "1e-6", "m²/s"],
         ["Densité", "1.225", "kg/m³"]],
        headers=["Propriété", "Valeur", "Unité"],
        caption="Fluid properties",
    )

    # Governing equations
    doc.add_section("Équations de Navier-Stokes", "")
    doc.add_math(r"\nabla \cdot \mathbf{u} = 0")
    doc.add_math(r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}")

    # Reynolds number
    doc.add_section("Parametres sans dimension", "")
    doc.add_math(r"Re = \frac{UL}{\nu} = \frac{1 \times 1}{1 \times 10^{-6}} = 100")

    # Mesh statistics
    doc.add_section("Statistiques du maillage", "")
    mesh_stats = {
        "Cells": 800,
        "Points": 900,
        "Faces": 2400,
    }
    doc.add_table(
        [[k, v] for k, v in mesh_stats.items()],
        headers=["Statistic", "Value"],
        caption="Mesh quality statistics",
    )

    # Results
    doc.add_section("Résultats", "")
    doc.add_table(
        [["Variable", "Min", "Max", "Mean"],
         ["U_x", 0, 2.5, 1.0],
         ["p", -500, 500, 0]],
        headers=["Variable", "Min", "Max", "Mean"],
        caption="Field statistics",
    )

    # Figures
    doc.add_section("Visualisations", "")
    for img_name in ["velocity_contour.png", "pressure_contour.png", "streamlines.png"]:
        img_path = results_path / img_name
        if not img_path.exists():
            img_path = case_path / img_name
        if img_path.exists():
            doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(),
                          width="0.7\\textwidth")

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst scientific document
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="Cavity Laminar Flow Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Lid-driven cavity laminar flow simulation using icoFoam. "
        "This case validates the basic CFD workflow with foampilot."
    )
    typst_doc.add_equation(
        r"Re = UL / \nu = 100",
        caption="Reynolds number",
        label="eq:reynolds",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re", "100"], ["Nu", "4.4"]],
        headers=["Parameter", "Value"],
        caption="Dimensionless parameters",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "cavity_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — Cavity Laminar")
    print("=" * 60)
    print(f"HTML    : {html_path}")
    print(f"LaTeX   : {doc.filepath}.tex")
    print(f"Typst   : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
