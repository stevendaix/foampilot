#!/usr/bin/env python3
"""Report generator for Tutorial 2: SimpleCar Turbulent (simpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/02_simpleCar_turbulent
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
        title="SimpleCar Turbulent Flow Report",
        author="FoamPilot",
    )

    report.add_statistic("Re_L", 9e6, "", "Reynolds number (based on car length)")
    report.add_statistic("Cd", 0.30, "", "Drag coefficient")
    report.add_statistic("Cl", 0.1, "", "Lift coefficient")
    report.add_statistic("U_inlet", 30.0, "m/s", "Inlet velocity")
    report.add_statistic("I_inlet", 0.05, "", "Turbulence intensity")

    html_path = report.save_html_report(filename="simplecar_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="SimpleCar Aerodynamics — Rapport complet",
        author="FoamPilot",
        filename="simplecar_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente les résultats d'une simulation aérodynamique "
        "turbulente autour d'une voiture simplifiée avec simpleFoam et le "
        "modèle k-omega SST."
    )

    # Fluid properties
    doc.add_section("Propriétés du fluide", "")
    doc.add_table(
        [["Fluide", "Air"],
         ["Vitesse entrée", "30", "m/s"],
         ["Turbulence", "k-omega SST", ""],
         ["Intensité turbulente", "5", "%"]],
        headers=["Propriété", "Valeur", "Unité"],
        caption="Fluid properties and turbulence model",
    )

    # Governing equations
    doc.add_section("Équations RANS", "")
    doc.add_math(r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nabla \cdot \left[ \nu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right]")

    # k-omega SST model
    doc.add_section("Modèle k-omega SST", "")
    doc.add_math(r"k = \frac{3}{2} (I \cdot U)^2")

    # Performance metrics
    doc.add_section("Performances aérodynamiques", "")
    doc.add_table(
        [["Coefficient", "Valeur"],
         ["Cd", "0.30"],
         ["Cl", "0.10"]],
        headers=["Coefficient", "Valeur"],
        caption="Aerodynamic coefficients",
    )

    # Figures
    doc.add_section("Visualisations", "")
    for img_name in ["pressure_contour.png", "velocity_vectors.png", "cp_distribution.png"]:
        img_path = results_path / img_name
        if not img_path.exists():
            img_path = case_path / img_name
        if img_path.exists():
            doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(),
                          width="0.7\\textwidth")

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="SimpleCar Turbulent Flow Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "External aerodynamics of a simplified car using RANS k-omega SST."
    )
    typst_doc.add_equation(
        r"C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}",
        caption="Drag coefficient",
        label="eq:drag",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re_L", "9e6"], ["Cd", "0.30"], ["Cl", "0.10"]],
        headers=["Parameter", "Value"],
        caption="Flow and performance parameters",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "simplecar_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — SimpleCar Turbulent")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
