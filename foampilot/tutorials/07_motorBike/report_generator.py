#!/usr/bin/env python3
"""Report generator for Tutorial 7: MotorBike External Aero (simpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/07_motorBike
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
        title="MotorBike Aerodynamics Report",
        author="FoamPilot",
    )

    report.add_statistic("Re_L", 4e6, "", "Reynolds number (car length L=2m)")
    report.add_statistic("U_inlet", 30.0, "m/s", "Inlet velocity (108 km/h)")
    report.add_statistic("Cd", 0.35, "", "Drag coefficient")
    report.add_statistic("Cl", 0.05, "", "Lift coefficient")
    report.add_statistic("F_drag", 225, "N", "Drag force (~)")
    report.add_statistic("A_frontal", 0.7, "m²", "Frontal area")

    html_path = report.save_html_report(filename="motorbike_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="MotorBike Aerodynamics — Rapport complet",
        author="FoamPilot",
        filename="motorbike_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation aérodynamique externe à haute vitesse "
        "autour d'une moto avec simpleFoam et le modèle k-omega SST."
    )

    # Drag equation
    doc.add_section("Equation de traînée", "")
    doc.add_math(r"C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}")

    # Reynolds number
    doc.add_section("Nombre de Reynolds", "")
    doc.add_math(r"Re_L = \frac{U L}{\nu} = \frac{30 \times 2}{1.5 \times 10^{-5}} = 4 \times 10^6")

    # Results
    doc.add_section("Resultats aerodynamiques", "")
    doc.add_table(
        [["Coefficient", "Valeur", "Unité"],
         ["Cd", "0.35", ""],
         ["Cl", "0.05", ""],
         ["Force traînée", "225", "N"],
         ["A frontale", "0.7", "m²"]],
        headers=["Coefficient", "Valeur", "Unité"],
        caption="Aerodynamic results",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="MotorBike Aerodynamics Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "High-speed external flow around a motorcycle using RANS k-omega SST."
    )
    typst_doc.add_equation(
        r"C_d = F_d / (\frac{1}{2} \rho U^2 A)",
        caption="Drag coefficient",
        label="eq:cd",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re_L", "4e6"], ["Cd", "0.35"], ["Cl", "0.05"]],
        headers=["Parameter", "Value"],
        caption="Aerodynamic coefficients",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "motorbike_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — MotorBike")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
