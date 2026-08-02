#!/usr/bin/env python3
"""Report generator for Tutorial 6: Building Aerodynamics (simpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/06_buildingAero
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
        title="Building Aerodynamics Report",
        author="FoamPilot",
    )

    report.add_statistic("Re_L", 6.7e6, "", "Reynolds number (building height)")
    report.add_statistic("U_inlet", 10.0, "m/s", "Inlet wind speed")
    report.add_statistic("I_inlet", 0.15, "", "Turbulence intensity")
    report.add_statistic("AR", 1.0, "", "Canyon aspect ratio (H/W)")
    report.add_statistic("Cd", 1.2, "", "Mean drag coefficient")

    html_path = report.save_html_report(filename="building_aero_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="Aérodynamique des bâtiments — Rapport complet",
        author="FoamPilot",
        filename="building_aero_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation aérodynamique urbaine autour "
        "d'un quartier de bâtiments avec simpleFoam et le modèle k-omega SST."
    )

    # Urban boundary layer profile
    doc.add_section("Profil de couche limite urbaine", "")
    doc.add_math(r"u(y) = u_* \frac{\ln(y / y_0)}{\kappa}")

    # Canyon aspect ratio
    doc.add_section("Ratio d'aspect du canyon", "")
    doc.add_math(r"AR = \frac{H_{building}}{W_{street}} = 1.0")

    # Results
    doc.add_section("Resultats", "")
    doc.add_table(
        [["Parametre", "Valeur", "Unité"],
         ["Vitesse entrée", "10", "m/s"],
         ["Intensité turbulente", "15", "%"],
         ["Cd moyen", "1.2", ""],
         ["AR canyon", "1.0", ""]],
        headers=["Paramètre", "Valeur", "Unité"],
        caption="Building aerodynamics parameters",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="Building Aerodynamics Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Urban boundary layer flow simulation around buildings using "
        "k-omega SST turbulence model."
    )
    typst_doc.add_equation(
        r"u(y) = u_* \frac{\ln(y/y_0)}{\kappa}",
        caption="Logarithmic wind profile",
        label="eq:log_profile",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re", "6.7e6"], ["AR", "1.0"], ["Cd", "1.2"]],
        headers=["Parameter", "Value"],
        caption="Simulation parameters",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "building_aero_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — Building Aerodynamics")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
