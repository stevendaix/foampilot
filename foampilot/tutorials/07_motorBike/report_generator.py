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

    # Load statistics from post-processing if available
    stats_file = case_path / "all_stats.json"
    stats = {}
    if stats_file.exists():
        import json
        with open(stats_file, "r") as f:
            stats = json.load(f)

    cell_stats = stats.get("cell_region_stats_U", {})
    mesh_stats = stats.get("mesh_stats", {})

    u_mean = cell_stats.get("mean", 20.0)
    u_min = cell_stats.get("min", 0.0)
    u_max = cell_stats.get("max", 30.0)
    num_cells = mesh_stats.get("num_cells", "N/A")
    num_points = mesh_stats.get("num_points", "N/A")

    # ------------------------------------------------------------------
    # 1. CFDReportGenerator — HTML report
    # ------------------------------------------------------------------
    report = CFDReportGenerator(
        case_path=case_path,
        title="MotorBike Aerodynamics Report",
        author="FoamPilot",
    )

    report.add_statistic("Re_L", 4e6, "", "Reynolds number (car length L=2m)")
    report.add_statistic("U_inlet", 20.0, "m/s", "Inlet velocity")
    report.add_statistic("U_mean", round(u_mean, 3), "m/s", "Mean velocity in domain")
    report.add_statistic("U_min", round(u_min, 3), "m/s", "Min velocity in domain")
    report.add_statistic("U_max", round(u_max, 3), "m/s", "Max velocity in domain")
    report.add_statistic("num_cells", num_cells, "", "Number of cells")
    report.add_statistic("num_points", num_points, "", "Number of points")

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
        "Ce rapport presente la simulation aerodynamique externe a haute vitesse "
        "autour d'une moto avec simpleFoam et le modele SpalartAllmaras."
    )

    # Drag equation
    doc.add_section("Equation de trainee", "")
    doc.add_math(r"C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}")

    # Reynolds number
    doc.add_section("Nombre de Reynolds", "")
    doc.add_math(r"Re_L = \frac{U L}{\nu} = \frac{20 \times 2}{1.5 \times 10^{-5}} \approx 2.7 \times 10^6")

    # Results
    doc.add_section("Resultats aerodynamiques", "")
    doc.add_table(
        [["Coefficient", "Valeur", "Unite"],
         ["U_mean", f"{u_mean:.3f}", "m/s"],
         ["U_min", f"{u_min:.3f}", "m/s"],
         ["U_max", f"{u_max:.3f}", "m/s"],
         ["Num cells", str(num_cells), ""],
         ["Num points", str(num_points), ""]],
        headers=["Coefficient", "Valeur", "Unite"],
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
        "High-speed external flow around a motorcycle using RANS SpalartAllmaras."
    )
    typst_doc.add_equation(
        r"C_d = F_d / (\frac{1}{2} \rho U^2 A)",
        caption="Drag coefficient",
        label="eq:cd",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re_L", "2.7e6"], ["U_mean", f"{u_mean:.3f}"], ["U_max", f"{u_max:.3f}"]],
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
