#!/usr/bin/env python3
"""Report generator for Tutorial 8: Thermal Buoyancy (buoyantSimpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/08_thermalBuoyancy
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
        title="Thermal Buoyancy Report",
        author="FoamPilot",
    )

    report.add_statistic("Ra", 9.7e9, "", "Rayleigh number")
    report.add_statistic("T_hot", 350.0, "K", "Hot wall temperature")
    report.add_statistic("T_cold", 300.0, "K", "Cold wall temperature")
    report.add_statistic("Delta_T", 50.0, "K", "Temperature difference")
    report.add_statistic("g", 9.81, "m/s²", "Gravity acceleration")
    report.add_statistic("L_char", 4.0, "m", "Characteristic length")

    html_path = report.save_html_report(filename="thermal_buoyancy_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="Convection thermique naturelle — Rapport complet",
        author="FoamPilot",
        filename="thermal_buoyancy_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation de convection naturelle "
        "dans une pièce chauffée avec buoyantSimpleFoam et l'approximation Boussinesq."
    )

    # Boussinesq approximation
    doc.add_section("Approximation Boussinesq", "")
    doc.add_math(r"\rho = \rho_0 [1 - \beta (T - T_0)]")

    # Modified pressure
    doc.add_section("Pression modifiée", "")
    doc.add_math(r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}")

    # Rayleigh number
    doc.add_section("Nombre de Rayleigh", "")
    doc.add_math(
        r"Ra = \frac{g \beta \Delta T L^3}{\nu \alpha} = \frac{9.81 \times 3.2 \times 10^{-3} \times 50 \times 4^3}{1.5 \times 10^{-5} \times 2.2 \times 10^{-5}} \approx 9.7 \times 10^9"
    )

    # Boundary conditions
    doc.add_section("Conditions aux limites", "")
    doc.add_table(
        [["Patch", "Temperature", "Condition"],
         ["hotWall", "350", "K fixedValue"],
         ["coldWall", "300", "K fixedValue"],
         ["other walls", "adiabatic", "zeroGradient"]],
        headers=["Patch", "Température", "Condition"],
        caption="Wall boundary conditions",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="Thermal Buoyancy Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Natural convection in a heated room with Boussinesq approximation "
        "and k-epsilon turbulence model."
    )
    typst_doc.add_equation(
        r"Ra = g \beta \Delta T L^3 / (\nu \alpha)",
        caption="Rayleigh number",
        label="eq:rayleigh",
    )
    typst_doc.add_equation(
        r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}",
        caption="Modified pressure",
        label="eq:pressure",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Ra", "9.7e9"], ["T_hot", "350 K"], ["T_cold", "300 K"]],
        headers=["Parameter", "Value"],
        caption="Simulation parameters",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "thermal_buoyancy_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — Thermal Buoyancy")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
