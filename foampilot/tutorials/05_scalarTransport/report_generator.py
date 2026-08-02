#!/usr/bin/env python3
"""Report generator for Tutorial 5: Scalar Transport (buoyantSimpleFoam).

Generates a complete report using foampilot's report engine:
- CFDReportGenerator (HTML with Plotly)
- LatexDocument (LaTeX/PDF via PyLaTeX)
- ScientificDocument + TypstRenderer (Typst PDF)

Usage:
    cd foampilot/tutorials/05_scalarTransport
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
        title="Scalar Transport Report",
        author="FoamPilot",
    )

    report.add_statistic("Re", 100, "", "Reynolds number")
    report.add_statistic("Pe", 71, "", "Peclet number")
    report.add_statistic("Pr", 0.71, "", "Prandtl number")
    report.add_statistic("T_inlet", 300.0, "K", "Inlet temperature")
    report.add_statistic("T_wall", 350.0, "K", "Wall temperature")

    html_path = report.save_html_report(filename="scalar_transport_report.html")
    print(f"HTML report generated: {html_path}")

    # ------------------------------------------------------------------
    # 2. LaTeX report
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="Transport de scalaire passif — Rapport complet",
        author="FoamPilot",
        filename="scalar_transport_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente la simulation du transport d'un scalaire passif "
        "(température) dans un écoulement laminaire, avec scalarTransportFoam."
    )

    # Scalar transport equation
    doc.add_section("Equation de transport du scalaire", "")
    doc.add_math(r"\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T")

    # Peclet number
    doc.add_section("Nombre de Peclet", "")
    doc.add_math(r"Pe = \frac{UL}{\alpha} = Re \cdot Pr = 100 \times 0.71 = 71")

    # Boundary conditions
    doc.add_section("Conditions aux limites", "")
    doc.add_table(
        [["Patch", "Condition", "T (K)"],
         ["inlet", "fixedValue", "300"],
         ["wall", "fixedValue", "350"],
         ["outlet", "zeroGradient", ""]],
        headers=["Patch", "Condition", "Value"],
        caption="Temperature boundary conditions",
    )

    # Results
    doc.add_section("Resultats", "")
    doc.add_table(
        [["T_mean", "325", "K"],
         ["T_bulk", "~325", "K"],
         ["q_wall", "~500", "W/m²"]],
        headers=["Paramètre", "Valeur", "Unité"],
        caption="Scalar transport results",
    )

    doc.generate_document(output_format="tex")
    print(f"LaTeX report generated: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 3. Typst report
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="Scalar Transport Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "Passive scalar (temperature) transport in laminar channel flow."
    )
    typst_doc.add_equation(
        r"Pe = UL / \alpha",
        caption="Peclet number",
        label="eq:peclet",
    )
    typst_doc.add_table(
        [["Parameter", "Value"], ["Re", "100"], ["Pr", "0.71"], ["Pe", "71"]],
        headers=["Parameter", "Value"],
        caption="Dimensionless numbers",
    )

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    typst_path = case_path / "report" / "scalar_transport_typst_report.typ"
    typst_path.parent.mkdir(exist_ok=True)
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report generated: {typst_path}")

    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE — Scalar Transport")
    print("=" * 60)
    print(f"HTML  : {html_path}")
    print(f"LaTeX : {doc.filepath}.tex")
    print(f"Typst : {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
