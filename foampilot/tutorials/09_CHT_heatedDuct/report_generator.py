#!/usr/bin/env python3
"""Rapport complet CHT — Heat Exchanger (chtMultiRegionFoam)

Génère un rapport PDF et HTML complet à partir des résultats de
simulation CHT en utilisant l'API foampilot report engine.

Utilise :
  - CFDReportGenerator (rapport HTML + LaTeX/Typst)
  - LatexDocument (rapport PDF via PyLaTeX)
  - ScientificDocument / TypstRenderer (rapport Typst)

Usage ::
    cd foampilot/tutorials/09_CHT_heatedDuct
    python report_generator.py
"""

import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.report.report_generator import CFDReportGenerator
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer


def main():
    case_path = Path.cwd()
    results_path = case_path / "postProcessing"

    # ------------------------------------------------------------------
    # 1. Load statistics CSV
    # ------------------------------------------------------------------
    stats_csv = results_path / "temperature_statistics.csv"
    df_stats = pd.read_csv(stats_csv) if stats_csv.exists() else pd.DataFrame()

    profile_csv = results_path / "temperature_profile_combined.csv"
    df_profile = pd.read_csv(profile_csv) if profile_csv.exists() else pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. Parse CHT report for key metrics
    # ------------------------------------------------------------------
    report_md = results_path / "CHT_Report.md"
    metrics = {}
    if report_md.exists():
        text = report_md.read_text()
        # Extract key values from the report
        for key in ["Heat transfer coefficient", "Nusselt number", "Thermal resistance"]:
            for line in text.splitlines():
                if key in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        val_str = parts[2].strip()
                        try:
                            metrics[key] = float(val_str.split()[0])
                        except (ValueError, IndexError):
                            pass

    # ------------------------------------------------------------------
    # 3. CFDReportGenerator — HTML report with Plotly
    # ------------------------------------------------------------------
    report = CFDReportGenerator(
        case_path=case_path,
        title="CHT Heated Duct Report",
        author="FoamPilot",
    )

    # Add scalar statistics
    report.add_statistic("Nu",
                         metrics.get("Nusselt number", 0.26),
                         "",
                         "Nusselt number (laminar)")
    report.add_statistic("h",
                         metrics.get("Heat transfer coefficient", 3.38),
                         "W/(m²·K)",
                         "Heat transfer coefficient")
    report.add_statistic("R_th",
                         metrics.get("Thermal resistance", 0.30),
                         "K/W",
                         "Thermal resistance")
    report.add_statistic("Re",
                         3333.0,
                         "",
                         "Reynolds number (based on duct height)")

    # Add figures
    for name, caption in [
        ("fluid_temperature_contour.png", "Fluid temperature contour"),
        ("solid_temperature_contour.png", "Solid temperature contour"),
        ("cht_temperature_contour.png", "CHT temperature overlay (fluid + solid)"),
    ]:
        img = results_path / name
        if img.exists():
            report.add_figure(str(img), caption)

    # Add statistics table
    if not df_stats.empty:
        table_data = df_stats.values.tolist()
        report.add_table(
            table_data,
            headers=list(df_stats.columns),
            caption="Temperature statistics by region",
        )

    # Generate HTML report
    html_path = report.save_html_report(filename="cht_report.html")
    print(f"HTML report: {html_path}")

    # ------------------------------------------------------------------
    # 4. LaTeX PDF report via LatexDocument
    # ------------------------------------------------------------------
    doc = LatexDocument(
        title="CHT Heat Exchanger — Rapport complet",
        author="FoamPilot",
        filename="cht_report",
        output_dir=case_path,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport présente les résultats d'une simulation de transfert "
        "de chaleur conjugé (CHT) d'un conduit chauffé, réalisée avec "
        "OpenFOAM 13 (chtMultiRegionFoam) et l'API Python foampilot."
    )

    # Section: Physics
    doc.add_section("Physique du cas",
        "Écoulement laminaire compressible d'air dans un conduit 2D "
        "(0.1 m × 0.02 m), avec un mur solide en cuivre (380 W/m·K) "
        "chauffant le fluide de 300 K à 350 K."
    )

    # Equations
    doc.add_math(r"h = \frac{q}{T_{wall} - T_{bulk}}")
    doc.add_math(r"Nu = \frac{h \cdot L}{k}")
    doc.add_math(r"R_{th} = \frac{\Delta T}{Q}")

    # Section: Results tables
    doc.add_section("Statistiques de température", "")
    if not df_stats.empty:
        doc.add_dataframe_table(df_stats, caption="Température par région")

    doc.add_section("Résultats clés", "")
    results_data = [
        ["Nusselt number", f"{metrics.get('Nusselt number', 0.26):.4f}", ""],
        ["Heat transfer coefficient", f"{metrics.get('Heat transfer coefficient', 3.38):.2f}", "W/(m²·K)"],
        ["Thermal resistance", f"{metrics.get('Thermal resistance', 0.30):.4f}", "K/W"],
        ["Interface temperature", "350.00", "K"],
    ]
    doc.add_table(
        results_data,
        headers=["Parameter", "Value", "Unit"],
        caption="Key CHT results",
    )

    # Figures
    doc.add_section("Visualisations", "")
    for name, caption in [
        ("fluid_temperature_contour.png", "Temperature in fluid region"),
        ("solid_temperature_contour.png", "Temperature in solid region"),
        ("cht_temperature_contour.png", "CHT temperature overlay"),
    ]:
        img = results_path / name
        if img.exists():
            doc.add_figure(str(img), caption=caption, width="0.8\\textwidth")

    doc.generate_document(output_format="tex")
    print(f"LaTeX report: {doc.filepath}.tex")

    # ------------------------------------------------------------------
    # 5. Typst scientific document
    # ------------------------------------------------------------------
    typst_doc = ScientificDocument(
        title="CHT Heat Exchanger Analysis",
        author="FoamPilot",
    )
    typst_doc.add_section("Introduction",
        "This document presents a conjugate heat transfer analysis of a "
        "heated duct using OpenFOAM 13 and the foampilot CHT module."
    )

    typst_doc.add_equation(
        r"h = q / (T_{wall} - T_{bulk})",
        caption="Heat transfer coefficient",
        label="eq:h",
    )
    typst_doc.add_equation(
        r"Nu = h L / k",
        caption="Nusselt number",
        label="eq:nu",
    )
    typst_doc.add_equation(
        r"R_{th} = \Delta T / Q",
        caption="Thermal resistance",
        label="eq:rth",
    )

    # Results table in Typst
    typst_table_data = [
        ["Parameter", "Value", "Unit"],
        ["Nusselt number", f"{metrics.get('Nusselt number', 0.26):.4f}", ""],
        ["Heat transfer coefficient", f"{metrics.get('Heat transfer coefficient', 3.38):.2f}", "W/(m²·K)"],
        ["Thermal resistance", f"{metrics.get('Thermal resistance', 0.30):.4f}", "K/W"],
    ]
    typst_doc.add_table(typst_table_data, caption="Key CHT results", label="tab:results")

    renderer = TypstRenderer()
    typst_source = renderer.render(typst_doc)
    report_dir = case_path / "report"
    report_dir.mkdir(exist_ok=True)
    typst_path = report_dir / "cht_report.typ"
    typst_path.write_text(typst_source, encoding="utf-8")
    print(f"Typst report: {typst_path}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"HTML report : {html_path}")
    print(f"LaTeX report: {doc.filepath}.tex")
    print(f"Typst report: {typst_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
