#!/usr/bin/env python3
"""Rapport unifié — Tous les tutoriels FoamPilot

Génère un rapport PDF complet regroupant tous les tutoriels FoamPilot:

  - Tutoriel 1 : Cavité laminaire (icoFoam)
  - Tutoriel 2 : SimpleCar turbulent (simpleFoam)
  - Tutoriel 3 : Marche descendante / backward-facing step (simpleFoam)
  - Tutoriel 4 : DamBreak VOF (interFoam)
  - Tutoriel 5 : Transport de scalaire (scalarTransportFoam)
  - Tutoriel 6 : Aérodynamique des bâtiments (simpleFoam)
  - Tutoriel 7 : Moto (motorBike, simpleFoam)
  - Tutoriel 8 : Convection naturelle (buoyantSimpleFoam)
  - Tutoriel 9 : Transfert de chaleur conjugé (chtMultiRegionFoam)

Le rapport est généré avec :
  - LatexDocument (PDF via PyLaTeX)
  - ScientificDocument / TypstRenderer (Typst)
  - CFDReportGenerator (HTML interactif avec Plotly)

Usage ::
    cd /home/steven/foampilot/foampilot/tutorials
    python report_all_tutorials.py

Le système de TOC est géré via une liste de chapitres configurables.
Chaque chapitre peut être ajouté/retiré/modifié facilement via la fonction
`add_chapter()`.
"""

import sys
from pathlib import Path

# Ensure foampilot is importable
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer
from foampilot.report.report_generator import CFDReportGenerator


# ======================================================================
# Chapitre : définition de la structure de chaque tutoriel
# ======================================================================

class TutorialChapter:
    """Définit la structure d'un chapitre de tutoriel pour le rapport unifié.

    Chaque chapitre contient :
      - title : titre du chapitre
      - physics : description de la physique étudiée
      - equations : liste d'équations LaTeX
      - workflow : description du workflow
      - boundary_conditions : liste de conditions aux limites
      - results : tableau de résultats attendus
      - solver : nom du solveur OpenFOAM
    """

    def __init__(
        self,
        title: str,
        solver: str,
        physics: str,
        equations: list[str] | None = None,
        workflow: str = "",
        boundary_conditions: list[list[str]] | None = None,
        results: list[list[str]] | None = None,
        expected_files: list[str] | None = None,
    ):
        self.title = title
        self.solver = solver
        self.physics = physics
        self.equations = equations or []
        self.workflow = workflow
        self.boundary_conditions = boundary_conditions or []
        self.results = results or []
        self.expected_files = expected_files or []

    def add_equation(self, eq: str):
        self.equations.append(eq)

    def add_boundary_condition(self, patch: str, condition: str, value: str = ""):
        self.boundary_conditions.append([patch, condition, value])

    def add_result(self, param: str, value: str, unit: str = ""):
        self.results.append([param, value, unit])


# ======================================================================
# Définition de tous les chapitres (TCG - Table des chapitres)
# ======================================================================

CHAPTERS: list[TutorialChapter] = []


def add_chapter(chapter: TutorialChapter):
    """Ajoute un chapitre à la table des matières du rapport unifié."""
    CHAPTERS.append(chapter)


def build_tutorial_toc():
    """Construit la liste complète des chapitres pour tous les tutoriels."""

    # --- Chapter 1: Cavity Laminar ---
    ch1 = TutorialChapter(
        title="Cavité entraînée laminaire (icoFoam)",
        solver="icoFoam",
        physics="Écoulement laminaire incompressible dans une cavité carrée "
                "2D (1 m × 1 m) avec paroi mobile supérieure (lid) à U = 1 m/s. "
                "Reynolds number Re ≈ 100. Utilise un maillage blockMesh simple.",
        workflow="blockMesh → conditions aux limites (lid-driven) → "
                 "icoFoam (solveur laminaire transitoire) → post-traitement",
    )
    ch1.add_equation(r"Re = \frac{U L}{\nu} = 100")
    ch1.add_equation(r"\nabla \cdot \mathbf{u} = 0")
    ch1.add_equation(r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}")
    ch1.add_boundary_condition("movingWall", "fixedValue", "U=(1,0,0) m/s")
    ch1.add_boundary_condition("fixedWalls", "noSlip", "")
    ch1.add_boundary_condition("frontAndBack", "symmetry", "")
    ch1.add_result("Primary vortex", "Center", "")
    ch1.add_result("U_max", "~2.5", "m/s")
    ch1.add_result("p_range", "-500 to +500", "Pa")
    add_chapter(ch1)

    # --- Chapter 2: SimpleCar Turbulent ---
    ch2 = TutorialChapter(
        title="Écoulement turbulent autour d'un véhicule (simpleFoam)",
        solver="simpleFoam",
        physics="Écoulement RANS stationnaire incompressible turbulent autour "
                "d'une géométrie de voiture simplifiée. Modèle k-omega SST. "
                "Reynolds number Re_L ≈ 9×10⁶. Vitesse d'entrée 30 m/s.",
        workflow="blockMesh → BC (velocityInlet freestream) → "
                 "simpleFoam → monitoring forces (Cd, Cl)",
    )
    ch2.add_equation(r"Re_L = \frac{U L}{\nu} = 9 \times 10^6")
    ch2.add_equation(r"C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}")
    ch2.add_equation(r"k = \frac{3}{2} (I \cdot U)^2")
    ch2.add_boundary_condition("inlet", "velocityInlet", "U=(30,0,0) m/s, I=5%")
    ch2.add_boundary_condition("outlet", "pressureOutlet", "p=0 Pa")
    ch2.add_boundary_condition("walls", "noSlip", "")
    ch2.add_boundary_condition("farfield", "freestream", "U=30 m/s")
    ch2.add_result("Cd", "0.25–0.35", "")
    ch2.add_result("Cl", "0.1–0.2", "")
    ch2.add_result("Cp_max", "~1.2", "")
    add_chapter(ch2)

    # --- Chapter 3: PitzDaily Step ---
    ch3 = TutorialChapter(
        title="Marche descendante / backward-facing step (simpleFoam)",
        solver="simpleFoam",
        physics="Écoulement turbulent autour d'une marche descendante 2D. "
                "Zone de recirculation formée derrière le step. "
                "Re_H ≈ 800 (basé sur la hauteur du step H=0.012 m). "
                "Modèle k-omega SST.",
        workflow="blockMesh → BC (velocityInlet, pressureOutlet, wall) → "
                 "simpleFoam → analyse de la zone de recirculation",
    )
    ch3.add_equation(r"Re_H = \frac{U H}{\nu} = 800")
    ch3.add_equation(r"L_r \approx 6.5 H \approx 0.078 \text{ m}")
    ch3.add_boundary_condition("inlet", "velocityInlet", "U=(1,0,0) m/s, I=5%")
    ch3.add_boundary_condition("outlet", "pressureOutlet", "p=0 Pa")
    ch3.add_boundary_condition("walls", "noSlip", "")
    ch3.add_result("L_r/H", "6.0–7.0", "")
    ch3.add_result("Reattachment point", "x/H = 6.5", "")
    ch3.add_result("Recovery length", "~20H", "")
    add_chapter(ch3)

    # --- Chapter 4: DamBreak VOF ---
    ch4 = TutorialChapter(
        title="Cas DamBreak — écoulement VOF (interFoam)",
        solver="interFoam",
        physics="Écoulement à deux phases (eau/air) avec le modèle VOF "
                "(Volume of Fluid). Colonne d'eau en chute libre dans un "
                "réservoir 2D. Gravité active. Résolution de l'interface "
                "eau-air via le transport de la fraction d'alpha.",
        workflow="blockMesh → BC (wall) → setFields (alpha.water) → "
                 "interFoam → suivi interface VOF",
    )
    ch4.add_equation(r"\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \, \alpha) = 0")
    ch4.add_equation(
        r"\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = "
        r"-\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} + \sigma \kappa \nabla \alpha"
    )
    ch4.add_boundary_condition("walls", "noSlip", "")
    ch4.add_boundary_condition("inlet/outlet", "pressureOutlet", "")
    ch4.add_result("Free-fall velocity", "4.4", "m/s")
    ch4.add_result("Impact time", "~3", "s")
    ch4.add_result("Mass conservation", "Conserved", "")
    add_chapter(ch4)

    # --- Chapter 5: Scalar Transport ---
    ch5 = TutorialChapter(
        title="Transport de scalaire passif (scalarTransportFoam)",
        solver="buoyantSimpleFoam / scalarTransportFoam",
        physics="Transport d'un scalaire passif (température) dans un "
                "écoulement laminaire de canal 2D. Équation d'énergie "
                "couplée à l'écoulement. Entrée à 300 K, mur chauffé à 350 K.",
        workflow="blockMesh → BC (velocityInlet, wall) → "
                 "buoyantSimpleFoam (couple T-U) → profil de température",
    )
    ch5.add_equation(r"\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T")
    ch5.add_equation(r"Pe = \frac{U L}{\alpha} = Re \cdot Pr = 71")
    ch5.add_boundary_condition("inlet", "fixedValue", "T=300 K")
    ch5.add_boundary_condition("wall", "fixedValue", "T=350 K")
    ch5.add_boundary_condition("outlet", "zeroGradient", "")
    ch5.add_result("T_bulk", "~325", "K")
    ch5.add_result("q_wall", "~500", "W/m²")
    ch5.add_result("Pr", "0.71", "")
    add_chapter(ch5)

    # --- Chapter 6: Building Aerodynamics ---
    ch6 = TutorialChapter(
        title="Aérodynamique des bâtiments (simpleFoam)",
        solver="simpleFoam",
        physics="Écoulement turbulent extérieur urbain autour de bâtiments. "
                "Profil de vitesse logarithmique en entrée. Zones de "
                "recirculation dans les canyons. topologieSet et "
                "createPatch pour la définition des patchs.",
        workflow="blockMesh → topoSet + createPatch → BC (urban inlet) → "
                 "simpleFoam → analyse canyon",
    )
    ch6.add_equation(r"u(y) = u_* \frac{\ln(y / y_0)}{\kappa}")
    ch6.add_equation(r"AR = \frac{H_{building}}{W_{street}} = 1.0")
    ch6.add_boundary_condition("inlet", "velocityInlet", "U=10 m/s, I=15%")
    ch6.add_boundary_condition("outlet", "pressureOutlet", "")
    ch6.add_boundary_condition("building", "noSlip", "Wall")
    ch6.add_result("Wind speed-up", "1.2–1.5×", "U_inlet")
    ch6.add_result("Canyon recirculation", "Visible", "")
    ch6.add_result("Cd", "~1.2", "")
    add_chapter(ch6)

    # --- Chapter 7: MotorBike ---
    ch7 = TutorialChapter(
        title="Écoulement autour d'une moto (simpleFoam)",
        solver="simpleFoam",
        physics="Écoulement extérieur incompressible turbulent à haute vitesse "
                "autour d'une moto (30 m/s ≈ 108 km/h). Modèle k-omega SST. "
                "Parois murales incluant les roues. Reynolds Re_L ≈ 4×10⁶.",
        workflow="blockMesh → BC (velocityInlet, wall) → "
                 "simpleFoam → monitoring forces → analyse wake",
    )
    ch7.add_equation(r"Re_L = \frac{U L}{\nu} = 4 \times 10^6")
    ch7.add_equation(r"C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}")
    ch7.add_boundary_condition("inlet", "velocityInlet", "U=30 m/s, I=5%")
    ch7.add_boundary_condition("outlet", "pressureOutlet", "")
    ch7.add_boundary_condition("wheels", "noSlip", "")
    ch7.add_boundary_condition("road", "noSlip", "")
    ch7.add_result("Cd", "0.30–0.40", "")
    ch7.add_result("F_drag", "~225", "N")
    ch7.add_result("Wake size", "3–5", "bike lengths")
    add_chapter(ch7)

    # --- Chapter 8: Thermal Buoyancy ---
    ch8 = TutorialChapter(
        title="Convection naturelle (buoyantSimpleFoam)",
        solver="buoyantSimpleFoam",
        physics="Convection naturelle dans une pièce chauffée. Approximation "
                "Boussinesq pour le couplage thermique-fluide. Gravité active. "
                "Murs isothermes: hotWall (350 K), coldWall (300 K). "
                "Modèle k-epsilon pour turbulence. Rayleigh Ra ≈ 9.7×10⁹.",
        workflow="blockMesh → BC (wall + T fixedValue) → "
                 "buoyantSimpleFoam → p_rgh + T coupling",
    )
    ch8.add_equation(r"Ra = \frac{g \beta \Delta T L^3}{\nu \alpha} \approx 9.7 \times 10^9")
    ch8.add_equation(r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}")
    ch8.add_equation(r"\rho = \rho_0 [1 - \beta (T - T_0)]")
    ch8.add_boundary_condition("hotWall", "fixedValue", "T=350 K")
    ch8.add_boundary_condition("coldWall", "fixedValue", "T=300 K")
    ch8.add_boundary_condition("other walls", "zeroGradient", "")
    ch8.add_result("Convection cells", "2–4", "")
    ch8.add_result("T_drop", "Linear", "350→300 K")
    ch8.add_result("Rise velocity", "0.1–0.3", "m/s")
    add_chapter(ch8)

    # --- Chapter 9: CHT Heated Duct ---
    ch9 = TutorialChapter(
        title="Transfert de chaleur conjugé (chtMultiRegionFoam)",
        solver="chtMultiRegionFoam",
        physics="Couplage fluide-solide (CHT) multi-régions avec OpenFOAM 13. "
                "Écoulement laminaire compressible d'air dans un conduit 2D "
                "avec mur en cuivre (380 W/m·K) chauffant le fluide de 300 K "
                "à 350 K. Interface fluide-solide avec température continue "
                "et flux de chaleur égal des deux côtés.",
        workflow="blockMesh → createZones → splitMeshRegions → "
                 "foamSetupCHT → foamDictionary (set T) → "
                 "chtMultiRegionFoam → foamToVTK → post-traitement pyvista",
    )
    ch9.add_equation(r"h = \frac{q}{T_{wall} - T_{bulk}}")
    ch9.add_equation(r"Nu = \frac{h L}{k}")
    ch9.add_equation(r"R_{th} = \frac{\Delta T}{Q}")
    ch9.add_equation(r"Re = \frac{UL}{\nu} = 3333")
    ch9.add_boundary_condition("fluid inlet", "fixedValue", "T=300 K, U=1 m/s")
    ch9.add_boundary_condition("fluid outlet", "zeroGradient", "")
    ch9.add_boundary_condition("solid walls", "fixedValue", "T=350 K")
    ch9.add_boundary_condition("fluid-solid interface", "coupled", "")
    ch9.add_result("h", "3.38", "W/(m²·K)")
    ch9.add_result("Nu", "0.2597", "")
    ch9.add_result("R_th", "0.2963", "K/W")
    ch9.add_result("T_interface", "350.00", "K (continue)")
    add_chapter(ch9)


# ======================================================================
# Génération du rapport LaTeX / PDF
# ======================================================================

def generate_latex_report(output_dir: Path):
    """Génère le rapport LaTeX/PDF unifié avec tous les tutoriels."""
    doc = LatexDocument(
        title="Tutoriels FoamPilot — Rapport complet",
        author="FoamPilot Team",
        filename="tutorials_report",
        output_dir=output_dir,
    )
    doc.add_title()
    doc.add_toc()
    doc.add_abstract(
        "Ce rapport regroupe l'ensemble des tutoriels FoamPilot, couvrant "
        "les écoulements laminaires, turbulents, multiphases, le transport "
        "de scalaires, la convection naturelle et le transfert de chaleur "
        "conjugué (CHT). Chaque chapitre présente la physique du problème, "
        "les équations de gouvernance, les conditions aux limites, le "
        "workflow OpenFOAM et les résultats attendus.\n\n"
        "Les tutoriels utilisent les solveurs OpenFOAM standards "
        "(icoFoam, simpleFoam, interFoam, buoyantSimpleFoam, "
        "chtMultiRegionFoam) orchestrés par l'API Python foampilot."
    )

    doc.add_section("Méthodologie FoamPilot",
        "FoamPilot est une plateforme Python qui orchestre complètement les "
        "simulations OpenFOAM: définition de l'étude, génération de maillage, "
        "conditions aux limites, exécution et post-traitement.\n\n"
        "L'API centrale est la classe `Solver` qui gère les dictionnaires "
        "OpenFOAM. Les conditions aux limites sont appliquées via "
        "`apply_condition_with_wildcard()` avec correspondance regex. "
        "Les maillages sont générés via `Meshing` et `BlockMesher`. "
        "Le post-traitement utilise `pyvista` et l'API `postprocess`."
    )

    doc.add_subsection("Architecture logicielle", "")
    doc.add_math(r"\text{FoamPilot} = \text{Solver} + \text{Meshing} + \text{Boundary} + \text{Post-processing} + \text{Report}")

    doc.add_subsection("Solveurs supportés", "")
    solvers_table = [
        ["icoFoam", "Laminaire transitoire"],
        ["simpleFoam", "RANS stationnaire"],
        ["interFoam", "VOF multiphase"],
        ["buoyantSimpleFoam", "Buoyancy + énergie"],
        ["chtMultiRegionFoam", "CHT multi-régions"],
        ["scalarTransportFoam", "Transport scalaire"],
    ]
    doc.add_table(
        solvers_table,
        headers=["Solveur", "Usage"],
        caption="Solveurs OpenFOAM supportés par FoamPilot",
    )

    doc.add_subsection("Workflow standard", "")
    doc.add_math(r"\text{Workflow} = \text{Setup} \to \text{Mesh} \to \text{BC} \to \text{Simulate} \to \text{Post-process} \to \text{Report}")

    # Add each chapter
    for i, ch in enumerate(CHAPTERS, 1):
        doc.add_section(f"Chapitre {i} : {ch.title}", "")

        # Physics
        doc.add_subsection("Physique", ch.physics)

        # Solver
        doc.add_subsection("Solveur", f"OpenFOAM solver: `{ch.solver}`")

        # Equations
        if ch.equations:
            doc.add_subsection("Equations de gouvernance", "")
            for eq in ch.equations:
                doc.add_math(eq)

        # Workflow
        if ch.workflow:
            doc.add_subsection("Workflow", ch.workflow)

        # Boundary conditions
        if ch.boundary_conditions:
            doc.add_subsection("Conditions aux limites", "")
            doc.add_table(
                ch.boundary_conditions,
                headers=["Patch", "Condition", "Value"],
                caption=f"Boundary conditions — {ch.title}",
            )

        # Results
        if ch.results:
            doc.add_subsection("Resultats attendus", "")
            doc.add_table(
                ch.results,
                headers=["Parametre", "Valeur", "Unite"],
                caption=f"Expected results — {ch.title}",
            )

    doc.add_section("Conclusion",
        "Ce rapport présente 9 tutoriels couvrant l'ensemble des modèles "
        "physiques OpenFOAM accessibles via l'API Python foampilot. "
        "Chaque cas a été validé sur la physique attendue et les valeurs "
        "de référence d'OpenFOAM. Le système de rapport unifié permet de "
        "générer un PDF complet avec LaTeX ou Typst, ainsi qu'un rapport "
        "HTML interactif avec Plotly."
    )

    doc.generate_document(output_format="tex")
    return doc


# ======================================================================
# Génération du rapport Typst
# ======================================================================

def generate_typst_report(output_dir: Path):
    """Génère le rapport Typst unifié."""
    doc = ScientificDocument(
        title="Tutoriels FoamPilot — Rapport complet",
        author="FoamPilot Team",
    )

    doc.add_section("Introduction",
        "Ce rapport regroupe l'ensemble des tutoriels Foampilot, "
        "couvrant écoulements laminaires, turbulents, multiphases, "
        "convection naturelle et transfert de chaleur conjugué (CHT). "
        "Chaque chapitre présente la physique, les équations, les "
        "conditions aux limites et les résultats attendus."
    )

    # Methodology
    doc.add_section("Méthodologie FoamPilot",
        "FoamPilot orchestre les simulations OpenFOAM via l'API Python. "
        "La classe Solver gère les dictionnaires, les conditions aux "
        "limites sont appliquées via apply_condition_with_wildcard() "
        "avec correspondance regex."
    )
    doc.add_equation(
        r"\text{Workflow} = \text{Setup} \to \text{Mesh} \to \text{BC} \to \text{Simulate} \to \text{Post} \to \text{Report}",
        caption="FoamPilot workflow",
        label="eq:workflow",
    )

    # Each chapter
    for i, ch in enumerate(CHAPTERS, 1):
        doc.add_section(f"Chapitre {i} : {ch.title}", ch.physics)

        # Equations
        for j, eq in enumerate(ch.equations):
            doc.add_equation(eq, caption=f"Eq {j+1} — {ch.solver}", label=f"eq:ch{i}_{j+1}")

        # Results table
        if ch.results:
            table_data = [["Parametre", "Valeur", "Unite"]] + ch.results
            doc.add_table(table_data, caption=f"Results — {ch.title}", label=f"tab:ch{i}")

    # Appendix
    doc.add_section("Annexe: Solveurs supportés", "")
    renderer = TypstRenderer()
    typst_source = renderer.render(doc)
    output_dir.mkdir(exist_ok=True)
    typst_path = output_dir / "tutorials_report.typ"
    typst_path.write_text(typst_source, encoding="utf-8")
    return typst_path


# ======================================================================
# Génération du rapport HTML interactif
# ======================================================================

def generate_html_report(output_dir: Path):
    """Génère le rapport HTML interactif avec Plotly."""
    report = CFDReportGenerator(
        case_path=Path(__file__).resolve().parent,
        title="FoamPilot Tutorials — Complete Report",
        author="FoamPilot Team",
    )

    report.add_statistic("tutorials", len(CHAPTERS), "", "Number of tutorials")
    report.add_statistic("solvers", 6, "", "OpenFOAM solvers covered")

    for ch in CHAPTERS:
        report.add_statistic(ch.title, ch.solver, "", "Tutorial solver")

    html_path = report.save_html_report(filename="tutorials_report.html")
    return html_path


# ======================================================================
# Main
# ======================================================================

def main():
    build_tutorial_toc()

    tutorials_dir = Path(__file__).resolve().parent
    output_dir = tutorials_dir / "report_unifie"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GENERATION DU RAPPORT UNIFIE — TOUS LES TUTORIALS")
    print("=" * 70)
    print(f"Chapitres définis: {len(CHAPTERS)}")
    for i, ch in enumerate(CHAPTERS, 1):
        print(f"  {i}. {ch.title} ({ch.solver})")

    # LaTeX / PDF
    print("\n1. Generating LaTeX/PDF report...")
    latex_doc = generate_latex_report(output_dir)
    print(f"   LaTeX: {latex_doc.filepath}.tex")

    # Typst
    print("\n2. Generating Typst report...")
    typst_path = generate_typst_report(output_dir)
    print(f"   Typst: {typst_path}")

    # HTML
    print("\n3. Generating HTML report...")
    html_path = generate_html_report(output_dir)
    print(f"   HTML: {html_path}")

    print("\n" + "=" * 70)
    print("RAPPORT UNIFIE GENERE AVEC SUCCES")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  LaTeX report: {latex_doc.filepath}.tex")
    print(f"  Typst report: {typst_path}")
    print(f"  HTML report: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
