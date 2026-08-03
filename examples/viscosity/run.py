#!/usr/bin/env python3
"""
Exemple complet : écoulement à deux phases (eau/air) avec solveur VoF
et foampilot.

Ce cas est une adaptation foampilot du cas Template fourni dans
``examples/viscosity/Template/``.  Il simule l'introduction d'un
liquide visqueux (propriétés proches de l'huile) au moyen d'un
injectionneur situé sur le fond d'une cuve cubique, sous l'effet de la
gravité, avec tension superficielle.

Cas :
    - Domaine cubique 0.1 m x 0.1 m x 0.1 m (blockMesh)
    - Solveur : incompressibleVoF (interFoam — OpenFOAM 13)
    - Phase 1 : eau   (nu=72e-06, rho=915)   — liquide visqueux
    - Phase 2 : air   (nu=1.48e-05, rho=1.2)
    - Gravite : (0 0 -9.81) m/s^2
    - Turbulence : laminaire
    - Algorithme : PIMPLE (transient)
    - AMR : dynamicRefine sur alpha.water

Usage :
    cd examples/viscosity
    python run.py

Author: foampilot
"""

from pathlib import Path
from typing import Dict, List, Any

from foampilot.solver import Solver
from foampilot import Meshing
from foampilot.base.openFOAMFile import OpenFOAMFile


CASE_REL_PATH = "case"


def write_two_phase_transport_properties(case_path: Path, sigma: float) -> None:
    """Écrire ``constant/phaseProperties`` au format two-phase VoF.

    Le solveur VoF lit ``phaseProperties`` (et non ``transportProperties``).
    Foampilot génère un ``transportProperties`` Newtonian simple, mais on
    écrase donc le fichier produit par ``solver.constant.write()`` avec le
    format two-phase attendu.
    """
    path = case_path / "constant" / "phaseProperties"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"constant\";\n"
        "    object      phaseProperties;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        f"phases          (water air);\n"
        "\n"
        f"sigma           {sigma};\n"
        "\n"
        "// ************************************************************************* //\n"
    )
    path.write_text(content)


def write_momentum_transport(case_path: Path, simulation_type: str = "laminar") -> None:
    """Écrire ``constant/momentumTransport`` pour le modèle de turbulence."""
    path = case_path / "constant" / "momentumTransport"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"constant\";\n"
        "    object      momentumTransport;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        f"simulationType  {simulation_type};\n"
        "\n"
        "// ************************************************************************* //\n"
    )
    path.write_text(content)


def write_physical_properties(
    case_path: Path, phase: str, nu: float, rho: float
) -> None:
    """Écrire ``constant/physicalProperties.<phase>`` pour un two-phase VoF."""
    path = case_path / "constant" / f"physicalProperties.{phase}"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"constant\";\n"
        f"    object      physicalProperties.{phase};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "viscosityModel  constant;\n"
        "\n"
        f"nu              {nu};\n"
        "\n"
        f"rho             {rho};\n"
        "\n"
        "// ************************************************************************* //\n"
    )
    path.write_text(content)


def write_setFields_dict(case_path: Path) -> None:
    """Écrire ``system/setFieldsDict`` pour initialiser un goutte d'eau."""
    path = case_path / "system" / "setFieldsDict"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"system\";\n"
        "    object      setFieldsDict;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
        "defaultValues\n"
        "{\n"
        "    alpha.water 0;\n"
        "}\n"
        "\n"
        "zones\n"
        "{\n"
        "    waterDrop\n"
        "    {\n"
        "        type        sphere;\n"
        "        centre      (0.05 0.05 0.05);\n"
        "        radius      0.005;\n"
        "        values\n"
        "        {\n"
        "            alpha.water 1;\n"
        "        }\n"
        "    }\n"
        "}\n"
        "\n"
        "// ************************************************************************* //\n"
    )
    path.write_text(content)


def write_functions_file(case_path: Path) -> None:
    """Écrire ``system/functions`` pour le suivi des résidus."""
    system_path = case_path / "system"
    system_path.mkdir(parents=True, exist_ok=True)
    path = system_path / "functions"
    content = (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |                                          \\\\\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox     \\\\\n"
        "   \\\\    /   O peration     | Version:  13                              \\\\\n"
        "    \\\\  /    A nd           | Website:  https://openfoam.org            \\\\\n"
        "     \\\\/     M anipulation  |                                             \\\\\n"
        " *---------------------------------------------------------------------------*\n"
        " FoamFile                                                                      \\\n"
        " {                                                                             \\\n"
        "     format      ascii;                                                        \\\n"
        "     class       dictionary;                                                   \\\n"
        "     location    \"system\";                                                      \\\n"
        "     object      functions;                                                      \\\n"
        " }                                                                             \\\n"
        " // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n"
        "                                                                               \n"
        " #includeFunc residuals(name=residuals, fields=(p_rgh U alpha.water))           \n"
        "                                                                               \n"
        "// ************************************************************************* //\n"
    )
    path.write_text(content)


def write_boundary_file(
    case_path: Path,
    field: str,
    boundaries: Dict[str, Dict[str, Any]],
    internal_field: str,
    dimensions: str,
) -> None:
    """Écrire manuellement un fichier de champ dans ``0/``.

    Utilisée pour p_rgh (dimensions de pression) et alpha.water/air
    (valeurs internes personnalisées).
    """
    folder_0 = case_path / "0"
    folder_0.mkdir(parents=True, exist_ok=True)
    file_path = folder_0 / field

    class_field = "volVectorField" if field == "U" else "volScalarField"

    lines: List[str] = []
    lines.append("FoamFile")
    lines.append("{")
    lines.append("    version     2.0;")
    lines.append("    format      ascii;")
    lines.append(f"    class       {class_field};")
    lines.append(f"    object      {field};")
    lines.append("}")
    lines.append("")
    lines.append(f"dimensions      {dimensions};")
    lines.append(f"internalField   {internal_field};")
    lines.append("")
    lines.append("boundaryField")
    lines.append("{")
    for patch, params in boundaries.items():
        lines.append(f"    {patch}")
        lines.append("    {")
        _write_dict(lines, params, indent=2)
        lines.append("    }")
        lines.append("")
    lines.append("}")
    lines.append("")
    lines.append("// ************************************************************************* //")

    file_path.write_text("\n".join(lines))


def _write_dict(lines: List[str], attrs: Dict[str, Any], indent: int = 0) -> None:
    """Écrire récursivement un dictionnaire au format OpenFOAM."""
    pad = "    " * indent
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{key}")
                lines.append(f"{pad}{{")
                _write_dict(lines, value, indent + 1)
                lines.append(f"{pad}}}")
            continue
        if isinstance(value, bool):
            lines.append(f"{pad}{key} {'true' if value else 'false'};")
        elif isinstance(value, (int, float)):
            lines.append(f"{pad}{key} {value};")
        elif isinstance(value, str):
            lines.append(f"{pad}{key} {value};")
        else:
            lines.append(f"{pad}{key} {value};")


def setup_boundary_conditions(solver: Solver, case_path: Path) -> None:
    """Configurer toutes les conditions aux limites pour le cas VoF."""
    solver.boundary.initialize_boundary()

    # Supprimer le champ T (généré par with_gravity mais non utilisé par VoF)
    solver.boundary.fields.pop("T", None)
    if "T" in solver._solver.fields_manager.fields:
        solver._solver.fields_manager.fields.pop("T")

    # ------------------------------------------------------------------
    # alpha.water — conditions aux limites
    # ------------------------------------------------------------------
    solver.boundary.set_raw_condition("inlet", "alpha.water", {
        "type": "fixedValue",
        "value": "uniform 1",
    })
    solver.boundary.set_raw_condition("atmosphere", "alpha.water", {
        "type": "inletOutlet",
        "inletValue": "uniform 0",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("walls", "alpha.water", {
        "type": "zeroGradient",
    })

    # ------------------------------------------------------------------
    # alpha.air — conditions aux limites
    # ------------------------------------------------------------------
    solver.boundary.set_raw_condition("inlet", "alpha.air", {
        "type": "fixedValue",
        "value": "uniform 0",
    })
    solver.boundary.set_raw_condition("atmosphere", "alpha.air", {
        "type": "inletOutlet",
        "inletValue": "uniform 1",
        "value": "uniform 1",
    })
    solver.boundary.set_raw_condition("walls", "alpha.air", {
        "type": "zeroGradient",
    })

    # ------------------------------------------------------------------
    # U — vitesse
    # ------------------------------------------------------------------
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (0 0 0.5)",
    })
    solver.boundary.set_raw_condition("atmosphere", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })
    solver.boundary.set_raw_condition("walls", "U", {
        "type": "noSlip",
    })

    # ------------------------------------------------------------------
    # p_rgh — pression
    # ------------------------------------------------------------------
    solver.boundary.set_raw_condition("inlet", "p_rgh", {
        "type": "prghTotalPressure",
        "psi": "none",
        "gamma": 1,
        "p0": "$internalField",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("atmosphere", "p_rgh", {
        "type": "prghTotalPressure",
        "psi": "none",
        "gamma": 1,
        "p0": "$internalField",
        "value": "$internalField",
    })
    solver.boundary.set_raw_condition("walls", "p_rgh", {
        "type": "fixedFluxPressure",
        "gradient": "$internalField",
        "value": "$internalField",
    })


def write_all_boundary_files(solver: Solver, case_path: Path) -> None:
    """Écrire tous les fichiers de champ 0/."""
    # Ajouter p_rgh aux dimensions connues
    OpenFOAMFile.FIELD_DIMENSIONS["p_rgh"] = "[1 -1 -2 0 0 0 0]"

    field_dims = {
        "U": "[0 1 -1 0 0 0 0]",
        "p_rgh": "[1 -1 -2 0 0 0 0]",
        "alpha.water": "[0 0 0 0 0 0 0]",
        "alpha.air": "[0 0 0 0 0 0 0]",
    }

    internal_fields = {
        "U": "uniform (0 0 0)",
        "p_rgh": "uniform 0",
        "alpha.water": "uniform 0",
        "alpha.air": "uniform 1",
    }

    for field in solver.boundary.fields:
        if field not in field_dims:
            continue
        write_boundary_file(
            case_path=case_path,
            field=field,
            boundaries=solver.boundary.fields[field],
            internal_field=internal_fields.get(field, "uniform 0"),
            dimensions=field_dims[field],
        )


def post_process(case_path: Path) -> None:
    """Post-traitement simple : vérifier les résidus et lister les temps."""
    log_file = case_path / "log.incompressibleVoF"
    if log_file.exists():
        from foampilot.utilities.residuals import ResidualsPost

        residuals = ResidualsPost(log_file)
        residuals.process(export_csv=True, export_png=True)
        print("Résidus exportés (CSV + PNG).")

    times = sorted(
        [d.name for d in case_path.iterdir()
         if d.is_dir() and _is_numeric(d.name)],
        key=float,
    )
    if times:
        print(f"Temps disponibles : {times}")

    # Vérifier la présence de alpha.water à la dernière itération
    if times:
        last_time = times[-1]
        alpha_file = case_path / last_time / "alpha.water"
        if alpha_file.exists():
            content = alpha_file.read_text()
            has_nonuniform = "nonuniform" in content
            print(f"alpha.water à t={last_time}: "
                  f"{'nonuniformList' if has_nonuniform else 'uniform'}")


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    case_path = Path(__file__).resolve().parent / CASE_REL_PATH
    case_path.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Initialisation du solveur VoF
    # ==================================================================
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = True
    solver.is_vof = True
    solver.transient = True
    solver.turbulence_model = "laminar"

    # ==================================================================
    # 2. Configuration du controlDict
    # ==================================================================
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 0.2
    solver.system.controlDict.deltaT = 0.0001
    solver.system.controlDict.writeControl = "adjustableRunTime"
    solver.system.controlDict.writeInterval = 0.01
    solver.system.controlDict.adjustTimeStep = "yes"
    solver.system.controlDict.maxCo = 0.5
    solver.system.controlDict.maxAlphaCo = 0.3
    solver.system.controlDict.maxDeltaT = 0.0005

    # PIMPLE
    solver.system.fvSolution.set_pimple(
        nOuterCorrectors=2,
        nCorrectors=3,
        nNonOrthogonalCorrectors=0,
        momentumPredictor=True,
    )

    solver.system.fvSolution.solvers.pop("alpha.water", None)
    solver.system.fvSolution.solvers.pop("alpha.air", None)
    solver.system.fvSolution.solvers["alpha.water"] = {
        "nCorrectors": "1",
        "nSubCycles": "8",
    }
    solver.system.fvSolution.solvers["alpha.air"] = {
        "nCorrectors": "1",
        "nSubCycles": "8",
    }
    solver.system.fvSolution.solvers["pcorr"] = {
        "solver": "PCG",
        "preconditioner": {
            "preconditioner": "GAMG",
            "tolerance": "1e-05",
            "relTol": "0",
            "smoother": "GaussSeidel",
        },
        "tolerance": "1e-05",
        "relTol": "0",
        "maxIter": "100",
    }
    solver.system.fvSolution.solvers["pcorr.*"] = {
        "solver": "PCG",
        "preconditioner": {
            "preconditioner": "GAMG",
            "tolerance": "1e-05",
            "relTol": "0",
            "smoother": "GaussSeidel",
        },
        "tolerance": "1e-05",
        "relTol": "0",
        "maxIter": "100",
    }

    solver.system.write()
    write_functions_file(case_path)

    # ==================================================================
    # 3. Maillage (blockMesh)
    # ==================================================================
    print("1. Génération du maillage (blockMesh) ...")
    mesh = Meshing(case_path, mesher="blockMesh")
    blockmesh = mesh.mesher

    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0, 0, 0],
        [0.1, 0, 0],
        [0.1, 0.1, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
        [0.1, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0, 0.1, 0.1],
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (20 20 40) simpleGrading (1 1 1)"
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"defaultFaces": "empty"}
    blockmesh.boundary = {
        "inlet": {
            "type": "patch",
            "faces": [[0, 3, 2, 1]],
        },
        "atmosphere": {
            "type": "patch",
            "faces": [[4, 5, 6, 7]],
        },
        "walls": {
            "type": "wall",
            "faces": [
                [0, 4, 7, 3],
                [1, 2, 6, 5],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
            ],
        },
    }
    blockmesh.mergePatchPairs = []

    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()

    # ==================================================================
    # 4. Fichiers constants (two-phase VoF)
    # ==================================================================
    print("2. Écriture des propriétés physiques (two-phase VoF) ...")
    solver.constant.write()

    # Overwrite transportProperties with VoF two-phase format
    write_two_phase_transport_properties(case_path, sigma=0.032)

    # Phase-specific physical properties
    write_physical_properties(case_path, "water", nu=72e-06, rho=915)
    write_physical_properties(case_path, "air", nu=1.48e-05, rho=1.2)

    # Momentum transport (laminar)
    write_momentum_transport(case_path, simulation_type="laminar")

    # ==================================================================
    # 5. Conditions aux limites
    # ==================================================================
    print("3. Configuration des conditions aux limites ...")
    setup_boundary_conditions(solver, case_path)
    write_all_boundary_files(solver, case_path)

    # ==================================================================
    # 6. setFields (initialisation avec un goutte d'eau)
    # ==================================================================
    # Skip setFields for initial validation to avoid sharp discontinuity;
    # the inlet BC will introduce alpha.water smoothly.
    # print("4. Initialisation avec setFields ...")
    # write_setFields_dict(case_path)
    # solver.run_command(
    #     ["setFields", "-case", str(case_path)],
    #     log_filename="log.setFields",
    # )

    # ==================================================================
    # 7. Lancement de la simulation
    # ==================================================================
    print("\n" + "=" * 60)
    print("Lancement de la simulation VoF (incompressibleVoF)")
    print("=" * 60)
    solver.run_simulation()

    # ==================================================================
    # 8. Post-traitement
    # ==================================================================
    print("\n" + "=" * 60)
    print("Post-traitement")
    print("=" * 60)
    post_process(case_path)

    print("\n" + "=" * 60)
    print("Simulation terminée avec succès !")
    print(f"  Cas      : {case_path}")
    print(f"  Log      : {case_path / 'log.incompressibleVoF'}")
    print(f"  Résultats: {case_path / 'postProcessing'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
