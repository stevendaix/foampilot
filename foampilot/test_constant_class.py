"""
Script de test complet pour toutes les classes du répertoire `constant` de foampilot.

Ce script va :
1. Initialiser un cas de test pour un solveur incompressible.
2. Initialiser un cas de test pour un solveur compressible.
3. Tester l'écriture de tous les fichiers du répertoire `constant`.
4. Vérifier la présence des fichiers générés.
5. Comparer leur contenu avec des fichiers de référence.
"""

from pathlib import Path
import shutil
import filecmp
import os

from foampilot.solver import Solver
from foampilot.utilities.manageunits import Quantity


REFERENCE_DIR = Path("reference_constant_files")


def compare_with_reference(file_path: Path) -> bool:
    """Compare un fichier généré avec son équivalent dans reference_constant_files."""
    ref_file = REFERENCE_DIR / file_path.name

    if not ref_file.exists():
        print(f"   ⚠ Aucun fichier de référence trouvé pour {file_path.name} → ignoré")
        return True  # On ignore la comparaison

    if filecmp.cmp(file_path, ref_file, shallow=False):
        print(f"      ✅ Contenu identique au fichier de référence.")
        return True

    print(f"      ❌ Contenu différent du fichier de référence !")
    return False


def run_test(case_name: str, compressible: bool, with_gravity: bool, with_radiation: bool):
    print(f"--- Démarrage du test : {case_name} ---")
    case_path = Path.cwd() / case_name
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialisation du solveur
    solver = Solver(case_path)
    solver.compressible = compressible
    solver.with_gravity = with_gravity

    # 2. Configuration des propriétés
    if compressible:
        solver.constant.physicalProperties.mu = Quantity("1.8e-5", "kg/m/s")
        solver.constant.physicalProperties.Cp = Quantity(1005, "J/kg/K")
        solver.constant.physicalProperties.energy = True
        solver.constant.pRef.value = Quantity(101325, "Pa")
    else:
        solver.constant.transportProperties.nu = Quantity("1.5e-5", "m^2/s")

    if with_radiation:
        solver.constant.enable_radiation(model="P1")

    # 3. Écriture
    solver.constant.write()

    # 4. Vérification
    print(f"\nVérification des fichiers pour le cas '{case_name}':")
    constant_dir = case_path / "constant"
    expected_files = []

    if compressible:
        expected_files.extend(["physicalProperties", "pRef"])
    else:
        expected_files.append("transportProperties")

    if with_gravity:
        expected_files.append("g")

    if with_radiation:
        expected_files.extend(["radiationProperties", "fvModels"])

    expected_files.append("turbulenceProperties")

    all_ok = True
    for f in expected_files:
        file_path = constant_dir / f

        if file_path.exists():
            print(f"  ✅ Fichier '{f}' trouvé.")
            if os.path.getsize(file_path) > 50:
                print(f"      - Contenu non vide.")

                ok = compare_with_reference(file_path)
                all_ok = all_ok and ok
            else:
                print(f"      ❌ Fichier '{f}' vide !")
                all_ok = False
        else:
            print(f"  ❌ Fichier '{f}' manquant !")
            all_ok = False

    if all_ok:
        print(f"\n✅ SUCCESS : {case_name}")
    else:
        print(f"\n❌ FAILED : {case_name}")

    print(f"--- Fin du test : {case_name} ---\n")

#############################



def build_reference_files(
    compressible: bool,
    with_gravity: bool,
    with_radiation: bool,
    overwrite: bool = True,
):
    """
    Construit un set de fichiers de référence dans reference_constant_files/
    selon les options du solveur (compressible, gravité, radiation).
    """

    tmp_case = Path("_reference_tmp_case")

    # Reset dossier
    if tmp_case.exists():
        shutil.rmtree(tmp_case)
    tmp_case.mkdir(parents=True)

    # Init solver
    solver = Solver(tmp_case)
    solver.compressible = compressible
    solver.with_gravity = with_gravity

    if compressible:
        solver.constant.physicalProperties.mu = Quantity("1.8e-5", "kg/m/s")
        solver.constant.physicalProperties.Cp = Quantity(1005, "J/kg/K")
        solver.constant.pRef.value = Quantity(101325, "Pa")
    else:
        solver.constant.transportProperties.nu = Quantity("1.5e-5", "m^2/s")

    if with_radiation:
        solver.constant.enable_radiation(model="P1")

    # Write constant files
    solver.constant.write()

    # Ensure reference dir exists
    REFERENCE_DIR.mkdir(exist_ok=True)

    # Copy generated files
    constant_dir = tmp_case / "constant"

    print("\nCréation des fichiers de référence :")
    for f in constant_dir.iterdir():
        dst = REFERENCE_DIR / f.name

        if dst.exists() and not overwrite:
            print(f"   ⚠ {f.name} existe déjà → ignoré")
            continue

        shutil.copy(f, dst)
        print(f"   ✅ Copié : {f.name}")

    print("\n✅ Références mises à jour dans reference_constant_files/")

    # Clean tmp dir
    shutil.rmtree(tmp_case)

def build_all_reference_sets(overwrite=True):

    print("\n=== Construction des références ===\n")

    cases = [
        ("incompressible", False, False),
        ("incompressible_gravity", False, True),
        ("compressible", True, False),
        ("compressible_gravity_radiation", True, True),
    ]

    for name, comp, grav in cases:
        print(f"--- {name} ---")
        build_reference_files(
            compressible=comp,
            with_gravity=grav,
            with_radiation=(comp and grav),
            overwrite=overwrite,
        )

    print("\n=== Fin construction références ===\n")

    # build_all_reference_sets(overwrite=True)





if __name__ == "__main__":
    run_test("test_incompressible", compressible=False, with_gravity=False, with_radiation=False)
    run_test("test_incompressible_gravity", compressible=False, with_gravity=True, with_radiation=False)
    run_test("test_compressible", compressible=True, with_gravity=False, with_radiation=False)
    run_test("test_compressible_gravity_radiation", compressible=True, with_gravity=True, with_radiation=True)