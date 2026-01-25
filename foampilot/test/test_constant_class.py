from pathlib import Path
import shutil
import filecmp
import os

from foampilot.solver.solver import Solver
from foampilot.utilities.manageunits import ValueWithUnit

BASE_REFERENCE_DIR = Path("reference/constant_files")
TESTS_DIR = Path("tests")


# ------------------------
# Comparaison avec référence
# ------------------------
def compare_with_reference(case_name: str, file_path: Path) -> bool:
    """Compare un fichier généré avec son équivalent dans reference_constant_files/<case_name>."""
    ref_file = BASE_REFERENCE_DIR / case_name / file_path.name

    if not ref_file.exists():
        print(f"   ⚠ Aucun fichier de référence trouvé pour {file_path.name} → ignoré")
        return True

    if filecmp.cmp(file_path, ref_file, shallow=False):
        print(f"      ✅ Contenu identique au fichier de référence.")
        return True

    print(f"      ❌ Contenu différent du fichier de référence !")
    return False


# ------------------------
# Run test case
# ------------------------
def run_test(case_name: str, compressible: bool, with_gravity: bool, with_radiation: bool):
    print(f"\n--- Démarrage du test : {case_name} ---")
    case_path = TESTS_DIR / case_name
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True, exist_ok=True)

    solver = Solver(case_path)
    solver.compressible = compressible
    solver.with_gravity = with_gravity

    # Propriétés
    if compressible:
        solver.constant.physicalProperties.mu = ValueWithUnit("1.8e-5", "kg/m/s")
        solver.constant.physicalProperties.Cp = ValueWithUnit(1005, "J/kg/K")
        solver.constant.pRef.value = ValueWithUnit(101325, "Pa")
    else:
        solver.constant.transportProperties.nu = ValueWithUnit("1.5e-5", "m^2/s")

    if with_radiation:
        solver.constant.enable_radiation(model="P1")

    solver.constant.write()

    # Vérification
    print(f"\nVérification des fichiers pour le cas '{case_name}':")
    constant_dir = case_path / "constant"
    expected_files = ["turbulenceProperties"]

    if compressible:
        expected_files.extend(["physicalProperties", "pRef"])
    else:
        expected_files.append("transportProperties")

    if with_gravity:
        expected_files.append("g")

    if with_radiation:
        expected_files.extend(["radiationProperties", "fvModels"])

    all_ok = True
    for f in expected_files:
        file_path = constant_dir / f
        
        if file_path.exists():
            print(f"  ✅ Fichier '{f}' trouvé.")
            if os.path.getsize(file_path) > 50:
                print(f"      - Contenu non vide.")
                ok = compare_with_reference(case_name, file_path)
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


# ------------------------
# Build reference files
# ------------------------
def build_reference_files(
    case_name: str,
    compressible: bool,
    with_gravity: bool,
    with_radiation: bool,
    overwrite: bool = True,
):
    tmp_case = Path("_reference_tmp_case")
    if tmp_case.exists():
        shutil.rmtree(tmp_case)
    tmp_case.mkdir(parents=True)

    solver = Solver(tmp_case)
    solver.compressible = compressible
    solver.with_gravity = with_gravity

    if compressible:
        solver.constant.physicalProperties.mu = ValueWithUnit("1.8e-5", "kg/m/s")
        solver.constant.physicalProperties.Cp = ValueWithUnit(1005, "J/kg/K")
        solver.constant.pRef.value = ValueWithUnit(101325, "Pa")
    else:
        solver.constant.transportProperties.nu = ValueWithUnit("1.5e-5", "m^2/s")

    if with_radiation:
        solver.constant.enable_radiation(model="P1")

    solver.constant.write()

    # Créer dossier de référence pour le cas
    case_ref_dir = BASE_REFERENCE_DIR / case_name
    case_ref_dir.mkdir(parents=True, exist_ok=True)

    constant_dir = tmp_case / "constant"
    print(f"\nCréation des fichiers de référence pour {case_name}:")
    for f in constant_dir.iterdir():
        dst = case_ref_dir / f.name
        if dst.exists() and not overwrite:
            print(f"   ⚠ {f.name} existe déjà → ignoré")
            continue
        shutil.copy(f, dst)
        print(f"   ✅ Copié : {f.name}")

    shutil.rmtree(tmp_case)
    print(f"\n✅ Références mises à jour pour {case_name}.\n")


# ------------------------
# Build all reference sets
# ------------------------
def build_all_reference_sets(overwrite=True):
    print("\n=== Construction des références ===\n")
    cases = [
        ("incompressible", False, False, False),
        ("incompressible_gravity", False, True, False),
        ("compressible", True, False, False),
        ("compressible_gravity_radiation", True, True, True),
    ]
    for name, comp, grav, rad in cases:
        build_reference_files(name, comp, grav, rad, overwrite=overwrite)
    print("\n=== Fin construction références ===\n")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # 1️⃣ Build references
    build_all_reference_sets(overwrite=True)

    # 2️⃣ Run all tests
    run_test("incompressible", False, False, False)
    run_test("incompressible_gravity", False, True, False)
    run_test("compressible", True, False, False)
    run_test("compressible_gravity_radiation", True, True, True)
