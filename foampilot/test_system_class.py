from pathlib import Path
import shutil
import filecmp
from foampilot.solver import Solver

REFERENCE_SYSTEM_DIR = Path("reference_system_files")


# ------------------------
# Gestion des fichiers de référence
# ------------------------
def collect_reference_system_files():
    """
    Collect all reference system files from the reference folder.
    """
    files = {}
    if not REFERENCE_SYSTEM_DIR.exists():
        return files
    for f in REFERENCE_SYSTEM_DIR.iterdir():
        if f.is_file():
            files[f.name] = f
    return files


def rebuild_reference_system(case_name="tmp_system"):
    """
    Rebuild reference system files from a generated system directory.
    """
    system_dir = prepare_case_system(case_name)
    REFERENCE_SYSTEM_DIR.mkdir(exist_ok=True)
    for file_path in system_dir.iterdir():
        target = REFERENCE_SYSTEM_DIR / file_path.name
        shutil.copy(file_path, target)
        print(f"🔄 Updated reference file: {target}")
    print("✅ Reference system rebuilt.")


# ------------------------
# Préparation du cas
# ------------------------
def prepare_case_system(case_name, transient=False, energy=False, simulation_type="incompressible", algorithm="SIMPLE"):
    """
    Initialize a solver and write the system directory for testing.
    """
    print(f"=== System test case: {case_name} ===")

    case_path = Path(case_name)
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir()

    solver = Solver(case_path)
    solver.transient = transient
    solver.energy_activated = energy
    solver.simulation_type = simulation_type
    solver.algorithm = algorithm

    system_dir = solver.system.write()
    return system_dir


# ------------------------
# Comparaison des fichiers
# ------------------------
def compare_system_files(generated_dir, ref_files):
    """
    Compare generated system files with reference files.
    """
    success = True
    for filename, reference in ref_files.items():
        generated = generated_dir / filename
        if not generated.exists():
            print(f"❌ Missing system file: {filename}")
            success = False
            continue
        if not filecmp.cmp(generated, reference, shallow=False):
            print(f"❌ File content mismatch: {filename}")
            success = False
        else:
            print(f"✅ {filename} matches reference")
    return success


# ------------------------
# Exécution du test
# ------------------------
def run_system_test(case_name, transient=False, energy=False, simulation_type="incompressible", algorithm="SIMPLE"):
    system_dir = prepare_case_system(case_name, transient, energy, simulation_type, algorithm)
    ref_files = collect_reference_system_files()
    if not ref_files:
        print("⚠️ No reference files found. You may want to rebuild them first using rebuild_reference_system().")
        return
    ok = compare_system_files(system_dir, ref_files)
    if ok:
        print(f"✅ SUCCESS: {case_name}\n")
    else:
        print(f"❌ FAILED: {case_name}\n")


# ------------------------
# Exemple d'utilisation
# ------------------------
if __name__ == "__main__":
    # Cas incompressible SIMPLE
    run_system_test("test_incompressible_simple", transient=False, energy=False, simulation_type="incompressible", algorithm="SIMPLE")

    # Cas incompressible transitoire SIMPLE
    run_system_test("test_incompressible_transient", transient=True, energy=False, simulation_type="incompressible", algorithm="SIMPLE")

    # Cas compressible avec énergie
    run_system_test("test_compressible_energy", transient=False, energy=True, simulation_type="compressible", algorithm="SIMPLE")

    # Cas Boussinesq transitoire
    run_system_test("test_boussinesq_transient", transient=True, energy=True, simulation_type="boussinesq", algorithm="PIMPLE")

    # Pour reconstruire les fichiers de référence à partir de n'importe quel cas
    # rebuild_reference_system("test_incompressible_simple")