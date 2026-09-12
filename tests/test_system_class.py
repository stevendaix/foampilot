from pathlib import Path
import shutil
import filecmp
from foampilot.solver.solver import Solver

BASE_REFERENCE_DIR = Path("reference/system_files")


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
# Gestion des fichiers de référence
# ------------------------
def rebuild_reference_system(case_name, transient=False, energy=False, simulation_type="incompressible", algorithm="SIMPLE"):
    """
    Rebuild reference system files for a given case.
    Each case has its own folder in reference/system_files.
    """
    system_dir = prepare_case_system(case_name, transient, energy, simulation_type, algorithm)

    case_ref_dir = BASE_REFERENCE_DIR / case_name
    case_ref_dir.mkdir(parents=True, exist_ok=True)

    for file_path in system_dir.iterdir():
        target = case_ref_dir / file_path.name
        shutil.copy(file_path, target)
        print(f"🔄 Updated reference file: {target}")

    print(f"✅ Reference system rebuilt for {case_name}.")


# ------------------------
# Comparaison des fichiers
# ------------------------
def compare_system_files(case_name, generated_dir):
    """
    Compare generated system files with reference files for a given case.
    """
    case_ref_dir = BASE_REFERENCE_DIR / case_name
    if not case_ref_dir.exists():
        print(f"⚠️ Reference files for {case_name} not found.")
        return False

    success = True
    for ref_file in case_ref_dir.iterdir():
        generated = generated_dir / ref_file.name
        if not generated.exists():
            print(f"❌ Missing system file: {ref_file.name}")
            success = False
            continue
        if not filecmp.cmp(generated, ref_file, shallow=False):
            print(f"❌ File content mismatch: {ref_file.name}")
            success = False
        else:
            print(f"✅ {ref_file.name} matches reference")
    return success


# ------------------------
# Exécution du test
# ------------------------
def run_system_test(case_name, transient=False, energy=False, simulation_type="incompressible", algorithm="SIMPLE"):
    system_dir = prepare_case_system(case_name, transient, energy, simulation_type, algorithm)
    ok = compare_system_files(case_name, system_dir)
    if ok:
        print(f"✅ SUCCESS: {case_name}\n")
    else:
        print(f"❌ FAILED: {case_name}\n")


# ------------------------
# Exemple d'utilisation
# ------------------------
if __name__ == "__main__":
    test_cases = [
        ("test_incompressible_simple", False, False, "incompressible", "SIMPLE"),
        ("test_incompressible_transient", True, False, "incompressible", "PIMPLE"),
        ("test_compressible_energy", False, True, "compressible", "SIMPLE"),
        ("test_boussinesq_transient", True, True, "boussinesq", "PIMPLE"),
    ]

    # # Rebuild references for all cases
    # for case_name, transient, energy, sim_type, algo in test_cases:
    #     rebuild_reference_system(case_name, transient, energy, sim_type, algo)

    # Run tests for all cases
    for case_name, transient, energy, sim_type, algo in test_cases:
        run_system_test(case_name, transient, energy, sim_type, algo)
