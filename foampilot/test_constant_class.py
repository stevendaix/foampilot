'''
Script de test complet pour toutes les classes du répertoire `constant` de foampilot.

Ce script va :
1.  Initialiser un cas de test pour un solveur incompressible.
2.  Initialiser un cas de test pour un solveur compressible.
3.  Tester l'écriture de tous les fichiers du répertoire `constant` dans les deux cas.
4.  Vérifier la présence et le contenu de base des fichiers générés.
'''
from pathlib import Path
import shutil
import os

from foampilot.solver import Solver
from foampilot.utilities.manageunits import Quantity


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
        # Pour un cas compressible, on modifie les propriétés physiques
        solver.constant.physicalProperties.mu = Quantity("1.8e-5", "kg/m/s")
        solver.constant.physicalProperties.Cp = Quantity(1005, "J/kg/K")
        solver.constant.physicalProperties.energy = True
        solver.constant.pRef.value = Quantity(101325, "Pa")
    else:
        # Pour un cas incompressible, on modifie les propriétés de transport
        solver.constant.transportProperties.nu = Quantity("1.5e-5", "m^2/s")

    # Activer la radiation si demandé
    if with_radiation:
        solver.constant.enable_radiation(model="P1")

    # 3. Écriture des fichiers `constant`
    solver.constant.write()

    # 4. Vérification des fichiers générés
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

    # Le fichier turbulenceProperties est toujours écrit
    expected_files.append("turbulenceProperties")

    all_files_found = True
    for f in expected_files:
        file_path = constant_dir / f
        if file_path.exists():
            print(f"  [✅] Fichier '{f}' trouvé.")
            # Vérifier que le fichier n'est pas vide
            if os.path.getsize(file_path) > 50: # 50 bytes pour l'en-tête
                print(f"      - Contenu valide (non vide).")
            else:
                print(f"      - [❌] ERREUR: Le fichier '{f}' est vide !")
                all_files_found = False
        else:
            print(f"  [❌] ERREUR: Le fichier '{f}' est manquant !")
            all_files_found = False

    if all_files_found:
        print(f"\n[✔] Succès du test pour le cas '{case_name}' ! Tous les fichiers attendus ont été créés.")
    else:
        print(f"\n[❌] Échec du test pour le cas '{case_name}'. Des fichiers sont manquants ou vides.")

    print(f"--- Fin du test : {case_name} ---\n")

if __name__ == "__main__":
    # Test 1: Cas incompressible simple
    run_test("test_incompressible", compressible=False, with_gravity=False, with_radiation=False)

    # Test 2: Cas incompressible avec gravité
    run_test("test_incompressible_gravity", compressible=False, with_gravity=True, with_radiation=False)

    # Test 3: Cas compressible simple
    run_test("test_compressible", compressible=True, with_gravity=False, with_radiation=False)

    # Test 4: Cas compressible avec gravité et radiation
    run_test("test_compressible_gravity_radiation", compressible=True, with_gravity=True, with_radiation=True)