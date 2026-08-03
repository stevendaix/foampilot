#!/usr/bin/env python3
"""
Vérification post-traitement PyVista pour les exemples CSV foampilot.

Uniform cases : vérification directe des fichiers de champ (pas de foamToVTK).
Spatial cases : utilise OpenFOAMDirectReader (les champs spatiaux ont des
                valeurs réelles dans les fichiers de champ).

Usage :
    cd examples/csv_example
    python verify_csv_post.py [--base-dir .]
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "foampilot" / "src"))

from foampilot.postprocess import OpenFOAMDirectReader
import pyvista as pv


def verify_uniform_scalar(case_path: Path, field: str = "T"):
    """Verify uniform scalar CSV BC by inspecting field file directly."""
    print(f"\n{'='*60}")
    print(f"Vérification uniforme scalaire : {case_path.name} / {field}")
    print(f"{'='*60}")

    field_file = case_path / "0" / field
    if not field_file.exists():
        print(f"  ERREUR: fichier de champ {field_file} non trouvé")
        return False

    content = field_file.read_text()

    if "uniformFixedValue" not in content:
        print("  ERREUR: type uniformFixedValue non trouvé")
        return False

    if "uniformValue" not in content:
        print("  ERREUR: uniformValue non trouvé")
        return False

    if 'file "constant/' not in content:
        print("  ERREUR: référence fichier CSV non trouvée")
        return False

    import re
    csv_match = re.search(r'file\s+"(constant/[^"]+)"', content)
    if not csv_match:
        print("  ERREUR: impossible d'extraire le nom du fichier CSV")
        return False

    csv_rel = csv_match.group(1)
    csv_path = case_path / csv_rel
    if not csv_path.exists():
        print(f"  ERREUR: fichier CSV {csv_path} non trouvé")
        return False

    lines = csv_path.read_text().strip().splitlines()
    if len(lines) < 1:
        print(f"  ERREUR: CSV vide")
        return False

    first_line = lines[0].strip().split(",")
    if len(first_line) != 2:
        print(f"  ERREUR: CSV doit avoir 2 colonnes, trouvé {len(first_line)}")
        return False

    try:
        times = [float(line.split(",")[0]) for line in lines]
        values = [float(line.split(",")[1]) for line in lines]
    except ValueError as e:
        print(f"  ERREUR: CSV invalide - {e}")
        return False

    print(f"  CSV: {csv_rel}")
    print(f"  Lignes: {len(lines)}")
    print(f"  Temps: [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"  Valeurs: [{values[0]:.4f}, {values[-1]:.4f}]")
    if len(lines) == 1:
        print("  -> OK (stationnaire, 1 ligne)")
    else:
        print("  -> OK")
    return True


def verify_uniform_vector(case_path: Path, field: str = "U"):
    """Verify uniform vector CSV BC by inspecting field file directly."""
    print(f"\n{'='*60}")
    print(f"Vérification uniforme vecteur : {case_path.name} / {field}")
    print(f"{'='*60}")

    field_file = case_path / "0" / field
    if not field_file.exists():
        print(f"  ERREUR: fichier de champ {field_file} non trouvé")
        return False

    content = field_file.read_text()

    # Check uniformFixedValue BC
    if "uniformFixedValue" not in content:
        print("  ERREUR: type uniformFixedValue non trouvé")
        return False

    # Check uniformValue table
    if "uniformValue" not in content:
        print("  ERREUR: uniformValue non trouvé")
        return False

    # Check CSV file reference
    if 'file "constant/' not in content:
        print("  ERREUR: référence fichier CSV non trouvée")
        return False

    # Check columns (0 (1 2 3))
    if "columns (0 (1 2 3))" not in content:
        print("  ERREUR: format des colonnes incorrect (attendu: (0 (1 2 3)))")
        return False

    # Extract CSV filename
    import re
    csv_match = re.search(r'file\s+"(constant/[^"]+)"', content)
    if not csv_match:
        print("  ERREUR: impossible d'extraire le nom du fichier CSV")
        return False

    csv_rel = csv_match.group(1)
    csv_path = case_path / csv_rel
    if not csv_path.exists():
        print(f"  ERREUR: fichier CSV {csv_path} non trouvé")
        return False

    # Check CSV format
    lines = csv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        print(f"  ERREUR: CSV trop court ({len(lines)} lignes)")
        return False

    first_line = lines[0].strip().split(",")
    if len(first_line) != 4:
        print(f"  ERREUR: CSV doit avoir 4 colonnes, trouvé {len(first_line)}")
        return False

    try:
        times = [float(line.split(",")[0]) for line in lines]
        ux = [float(line.split(",")[1]) for line in lines]
        uy = [float(line.split(",")[2]) for line in lines]
        uz = [float(line.split(",")[3]) for line in lines]
    except ValueError as e:
        print(f"  ERREUR: CSV invalide - {e}")
        return False

    print(f"  CSV: {csv_rel}")
    print(f"  Lignes: {len(lines)}")
    print(f"  Temps: [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"  U initial: ({ux[0]:.4f}, {uy[0]:.4f}, {uz[0]:.4f})")
    print(f"  U final: ({ux[-1]:.4f}, {uy[-1]:.4f}, {uz[-1]:.4f})")
    print("  -> OK")
    return True


def verify_spatial(case_path: Path, field: str = "T"):
    """Verify spatial CSV BC using OpenFOAMDirectReader."""
    print(f"\n{'='*60}")
    print(f"Vérification spatiale : {case_path.name} / {field}")
    print(f"{'='*60}")

    reader = OpenFOAMDirectReader(case_path)
    time_steps = reader.get_time_steps()
    print(f"  Pas de temps disponibles: {time_steps}")

    success = False
    for ts in time_steps:
        try:
            mesh = reader.to_pyvista(fields=[field], time_step=ts)
            data = None
            if field in mesh.point_data:
                data = mesh.point_data[field]
            elif field in mesh.cell_data:
                data = mesh.cell_data[field]
            
            if data is not None:
                print(f"  t={ts}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
                success = True
            else:
                print(f"  t={ts}: champ '{field}' non trouvé")
        except Exception as e:
            error_msg = str(e)
            if "Invalid array shape" in error_msg or "Field file not found" in error_msg:
                print(f"  t={ts}: ignoré (fichier incomplet ou absent)")
            else:
                print(f"  t={ts}: ERREUR - {e}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Vérification post-traitement CSV")
    parser.add_argument("--base-dir", default=".", help="Répertoire de base des exemples")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if base.name == "csv_example":
        examples_dir = base
    else:
        examples_dir = base / "examples" / "csv_example"

    results = {}

    uniform_cases = [
        ("case_uniform_scalar_steady", "T", verify_uniform_scalar),
        ("case_uniform_scalar", "T", verify_uniform_scalar),
        ("case_uniform_vector", "U", verify_uniform_vector),
    ]

    spatial_cases = [
        ("case_spatial", "T", verify_spatial),
        ("case_spatial_steady", "T", verify_spatial),
    ]

    for case_name, field, func in uniform_cases + spatial_cases:
        case_path = examples_dir / case_name
        if case_path.exists():
            try:
                ok = func(case_path, field)
                results[case_name] = "OK" if ok else "FAIL"
            except Exception as e:
                print(f"ERREUR sur {case_name}: {e}")
                results[case_name] = f"ERROR: {e}"
        else:
            print(f"Cas non trouvé: {case_path}")
            results[case_name] = "MISSING"

    print(f"\n{'='*60}")
    print("Résumé")
    print(f"{'='*60}")
    for case_name, status in results.items():
        print(f"  {case_name}: {status}")

    all_ok = all(s == "OK" for s in results.values())
    if all_ok:
        print("\nToutes les vérifications ont réussi !")
    else:
        print("\nCertaines vérifications ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
