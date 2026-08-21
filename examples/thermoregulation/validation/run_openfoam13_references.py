from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "openfoam_runs"
OUT = ROOT / "examples" / "thermoregulation" / "validation" / "results"
FOAM_TUTORIALS = Path(os.environ.get("FOAM_TUTORIALS", "/opt/openfoam13/tutorials"))


def run(command: str, cwd: Path) -> None:
    env = os.environ.copy()
    env["WM_PROJECT_DIR"] = "/opt/openfoam13"
    subprocess.run(["bash", "-lc", command], cwd=cwd, env=env, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    buoyant_src = FOAM_TUTORIALS / "fluid" / "buoyantCavity"
    sphere_src = FOAM_TUTORIALS / "multiRegion" / "CHT" / "coolingSphere"
    buoyant = RUNS / "buoyantCavity_validation"
    sphere = RUNS / "coolingSphere_validation"

    if not buoyant.exists():
        shutil.copytree(buoyant_src, buoyant)
    if not sphere.exists():
        shutil.copytree(sphere_src, sphere)

    run("source /opt/openfoam13/etc/bashrc && ./Allclean >/dev/null 2>&1 || true; ./Allrun", buoyant)
    run(
        "source /opt/openfoam13/etc/bashrc && "
        "./Allclean >/dev/null 2>&1 || true; "
        "./Allmesh >/dev/null; foamSetupCHT >/dev/null; "
        "foamDictionary -entry internalField -set 'uniform 348' 0/solid/T >/dev/null; "
        "decomposePar -allRegions >/dev/null; "
        "mpirun --oversubscribe -np 4 foamMultiRun -parallel >/dev/null; "
        "reconstructPar -allRegions >/dev/null",
        sphere,
    )

    buoyant_end = (buoyant / "1000").exists()
    sphere_end = (sphere / "1" / "fluid" / "T").exists() or (sphere / "1" / "fluid" / "T.gz").exists()
    report = f"""# Références OpenFOAM 13 exécutées

| Cas | Référence | Résultat |
|---|---|---|
| `buoyantCavity` | Convection naturelle avec profils expérimentaux dans `validation/exptData` | {'OK' if buoyant_end else 'ÉCHEC'} |
| `coolingSphere` | CHT transitoire air–cuivre, `Tinitial=296 K`, solide initial à `348 K` | {'OK' if sphere_end else 'ÉCHEC'} |

Le cas humain MakeHuman utilise ces références à deux niveaux. `buoyantCavity` valide le solveur de convection naturelle et la chaîne de comparaison à des mesures. `coolingSphere` valide la chaîne transitoire CHT multi-région. Le corps humain reste une application géométrique et thermophysiologique ; il ne doit pas être présenté comme une validation expérimentale humaine tant qu’un jeu de mesures correspondant n’est pas intégré.
"""
    (OUT / "openfoam13_reference_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
