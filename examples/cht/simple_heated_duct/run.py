#!/usr/bin/env python3
"""Tutorial CHT : Heated Duct (chtMultiRegionFoam)

Demonstrates a complete conjugate heat transfer simulation using
foampilot and OpenFOAM 13 (chtMultiRegionFoam).

Workflow:
  1. blockMesh   — create base mesh
  2. createZones — define solid/fluid cellZones
  3. splitMeshRegions — split into region meshes
  4. foamSetupCHT — generate field templates and material properties
  5. Post-setup adjustments (BCs, initial T, fvSchemes, fvSolution)
  6. chtMultiRegionFoam — run the simulation
  7. foamToVTK — convert results for post-processing
  8. pyvista analysis

Usage ::
    cd examples/cht/simple_heated_duct
    python run.py
"""

import sys
import subprocess
from pathlib import Path

case_path = Path(__file__).resolve().parent


def run(cmd):
    """Run a shell command in the case directory."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=case_path, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def main():
    print("=" * 60)
    print("CHT Tutorial: Heated Duct (chtMultiRegionFoam)")
    print("=" * 60)

    # --- Step 1: Generate mesh ---
    print("\n1. Generating mesh (blockMesh) ...")
    run(["blockMesh"])

    # --- Step 2: Create cell zones ---
    print("\n2. Creating cell zones (createZones) ...")
    run(["createZones"])

    # --- Step 3: Split into regions ---
    print("\n3. Splitting mesh into regions (splitMeshRegions) ...")
    run(["splitMeshRegions", "-cellZones", "-defaultRegionName", "fluid"])

    # --- Step 4: Set up CHT ---
    print("\n4. Setting up CHT case (foamSetupCHT) ...")
    run(["foamSetupCHT"])

    # --- Step 5: Copy additional files ---
    print("\n5. Copying auxiliary files ...")
    of_dir = Path("/opt/openfoam13/tutorials/multiRegion/CHT/coolingSphere/templates")

    run(["cp", str(of_dir / "constant/fluid/g"),
         str(case_path / "constant/fluid/g")])
    run(["cp", str(of_dir / "constant/fluid/pRef"),
         str(case_path / "constant/fluid/pRef")])
    run(["cp", str(of_dir / "materials/air/thermophysicalTransport"),
         str(case_path / "constant/fluid/thermophysicalTransport")])

    # Region-specific fvSchemes/fvSolution
    run(["cp", str(of_dir / "system/fluid/fvSchemes"),
         str(case_path / "system/fluid/fvSchemes")])
    run(["cp", str(of_dir / "system/fluid/fvSolution"),
         str(case_path / "system/fluid/fvSolution")])
    run(["cp", str(of_dir / "system/solid/fvSchemes"),
         str(case_path / "system/solid/fvSchemes")])
    run(["cp", str(of_dir / "system/solid/fvSolution"),
         str(case_path / "system/solid/fvSolution")])

    # --- Step 6: Set boundary conditions and initial fields ---
    print("\n6. Setting initial conditions ...")
    run(["foamDictionary", "-entry", "internalField",
         "-set", "uniform 350", "0/solid/T"])
    run(["foamDictionary", "-entry", "internalField",
         "-set", "uniform 300", "0/fluid/T"])
    run(["foamDictionary", "-entry", "internalField",
         "-set", "uniform 1e5", "0/fluid/p"])

    # --- Step 7: Run solver ---
    print("\n7. Running chtMultiRegionFoam ...")
    run(["chtMultiRegionFoam"])

    # --- Step 8: Convert to VTK ---
    print("\n8. Converting to VTK (foamToVTK) ...")
    run(["foamToVTK", "-region", "fluid", "-latestTime",
         "-fields", "(T U p k omega)"])
    run(["foamToVTK", "-region", "solid", "-latestTime",
         "-fields", "(T)"])

    # --- Step 9: Post-process ---
    print("\n9. Running post-processing ...")
    sys.path.insert(0, str(case_path.parent.parent.parent / "foampilot" / "src"))
    run_post = subprocess.run(
        [sys.executable, str(case_path / "run_post.py")],
        cwd=case_path, capture_output=True, text=True
    )
    print(run_post.stdout)
    if run_post.returncode != 0:
        print(f"Post-processing warning: {run_post.stderr[-500:]}")

    print("\n" + "=" * 60)
    print("Tutorial complete!")
    print(f"  Case:    {case_path}")
    print(f"  Results: {case_path / 'postProcessing'}")
    print(f"  Plots:   {case_path / 'postProcessing' / '*.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
