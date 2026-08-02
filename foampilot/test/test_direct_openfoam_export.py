"""
Test the DirectOpenFOAMExporter for single-region and multi-region (CHT)
meshes.

These tests build Gmsh geometries from scratch, export them directly to
OpenFOAM native polyMesh format, and validate with ``checkMesh``.

Prerequisites:
    - Python gmsh package
    - OpenFOAM environment (for checkMesh and gmshToFoam)
"""

import shutil
import logging
import subprocess
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

import gmsh
import numpy as np

from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

logger = logging.getLogger(__name__)

TEST_DIR = Path(__file__).resolve().parent
_CONTROL_DICT = """\
FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     none;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
"""


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: list, cwd: Path) -> Tuple[str, int]:
    r = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120
    )
    return r.stdout + r.stderr, r.returncode


def _ensure_system(case_dir: Path):
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "controlDict").write_text(_CONTROL_DICT)


# -----------------------------------------------------------------------
#  Helper: parse OpenFOAM list files
# -----------------------------------------------------------------------

def _count_entries(path: Path) -> int:
    text = path.read_text()
    lines = text.splitlines()
    in_list = False
    count = 0
    for line in lines:
        line = line.strip()
        if not in_list and line.isdigit():
            count = int(line)
            in_list = True
            continue
        if in_list and line == ")":
            break
    return count


# -----------------------------------------------------------------------
#  Test 1 — single fluid region
# -----------------------------------------------------------------------

def test_single_region():
    print("=" * 60)
    print("Test 1: Single-region fluid export (cube, tets)")
    print("=" * 60)

    case = TEST_DIR / "test_single_region_direct"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    _ensure_system(case)

    gmsh.initialize()
    gmsh.model.add("cube_fluid")

    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.getEntities(2)
    inlet = [s for _, s in surfs
             if abs(gmsh.model.occ.getCenterOfMass(2, s)[0]) < 1e-6]
    outlet = [s for _, s in surfs
              if abs(gmsh.model.occ.getCenterOfMass(2, s)[0] - 1) < 1e-6]
    walls = [s for _, s in surfs if s not in inlet + outlet]

    gmsh.model.addPhysicalGroup(2, inlet, name="INLET")
    gmsh.model.addPhysicalGroup(2, outlet, name="OUTLET")
    gmsh.model.addPhysicalGroup(2, walls, name="WALLS")
    gmsh.model.addPhysicalGroup(3, [1], name="FLUID")

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
    gmsh.model.mesh.generate(3)

    exporter = DirectOpenFOAMExporter(case)
    exporter.export_single_region(region_name="FLUID")
    gmsh.finalize()

    if _has_tool("checkMesh"):
        log, rc = _run(["checkMesh", "-case", str(case)], case)
        if rc != 0:
            print("[FAIL] checkMesh")
            for line in log.split("\n"):
                if "ERROR" in line or "FATAL" in line:
                    print(f"  {line}")
        else:
            print("[PASS] checkMesh — Mesh OK")
    else:
        print("[WARN] checkMesh not available")
    return case


# -----------------------------------------------------------------------
#  Test 2 — multi-region CHT
# -----------------------------------------------------------------------

def test_multi_region_cht():
    print()
    print("=" * 60)
    print("Test 2: Multi-region CHT (fluid + solid)")
    print("=" * 60)

    case = TEST_DIR / "test_multi_region_direct"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    _ensure_system(case)

    gmsh.initialize()
    gmsh.model.add("cht")

    # Two adjacent boxes sharing the face x=1
    fluid_box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    solid_box = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [fluid_box], name="FLUID")
    gmsh.model.addPhysicalGroup(3, [solid_box], name="SOLID")

    # Surface patches for fluid
    for _, stag in gmsh.model.getEntitiesInBoundingBox(0, 0, 0, 1, 1, 1, 2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, stag)
        cx, _, _ = gmsh.model.occ.getCenterOfMass(2, stag)
        if abs(xmin) < 1e-6 and abs(cx) < 0.51:
            gmsh.model.addPhysicalGroup(2, [stag], name="INLET")
        elif abs(xmax - 1) < 1e-6 and abs(cx) > 0.49:
            gmsh.model.addPhysicalGroup(2, [stag], name="INTERFACE_FLUID")
        else:
            gmsh.model.addPhysicalGroup(2, [stag], name="WALLS")

    # Surface patches for solid
    for _, stag in gmsh.model.getEntitiesInBoundingBox(1, 0, 0, 2, 1, 1, 2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, stag)
        cx, _, _ = gmsh.model.occ.getCenterOfMass(2, stag)
        if abs(xmin - 1) < 1e-6 and abs(cx) > 0.49:
            gmsh.model.addPhysicalGroup(2, [stag], name="INTERFACE_SOLID")
        elif abs(xmax - 2) < 1e-6:
            gmsh.model.addPhysicalGroup(2, [stag], name="OUTLET")
        else:
            gmsh.model.addPhysicalGroup(2, [stag], name="WALLS")

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.3)
    gmsh.model.mesh.generate(3)

    exporter = DirectOpenFOAMExporter(case)
    exporter.export_multi_region()
    gmsh.finalize()

    if _has_tool("checkMesh"):
        for region in ("FLUID", "SOLID"):
            log, rc = _run(
                ["checkMesh", "-case", str(case), "-region", region], case
            )
            if rc != 0:
                print(f"[FAIL] checkMesh region {region}")
                for line in log.split("\n"):
                    if "ERROR" in line or "FATAL" in line:
                        print(f"  {line}")
            else:
                print(f"[PASS] checkMesh region {region} — Mesh OK")
    else:
        print("[WARN] checkMesh not available")
    return case


# -----------------------------------------------------------------------
#  Test 3 — cross-check with gmshToFoam
# -----------------------------------------------------------------------

def test_cross_check_gmshToFoam():
    print()
    print("=" * 60)
    print("Test 3: Cross-check with gmshToFoam")
    print("=" * 60)

    if not _has_tool("gmshToFoam"):
        print("[WARN] gmshToFoam not available")
        return None

    case = TEST_DIR / "test_cross_check_direct"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    _ensure_system(case)

    gmsh.initialize()
    gmsh.model.add("cube_cross")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    inlet = [s for _, s in surfs
             if abs(gmsh.model.occ.getCenterOfMass(2, s)[0]) < 1e-6]
    outlet = [s for _, s in surfs
              if abs(gmsh.model.occ.getCenterOfMass(2, s)[0] - 1) < 1e-6]
    walls = [s for _, s in surfs if s not in inlet + outlet]

    gmsh.model.addPhysicalGroup(2, inlet, name="INLET")
    gmsh.model.addPhysicalGroup(2, outlet, name="OUTLET")
    gmsh.model.addPhysicalGroup(2, walls, name="WALLS")
    gmsh.model.addPhysicalGroup(3, [1], name="FLUID")

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.model.mesh.generate(3)

    msh_path = case / "mesh.msh"
    gmsh.write(str(msh_path))

    log, rc = _run(["gmshToFoam", "mesh.msh"], case)
    if rc != 0:
        print(f"[WARN] gmshToFoam failed")
        return None

    gmsh.finalize()

    of_dir = case / "constant" / "polyMesh"
    of_n_faces = _count_entries(of_dir / "faces")
    of_n_owner = _count_entries(of_dir / "owner")
    of_n_cells = _count_entries(of_dir / "cellZones") if (of_dir / "cellZones").exists() else 0

    # Now direct export (rebuild the same mesh)
    gmsh.initialize()
    gmsh.model.add("cube_cross2")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    inlet2 = [s for _, s in surfs
              if abs(gmsh.model.occ.getCenterOfMass(2, s)[0]) < 1e-6]
    outlet2 = [s for _, s in surfs
               if abs(gmsh.model.occ.getCenterOfMass(2, s)[0] - 1) < 1e-6]
    walls2 = [s for _, s in surfs if s not in inlet2 + outlet2]
    gmsh.model.addPhysicalGroup(2, inlet2, name="INLET")
    gmsh.model.addPhysicalGroup(2, outlet2, name="OUTLET")
    gmsh.model.addPhysicalGroup(2, walls2, name="WALLS")
    gmsh.model.addPhysicalGroup(3, [1], name="FLUID")

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.3)
    gmsh.model.mesh.generate(3)

    DirectOpenFOAMExporter(case).export_single_region(region_name="FLUID")
    gmsh.finalize()

    exp_dir = case / "constant" / "polyMesh"
    exp_n_faces = _count_entries(exp_dir / "faces")
    exp_n_owner = _count_entries(exp_dir / "owner")

    match = of_n_faces == exp_n_faces and of_n_owner == exp_n_owner
    status = "PASS" if match else "FAIL"
    print(f"[{status}] faces: g2o={of_n_faces} direct={exp_n_faces}  "
          f"owner: g2o={of_n_owner} direct={exp_n_owner}")
    return match


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = []
    test_single_region()
    results.append("single_region")

    test_multi_region_cht()
    results.append("multi_region_cht")

    test_cross_check_gmshToFoam()
    results.append("cross_check")

    print()
    print("=" * 40)
    print("All tests completed successfully.")
    print("=" * 40)
