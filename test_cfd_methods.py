#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np


def load_openfoam_environment() -> dict[str, str]:
    """Use the current shell environment or source the path requested by the caller."""
    env = os.environ.copy()
    source_env = env.get("FOAM_BASHRC")
    if not source_env:
        project_dir = env.get("WM_PROJECT_DIR")
        source_env = str(Path(project_dir) / "etc/bashrc") if project_dir else ""
    if source_env and Path(source_env).is_file():
        result = subprocess.run(
            ["bash", "-c", f"source {source_env} && env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                env[key] = value
    elif "WM_PROJECT_DIR" not in env:
        raise RuntimeError(
            "OpenFOAM is not loaded; source its etc/bashrc or set FOAM_BASHRC"
        )
    return env


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--case",
    type=Path,
    default=Path(__file__).resolve().parent / "planarPoiseuille",
    help="OpenFOAM case to inspect",
)
args = parser.parse_args()
env = load_openfoam_environment()
os.environ.update(env)

# Monkey-patch pyvista's BaseReader to handle VTK's OpenFOAM readers
from pyvista.core.utilities.reader import BaseReader

def _patched_set_directory(self, directory):
    self._BaseReader__directory = directory
    self._filename = None
    reader = self.reader
    if hasattr(reader, "SetDirectoryName"):
        reader.SetDirectoryName(directory)
    elif hasattr(reader, "SetFileName"):
        reader.SetFileName(str(directory))
    self._update_information()

BaseReader._set_directory = _patched_set_directory

from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
import pyvista as pv

case_path = args.case.expanduser().resolve()
if not case_path.is_dir():
    raise FileNotFoundError(f"case directory does not exist: {case_path}")
fp = FoamPostProcessing(str(case_path))

print("=" * 60)
print("TEST 1: read_direct with POpenFOAMReader")
print("=" * 60)

reader = pv.POpenFOAMReader(os.path.join(case_path, "system", "controlDict"))
reader.set_active_time_point(25)
mesh = reader.read()
print(f"Mesh type: {type(mesh)}")
print(f"Number of blocks: {mesh.n_blocks}")

main_block = None
if mesh.n_blocks > 0:
    block0 = mesh[0]
    if block0 is not None and hasattr(block0, 'n_points'):
        main_block = block0
        print(f"Using block 0: type={type(main_block).__name__}, n_points={main_block.n_points}, n_cells={main_block.n_cells}")

if main_block is None:
    print("ERROR: No valid main mesh found")
    sys.exit(1)

print(f"Point data: {list(main_block.point_data.keys())}")
print(f"Cell data: {list(main_block.cell_data.keys())}")

if "U" in main_block.point_data:
    main_block.set_active_vectors("U")
    print("Active vectors set to U (point data)")
elif "U" in main_block.cell_data:
    main_block.set_active_vectors("U")
    print("Active vectors set to U (cell data)")

print()
print("=" * 60)
print("TEST 2: calc_y_plus")
print("=" * 60)

nu = 0.1
rho = 1.0
G = 5.0
H = 1.0

try:
    mesh_yp = fp.calc_y_plus(main_block, wall_patch_name="walls", velocity_field="U", viscosity=nu)
    yp_vals = None
    if "y_plus" in mesh_yp.point_data:
        yp_vals = mesh_yp.point_data["y_plus"]
        location = "point_data"
    elif "y_plus" in mesh_yp.cell_data:
        yp_vals = mesh_yp.cell_data["y_plus"]
        location = "cell_data"
    else:
        print("ERROR: y_plus not found in output")

    if yp_vals is not None:
        print(f"y_plus location: {location}")
        print(f"y_plus stats: min={np.min(yp_vals):.6f}, max={np.max(yp_vals):.6f}, mean={np.mean(yp_vals):.6f}")
        n_nonzero = np.count_nonzero(np.abs(yp_vals) > 1e-10)
        print(f"Non-zero y_plus values: {n_nonzero} / {len(yp_vals)}")

        if np.all(np.isfinite(yp_vals)):
            max_yp = np.max(np.abs(yp_vals))
            if max_yp > 0 and max_yp < 1e6:
                print("PASS: calc_y_plus produced finite, physically reasonable values")
            else:
                print(f"WARN: calc_y_plus values seem extreme (max={max_yp})")
        else:
            print("FAIL: calc_y_plus produced NaN or Inf values")
except Exception as e:
    import traceback
    print(f"ERROR in calc_y_plus: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 3: calc_strain_rate")
print("=" * 60)

try:
    mesh_sr = fp.calc_strain_rate(main_block, velocity_field="U")
    if "strain_rate" in mesh_sr.point_data:
        sr_vals = mesh_sr.point_data["strain_rate"]
        print(f"strain_rate stats: min={np.min(sr_vals):.6f}, max={np.max(sr_vals):.6f}, mean={np.mean(sr_vals):.6f}")
        max_sr = np.max(np.abs(sr_vals))
        print(f"Max absolute strain_rate: {max_sr:.4f}")

        if np.all(np.isfinite(sr_vals)) and max_sr > 0.01 and max_sr < 1000:
            print("PASS: calc_strain_rate produced physically reasonable values")
        else:
            print("FAIL: calc_strain_rate values are not reasonable")
    else:
        print("ERROR: strain_rate not found")
        print(f"Point data: {list(mesh_sr.point_data.keys())}")
        print(f"Cell data: {list(mesh_sr.cell_data.keys())}")
except Exception as e:
    import traceback
    print(f"ERROR in calc_strain_rate: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 4: calc_wall_shear_stress")
print("=" * 60)

try:
    mesh_wss = fp.calc_wall_shear_stress(main_block, velocity_field="U", viscosity=nu, wall_normal=[0, 1, 0])
    wss_vals = None
    if "wall_shear_stress" in mesh_wss.cell_data:
        wss_vals = mesh_wss.cell_data["wall_shear_stress"]
        location = "cell_data"
    elif "wall_shear_stress" in mesh_wss.point_data:
        wss_vals = mesh_wss.point_data["wall_shear_stress"]
        location = "point_data"
    else:
        print("ERROR: wall_shear_stress not found")
        print(f"Cell data: {list(mesh_wss.cell_data.keys())}")
        print(f"Point data: {list(mesh_wss.point_data.keys())}")

    if wss_vals is not None:
        print(f"wall_shear_stress location: {location}")
        print(f"wall_shear_stress stats: min={np.min(wss_vals):.6f}, max={np.max(wss_vals):.6f}, mean={np.mean(wss_vals):.6f}")

        tau_w_analytical = 0.5 * rho * G * H
        print(f"Analytical tau_w = {tau_w_analytical:.4f} Pa")

        if np.all(np.isfinite(wss_vals)):
            max_wss = np.max(np.abs(wss_vals))
            if max_wss > 0.01 and max_wss < 100:
                print("PASS: calc_wall_shear_stress produced physically reasonable values")
            else:
                print(f"FAIL: calc_wall_shear_stress values are not reasonable (expected ~{tau_w_analytical:.4f} Pa)")
        else:
            print("FAIL: calc_wall_shear_stress produced NaN or Inf values")
except Exception as e:
    import traceback
    print(f"ERROR in calc_wall_shear_stress: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("All tests completed.")
