"""Regression test for importing post-processing without CAD dependencies."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_postprocess_import_does_not_load_cad_dependencies():
    """The post-processing namespace must be usable without build123d/Gmsh/CAD."""
    src = Path(__file__).resolve().parents[1] / "src"
    script = r'''
import sys

class BlockCADImports:
    blocked = {
        "build123d", "gmsh", "cadquery", "OCC", "vmtk", "pygmsh",
        "trimesh", "shapely", "pyacvd", "open3d",
    }
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self.blocked:
            raise ImportError(f"CAD dependency forbidden in isolated test: {fullname}")
        return None

sys.meta_path.insert(0, BlockCADImports())
from foampilot.postprocess import CFDMonitor, ResultMetadata, EngineeringReport
assert CFDMonitor is not None
assert ResultMetadata is not None
assert EngineeringReport is not None
assert not any(name.split(".", 1)[0] in BlockCADImports.blocked for name in sys.modules)
print("postprocess import without CAD: OK")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "postprocess import without CAD: OK" in completed.stdout
