#!/usr/bin/env python3
"""
Gmsh geometry importer for build123d solids.

Encapsulates the temporary BREP transfer between build123d and Gmsh.
The rest of the pipeline never sees the BREP file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import gmsh

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "foampilot" / "src"))

from build123d import Solid


class GmshGeometryImporter:
    """Import build123d solids into Gmsh via temporary BREP file."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path("/tmp")
        self._brep_path: Optional[Path] = None

    def import_build123d(self, solid: Solid) -> list[tuple[int, int]]:
        """Import a build123d solid into Gmsh.

        Args:
            solid: build123d Solid to import.

        Returns:
            List of (dim, tag) pairs for imported entities.

        Raises:
            RuntimeError: If import fails or no volume is found.
        """
        from build123d import export_step

        self._brep_path = self.temp_dir / f"build123_import_{id(solid)}.step"

        try:
            export_step(solid, str(self._brep_path))

            gmsh.initialize()
            gmsh.model.add("build123_import")
            gmsh.option.setNumber("Geometry.Tolerance", 1e-6)

            entities = gmsh.model.occ.importShapes(
                str(self._brep_path),
                highestDimOnly=True,
            )
            gmsh.model.occ.synchronize()

            volumes = [tag for dim, tag in entities if dim == 3]
            if not volumes:
                raise RuntimeError(
                    f"No volume found after importing build123d solid. "
                    f"Entities: {entities}"
                )

            return entities

        except Exception as exc:
            raise RuntimeError(f"Failed to import build123d solid into Gmsh: {exc}") from exc

        finally:
            if self._brep_path and self._brep_path.exists():
                self._brep_path.unlink(missing_ok=True)
                self._brep_path = None

    def cleanup(self):
        """Clean up temporary files."""
        if self._brep_path and self._brep_path.exists():
            self._brep_path.unlink(missing_ok=True)
            self._brep_path = None
