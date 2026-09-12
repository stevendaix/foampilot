from typing import Tuple, List, Dict
import gmsh

from foampilot.urban.geometry.cfd_geometry import CFDGeometry


class PatchAssigner:
    def assign(self, builder) -> None:
        geometry = builder.geometry
        xmin, ymin, zmin, xmax, ymax, zmax = geometry.domain_box

        all_faces = gmsh.model.getEntities(dim=2)
        patch_to_surfaces: Dict[str, List[int]] = {}

        for _, face in all_faces:
            try:
                com = gmsh.model.occ.getCenterOfMass(2, face)
            except Exception:
                continue

            if com is None:
                continue

            cx, cy, cz = com
            patch = self._classify(cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax)
            patch_to_surfaces.setdefault(patch, []).append(face)

        for patch_name, tags in patch_to_surfaces.items():
            if tags:
                gmsh.model.addPhysicalGroup(2, tags, name=patch_name)

        fluid_volumes = gmsh.model.getEntities(dim=3)
        fluid_tags = [tag for _, tag in fluid_volumes]
        if fluid_tags:
            gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

    def _classify(
        self,
        cx: float, cy: float, cz: float,
        xmin: float, ymin: float, zmin: float,
        xmax: float, ymax: float, zmax: float,
    ) -> str:
        tol = 1e-6

        if abs(cx - xmin) < tol:
            return "inlet"
        if abs(cx - xmax) < tol:
            return "outlet"
        if abs(cy - ymin) < tol:
            return "side_left"
        if abs(cy - ymax) < tol:
            return "side_right"
        if abs(cz - zmax) < tol:
            return "top"
        if abs(cz - zmin) < tol:
            return "ground"
        return "buildings"
