"""
Direct export from Gmsh to OpenFOAM native polyMesh format.

This module writes OpenFOAM ``constant/polyMesh`` files (points, faces, owner,
neighbour, boundary, cellZones) directly from the Gmsh Python API, bypassing
the external ``gmshToFoam`` utility.

Supported 3-D cell element types:
    * tetrahedra  (Gmsh type 4)
    * hexahedra   (Gmsh type 5)

Supported 2-D boundary face element types:
    * triangles   (Gmsh type 2)
    * quadrangles (Gmsh type 3)

For CHT multi-region meshes each volume physical group is written to its own
``constant/<regionName>/polyMesh`` directory.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import logging

import gmsh
import numpy as np

logger = logging.getLogger(__name__)

# Gmsh element-type codes
GMSH_TRI = 2
GMSH_QUAD = 3
GMSH_TET = 4
GMSH_HEX = 5
GMSH_PRI = 6
GMSH_PYR = 7

# ---------------------------------------------------------------------------
#  Element topology
# ---------------------------------------------------------------------------

_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
    GMSH_PRI: 6,
    GMSH_PYR: 5,
}

_GMSH_TO_OPENFOAM_CELL = {
    GMSH_TET: "tet",
    GMSH_HEX: "hex",
    GMSH_PRI: "wedge",
    GMSH_PYR: "pyr",
}

# For a tet (nodes n0..n3) the four faces in outward winding order are:
# face opposite to n0:  (n1, n2, n3)
# face opposite to n1:  (n0, n3, n2)
# face opposite to n2:  (n0, n1, n3)
# face opposite to n3:  (n0, n2, n1)
TET_FACES: List[Tuple[int, ...]] = [
    (1, 2, 3),
    (0, 3, 2),
    (0, 1, 3),
    (0, 2, 1),
]

# For a hex (nodes n0..n7) the six outward-facing faces are:
#   bottom  (-z): (0,3,2,1)
#   top    (+z): (4,5,6,7)
#   front  (-y): (0,1,5,4)
#   back   (+y): (3,2,6,7)
#   left   (-x): (0,3,7,4)
#   right  (+x): (1,2,6,5)
HEX_FACES: List[Tuple[int, ...]] = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
]

_FACE_TABLE = {
    GMSH_TET: TET_FACES,
    GMSH_HEX: HEX_FACES,
}


def _face_key(node_ids: List[int]) -> Tuple[int, ...]:
    """Canonical, order-independent face identifier."""
    return tuple(sorted(node_ids))


def _to_upper_triangular(face_nodes: List[int]) -> List[int]:
    """Cyclically rotate *face_nodes* so the minimum vertex is first.

    This preserves the winding order (and hence the normal direction)
    but satisfies OpenFOAM's internal-face ordering requirement.
    """
    if not face_nodes:
        return face_nodes
    min_idx = face_nodes.index(min(face_nodes))
    return face_nodes[min_idx:] + face_nodes[:min_idx]


def _face_normal(pts: np.ndarray) -> np.ndarray:
    """Approximate geometric normal for a polygonal face (3–8 nodes)."""
    n = len(pts)
    if n == 3:
        return np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if n == 4:
        n1 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n2 = np.cross(pts[2] - pts[0], pts[3] - pts[0])
        return n1 + n2
    normal = np.zeros(3)
    for i in range(1, n - 1):
        normal += np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
    return normal


def _orient_face_outward(
    face_nodes: List[int],
    cell_nodes: List[int],
    node_coords: np.ndarray,
) -> List[int]:
    """Return *face_nodes* reordered so the computed normal points
    outward from the cell defined by *cell_nodes*.

    Outward is verified via the dot product between the geometric
    normal and the vector from cell-centroid to face-centroid.

    .. note::
        *face_nodes* and *cell_nodes* are already 0-based OpenFOAM
        indices — no tag remapping needed here.
    """
    cell_pts = node_coords[np.array(cell_nodes)]
    cell_centroid = cell_pts.mean(axis=0)

    face_pts = node_coords[np.array(face_nodes)]
    face_centroid = face_pts.mean(axis=0)
    to_face = face_centroid - cell_centroid

    normal = _face_normal(face_pts)
    if np.dot(normal, to_face) > 0:
        return face_nodes
    return list(reversed(face_nodes))


# ---------------------------------------------------------------------------
#  OpenFOAM ASCII header helpers
# ---------------------------------------------------------------------------

_OF_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |                                                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
   \\\\    /   O peration      | Version:  13                                  |
    \\\\  /    A nd           | Website:  https://openfoam.org                  |
     \\\\/     M anipulation   |                                                 |
\\*---------------------------------------------------------------------------*/
"""

_OF_FOOTER = "// ************************************************************************* //\n"


def _of_header(cls: str, obj: str) -> str:
    return (
        _OF_HEADER
        + "\n"
        + "FoamFile\n"
        "{\n"
        f"    format      ascii;\n"
        f"    class       {cls};\n"
        f'    location    "constant/polyMesh";\n'
        f"    object      {obj};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


# ---------------------------------------------------------------------------
#  Core exporter
# ---------------------------------------------------------------------------

class DirectOpenFOAMExporter:
    """Write OpenFOAM polyMesh files directly from a Gmsh model.

    Usage (single fluid region)::

        gmsh.initialize()
        gmsh.model.add("case")
        # build geometry + assign physical groups
        gmsh.model.mesh.generate(3)
        DirectOpenFOAMExporter("/path/to/case").export_single_region()
        gmsh.finalize()

    Usage (multi-region CHT)::

        gmsh.initialize()
        gmsh.model.add("case")
        # build geometry with FLUID + SOLID volume physical groups
        gmsh.model.mesh.generate(3)
        DirectOpenFOAMExporter("/path/to/case").export_multi_region()
        gmsh.finalize()
    """

    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)

    # ==================================================================
    #  Public API
    # ==================================================================

    def export_single_region(self, region_name: str = "fluid") -> Path:
        """Export a single-region mesh to ``constant/polyMesh``.

        Parameters
        ----------
        region_name : str
            Name used as the cell-zone label.

        Returns
        -------
        Path
            Directory containing the written polyMesh files.
        """
        polyMesh_dir = self.case_path / "constant" / "polyMesh"
        polyMesh_dir.mkdir(parents=True, exist_ok=True)

        data = self._build_mesh_data(None)
        self._write_all(polyMesh_dir, data)
        self._log_stats(region_name, *data)
        logger.info("Direct export complete -> %s", polyMesh_dir)
        return polyMesh_dir

    def export_multi_region(
        self, region_map: Optional[Dict[str, str]] = None
    ) -> List[Path]:
        """Export a multi-region CHT mesh.
        region *X* is written to ``constant/<region_map[X]>/polyMesh/``.

        Parameters
        ----------
        region_map : dict, optional
            Mapping ``{volume_physical_name: directory_name}``.
            If *None*, the physical-group name is used directly.

        Returns
        -------
        list of Path
            Directories of every region that was written.
        """
        vol_groups = gmsh.model.getPhysicalGroups(dim=3)
        vol_names = {
            gmsh.model.getPhysicalName(3, tag): tag
            for _, tag in vol_groups
            if gmsh.model.getPhysicalName(3, tag)
        }
        if not vol_names:
            raise RuntimeError(
                "No 3-D physical groups found -- cannot build CHT regions."
            )

        written: List[Path] = []
        for vol_name, vol_tag in vol_names.items():
            region_dir_name = (region_map or {}).get(vol_name, vol_name)
            polyMesh_dir = (
                self.case_path / "constant" / region_dir_name / "polyMesh"
            )
            polyMesh_dir.mkdir(parents=True, exist_ok=True)

            data = self._build_mesh_data({vol_name: vol_tag})
            self._write_all(polyMesh_dir, data)
            self._log_stats(vol_name, *data)
            logger.info("Region '%s' -> %s", vol_name, polyMesh_dir)
            written.append(polyMesh_dir)

        return written

    # ==================================================================
    #  Mesh data construction (pure Gmsh extraction)
    # ==================================================================

    def _get_node_coords(
        self,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Return ``(points_array, tag_to_index)``.

        *points_array* has shape ``(N, 3)`` and is 0-indexed (OpenFOAM
        convention).  *tag_to_index* maps raw Gmsh node tags to the
        contiguous 0-based indices.
        """
        node_tags, coords, _params = gmsh.model.mesh.getNodes()
        node_tags = [int(t) for t in list(node_tags)]
        coords_list = list(coords)

        # Build tag→index and coordinate lookup in one pass
        coord_lookup: Dict[int, Tuple[float, float, float]] = {}
        for i, tag in enumerate(node_tags):
            coord_lookup[tag] = (
                float(coords_list[3 * i]),
                float(coords_list[3 * i + 1]),
                float(coords_list[3 * i + 2]),
            )

        sorted_tags = sorted(node_tags)
        tag_to_index: Dict[int, int] = {
            tag: i for i, tag in enumerate(sorted_tags)
        }
        points = np.array([coord_lookup[t] for t in sorted_tags], dtype=np.float64)
        return points, tag_to_index

    def _get_entities_for_volumes(
        self, volume_phys: Optional[Dict[str, int]]
    ) -> List[Tuple[int, str]]:
        """Return ``(entity_tag, volume_name)`` pairs for the requested
        volume physical groups."""
        result: List[Tuple[int, str]] = []

        if volume_phys is None:
            vol_groups = gmsh.model.getPhysicalGroups(dim=3)
            for _, tag in vol_groups:
                name = gmsh.model.getPhysicalName(3, tag) or "FLUID"
                result.extend(
                    (ent, name)
                    for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag)
                )
        else:
            for name, ptag in volume_phys.items():
                result.extend(
                    (ent, name)
                    for ent in gmsh.model.getEntitiesForPhysicalGroup(3, ptag)
                )
        return result

    def _get_surface_patch_map(
        self, tag_to_index: Dict[int, int]
    ) -> Dict[Tuple[int, ...], str]:
        """Build a mapping ``sorted_face_key -> patch_name`` from 2-D physical
        groups when available, falling back to face-centroid classification.
        """
        patch_map: Dict[Tuple[int, ...], str] = {}

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = [int(t) for t in list(node_tags)]
        coords_list = list(coords)
        coord_lookup: Dict[int, Tuple[float, float, float]] = {}
        for i, tag in enumerate(node_tags):
            coord_lookup[tag] = (
                float(coords_list[3 * i]),
                float(coords_list[3 * i + 1]),
                float(coords_list[3 * i + 2]),
            )
        sorted_tags = sorted(node_tags)
        index_to_tag = {i: tag for i, tag in enumerate(sorted_tags)}

        surf_groups = gmsh.model.getPhysicalGroups(dim=2)
        if surf_groups:
            all_named = True
            for _, stag in surf_groups:
                name = gmsh.model.getPhysicalName(2, stag)
                if not name:
                    all_named = False
                    break
            if all_named:
                for _, stag in surf_groups:
                    name = gmsh.model.getPhysicalName(2, stag) or "patch"
                    entities = gmsh.model.getEntitiesForPhysicalGroup(2, stag)
                    for entity_tag in entities:
                        etypes, _, enodes = gmsh.model.mesh.getElements(2, entity_tag)
                        for etype, node_list in zip(etypes, enodes):
                            npp = _NODES_PER_ELEM.get(etype, 3)
                            count = len(node_list) // npp
                            for idx in range(count):
                                start = idx * npp
                                raw_nodes = [int(n) for n in node_list[start : start + npp]]
                                of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                                key = _face_key(of_nodes)
                                patch_map.setdefault(key, name)
                return patch_map

        vol_groups = gmsh.model.getPhysicalGroups(dim=3)
        entities = []
        for _, tag in vol_groups:
            name = gmsh.model.getPhysicalName(3, tag) or "FLUID"
            entities.extend(
                (ent, name) for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag)
            )
        if not entities:
            return patch_map

        all_coords = np.array(
            [coord_lookup.get(index_to_tag.get(i, 0), (0.0, 0.0, 0.0)) for i in range(len(tag_to_index))]
        )
        if all_coords.shape[0] == 0:
            return patch_map

        xmin = float(all_coords[:, 0].min())
        ymin = float(all_coords[:, 1].min())
        zmin = float(all_coords[:, 2].min())
        xmax = float(all_coords[:, 0].max())
        ymax = float(all_coords[:, 1].max())
        zmax = float(all_coords[:, 2].max())

        TET_FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
        HEX_FACES = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (3, 2, 6, 7),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]
        _FACE_TABLE = {
            GMSH_TET: TET_FACES,
            GMSH_HEX: HEX_FACES,
        }

        for ent_tag, _zone_name in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(3, ent_tag)
            for etype, node_list in zip(etypes, enodes):
                npp = _NODES_PER_ELEM.get(etype)
                if npp is None:
                    continue
                count = len(node_list) // npp
                for idx in range(count):
                    start = idx * npp
                    raw_nodes = [int(n) for n in node_list[start : start + npp]]
                    of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                    face_defs = _FACE_TABLE.get(etype, [])
                    for fdef in face_defs:
                        face_nodes = [of_nodes[i] for i in fdef]
                        face_pts = all_coords[np.array(face_nodes)]
                        face_centroid = face_pts.mean(axis=0)
                        cx, cy, cz = (
                            float(face_centroid[0]),
                            float(face_centroid[1]),
                            float(face_centroid[2]),
                        )
                        tol = 1e-4
                        if abs(cx - xmin) <= tol:
                            patch = "inlet"
                        elif abs(cx - xmax) <= tol:
                            patch = "outlet"
                        elif abs(cy - ymin) <= tol:
                            patch = "side_left"
                        elif abs(cy - ymax) <= tol:
                            patch = "side_right"
                        elif abs(cz - zmax) <= tol:
                            patch = "top"
                        elif abs(cz - zmin) <= tol:
                            patch = "ground"
                        else:
                            patch = "buildings"
                        key = _face_key(face_nodes)
                        patch_map.setdefault(key, patch)

        return patch_map

        surf_groups = gmsh.model.getPhysicalGroups(dim=2)
        if surf_groups:
            all_named = True
            for _, stag in surf_groups:
                name = gmsh.model.getPhysicalName(2, stag)
                if not name:
                    all_named = False
                    break
            if all_named:
                for _, stag in surf_groups:
                    name = gmsh.model.getPhysicalName(2, stag) or "patch"
                    entities = gmsh.model.getEntitiesForPhysicalGroup(2, stag)
                    for entity_tag in entities:
                        etypes, _, enodes = gmsh.model.mesh.getElements(2, entity_tag)
                        for etype, node_list in zip(etypes, enodes):
                            npp = _NODES_PER_ELEM.get(etype, 3)
                            count = len(node_list) // npp
                            for idx in range(count):
                                start = idx * npp
                                raw_nodes = [int(n) for n in node_list[start : start + npp]]
                                of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                                key = _face_key(of_nodes)
                                patch_map.setdefault(key, name)
                return patch_map

        vol_groups = gmsh.model.getPhysicalGroups(dim=3)
        entities = []
        for _, tag in vol_groups:
            name = gmsh.model.getPhysicalName(3, tag) or "FLUID"
            entities.extend(
                (ent, name) for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag)
            )
        if not entities:
            return patch_map

        all_coords = np.array(
            [coord_lookup.get(index_to_tag.get(i, 0), (0.0, 0.0, 0.0)) for i in range(len(tag_to_index))]
        )
        if all_coords.shape[0] == 0:
            return patch_map

        xmin = float(all_coords[:, 0].min())
        ymin = float(all_coords[:, 1].min())
        zmin = float(all_coords[:, 2].min())
        xmax = float(all_coords[:, 0].max())
        ymax = float(all_coords[:, 1].max())
        zmax = float(all_coords[:, 2].max())

        TET_FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
        HEX_FACES = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (3, 2, 6, 7),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]
        _FACE_TABLE = {
            GMSH_TET: TET_FACES,
            GMSH_HEX: HEX_FACES,
        }

        for ent_tag, _zone_name in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(3, ent_tag)
            for etype, node_list in zip(etypes, enodes):
                npp = _NODES_PER_ELEM.get(etype)
                if npp is None:
                    continue
                count = len(node_list) // npp
                for idx in range(count):
                    start = idx * npp
                    raw_nodes = [int(n) for n in node_list[start : start + npp]]
                    of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                    face_defs = _FACE_TABLE.get(etype, [])
                    for fdef in face_defs:
                        face_nodes = [of_nodes[i] for i in fdef]
                        face_pts = all_coords[np.array(face_nodes)]
                        face_centroid = face_pts.mean(axis=0)
                        cx, cy, cz = (
                            float(face_centroid[0]),
                            float(face_centroid[1]),
                            float(face_centroid[2]),
                        )
                        tol = 1e-4
                        if abs(cx - xmin) <= tol:
                            patch = "inlet"
                        elif abs(cx - xmax) <= tol:
                            patch = "outlet"
                        elif abs(cy - ymin) <= tol:
                            patch = "side_left"
                        elif abs(cy - ymax) <= tol:
                            patch = "side_right"
                        elif abs(cz - zmax) <= tol:
                            patch = "top"
                        elif abs(cz - zmin) <= tol:
                            patch = "ground"
                        else:
                            patch = "buildings"
                        key = _face_key(face_nodes)
                        patch_map.setdefault(key, patch)

        return patch_map

    def _collect_cells(
        self,
        volume_phys: Optional[Dict[str, int]],
        tag_to_index: Dict[int, int],
    ) -> Tuple[List[List[int]], List[str], List[int]]:
        """Extract all 3-D cells from the Gmsh model.

        Returns ``(cell_nodes, zone_names, cell_types)``.  Node indices
        are remapped to OpenFOAM 0-based indices.
        """
        entities = self._get_entities_for_volumes(volume_phys)

        cell_nodes_list: List[List[int]] = []
        zone_names: List[str] = []
        cell_types: List[int] = []

        for ent_tag, zone_name in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(3, ent_tag)
            for etype, node_list in zip(etypes, enodes):
                npp = _NODES_PER_ELEM.get(etype)
                if npp is None:
                    continue
                count = len(node_list) // npp
                for idx in range(count):
                    start = idx * npp
                    raw_nodes = [int(n) for n in node_list[start : start + npp]]
                    of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                    cell_nodes_list.append(of_nodes)
                    zone_names.append(zone_name)
                    cell_types.append(etype)

        if not cell_nodes_list:
            raise RuntimeError(
                "No 3-D volume elements found in the Gmsh model."
            )
        return cell_nodes_list, zone_names, cell_types

    def _build_mesh_data(
        self, volume_phys: Optional[Dict[str, int]] = None
    ) -> Tuple[
        np.ndarray,
        List[Tuple[int, List[int]]],
        List[int],
        List[int],
        List[Tuple[str, str, int, int]],
        List[Tuple[str, List[int]]],
    ]:
        """Assemble OpenFOAM mesh data from the current Gmsh model.

        Returns
        -------
        points : ndarray (N, 3)
        faces : list of (nVerts, [v0, ...])
        owner : list of int  -- one entry per face
        neighbour : list of int -- one entry per internal face
        boundary : list of (name, type, nFaces, startFace)
        cell_zones : list of (zoneName, [cellIndices])
        """
        node_coords, tag_to_index = self._get_node_coords()
        patch_map = self._get_surface_patch_map(tag_to_index)
        cell_nodes_list, zone_names, cell_types = self._collect_cells(
            volume_phys, tag_to_index
        )

        # ---- build face -> {cell_id: oriented_face_nodes} dictionary ----
        # Each cell computes faces with outward normals.  For shared
        # (internal) faces the two cells produce opposite windings; we
        # keep both and pick the *owner* cell’s orientation later
        # (owner normal must point toward neighbour, which equals the
        # owner cell’s outward normal — exactly what we compute).
        face_orientations: Dict[Tuple[int, ...], Dict[int, List[int]]] = {}

        for cell_id in range(len(cell_nodes_list)):
            etype = cell_types[cell_id]
            cnodes = cell_nodes_list[cell_id]
            face_defs = _FACE_TABLE.get(etype, [])
            for fdef in face_defs:
                face_nodes = [cnodes[i] for i in fdef]
                oriented = _orient_face_outward(
                    face_nodes, cnodes, node_coords
                )
                key = _face_key(oriented)
                if key not in face_orientations:
                    face_orientations[key] = {}
                face_orientations[key][cell_id] = oriented

        # ---- classify faces ----
        internal_faces_list: List[Tuple[int, int, List[int]]] = []
        boundary_faces_list: List[Tuple[str, List[int]]] = []

        xmin = float(node_coords[:, 0].min())
        ymin = float(node_coords[:, 1].min())
        zmin = float(node_coords[:, 2].min())
        xmax = float(node_coords[:, 0].max())
        ymax = float(node_coords[:, 1].max())
        zmax = float(node_coords[:, 2].max())

        for key, cell_faces in face_orientations.items():
            cell_ids = list(cell_faces.keys())
            if len(cell_ids) == 1:
                face_nodes = cell_faces[cell_ids[0]]
                patch = patch_map.get(_face_key(face_nodes))
                if patch is None:
                    face_pts = node_coords[np.array(face_nodes)]
                    centroid = face_pts.mean(axis=0)
                    cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
                    tol = 1e-4
                    if abs(cx - xmin) <= tol:
                        patch = "inlet"
                    elif abs(cx - xmax) <= tol:
                        patch = "outlet"
                    elif abs(cy - ymin) <= tol:
                        patch = "side_left"
                    elif abs(cy - ymax) <= tol:
                        patch = "side_right"
                    elif abs(cz - zmax) <= tol:
                        patch = "top"
                    elif abs(cz - zmin) <= tol:
                        patch = "ground"
                    else:
                        patch = "buildings"
                boundary_faces_list.append((patch, face_nodes))
            else:
                owner_cell = min(cell_ids)
                neighbour_cell = max(cell_ids)
                owner_face = cell_faces[owner_cell]
                face_nodes = owner_face
                
                internal_faces_list.append(
                    (owner_cell, neighbour_cell, face_nodes)
                )

        # OpenFOAM requires internal faces sorted by (owner, neighbour)
        internal_faces_list.sort(key=lambda x: (x[0], x[1]))

        faces_out: List[Tuple[int, List[int]]] = []
        owner_out: List[int] = []
        neighbour_out: List[int] = []

        for owner_cell, neighbour_cell, face_nodes in internal_faces_list:
            face_nodes = _to_upper_triangular(face_nodes)
            faces_out.append((len(face_nodes), face_nodes))
            owner_out.append(owner_cell)
            neighbour_out.append(neighbour_cell)

        patch_order: "OrderedDict[str, List[List[int]]]" = OrderedDict()
        for patch, face_nodes in boundary_faces_list:
            patch_order.setdefault(patch, []).append(face_nodes)

        print(f"DEBUG boundary patches: {list(patch_order.keys())}")
        for patch, flist in patch_order.items():
            print(f"  DEBUG {patch}: {len(flist)} faces")

        boundary_out: List[Tuple[str, str, int, int]] = []
        for patch, flist in patch_order.items():
            start_face = len(faces_out)
            for face_nodes in flist:
                face_nodes = _to_upper_triangular(face_nodes)
                faces_out.append((len(face_nodes), face_nodes))
                key = _face_key(face_nodes)
                cell_ids = list(face_orientations[key].keys())
                owner_out.append(cell_ids[0])
            boundary_out.append((patch, "patch", len(flist), start_face))

        cell_zones_out: List[Tuple[str, List[int]]] = []
        if cell_nodes_list:
            zone_map: Dict[str, List[int]] = {}
            for idx, zname in enumerate(zone_names):
                zone_map.setdefault(zname, []).append(idx)
            for zname, indices in zone_map.items():
                cell_zones_out.append((zname, indices))

        points_out, faces_out = self._compact_points(
            node_coords, faces_out
        )

        return (
            points_out,
            faces_out,
            owner_out,
            neighbour_out,
            boundary_out,
            cell_zones_out,
        )

    def _compact_points(
        self,
        node_coords: np.ndarray,
        faces: List[Tuple[int, List[int]]],
    ) -> Tuple[np.ndarray, List[Tuple[int, List[int]]]]:
        """Remove unused points and remap face vertex indices."""
        used = set()
        for _, verts in faces:
            used.update(verts)
        if not used:
            return node_coords, faces

        sorted_used = sorted(used)
        old_to_new = {old: new for new, old in enumerate(sorted_used)}
        new_coords = node_coords[np.array(sorted_used)]
        new_faces = [(n, [old_to_new[v] for v in verts]) for n, verts in faces]
        return new_coords, new_faces

    def _write_all(self, polyMesh_dir: Path, data: Tuple) -> None:
        points, faces, owner, neighbour, boundary, cell_zones = data
        self._write_points(polyMesh_dir / "points", points)
        self._write_faces(polyMesh_dir / "faces", faces)
        self._write_label_list(
            polyMesh_dir / "owner", owner, "owner"
        )
        self._write_label_list(
            polyMesh_dir / "neighbour", neighbour, "neighbour"
        )
        self._write_boundary(
            polyMesh_dir / "boundary", boundary
        )
        self._write_cell_zones(polyMesh_dir / "cellZones", cell_zones)

    def _write_label_list(
        self,
        filepath: Path,
        values: List[int],
        name: str,
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("labelList", name))
            f.write(f"{len(values)}\n(\n")
            for v in values:
                f.write(f"{v}\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_points(
        self,
        filepath: Path,
        points: np.ndarray,
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("vectorField", "points"))
            f.write(f"{len(points)}\n")
            f.write("(\n")
            for x, y, z in points:
                f.write(f"({x} {y} {z})\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_faces(
        self,
        filepath: Path,
        faces: List[Tuple[int, List[int]]],
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("faceList", "faces"))
            f.write(f"{len(faces)}\n(\n")
            for nverts, verts in faces:
                f.write(f"{nverts}(")
                f.write(" ".join(str(v) for v in verts))
                f.write(")\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_boundary(
        self,
        filepath: Path,
        boundary: List[Tuple[str, str, int, int]],
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("polyBoundaryMesh", "boundary"))
            f.write(f"{len(boundary)}\n(\n")
            for pname, ptype, nfaces, start in boundary:
                f.write(f"    {pname}\n")
                f.write("    {\n")
                f.write(f"        type            {ptype};\n")
                f.write(f"        nFaces          {nfaces};\n")
                f.write(f"        startFace       {start};\n")
                f.write("    }\n\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_cell_zones(
        self, filepath: Path, zones: List[Tuple[str, List[int]]]
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("cellZoneList", "cellZones"))
            f.write(f"{len(zones)}\n(\n")
            for zname, members in zones:
                f.write(f"{zname}\n")
                f.write("{\n")
                f.write(f"    cellLabels              List<label>\n")
                f.write(f"{len(members)}\n(\n")
                for m in members:
                    f.write(f"    {m}\n")
                f.write(")\n")
                f.write(";\n")
                f.write("}\n")
                f.write("\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    # ==================================================================
    #  Diagnostics
    # ==================================================================

    @staticmethod
    def _log_stats(
        region_name: str,
        points: np.ndarray,
        faces: List,
        owner: List[int],
        neighbour: List[int],
        boundary: List[Tuple[str, str, int, int]],
        cell_zones: List,
    ) -> None:
        n_bnd = sum(b[2] for b in boundary)
        n_internal = len(owner) - n_bnd
        n_cells = len(cell_zones[0][1]) if cell_zones else 0
        logger.info(
            "[Export '%s'] points=%d  cells=%d  faces=%d  "
            "internalFaces=%d  boundaryFaces=%d  patches=%d",
            region_name,
            len(points),
            n_cells,
            len(faces),
            n_internal,
            n_bnd,
            len(boundary),
        )
