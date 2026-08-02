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
GMSH_TET = 4
GMSH_HEX = 5
GMSH_TRI = 2
GMSH_QUAD = 3

# ---------------------------------------------------------------------------
#  Element topology
# ---------------------------------------------------------------------------

_NODES_PER_ELEM = {
    GMSH_TRI: 3,
    GMSH_QUAD: 4,
    GMSH_TET: 4,
    GMSH_HEX: 8,
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
        """Build a mapping ``sorted_face_key -> patch_name`` from all
        2-D physical groups (surface elements).

        Node indices in the face key are remapped to OpenFOAM 0-based
        indices so they are consistent with cell-face keys.
        """
        patch_map: Dict[Tuple[int, ...], str] = {}
        surf_groups = gmsh.model.getPhysicalGroups(dim=2)
        for _, stag in surf_groups:
            pname = gmsh.model.getPhysicalName(2, stag)
            if not pname:
                continue
            for ent in gmsh.model.getEntitiesForPhysicalGroup(2, stag):
                etypes, _, enodes = gmsh.model.mesh.getElements(2, ent)
                offset = 0
                for etype, node_list in zip(etypes, enodes):
                    npp = _NODES_PER_ELEM.get(etype, 0)
                    if npp == 0:
                        continue
                    count = len(node_list) // npp
                    for idx in range(count):
                        start = offset + idx * npp
                        raw_face = [int(n) for n in node_list[start : start + npp]]
                        of_face = [tag_to_index.get(n, n) for n in raw_face]
                        key = _face_key(of_face)
                        patch_map.setdefault(key, pname)
                    offset += npp * count
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
            offset = 0
            for etype, node_list in zip(etypes, enodes):
                npp = _NODES_PER_ELEM.get(etype, 0)
                if npp == 0:
                    continue
                count = len(node_list) // npp
                for idx in range(count):
                    start = offset + idx * npp
                    raw_nodes = [int(n) for n in node_list[start : start + npp]]
                    of_nodes = [tag_to_index.get(n, n) for n in raw_nodes]
                    cell_nodes_list.append(of_nodes)
                    zone_names.append(zone_name)
                    cell_types.append(etype)
                offset += npp * count

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

        for key, cell_faces in face_orientations.items():
            cell_ids = list(cell_faces.keys())
            if len(cell_ids) == 1:
                patch = patch_map.get(key, "patch0")
                boundary_faces_list.append((patch, cell_faces[cell_ids[0]]))
            else:
                owner_cell = min(cell_ids)
                neighbour_cell = max(cell_ids)
                # owner's outward normal already points toward neighbour
                internal_faces_list.append(
                    (owner_cell, neighbour_cell, cell_faces[owner_cell])
                )

        # OpenFOAM requires internal faces sorted by (owner, neighbour)
        # — this is the "upper triangular" ordering.
        internal_faces_list.sort(key=lambda x: (x[0], x[1]))

        # ---- assemble faces / owner / neighbour ----
        faces_out: List[Tuple[int, List[int]]] = []
        owner_out: List[int] = []
        neighbour_out: List[int] = []

        for owner_cell, neighbour_cell, face_nodes in internal_faces_list:
            face_nodes = _to_upper_triangular(face_nodes)
            faces_out.append((len(face_nodes), face_nodes))
            owner_out.append(owner_cell)
            neighbour_out.append(neighbour_cell)

        # boundary faces -- grouped by patch
        patch_order: "OrderedDict[str, List[List[int]]]" = OrderedDict()
        for patch, face_nodes in boundary_faces_list:
            patch_order.setdefault(patch, []).append(face_nodes)

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

        # ---- cell zones ----
        zone_groups: Dict[str, List[int]] = {}
        for cell_id, zn in enumerate(zone_names):
            zone_groups.setdefault(zn, []).append(cell_id)
        cell_zones_out = [
            (zn, sorted(members)) for zn, members in zone_groups.items()
        ]

        # ---- compact points (remove unused, remap indices) ----
        # Collect all vertex labels actually referenced by faces
        used_vertices: Set[int] = set()
        for _, face_nodes in faces_out:
            used_vertices.update(face_nodes)

        # Build old-index -> new-index map (sorted for determinism)
        remap: Dict[int, int] = {
            old: new for new, old in enumerate(sorted(used_vertices))
        }

        compacted_points = node_coords[np.array(sorted(used_vertices), dtype=int)]

        # Remap face vertex lists
        compacted_faces = [
            (n, [remap[v] for v in face_nodes])
            for n, face_nodes in faces_out
        ]
        faces_out = compacted_faces

        return (
            compacted_points,
            faces_out,
            owner_out,
            neighbour_out,
            boundary_out,
            cell_zones_out,
        )

    # ==================================================================
    #  File writers
    # ==================================================================

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
        self._write_boundary(polyMesh_dir / "boundary", boundary)
        self._write_cell_zones(polyMesh_dir / "cellZones", cell_zones)

    def _write_points(self, filepath: Path, points: np.ndarray) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("vectorField", "points"))
            f.write(f"{len(points)}\n(\n")
            for p in points:
                f.write(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_faces(
        self, filepath: Path, faces: List[Tuple[int, List[int]]]
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("faceList", "faces"))
            f.write(f"{len(faces)}\n(\n")
            for nVerts, face_nodes in faces:
                nodes_str = " ".join(str(n) for n in face_nodes)
                f.write(f"{nVerts}({nodes_str})\n")
            f.write(")\n\n")
            f.write(_OF_FOOTER)

    def _write_label_list(
        self, filepath: Path, values: List[int], name: str
    ) -> None:
        with filepath.open("w") as f:
            f.write(_of_header("labelList", name))
            if not values:
                f.write("0()\n\n")
            else:
                f.write(f"{len(values)}\n(\n")
                for v in values:
                    f.write(f"{v}\n")
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
