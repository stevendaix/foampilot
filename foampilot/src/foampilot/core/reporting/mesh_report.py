import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from foampilot.base.openFOAMFile import OpenFOAMFile

logger = logging.getLogger(__name__)


class MeshQualityReport:
    def __init__(self, case_dir: str | Path):
        self.case_dir = Path(case_dir)
        self.mesh_stats: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, Any] = {}
        self.block_mesh_log: Optional[Path] = None
        self.poly_mesh_dir = self.case_dir / "constant" / "polyMesh"

    def _find_blockmesh_log(self) -> Optional[Path]:
        candidates = list(self.case_dir.glob("log.blockMesh"))
        if candidates:
            self.block_mesh_log = candidates[0]
            return self.block_mesh_log
        candidates = list(self.case_dir.glob("log.*"))
        for c in candidates:
            if "blockMesh" in c.name:
                self.block_mesh_log = c
                return self.block_mesh_log
        return None

    def _parse_blockmesh_log(self) -> None:
        if self.block_mesh_log is None:
            self._find_blockmesh_log()
        if self.block_mesh_log is None or not self.block_mesh_log.exists():
            return

        with self.block_mesh_log.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        internal_faces = re.search(r"Number of internal faces\s*:\s*(\d+)", content)
        if internal_faces:
            self.mesh_stats["internal_faces"] = int(internal_faces.group(1))

        boundary_faces = re.search(r"Number of boundary faces\s*:\s*(\d+)", content)
        if boundary_faces:
            self.mesh_stats["boundary_faces"] = int(boundary_faces.group(1))

        defined_bnd = re.search(
            r"Number of defined boundary faces\s*:\s*(\d+)", content
        )
        if defined_bnd:
            self.mesh_stats["defined_boundary_faces"] = int(defined_bnd.group(1))

        undefined_bnd = re.search(
            r"Number of undefined boundary faces\s*:\s*(\d+)", content
        )
        if undefined_bnd:
            self.mesh_stats["undefined_boundary_faces"] = int(undefined_bnd.group(1))

        cells_match = re.search(r"Creating cells", content)
        if cells_match:
            cell_count_match = re.search(
                r"Number of cells\s*:\s*(\d+)", content
            )
            if cell_count_match:
                self.mesh_stats["cell_count"] = int(cell_count_match.group(1))

        points_match = re.search(r"Number of points\s*:\s*(\d+)", content)
        if points_match:
            self.mesh_stats["points"] = int(points_match.group(1))

        block_info = re.findall(r"Block (\d+) cell size", content)
        if block_info:
            self.mesh_stats["num_blocks"] = len(block_info)

        skewness = re.search(r"Skewness\s*:\s*([\d\.]+)", content)
        if skewness:
            self.quality_metrics["skewness"] = float(skewness.group(1))

        non_orth = re.search(r"non-orthogonality\s*:\s*([\d\.]+)", content, re.IGNORECASE)
        if non_orth:
            self.quality_metrics["non_orthogonality"] = float(non_orth.group(1))

        vol_match = re.search(r"Total volume\s*:\s*([\d\.Ee\+\-]+)", content)
        if vol_match:
            self.quality_metrics["total_volume"] = float(vol_match.group(1))

        min_vol_match = re.search(r"Min volume\s*:\s*([\d\.Ee\+\-]+)", content)
        if min_vol_match:
            self.quality_metrics["min_volume"] = float(min_vol_match.group(1))

        max_vol_match = re.search(r"Max volume\s*:\s*([\d\.Ee\+\-]+)", content)
        if max_vol_match:
            self.quality_metrics["max_volume"] = float(max_vol_match.group(1))

    def _parse_polyMesh_files(self) -> None:
        if not self.poly_mesh_dir.exists():
            return

        owner_file = self.poly_mesh_dir / "owner"
        if owner_file.exists():
            try:
                of = OpenFOAMFile("owner", object_name="owner")
                if hasattr(of, "nb_cell"):
                    self.mesh_stats["cell_count"] = of.nb_cell
                if hasattr(of, "nb_faces"):
                    self.mesh_stats["face_count"] = of.nb_faces
            except Exception:
                pass

        faces_file = self.poly_mesh_dir / "faces"
        if faces_file.exists():
            try:
                ff = OpenFOAMFile("faces", object_name="faces")
                if hasattr(ff, "nfaces"):
                    self.mesh_stats["face_count"] = ff.nfaces
            except Exception:
                pass

        points_file = self.poly_mesh_dir / "points"
        if points_file.exists():
            try:
                pf = OpenFOAMFile("points", object_name="points")
                if hasattr(pf, "nb_pts"):
                    self.mesh_stats["points_count"] = pf.nb_pts
            except Exception:
                pass

        boundary_file = self.poly_mesh_dir / "boundary"
        if boundary_file.exists():
            try:
                bcf = OpenFOAMFile("boundary", object_name="boundary")
                if hasattr(bcf, "boundaryField") and isinstance(
                    bcf.boundaryField, dict
                ):
                    self.mesh_stats["boundary_patches"] = list(
                        bcf.boundaryField.keys()
                    )
                    self.mesh_stats["num_patches"] = len(
                        bcf.boundaryField
                    )
            except Exception:
                pass

    def _compute_quality_metrics(self) -> None:
        cc = self.mesh_stats.get("cell_count", 0)
        if cc == 0:
            cc = self.mesh_stats.get("internal_cells", 0)
        self.quality_metrics["cell_count"] = cc

        if "points_count" in self.mesh_stats:
            pc = self.mesh_stats["points_count"]
            if cc > 0:
                self.quality_metrics["points_per_cell"] = round(pc / cc, 2)

        if "internal_faces" in self.mesh_stats:
            if cc > 0:
                self.quality_metrics["faces_per_cell"] = round(
                    self.mesh_stats["internal_faces"] / cc, 2
                )

        if (
            "boundary_faces" in self.mesh_stats
            and self.mesh_stats.get("num_patches", 0) > 0
        ):
            self.quality_metrics["boundary_faces_per_patch"] = round(
                self.mesh_stats["boundary_faces"] / self.mesh_stats["num_patches"], 2
            )

        tv = self.quality_metrics.get("total_volume", 0.0)
        mv = self.quality_metrics.get("min_volume", 0.0)
        xv = self.quality_metrics.get("max_volume", 0.0)
        if tv > 0 and cc > 0:
            self.quality_metrics["average_cell_volume"] = tv / cc
        if mv > 0 and xv > 0 and mv != xv:
            self.quality_metrics["volume_range"] = f"{mv:.6e} .. {xv:.6e}"
            self.quality_metrics["volume_ratio"] = round(xv / max(mv, 1e-100), 2)

    def generate_report(self) -> str:
        self._parse_blockmesh_log()
        self._parse_polyMesh_files()
        self._compute_quality_metrics()

        lines: list[str] = []
        lines.append("## Mesh Quality Report")
        lines.append("")

        if self.mesh_stats:
            lines.append("### Mesh Topology")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, val in self.mesh_stats.items():
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                lines.append(f"| `{key}` | `{val}` |")
            lines.append("")

        if self.quality_metrics:
            lines.append("### Quality Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, val in self.quality_metrics.items():
                if isinstance(val, dict):
                    val = str(val)
                lines.append(f"| `{key}` | `{val}` |")
            lines.append("")
        else:
            lines.append("No mesh quality metrics available.")
            lines.append("")

        if self.block_mesh_log and self.block_mesh_log.exists():
            lines.append("### BlockMesh Summary")
            lines.append("")
            lines.append(f"Log file: `{self.block_mesh_log.name}`")
            lines.append("")
            with self.block_mesh_log.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Creating") or stripped.startswith("Checking"):
                        lines.append(f"- `{stripped}`")
            lines.append("")

        return "\n".join(lines)

    def write_report(self, output_path: str | Path | None = None) -> Path:
        report_content = self.generate_report()
        if output_path is None:
            output_path = self.case_dir / "report.md"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(report_content)
        return output_path
