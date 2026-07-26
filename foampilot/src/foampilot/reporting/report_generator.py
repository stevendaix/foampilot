import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from foampilot.base.openFOAMFile import OpenFOAMFile

logger = logging.getLogger(__name__)


class SimulationReport:
    def __init__(self, case_dir: str | Path):
        self.case_dir = Path(case_dir)
        self.log_file: Optional[Path] = None
        self.solver_name: str = "unknown"
        self.metrics: Dict[str, Any] = {}
        self.residual_data: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"time": [], "initial": [], "final": [], "iterations": []}
        )
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.execution_time: float = 0.0
        self.clock_time: int = 0
        self.iteration_count: int = 0
        self.courant_mean: float = 0.0
        self.courant_max: float = 0.0
        self.mesh_stats: Dict[str, Any] = {}
        self.solver_settings: Dict[str, Any] = {}
        self.bc_summary: Dict[str, Any] = {}

    def _find_log_file(self) -> Optional[Path]:
        if not self.case_dir.exists():
            logger.warning("Case directory does not exist: %s", self.case_dir)
            return None
        candidates = list(self.case_dir.glob("log.*"))
        if candidates:
            self.log_file = candidates[0]
            match = re.search(r"log\.(.+)", self.log_file.name)
            if match:
                self.solver_name = match.group(1)
            return self.log_file
        return None

    def _parse_log(self) -> None:
        if self.log_file is None:
            self._find_log_file()
        if self.log_file is None or not self.log_file.exists():
            logger.warning("No log file found for case: %s", self.case_dir)
            return

        time_pattern = re.compile(r"Time = (\d+\.?\d*)s?")
        solver_pattern = re.compile(
            r"(smoothSolver|GAMG|PCG|PBiCGStab|geometricOneWire):\s+"
            r"Solving for (\w+),\s+"
            r"Initial residual = ([\d\.Ee\+\-]+),\s+"
            r"Final residual = ([\d\.Ee\+\-]+),\s+"
            r"No Iterations (\d+)"
        )
        execution_pattern = re.compile(
            r"ExecutionTime = ([\d\.]+) s\s+ClockTime = (\d+) s"
        )
        courant_pattern = re.compile(
            r"Courant Number mean:\s+([\d\.Ee\+\-]+)\s+max:\s+([\d\.Ee\+\-]+)"
        )
        continuity_pattern = re.compile(
            r"time step continuity errors :\s+"
            r"sum local = ([\d\.Ee\+\-]+),\s+"
            r"global = ([\d\.Ee\+\-]+),\s+"
            r"cumulative = ([\d\.Ee\+\-]+)"
        )
        convergence_pattern = re.compile(
            r"converged\s*=\s*(true|false|1|0)"
        )
        warning_pattern = re.compile(
            r"(WARNING|warning|warn|WARNING:)", re.IGNORECASE
        )
        error_pattern = re.compile(
            r"(ERROR|error|ERR|fatal|FATAL|exception|Exception)"
        )
        end_pattern = re.compile(r"^\s*End\s*$", re.MULTILINE)

        current_time: Optional[float] = None
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()

                if (t_match := time_pattern.search(stripped)):
                    current_time = float(t_match.group(1))
                    self.iteration_count += 1

                if solver_match := solver_pattern.search(stripped):
                    solver_type, field, init_res, final_res, n_iter = (
                        solver_match.group(1),
                        solver_match.group(2),
                        float(solver_match.group(3)),
                        float(solver_match.group(4)),
                        int(solver_match.group(5)),
                    )
                    self.residual_data[field]["time"].append(
                        current_time if current_time is not None else 0.0
                    )
                    self.residual_data[field]["initial"].append(init_res)
                    self.residual_data[field]["final"].append(final_res)
                    self.residual_data[field]["iterations"].append(n_iter)

                if (exec_match := execution_pattern.search(stripped)):
                    self.execution_time = float(exec_match.group(1))
                    self.clock_time = int(exec_match.group(2))

                if (c_match := courant_pattern.search(stripped)):
                    self.courant_mean = float(c_match.group(1))
                    self.courant_max = float(c_match.group(2))

                if continuity_match := continuity_pattern.search(stripped):
                    if current_time is not None:
                        self.residual_data["continuity"][
                            "time"
                        ].append(current_time)
                        self.residual_data["continuity"][
                            "initial"
                        ].append(0.0)
                        self.residual_data["continuity"][
                            "final"
                        ].append(float(continuity_match.group(2)))
                        self.residual_data["continuity"][
                            "iterations"
                        ].append(0)

                if warning_pattern.search(stripped):
                    self.warnings.append(stripped)

                if error_pattern.search(stripped):
                    self.errors.append(stripped)

                if end_pattern.search(stripped):
                    pass

    def _extract_final_residuals(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for field, data in self.residual_data.items():
            if field == "continuity":
                continue
            if data["final"]:
                result[field] = {
                    "initial": data["initial"][-1],
                    "final": data["final"][-1],
                    "iterations": data["iterations"][-1],
                    "converged": data["final"][-1] < 1e-4,
                }
        if self.residual_data["continuity"]["final"]:
            result["continuity"] = {
                "initial": 0.0,
                "final": self.residual_data["continuity"]["final"][-1],
                "iterations": 0,
                "converged": abs(
                    self.residual_data["continuity"]["final"][-1]
                ) < 1e-4,
            }
        return result

    def _extract_solver_settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {}
        control_dict_path = self.case_dir / "system" / "controlDict"
        if control_dict_path.exists():
            try:
                cf = OpenFOAMFile("controlDict", object_name="controlDict")
                for attr in [
                    "application",
                    "startFrom",
                    "startTime",
                    "stopAt",
                    "endTime",
                    "deltaT",
                    "writeControl",
                    "writeInterval",
                    "purgeWrite",
                    "writeFormat",
                    "writePrecision",
                    "writeCompression",
                    "timeFormat",
                    "timePrecision",
                    "runTimeModifiable",
                ]:
                    if hasattr(cf, attr):
                        settings[attr] = getattr(cf, attr)
            except Exception:
                pass

        fv_solution_path = self.case_dir / "system" / "fvSolution"
        if fv_solution_path.exists():
            try:
                sf = OpenFOAMFile("fvSolution", object_name="fvSolution")
                settings["solvers"] = {}
                if hasattr(sf, "solvers") and isinstance(sf.solvers, dict):
                    for solver_name, solver_params in sf.solvers.items():
                        if isinstance(solver_params, dict):
                            settings["solvers"][solver_name] = {
                                k: str(v) for k, v in solver_params.items()
                            }
            except Exception:
                pass

        fv_schemes_path = self.case_dir / "system" / "fvSchemes"
        if fv_schemes_path.exists():
            try:
                sch = OpenFOAMFile(
                    "fvSchemes", object_name="fvSchemes"
                )
                for attr_name in [
                    "ddtSchemes",
                    "gradSchemes",
                    "divSchemes",
                    "laplacianSchemes",
                    "interpolationSchemes",
                    "snGradSchemes",
                ]:
                    if hasattr(sch, attr_name):
                        settings[attr_name] = getattr(sch, attr_name)
            except Exception:
                pass

        return settings

    def _extract_bc_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        boundary_path = self.case_dir / "0" / "boundary"
        if boundary_path.exists():
            try:
                bf = OpenFOAMFile(
                    "boundary", object_name="boundary"
                )
                if hasattr(bf, "boundaryField") and isinstance(
                    bf.boundaryField, dict
                ):
                    for patch_name, patch_data in bf.boundaryField.items():
                        patch_type = (
                            patch_data.get("type", "unknown")
                            if isinstance(patch_data, dict)
                            else "unknown"
                        )
                        summary[patch_name] = {"type": patch_type}
                        if isinstance(patch_data, dict):
                            for key in [
                                "value",
                                "uniformValue",
                                "phi",
                                "rate",
                                "flux",
                            ]:
                                if key in patch_data:
                                    summary[patch_name][key] = str(
                                        patch_data[key]
                                    )
            except Exception:
                pass
        return summary

    def _compute_convergence_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        final_res = self._extract_final_residuals()
        if not final_res:
            return metrics

        all_converged = all(
            v.get("converged", False) for v in final_res.values()
        )
        max_final = 0.0
        for field, data in final_res.items():
            fval = abs(data.get("final", 1.0))
            if fval > max_final:
                max_final = fval
        metrics["all_converged"] = all_converged
        metrics["max_final_residual"] = max_final
        metrics["total_iterations"] = sum(
            v.get("iterations", 0) for v in final_res.values()
        )
        metrics["num_fields"] = len(final_res)
        metrics["field_details"] = final_res
        return metrics

    def _extract_mesh_stats_from_log(self) -> None:
        if self.log_file is None:
            return
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        cell_match = re.search(
            r"Number of (internal|boundary) cells\s*:\s*(\d+)", content
        )
        if cell_match:
            self.mesh_stats["internal_cells"] = int(cell_match.group(2))

        face_match = re.search(
            r"Number of internal faces\s*:\s*(\d+)", content
        )
        if face_match:
            self.mesh_stats["internal_faces"] = int(face_match.group(1))

        bnd_match = re.search(
            r"Number of (boundary|processor) faces\s*:\s*(\d+)", content
        )
        if bnd_match:
            self.mesh_stats["boundary_faces"] = int(bnd_match.group(2))

        point_match = re.search(r"Number of points\s*:\s*(\d+)", content)
        if point_match:
            self.mesh_stats["points"] = int(point_match.group(1))

        mesh_dim_match = re.search(
            r"Creating\s+(\d+)\s*dimensional mesh", content
        )
        if mesh_dim_match:
            self.mesh_stats["dimension"] = int(mesh_dim_match.group(1))

        block_msg = re.search(
            r"Creating block mesh from\s+\"([^\"]+)\"", content
        )
        if block_msg:
            self.mesh_stats["block_mesh_dict"] = block_msg.group(1)

    def _extract_mesh_stats_from_polyMesh(self) -> None:
        poly_mesh_dir = self.case_dir / "constant" / "polyMesh"
        if not poly_mesh_dir.exists():
            return

        points_file = poly_mesh_dir / "points"
        if points_file.exists():
            try:
                pf = OpenFOAMFile("points", object_name="points")
                if hasattr(pf, "points"):
                    pts = pf.points
                    if isinstance(pts, list) and len(pts) > 0:
                        self.mesh_stats["points_count"] = len(pts)
                        try:
                            import numpy as np

                            arr = np.array(pts, dtype=float)
                            self.mesh_stats[
                                "bounds_x"
                            ] = f"{arr[:, 0].min():.6f} .. {arr[:, 0].max():.6f}"
                            self.mesh_stats[
                                "bounds_y"
                            ] = f"{arr[:, 1].min():.6f} .. {arr[:, 1].max():.6f}"
                            self.mesh_stats[
                                "bounds_z"
                            ] = f"{arr[:, 2].min():.6f} .. {arr[:, 2].max():.6f}"
                        except Exception:
                            pass
            except Exception:
                pass

        owner_file = poly_mesh_dir / "owner"
        if owner_file.exists():
            try:
                of = OpenFOAMFile("owner", object_name="owner")
                if hasattr(of, "nb_cell"):
                    self.mesh_stats["cell_count"] = of.nb_cell
                if hasattr(of, "nb_faces"):
                    self.mesh_stats["face_count"] = of.nb_faces
            except Exception:
                pass

        boundary_file = poly_mesh_dir / "boundary"
        if boundary_file.exists():
            try:
                bcf = OpenFOAMFile(
                    "boundary", object_name="boundary"
                )
                if hasattr(bcf, "boundaryField") and isinstance(
                    bcf.boundaryField, dict
                ):
                    self.mesh_stats["boundary_patches"] = list(
                        bcf.boundaryField.keys()
                    )
                    self.mesh_stats[
                        "num_patches"
                    ] = len(bcf.boundaryField)
            except Exception:
                pass

    def _extract_warnings_errors(self) -> None:
        if self.log_file is None:
            return
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if re.search(
                    r"(WARNING|warning|warn)", stripped, re.IGNORECASE
                ):
                    if stripped not in self.warnings:
                        self.warnings.append(stripped)
                if re.search(
                    r"(ERROR|error|ERR|fatal|FATAL)", stripped, re.IGNORECASE
                ):
                    if stripped not in self.errors:
                        self.errors.append(stripped)

    def generate_report(self) -> str:
        self._parse_log()
        self._extract_warnings_errors()
        self._extract_mesh_stats_from_log()
        self._extract_mesh_stats_from_polyMesh()
        self._extract_solver_settings()
        self._extract_bc_summary()

        final_residuals = self._extract_final_residuals()
        convergence = self._compute_convergence_metrics()

        lines: List[str] = []
        lines.append(f"# Simulation Report")
        lines.append("")
        lines.append(f"**Case:** `{self.case_dir}`")
        lines.append(f"**Solver:** `{self.solver_name}`")
        lines.append(f"**Generated:** {self.log_file.name if self.log_file else 'N/A'}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Execution Time | {self.execution_time:.3f} s |")
        lines.append(f"| Clock Time | {self.clock_time} s |")
        lines.append(f"| Iteration Count | {self.iteration_count} |")
        lines.append(f"| Courant Number (mean) | {self.courant_mean:.6e} |")
        lines.append(f"| Courant Number (max) | {self.courant_max:.6e} |")
        lines.append(
            f"| Converged | {'Yes' if convergence.get('all_converged', False) else 'No'} |"
        )
        lines.append(
            f"| Max Final Residual | {convergence.get('max_final_residual', 'N/A')} |"
        )
        lines.append(
            f"| Total Solver Iterations | {convergence.get('total_iterations', 0)} |"
        )
        lines.append(f"| Cell Count | {self.mesh_stats.get('cell_count', self.mesh_stats.get('internal_cells', 'N/A'))} |")
        lines.append(f"| Internal Faces | {self.mesh_stats.get('internal_faces', 'N/A')} |")
        lines.append(f"| Boundary Faces | {self.mesh_stats.get('boundary_faces', 'N/A')} |")
        lines.append("")

        lines.append("## Mesh Info")
        lines.append("")
        if self.mesh_stats:
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, val in self.mesh_stats.items():
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                lines.append(f"| `{key}` | `{val}` |")
        else:
            lines.append("No mesh statistics available.")
        lines.append("")

        lines.append("## Solver Settings")
        lines.append("")
        if self.solver_settings:
            for key, val in self.solver_settings.items():
                if isinstance(val, dict):
                    lines.append(f"### `{key}`")
                    lines.append("")
                    lines.append("| Key | Value |")
                    lines.append("|-----|-------|")
                    for k, v in val.items():
                        lines.append(f"| `{k}` | `{v}` |")
                    lines.append("")
                elif isinstance(val, list):
                    lines.append(f"### `{key}`")
                    lines.append("")
                    lines.append(f"```")
                    for item in val:
                        lines.append(f"{item}")
                    lines.append("```")
                    lines.append("")
                else:
                    lines.append(f"- **{key}**: `{val}`")
        else:
            lines.append("No solver settings found.")
        lines.append("")

        lines.append("## Boundary Conditions Summary")
        lines.append("")
        if self.bc_summary:
            lines.append("| Patch | Type |")
            lines.append("|-------|------|")
            for patch, data in self.bc_summary.items():
                btype = data.get("type", "unknown")
                lines.append(f"| `{patch}` | `{btype}` |")
        else:
            lines.append("No boundary condition data found.")
        lines.append("")

        lines.append("## Convergence Metrics")
        lines.append("")
        if final_residuals:
            lines.append("| Field | Initial Residual | Final Residual | Iterations | Converged |")
            lines.append("|-------|-----------------|----------------|------------|-----------|")
            for field, data in final_residuals.items():
                converged_str = "Yes" if data.get("converged", False) else "No"
                lines.append(
                    f"| `{field}` | {data.get('initial', 0):.4e} | {data.get('final', 0):.4e} | {data.get('iterations', 0)} | {converged_str} |"
                )
        else:
            lines.append("No convergence data found.")
        lines.append("")

        lines.append("## Residuals Plot Data")
        lines.append("")
        if self.residual_data:
            lines.append("```json")
            for field, data in self.residual_data.items():
                if field == "continuity":
                    continue
                lines.append(f'"{field}": {{')
                lines.append("  \"time\": [" + ", ".join(f"{v:.4f}" for v in data["time"]) + "],")
                lines.append("  \"initial\": [" + ", ".join(f"{v:.6e}" for v in data["initial"]) + "],")
                lines.append("  \"final\": [" + ", ".join(f"{v:.6e}" for v in data["final"]) + "],")
                lines.append("  \"iterations\": [" + ", ".join(str(v) for v in data["iterations"]) + "]")
                lines.append("},")
            lines.append("```")
        else:
            lines.append("No residual data available.")
        lines.append("")

        lines.append("## Warnings")
        lines.append("")
        if self.warnings:
            for w in self.warnings[:50]:
                lines.append(f"- `{w}`")
        else:
            lines.append("No warnings found.")
        lines.append("")

        lines.append("## Errors")
        lines.append("")
        if self.errors:
            for e in self.errors[:50]:
                lines.append(f"- `{e}`")
        else:
            lines.append("No errors found.")
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