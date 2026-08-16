import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')  # Force headless backend before importing pyplot
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ResidualsPost:
    """
    Post-processing tool for OpenFOAM residuals.

    Extracts solver residuals from an OpenFOAM log file
    and exports them in multiple formats:
    - CSV
    - JSON
    - PNG (Matplotlib)
    - HTML (Plotly)
    """

    def __init__(self, log_file: str | Path):
        self.log_file = Path(log_file)
        self.patterns = {
            "Ux": r"Solving for Ux",
            "Uy": r"Solving for Uy",
            "Uz": r"Solving for Uz",
            "p": r"Solving for p",
            "k": r"Solving for k",
            "omega": r"Solving for omega",
            "epsilon": r"Solving for epsilon",
        }
        self.residuals = {var: {"time": [], "initial": [], "final": []} for var in self.patterns}
        self.df: pd.DataFrame | None = None

        # Centralize output dir and base filename
        self.output_dir = self.log_file.parent / "residuals"
        self.output_dir.mkdir(exist_ok=True)
        self.base_name = f"{self.log_file.stem}_residuals"

    # -------------------------
    # Internal helpers
    # -------------------------

    def _check_data(self) -> bool:
        """Return True if residuals DataFrame exists and is not empty."""
        return self.df is not None and not self.df.empty

    def _save_file(self, suffix: str) -> Path:
        """Return the output file path with given suffix (e.g. '.csv')."""
        return self.output_dir / f"{self.base_name}{suffix}"

    # -------------------------
    # Core parsing
    # -------------------------

    def extract_residuals(self) -> None:
        """Parse the log file and build DataFrame of residuals."""
        if not self.log_file.exists():
            logging.error("Log file %s does not exist.", self.log_file)
            return

        time_pattern = re.compile(r"Time = (\d+\.?\d*)")
        solver_pattern = re.compile(
            r"Solving for (\w+),.*?Initial residual = ([\d\.Ee\+\-]+), Final residual = ([\d\.Ee\+\-]+)"
        )

        current_time = None
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if (t_match := time_pattern.search(line)):
                    current_time = float(t_match.group(1))

                if current_time is not None and (s_match := solver_pattern.search(line)):
                    field, init_res, final_res = s_match.groups()
                    if field in self.residuals:
                        self.residuals[field]["time"].append(current_time)
                        self.residuals[field]["initial"].append(float(init_res))
                        self.residuals[field]["final"].append(float(final_res))

        # Build DataFrame
        records = [
            {"time": t, "field": field, "initial": init, "final": final}
            for field, data in self.residuals.items()
            for t, init, final in zip(data["time"], data["initial"], data["final"])
        ]
        if records:
            self.df = pd.DataFrame(records)
            logging.info("Extracted residuals for %d fields.", self.df["field"].nunique())
        else:
            logging.warning("No residuals found in log file.")

    # -------------------------
    # Exporters
    # -------------------------

    def export_csv(self) -> None:
        if not self._check_data():
            logging.warning("No data to export as CSV.")
            return
        output = self._save_file(".csv")
        self.df.to_csv(output, index=False)
        logging.info("CSV exported: %s", output)

    def export_json(self) -> None:
        if not self._check_data():
            logging.warning("No data to export as JSON.")
            return
        output = self._save_file(".json")
        self.df.to_json(output, orient="records", indent=2)
        logging.info("JSON exported: %s", output)

    def export_matplotlib_png(self) -> None:
        if not self._check_data():
            logging.warning("No data to export as PNG.")
            return

        plt.figure(figsize=(10, 6))
        for field, sub in self.df.groupby("field"):
            plt.semilogy(sub["time"], sub["initial"], label=f"{field} initial")
            plt.semilogy(sub["time"], sub["final"], "--", label=f"{field} final")

        plt.xlabel("Time (s)")
        plt.ylabel("Residuals")
        plt.title(f"Residuals - {self.log_file.stem}")
        plt.legend()
        plt.grid(True, which="both")

        output = self._save_file(".png")
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("PNG exported: %s", output)

    def export_plotly_html(self, fig: go.Figure | None) -> None:
        if fig is None:
            logging.warning("No Plotly figure to export.")
            return
        output = self._save_file(".html")
        fig.write_html(output)
        logging.info("HTML exported: %s", output)

    # -------------------------
    # Plot preparation
    # -------------------------

    def prepare_plotly_figure(self) -> go.Figure | None:
        if not self._check_data():
            logging.warning("No data to plot.")
            return None

        colors = px.colors.qualitative.Set1
        color_map = {f: colors[i % len(colors)] for i, f in enumerate(self.df["field"].unique())}

        traces = []
        for field, sub in self.df.groupby("field"):
            traces.append(go.Scatter(
                x=sub["time"], y=sub["initial"], mode="lines",
                name=f"{field} initial",
                line=dict(color=color_map[field], dash="solid"),
            ))
            traces.append(go.Scatter(
                x=sub["time"], y=sub["final"], mode="lines",
                name=f"{field} final",
                line=dict(color=color_map[field], dash="dash"),
            ))

        layout = go.Layout(
            title=f"{self.log_file.name} - OpenFOAM Residuals",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Residuals (log scale)", type="log"),
        )
        fig = go.Figure(data=traces, layout=layout)
        logging.info("Plotly figure prepared.")
        return fig

    # -------------------------
    # Full pipeline
    # -------------------------

    def process(self, export_json=True, export_csv=True, export_png=True, export_html=True) -> None:
        """Run the full residuals processing pipeline."""
        self.extract_residuals()
        fig = self.prepare_plotly_figure()

        if export_csv:
            self.export_csv()
        if export_json:
            self.export_json()
        if export_png:
            self.export_matplotlib_png()
        if export_html:
            self.export_plotly_html(fig)

        logging.info("Residuals processing completed.")


# ---------------------------------------------------------------------------
# Convergence monitoring
# ---------------------------------------------------------------------------

class ConvergenceState(str, Enum):
    STABLE = "stable"
    SLOW_CONVERGENCE = "slow_convergence"
    OSCILLATING = "oscillating"
    DIVERGING = "diverging"
    PHYSICAL_BLOWUP = "physical_blowup"
    UNKNOWN = "unknown"


@dataclass
class IterationData:
    time: float
    residuals: Dict[str, float] = field(default_factory=dict)
    continuity: float = 0.0
    max_u: float = 0.0
    max_k: float = 0.0
    max_epsilon: float = 0.0
    cd: float = 0.0
    cl: float = 0.0
    mean_u: float = 0.0


@dataclass
class RelaxationPhase:
    name: str
    p: float
    u: float
    k: float
    epsilon: float
    n_non_orthogonal_correctors: int = 2
    relaxation_factor: float = 1.0


RELAXATION_PHASES = {
    1: RelaxationPhase("phase1", p=0.2, u=0.5, k=0.5, epsilon=0.5),
    2: RelaxationPhase("phase2", p=0.3, u=0.7, k=0.7, epsilon=0.7),
    3: RelaxationPhase("phase3", p=0.5, u=0.9, k=0.9, epsilon=0.9),
}


class ConvergenceMonitor:
    """Monitor OpenFOAM simpleFoam convergence in real-time."""

    def __init__(
        self,
        case_path: str | Path,
        log_name: str = "log.incompressibleFluid",
        window_size: int = 20,
        convergence_tol: float = 1e-4,
        slope_threshold: float = -0.1,
        oscillation_threshold: float = 0.3,
        continuity_threshold: float = 1e-3,
        blowup_multiplier: float = 2.0,
    ):
        self.case_path = Path(case_path)
        self.log_path = self.case_path / log_name
        self.window_size = window_size
        self.convergence_tol = convergence_tol
        self.slope_threshold = slope_threshold
        self.oscillation_threshold = oscillation_threshold
        self.continuity_threshold = continuity_threshold
        self.blowup_multiplier = blowup_multiplier

        self.history: List[IterationData] = []
        self.current_state = ConvergenceState.UNKNOWN
        self.current_phase = 1
        self.recovery_actions: List[str] = []

        self._file_pos: int = 0
        self._last_time: Optional[float] = None
        self._last_residuals: Dict[str, float] = {}

        self._patterns = {
            "time": re.compile(r"Time\s*=\s*([\d\.]+)s?"),
            "solver": re.compile(
                r"(?:smoothSolver|GAMG|PCG|PBiCGStab|geometricOneWire):\s*"
                r"Solving\s+for\s+(\w+),\s*"
                r"Initial\s+residual\s*=\s*([\d\.Ee\+\-]+),\s*"
                r"Final\s+residual\s*=\s*([\d\.Ee\+\-]+),\s*"
                r"No\s+Iterations\s+(\d+)"
            ),
            "continuity": re.compile(
                r"time\s+step\s+continuity\s+errors\s*:\s*"
                r"sum\s+local\s*=\s*([\d\.Ee\+\-]+),\s*"
                r"global\s*=\s*([\d\.Ee\+\-]+),\s*"
                r"cumulative\s*=\s*([\d\.Ee\+\-]+)"
            ),
            "execution": re.compile(
                r"ExecutionTime\s*=\s*([\d\.]+)\s+s\s+ClockTime\s*=\s*(\d+)\s+s"
            ),
            "force_coeffs": re.compile(
                r"Coefficients\s*:\s*Cd\s*=\s*([\d\.Ee\+\-]+)\s+Cl\s*=\s*([\d\.Ee\+\-]+)"
            ),
            "max_field": re.compile(
                r"Max\s+([Uk]|epsilon)\s*=\s*([\d\.Ee\+\-]+)"
            ),
            "mean_velocity": re.compile(
                r"mean\s+velocity\s+\(U\)\s*=\s*([\d\.Ee\+\-]+)"
            ),
        }

    def _read_new_lines(self) -> List[str]:
        if not self.log_path.exists():
            return []
        lines: List[str] = []
        try:
            with self.log_path.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(self._file_pos)
                for line in f:
                    lines.append(line.rstrip("\n"))
                self._file_pos = f.tell()
        except OSError:
            pass
        return lines

    def _parse_residuals(self, lines: List[str]) -> IterationData:
        data = IterationData(time=self._last_time or 0.0)
        current_time = self._last_time
        residuals: Dict[str, float] = {}

        for line in lines:
            t_match = self._patterns["time"].search(line)
            if t_match:
                current_time = float(t_match.group(1))
                data.time = current_time
                self._last_time = current_time

            solver_match = self._patterns["solver"].search(line)
            if solver_match:
                field = solver_match.group(1)
                final_res = float(solver_match.group(3))
                residuals[field] = final_res
                self._last_residuals[field] = final_res

            cont_match = self._patterns["continuity"].search(line)
            if cont_match:
                data.continuity = float(cont_match.group(2))

            force_match = self._patterns["force_coeffs"].search(line)
            if force_match:
                data.cd = float(force_match.group(1))
                data.cl = float(force_match.group(2))

            max_match = self._patterns["max_field"].search(line)
            if max_match:
                field_name = max_match.group(1)
                val = float(max_match.group(2))
                if field_name == "U":
                    data.max_u = val
                elif field_name == "k":
                    data.max_k = val
                elif field_name == "epsilon":
                    data.max_epsilon = val

            mean_match = self._patterns["mean_velocity"].search(line)
            if mean_match:
                data.mean_u = float(mean_match.group(1))

        data.residuals = dict(self._last_residuals)
        if current_time is not None:
            data.time = current_time
        return data

    def update(self) -> IterationData:
        lines = self._read_new_lines()
        if not lines:
            return IterationData(time=self._last_time or 0.0)

        data = self._parse_residuals(lines)
        if data.time > 0 or data.residuals:
            self.history.append(data)
            self._trim_history()
            self._update_state()
        return data

    def _trim_history(self) -> None:
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size :]

    def _compute_residual_slope(self, field: str) -> Optional[float]:
        values = [d.residuals.get(field, np.nan) for d in self.history]
        values = [v for v in values if not np.isnan(v)]
        if len(values) < 2:
            return None
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, np.log10(values), 1)
        return float(slope)

    def _compute_oscillation_index(self, field: str) -> Optional[float]:
        values = [d.residuals.get(field, np.nan) for d in self.history]
        values = [v for v in values if not np.isnan(v)]
        if len(values) < 3:
            return None
        diffs = np.diff(np.log10(values))
        return float(np.std(diffs))

    def _compute_force_delta(self, field: str) -> Optional[float]:
        values = [getattr(d, field) for d in self.history]
        if len(values) < 2:
            return None
        recent = values[-int(self.window_size / 2) :]
        if len(recent) < 2:
            return None
        return float(abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-12))

    def _detect_state(self) -> ConvergenceState:
        if not self.history:
            return ConvergenceState.UNKNOWN

        latest = self.history[-1]

        for field in ["U", "p", "k", "epsilon"]:
            slope = self._compute_residual_slope(field)
            oscillation = self._compute_oscillation_index(field)
            if slope is not None and slope > 0:
                return ConvergenceState.DIVERGING
            if oscillation is not None and oscillation > self.oscillation_threshold:
                return ConvergenceState.OSCILLATING

        if latest.continuity > self.continuity_threshold:
            return ConvergenceState.DIVERGING

        for field in ["U", "p", "k", "epsilon"]:
            if latest.residuals.get(field, 0.0) > 1.0:
                return ConvergenceState.PHYSICAL_BLOWUP

        all_converged = all(
            v < self.convergence_tol for v in latest.residuals.values()
        )
        if all_converged:
            return ConvergenceState.STABLE

        avg_slope = np.mean(
            [s for f in ["U", "p", "k", "epsilon"] if (s := self._compute_residual_slope(f)) is not None]
        )
        if avg_slope is not None and avg_slope > self.slope_threshold:
            return ConvergenceState.SLOW_CONVERGENCE

        return ConvergenceState.UNKNOWN

    def _update_state(self) -> None:
        self.current_state = self._detect_state()
        self._check_multi_criteria_convergence()
        self._update_phase()

    def _check_multi_criteria_convergence(self) -> None:
        if len(self.history) < self.window_size:
            return

        convergences = []

        for field in ["U", "p", "k", "epsilon"]:
            slope = self._compute_residual_slope(field)
            if slope is not None and slope < self.slope_threshold:
                convergences.append(True)
            else:
                convergences.append(False)

        latest = self.history[-1]
        if abs(latest.continuity) < self.continuity_threshold:
            convergences.append(True)
        else:
            convergences.append(False)

        for force_field in ["cd", "cl", "mean_u"]:
            delta = self._compute_force_delta(force_field)
            if delta is not None and delta < 0.005:
                convergences.append(True)
            else:
                convergences.append(False)

        if all(convergences):
            logger.info("Multi-criteria convergence achieved.")

    def _update_phase(self) -> None:
        if self.current_state == ConvergenceState.STABLE:
            return
        if self.current_state in (
            ConvergenceState.DIVERGING,
            ConvergenceState.PHYSICAL_BLOWUP,
        ):
            self.current_phase = max(1, self.current_phase - 1)
        elif self.current_state == ConvergenceState.SLOW_CONVERGENCE:
            self.current_phase = min(3, self.current_phase + 1)

    def get_recovery_actions(self) -> List[str]:
        actions: List[str] = []
        phase = RELAXATION_PHASES.get(self.current_phase, RELAXATION_PHASES[1])

        if self.current_state == ConvergenceState.OSCILLATING:
            actions.append("switch_convection_scheme")
            actions.append("increase_n_non_orthogonal_correctors")
            actions.append(f"apply_relaxation_{phase.name}")

        elif self.current_state == ConvergenceState.DIVERGING:
            actions.append("reduce_relaxation")
            actions.append("increase_n_non_orthogonal_correctors")
            actions.append("restart")

        elif self.current_state == ConvergenceState.PHYSICAL_BLOWUP:
            actions.append("reduce_relaxation")
            actions.append("restart")
            actions.append("check_bc")

        elif self.current_state == ConvergenceState.SLOW_CONVERGENCE:
            if self.current_phase < 3:
                actions.append(f"apply_relaxation_{RELAXATION_PHASES[self.current_phase + 1].name}")
            else:
                actions.append("switch_solver_preconditioner")

        self.recovery_actions = actions
        return actions

    def apply_recovery(self, action: str) -> bool:
        logger.info("Applying recovery action: %s", action)

        if action.startswith("apply_relaxation_"):
            phase_name = action.replace("apply_relaxation_", "")
            phase = next((p for p in RELAXATION_PHASES.values() if p.name == phase_name), None)
            if phase:
                self._set_relaxation(phase)
                return True

        if action == "reduce_relaxation":
            self._set_relaxation(RELAXATION_PHASES[1])
            return True

        if action == "increase_n_non_orthogonal_correctors":
            self._set_n_non_orthogonal_correctors(5)
            return True

        if action == "switch_convection_scheme":
            self._switch_convection_scheme()
            return True

        if action == "restart":
            self._restart_simulation()
            return True

        if action == "switch_solver_preconditioner":
            self._switch_solver_preconditioner()
            return True

        if action == "check_bc":
            logger.warning("Manual check required for boundary conditions.")
            return True

        return False

    def _set_relaxation(self, phase: RelaxationPhase) -> None:
        fv_solution = self.case_path / "system" / "fvSolution"
        if not fv_solution.exists():
            logger.warning("fvSolution not found; cannot set relaxation.")
            return

        content = fv_solution.read_text()
        content = re.sub(
            r"(relaxationFactors\s*\{\s*\n\s*p\s+)[\d\.]+", r"\g<1>{:.1f}".format(phase.p), content
        )
        content = re.sub(
            r"(\s+U\s+)[\d\.]+", lambda m: m.group(0).replace(m.group(1) + re.search(r"[\d\.]+", m.group(0)).group(0), m.group(1) + "{:.1f}".format(phase.u)), content
        )
        fv_solution.write_text(content)
        logger.info("Relaxation set to phase %s: p=%.1f, U=%.1f", phase.name, phase.p, phase.u)

    def _set_n_non_orthogonal_correctors(self, value: int) -> None:
        fv_solution = self.case_path / "system" / "fvSolution"
        if not fv_solution.exists():
            return
        content = fv_solution.read_text()
        content = re.sub(
            r"(nNonOrthogonalCorrectors\s+)\d+",
            r"\g<1>{}".format(value),
            content,
        )
        fv_solution.write_text(content)
        logger.info("nNonOrthogonalCorrectors set to %d.", value)

    def _switch_convection_scheme(self) -> None:
        fv_schemes = self.case_path / "system" / "fvSchemes"
        if not fv_schemes.exists():
            return
        content = fv_schemes.read_text()
        content = re.sub(
            r"(div\(phi,U\)\s+)(limitedCubic|upwind|linearUpwind)",
            r"\g<1>limitedLinear",
            content,
        )
        fv_schemes.write_text(content)
        logger.info("Convection scheme switched to limitedLinear.")

    def _switch_solver_preconditioner(self) -> None:
        fv_solution = self.case_path / "system" / "fvSolution"
        if not fv_solution.exists():
            return
        content = fv_solution.read_text()
        content = re.sub(
            r"(solver\s+)(GAMG|PCG|smoothSolver)",
            lambda m: m.group(1) + "GAMG" if m.group(2) != "GAMG" else m.group(1) + "PCG",
            content,
        )
        fv_solution.write_text(content)
        logger.info("Solver preconditioner switched.")

    def _restart_simulation(self) -> None:
        logger.warning("Restart requested. Restart logic must be implemented externally.")
        self.history.clear()
        self._last_time = None
        self._last_residuals.clear()
        self._file_pos = 0

    def get_phase(self) -> RelaxationPhase:
        return RELAXATION_PHASES.get(self.current_phase, RELAXATION_PHASES[1])

    def get_summary(self) -> Dict[str, Any]:
        latest = self.history[-1] if self.history else IterationData(time=0.0)
        return {
            "state": self.current_state.value,
            "phase": self.current_phase,
            "phase_name": self.get_phase().name,
            "iteration": len(self.history),
            "time": latest.time,
            "residuals": latest.residuals,
            "continuity": latest.continuity,
            "max_u": latest.max_u,
            "max_k": latest.max_k,
            "max_epsilon": latest.max_epsilon,
            "cd": latest.cd,
            "cl": latest.cl,
            "mean_u": latest.mean_u,
            "recovery_actions": self.recovery_actions,
        }

    def monitor_loop(
        self,
        interval: float = 1.0,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        iteration = 0
        while True:
            data = self.update()
            summary = self.get_summary()
            if callback:
                callback(summary)
            else:
                logger.info("State=%s Phase=%d Residuals=%s", summary["state"], summary["phase"], summary["residuals"])

            if self.current_state == ConvergenceState.STABLE:
                logger.info("Simulation converged.")
                break

            if max_iterations is not None and iteration >= max_iterations:
                logger.info("Max iterations reached.")
                break

            if data.time == 0.0 and not data.residuals:
                if not self.log_path.exists():
                    logger.warning("Log file not found: %s", self.log_path)

            time.sleep(interval)
            iteration += 1


def main():
    parser = argparse.ArgumentParser(description="OpenFOAM convergence monitor")
    parser.add_argument("--case", type=str, required=True, help="Path to OpenFOAM case")
    parser.add_argument("--log", type=str, default="log.incompressibleFluid", help="Log filename")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum iterations")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    monitor = ConvergenceMonitor(case_path=args.case, log_name=args.log)
    results: List[Dict[str, Any]] = []

    def save_callback(summary: Dict[str, Any]) -> None:
        results.append(summary)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))

    monitor.monitor_loop(
        interval=args.interval,
        callback=save_callback,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
