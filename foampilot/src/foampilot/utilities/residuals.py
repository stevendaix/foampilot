import re
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')  # Force headless backend before importing pyplot
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
            r"Solving for (\w+), Initial residual = ([\d\.Ee\+\-]+), Final residual = ([\d\.Ee\+\-]+)"
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