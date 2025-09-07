import re
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path
import json
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
matplotlib.use('Agg')  # Force headless backend before importing pyplot
import matplotlib.pyplot as plt

class ResidualsPost:
    """
    Post-processing tool for OpenFOAM residuals.

    This class extracts solver residuals from an OpenFOAM log file
    and exports them in multiple formats for visualization and analysis:
    - CSV
    - JSON
    - PNG (Matplotlib)
    - HTML (Plotly)

    Attributes
    ----------
    log_file : Path
        Path to the OpenFOAM log file.
    patterns : dict
        Dictionary of regex patterns used to detect solver fields.
    residuals : dict
        Parsed residuals organized by field, containing:
        - time
        - initial residuals
        - final residuals
    df : pandas.DataFrame or None
        DataFrame storing extracted residuals (columns: time, field, initial, final).
    output_dir : Path
        Output directory where CSV, JSON, PNG, and HTML files will be stored.
    """
    def __init__(self, log_file: str):
        """
        Initialize residuals post-processor.

        Parameters
        ----------
        log_file : str
            Path to the OpenFOAM log file to parse.
        """
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
        self.df = None
        self.output_dir = self.log_file.parent / "residuals"
        self.output_dir.mkdir(exist_ok=True)

    def check_log_exists(self) -> bool:
        """
        Check if the log file exists.

        Returns
        -------
        bool
            True if the log file exists, False otherwise.
        """
        return self.log_file.exists()

    def extract_residuals(self):
        """
        Parse the OpenFOAM log file and extract residuals.

        The method updates the `residuals` dictionary and builds
        a pandas DataFrame (`self.df`) containing the parsed data.

        Notes
        -----
        - Only fields defined in `self.patterns` are extracted.
        - Both initial and final residuals are stored.
        """
        if not self.check_log_exists():
            print(f"[ERROR] Log file {self.log_file} does not exist.")
            return

        time_pattern = re.compile(r"Time = (\d+\.?\d*)")
        solver_pattern = re.compile(
            r"Solving for (\w+), Initial residual = ([\d\.Ee\+\-]+), Final residual = ([\d\.Ee\+\-]+)"
        )

        current_time = None
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                t_match = time_pattern.search(line)
                if t_match:
                    current_time = float(t_match.group(1))

                s_match = solver_pattern.search(line)
                if s_match and current_time is not None:
                    field, init_res, final_res = s_match.groups()
                    if field in self.residuals:
                        self.residuals[field]["time"].append(current_time)
                        self.residuals[field]["initial"].append(float(init_res))
                        self.residuals[field]["final"].append(float(final_res))

        records = []
        for field, data in self.residuals.items():
            for t, init, final in zip(data["time"], data["initial"], data["final"]):
                records.append({"time": t, "field": field, "initial": init, "final": final})
        if records:
            self.df = pd.DataFrame(records)
            print(f"[INFO] Extracted residuals for {len(self.df['field'].unique())} fields.")

    def prepare_plotly_figure(self):
        """
        Prepare a Plotly figure of residuals.

        Returns
        -------
        plotly.graph_objs.Figure or None
            Interactive Plotly figure with log-scale residuals over time,
            or None if no data is available.
        """
        if self.df is None or self.df.empty:
            print("[WARNING] No data to plot.")
            return None

        colors = px.colors.qualitative.Set1
        color_map = {field: colors[i % len(colors)] for i, field in enumerate(self.df["field"].unique())}

        traces = []
        for field in self.df["field"].unique():
            sub = self.df[self.df["field"] == field]
            traces.append(go.Scatter(
                x=sub["time"], y=sub["initial"], mode="lines",
                name=f"{field} initial",
                line=dict(color=color_map[field], dash="solid")
            ))
            traces.append(go.Scatter(
                x=sub["time"], y=sub["final"], mode="lines",
                name=f"{field} final",
                line=dict(color=color_map[field], dash="dash")
            ))

        layout = go.Layout(
            title=f"{self.log_file.name} - OpenFOAM Residuals",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Residuals (log scale)", type="log"),
        )
        fig = go.Figure(data=traces, layout=layout)
        print("[INFO] Plotly figure prepared.")
        return fig

    def export_csv(self):
        """
        Export residuals as a CSV file.

        The file is stored in `output_dir` with suffix `_residuals.csv`.
        """
        if self.df is None or self.df.empty:
            print("[WARNING] No data to export as CSV.")
            return
        output_file = self.output_dir / f"{self.log_file.stem}_residuals.csv"
        self.df.to_csv(output_file, index=False)
        print(f"[SUCCESS] CSV exported: {output_file}")

    def export_json(self):
        """
        Export residuals as a JSON file.

        The file is stored in `output_dir` with suffix `_residuals.json`.
        """
        if self.df is None or self.df.empty:
            print("[WARNING] No data to export as JSON.")
            return
        output_file = self.output_dir / f"{self.log_file.stem}_residuals.json"
        self.df.to_json(output_file, orient="records", indent=2)
        print(f"[SUCCESS] JSON exported: {output_file}")

    def export_matplotlib_png(self):
        """
        Export residuals as a PNG image using Matplotlib.

        The file is stored in `output_dir` with suffix `_residuals.png`.
        """
        if self.df is None or self.df.empty:
            print("[WARNING] No data to export as PNG.")
            return

        plt.figure(figsize=(10, 6))
        for field in self.df["field"].unique():
            sub = self.df[self.df["field"] == field]
            plt.semilogy(sub["time"], sub["initial"], label=f"{field} initial")
            plt.semilogy(sub["time"], sub["final"], "--", label=f"{field} final")

        plt.xlabel("Time (s)")
        plt.ylabel("Residuals")
        plt.title(f"Residuals - {self.log_file.stem}")
        plt.legend()
        plt.grid(True, which="both")

        output_file = self.output_dir / f"{self.log_file.stem}_residuals.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] PNG exported: {output_file}")

    def export_plotly_html(self, fig):
        """
        Export residuals as an interactive Plotly HTML file.

        Parameters
        ----------
        fig : plotly.graph_objs.Figure
            The prepared Plotly figure to save.

        Notes
        -----
        The file is stored in `output_dir` with suffix `_residuals.html`.
        """
        if fig is None:
            return
        output_file = self.output_dir / f"{self.log_file.stem}_residuals.html"
        fig.write_html(output_file)
        print(f"[SUCCESS] HTML exported: {output_file}")

    def process(self, export_json=True, export_csv=True, export_png=True, export_html=True):
        """
        Run the full residuals processing pipeline.

        This method extracts residuals, prepares a Plotly figure,
        and exports the results in multiple formats.

        Parameters
        ----------
        export_json : bool, optional
            Whether to export residuals as JSON (default is True).
        export_csv : bool, optional
            Whether to export residuals as CSV (default is True).
        export_png : bool, optional
            Whether to export residuals as PNG (default is True).
        export_html : bool, optional
            Whether to export residuals as HTML (default is True).

        Notes
        -----
        - If the log file does not exist, no processing is performed.
        - If no residuals are found, no files are exported.
        """
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
        print("[INFO] Residuals processing completed.")
