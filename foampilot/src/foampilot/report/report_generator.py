import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

logger = logging.getLogger(__name__)


class CFDReportGenerator:
    """Generates technical PDF/HTML reports from CFD post-processing results.

    Aggregates statistics from a :class:`FoamPostProcessing` instance
    and produces LaTeX/PDF reports, Typst documents, and interactive
    HTML pages with embedded Plotly figures.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    output_dir : str or Path, optional
        Root directory for generated reports. Defaults to ``case_path / "report"``.
    title : str, optional
        Report title. Defaults to the case name.
    author : str, optional
        Report author. Defaults to ``"foampilot"``.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        author: str = "foampilot",
    ):
        self.case_path = Path(case_path)
        self.output_dir = Path(output_dir) if output_dir else self.case_path / "report"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title or self.case_path.name
        self.author = author
        self._statistics: Dict[str, Any] = {}
        self._figures: List[Dict[str, str]] = []
        self._tables: List[Dict[str, Any]] = []

    def add_statistic(
        self, name: str, value: Any, unit: str = "", description: str = ""
    ) -> None:
        """Register a scalar statistic for the report.

        Parameters
        ----------
        name : str
            Statistic name (e.g., ``"Re"``).
        value : Any
            Numeric value.
        unit : str, optional
            Physical unit.
        description : str, optional
            Human-readable description.
        """
        self._statistics[name] = {
            "value": value,
            "unit": unit,
            "description": description,
        }

    def add_figure(self, path: str, caption: str, label: str = "") -> None:
        """Register a figure for inclusion in the report.

        Parameters
        ----------
        path : str
            Path to the image file (PNG, SVG, PDF).
        caption : str
            Figure caption.
        label : str, optional
            Cross-reference label.
        """
        self._figures.append(
            {"path": path, "caption": caption, "label": label or f"fig_{len(self._figures)+1}"}
        )

    def add_table(self, data: List[List[Any]], headers: List[str], caption: str = "") -> None:
        """Register a table for inclusion in the report.

        Parameters
        ----------
        data : list of list
            Table rows (first row treated as header if ``headers`` is provided).
        headers : list of str
            Column headers.
        caption : str, optional
            Table caption.
        """
        self._tables.append(
            {"data": data, "headers": headers, "caption": caption}
        )

    def collect_time_series(
        self, postprocessor: "FoamPostProcessing", scalar_field: str
    ) -> pd.DataFrame:
        """Collect a scalar field time series across all time steps.

        Parameters
        ----------
        postprocessor : FoamPostProcessing
            The post-processing instance.
        scalar_field : str
            Name of the scalar field to collect.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``time`` and ``{scalar_field}_mean``,
            ``{scalar_field}_max``, ``{scalar_field}_min``.
        """
        time_steps = postprocessor.get_all_time_steps()
        rows = []
        for step in time_steps:
            structure = postprocessor.load_time_step(step)
            mesh = structure["cell"]
            if scalar_field not in mesh.point_data:
                logger.warning(
                    "Field '%s' not found at time step %d, skipping.", scalar_field, step
                )
                continue
            data = mesh.point_data[scalar_field]
            rows.append(
                {
                    "time": step,
                    f"{scalar_field}_mean": float(np.mean(data)),
                    f"{scalar_field}_max": float(np.max(data)),
                    f"{scalar_field}_min": float(np.min(data)),
                    f"{scalar_field}_std": float(np.std(data)),
                }
            )
        df = pd.DataFrame(rows)
        return df

    def collect_region_statistics(
        self, postprocessor: "FoamPostProcessing", scalar_fields: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Collect statistics per region for given scalar fields.

        Parameters
        ----------
        postprocessor : FoamPostProcessing
            The post-processing instance.
        scalar_fields : list of str
            Scalar field names to collect.

        Returns
        -------
        dict of str → pd.DataFrame
            Keys are region names (``"cell"``, boundary names); values
            are DataFrames with statistics per field.
        """
        time_steps = postprocessor.get_all_time_steps()
        if not time_steps:
            return {}
        structure = postprocessor.load_time_step(time_steps[-1])
        results: Dict[str, pd.DataFrame] = {}
        for region_name in list(structure.keys()):
            if region_name == "boundaries":
                for bname, bmesh in structure["boundaries"].items():
                    if bname not in results:
                        results[bname] = []
                    for field in scalar_fields:
                        if field in bmesh.point_data:
                            data = bmesh.point_data[field]
                            results[bname].append(
                                {
                                    "region": bname,
                                    "field": field,
                                    "mean": float(np.mean(data)),
                                    "max": float(np.max(data)),
                                    "min": float(np.min(data)),
                                    "std": float(np.std(data)),
                                }
                            )
            elif region_name == "cell":
                if "cell" not in results:
                    results["cell"] = []
                for field in scalar_fields:
                    if field in structure["cell"].point_data:
                        data = structure["cell"].point_data[field]
                        results["cell"].append(
                            {
                                "region": "cell",
                                "field": field,
                                "mean": float(np.mean(data)),
                                "max": float(np.max(data)),
                                "min": float(np.min(data)),
                                "std": float(np.std(data)),
                            }
                        )
        return {k: pd.DataFrame(v) for k, v in results.items()}

    def generate_plotly_time_series(
        self,
        df: pd.DataFrame,
        scalar_field: str,
        title: str = "",
    ) -> go.Figure:
        """Create an interactive Plotly time-series figure.

        Parameters
        ----------
        df : pd.DataFrame
            Time series data (as returned by :meth:`collect_time_series`).
        scalar_field : str
            Field name used in the DataFrame columns.
        title : str, optional
            Plot title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                f"{scalar_field} — Mean",
                f"{scalar_field} — Max",
                f"{scalar_field} — Min",
            ),
            vertical_spacing=0.08,
        )
        if df.empty:
            fig.update_layout(title=title or scalar_field)
            return fig

        cols = [c for c in df.columns if c.startswith(scalar_field)]
        labels = {
            f"{scalar_field}_mean": "Mean",
            f"{scalar_field}_max": "Max",
            f"{scalar_field}_min": "Min",
            f"{scalar_field}_std": "Std",
        }
        row_map = {
            f"{scalar_field}_mean": 1,
            f"{scalar_field}_max": 2,
            f"{scalar_field}_min": 3,
            f"{scalar_field}_std": 1,
        }

        for col in cols:
            fig.add_trace(
                go.Scatter(x=df["time"], y=df[col], name=labels.get(col, col)),
                row=row_map.get(col, 1),
                col=1,
            )

        fig.update_layout(
            title=title or scalar_field,
            height=700,
            hovermode="x unified",
        )
        return fig

    def generate_plotly_contour(
        self,
        mesh: pv.DataSet,
        scalars: str,
        title: str = "",
        cmap: str = "viridis",
    ) -> go.Figure:
        """Create an interactive Plotly contour plot from a PyVista mesh.

        Parameters
        ----------
        mesh : pv.DataSet
            The mesh with point or cell data.
        scalars : str
            Name of the scalar array to plot.
        title : str, optional
            Plot title.
        cmap : str, optional
            Plotly color scale name.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if scalars not in mesh.point_data and scalars not in mesh.cell_data:
            raise ValueError(f"Scalar field '{scalars}' not found in mesh data.")

        points = mesh.points
        cells = mesh.cells
        if cells.size == 0:
            raise ValueError("Mesh has no cells — cannot generate contour plot.")

        if scalars in mesh.cell_data:
            values = mesh.cell_data[scalars]
            # Map cell values to cell centroids
            centroids = mesh.cell_centers().points
        else:
            values = mesh.point_data[scalars]
            centroids = points

        fig = go.Figure(
            data=go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=values,
                    colorscale=cmap,
                    colorbar=dict(title=scalars),
                    opacity=0.8,
                ),
                hovertemplate=(
                    f"x: {{x:.4f}}<br>y: {{y:.4f}}<br>z: {{z:.4f}}<br>"
                    f"{scalars}: {{marker.color:.4g}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title or scalars,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
            ),
            height=700,
        )
        return fig

    def generate_plotly_vector_plot(
        self,
        mesh: pv.DataSet,
        vectors: str,
        title: str = "",
        subsample: int = 10,
    ) -> go.Figure:
        """Create an interactive Plotly vector (quiver) plot.

        Parameters
        ----------
        mesh : pv.DataSet
            The mesh with vector data.
        vectors : str
            Name of the vector field.
        title : str, optional
            Plot title.
        subsample : int, optional
            Downsample factor for clarity.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if vectors not in mesh.point_data:
            raise ValueError(f"Vector field '{vectors}' not found in mesh point data.")
        if not isinstance(subsample, int) or subsample < 1:
            raise ValueError("subsample must be a positive integer.")

        vecs = np.asarray(mesh.point_data[vectors])
        points = np.asarray(mesh.points)
        if vecs.ndim == 1:
            if vecs.size % 3 != 0:
                raise ValueError(
                    f"Vector field '{vectors}' must contain 3 components per point."
                )
            vecs = vecs.reshape(-1, 3)
        if vecs.ndim != 2 or vecs.shape[1] != 3:
            raise ValueError(
                f"Vector field '{vectors}' must have shape (n_points, 3)."
            )
        if len(points) != len(vecs):
            raise ValueError(
                f"Vector field '{vectors}' has {len(vecs)} values for "
                f"{len(points)} mesh points."
            )

        xs, ys, zs = points[::subsample].T
        u, v, w = vecs[::subsample].T

        fig = go.Figure(
            data=go.Cone(
                x=xs.flatten(),
                y=ys.flatten(),
                z=zs.flatten(),
                u=u.flatten(),
                v=v.flatten(),
                w=w.flatten(),
                sizemode="absolute",
                sizeref=0.01,
                anchor="tail",
                colorscale="Blues",
            )
        )
        fig.update_layout(
            title=title or vectors,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
            ),
            height=700,
        )
        return fig

    def save_html_report(
        self,
        filename: Union[str, Path] = "cfd_report.html",
        include_time_series: bool = True,
    ) -> Path:
        """Generate a self-contained interactive HTML report.

        Parameters
        ----------
        filename : str or Path, optional
            Output HTML file name. Defaults to ``"cfd_report.html"``.
        include_time_series : bool, optional
            Whether to include time-series plots.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        html_parts: List[str] = []

        html_parts.append("<!DOCTYPE html>")
        html_parts.append('<html lang="en">')
        html_parts.append("<head>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append(f"<title>{self.title}</title>")
        html_parts.append(
            "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>"
        )
        html_parts.append(
            "<style>"
            "body { font-family: sans-serif; margin: 2em; max-width: 1200px; }"
            "h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }"
            "h2 { margin-top: 1.5em; }"
            "table { border-collapse: collapse; margin: 1em 0; }"
            "th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }"
            "th { background: #f0f0f0; }"
            ".figure-container { margin: 2em 0; }"
            "</style>"
        )
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append(f"<h1>{self.title}</h1>")
        html_parts.append(f"<p><strong>Author:</strong> {self.author}</p>")
        html_parts.append(f"<p><strong>Case:</strong> {self.case_path}</p>")

        # Statistics summary
        if self._statistics:
            html_parts.append("<h2>Summary Statistics</h2>")
            html_parts.append("<table><tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Description</th></tr>")
            for name, stat in self._statistics.items():
                html_parts.append(
                    f"<tr><td>{name}</td><td>{stat['value']}</td>"
                    f"<td>{stat['unit']}</td><td>{stat['description']}</td></tr>"
                )
            html_parts.append("</table>")

        # Figures
        if self._figures:
            html_parts.append("<h2>Figures</h2>")
            for fig in self._figures:
                html_parts.append(f'<div class="figure-container">')
                html_parts.append(f'<h3>{fig["caption"]}</h3>')
                html_parts.append(f'<img src="{fig["path"]}" alt="{fig["caption"]}" style="max-width:100%;">')
                html_parts.append(f'</div>')

        # Tables
        if self._tables:
            html_parts.append("<h2>Tables</h2>")
            for tbl in self._tables:
                html_parts.append(f'<h3>{tbl["caption"]}</h3>')
                html_parts.append("<table>")
                html_parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in tbl["headers"]) + "</tr>")
                for row in tbl["data"]:
                    html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
                html_parts.append("</table>")

        html_parts.append("</body>")
        html_parts.append("</html>")

        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info("HTML report saved to %s", out_path)
        return out_path

    def save_latex_report(
        self,
        filename: str = "cfd_report.tex",
        compile_pdf: bool = False,
    ) -> Path:
        """Generate a LaTeX source file for a PDF report.

        Parameters
        ----------
        filename : str, optional
            Output ``.tex`` file name. Defaults to ``"cfd_report.tex"``.
        compile_pdf : bool, optional
            If ``True``, compile to PDF using PyLaTeX (requires a working LaTeX
            installation).

        Returns
        -------
        Path
            Path to the generated ``.tex`` file (and PDF if ``compile_pdf=True``).
        """
        output_name = Path(filename)
        if output_name.suffix == ".tex":
            output_name = output_name.with_suffix("")
        doc = LatexDocument(
            title=self.title,
            author=self.author,
            filename=str(output_name),
            output_dir=str(self.output_dir.parent),
        )
        doc.add_abstract(
            f"CFD post-processing report for case: {self.case_path}"
        )

        if self._statistics:
            doc.add_section("Summary Statistics", "")
            rows = [[str(s["value"]), s["unit"], s["description"]] for s in self._statistics.values()]
            doc.add_dataframe_table(
                pd.DataFrame(
                    rows,
                    columns=["Value", "Unit", "Description"],
                ),
                caption="Summary of computed statistics",
            )

        if self._tables:
            doc.add_section("Tables", "")
            for tbl in self._tables:
                doc.add_dataframe_table(
                    pd.DataFrame(tbl["data"], columns=tbl["headers"]),
                    caption=tbl["caption"],
                )

        if self._figures:
            doc.add_section("Figures", "")
            for fig in self._figures:
                doc.add_figure(fig["path"], fig["caption"], label=fig["label"])

        tex_path = doc.generate_tex()
        if compile_pdf:
            doc.generate_pdf()

        return tex_path

    def save_typst_report(
        self,
        filename: str = "cfd_report.typ",
    ) -> Path:
        """Generate a Typst source file for a PDF report.

        Parameters
        ----------
        filename : str, optional
            Output ``.typ`` file name. Defaults to ``"cfd_report.typ"``.

        Returns
        -------
        Path
            Path to the generated ``.typ`` file.
        """
        if not self._statistics and not self._figures and not self._tables:
            logger.warning("No data has been added to the report yet.")
            return self.output_dir / filename

        doc = ScientificDocument(
            title=self.title,
            author=self.author,
        )

        if self._statistics:
            doc.add_section("Summary Statistics", level=1)
            table_data = [["Parameter", "Value", "Unit", "Description"]]
            for name, stat in self._statistics.items():
                table_data.append([
                    name,
                    str(stat["value"]),
                    stat["unit"],
                    stat["description"],
                ])
            doc.add_table(table_data, caption="Summary statistics", label="tab:statistics")

        for index, tbl in enumerate(self._tables, start=1):
            doc.add_table(
                tbl["data"],
                headers=tbl["headers"],
                caption=tbl["caption"],
                label=f"tab_{index}",
            )

        for fig in self._figures:
            doc.add_figure(fig["path"], fig["caption"], label=fig["label"])

        renderer = TypstRenderer()
        typst_content = renderer.render(doc)
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(typst_content, encoding="utf-8")

        return out_path