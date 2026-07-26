import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv

logger = logging.getLogger(__name__)


def plotly_contour_from_mesh(
    mesh: pv.DataSet,
    scalars: str,
    title: str = "",
    cmap: str = "viridis",
    resolution: int = 50,
) -> go.Figure:
    """Generate an interactive Plotly contour from a PyVista mesh by
    sampling onto a regular grid.

    Parameters
    ----------
    mesh : pv.DataSet
        The source mesh.
    scalars : str
        Scalar field name.
    title : str, optional
        Plot title.
    cmap : str, optional
        Plotly colorscale name.
    resolution : int, optional
        Grid resolution for sampling.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bounds = np.array(mesh.bounds)
    x_center = bounds[0:3].mean()
    y_center = bounds[3:6].mean()
    z_center = bounds[6:9].mean()
    x_extent = (bounds[1] - bounds[0]) * 0.6
    y_extent = (bounds[4] - bounds[3]) * 0.6

    x = np.linspace(bounds[0] - x_extent, bounds[1] + x_extent, resolution)
    y = np.linspace(bounds[3] - y_extent, bounds[4] + y_extent, resolution)
    xx, yy = np.meshgrid(x, y)

    grid = pv.ImageData(dimensions=(resolution, resolution, 1),
                        spacing=(x[1] - x[0], y[1] - y[0], 1),
                        origin=(x[0], y[0], z_center))
    grid = grid.sample(mesh)
    scalars_grid = grid.point_data.get(scalars)
    if scalars_grid is None:
        raise ValueError(f"Scalar field '{scalars}' not available after sampling.")

    z_vals = np.full((resolution, resolution), z_center)

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=scalars_grid.reshape(resolution, resolution),
            colorscale=cmap,
            colorbar=dict(title=scalars),
            hovertemplate=(
                f"x: {{x:.4f}}<br>y: {{y:.4f}}<br>"
                f"{scalars}: {{z:.4g}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title or scalars,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=600,
    )
    return fig


def plotly_velocity_magnitude(
    mesh: pv.DataSet,
    velocity_field: str = "U",
    title: str = "Velocity Magnitude",
) -> go.Figure:
    """Plot velocity magnitude as a contour heatmap.

    Parameters
    ----------
    mesh : pv.DataSet
        The mesh.
    velocity_field : str, optional
        Velocity field name (default: ``"U"``).
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if velocity_field not in mesh.point_data:
        raise ValueError(f"Velocity field '{velocity_field}' not found.")

    vel = mesh.point_data[velocity_field]
    if vel.ndim == 1:
        u_mag = np.abs(vel)
    else:
        u_mag = np.linalg.norm(vel, axis=1)

    mesh.point_data["_u_mag_tmp_"] = u_mag
    fig = plotly_contour_from_mesh(mesh, "_u_mag_tmp_", title=title)
    mesh.point_data.remove("_u_mag_tmp_")
    return fig


def plotly_temperature_contour(
    mesh: pv.DataSet,
    temperature_field: str = "T",
    title: str = "Temperature Field",
) -> go.Figure:
    """Plot temperature as a contour heatmap.

    Parameters
    ----------
    mesh : pv.DataSet
        The mesh.
    temperature_field : str, optional
        Temperature field name (default: ``"T"``).
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return plotly_contour_from_mesh(
        mesh, temperature_field, title=title, cmap="thermal"
    )


def plotly_pressure_contour(
    mesh: pv.DataSet,
    pressure_field: str = "p",
    title: str = "Pressure Field",
) -> go.Figure:
    """Plot pressure as a contour heatmap.

    Parameters
    ----------
    mesh : pv.DataSet
        The mesh.
    pressure_field : str, optional
        Pressure field name (default: ``"p"``).
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return plotly_contour_from_mesh(
        mesh, pressure_field, title=title, cmap="RdBu_r"
    )


class CFDDashboard:
    """Interactive dashboard for CFD post-processing using Plotly and Streamlit.

    Parameters
    ----------
    case_path : str or Path
        Path to the OpenFOAM case directory.
    """

    def __init__(self, case_path: Union[str, Path]):
        self.case_path = Path(case_path)
        self._figures: Dict[str, go.Figure] = {}
        self._data: Dict[str, Any] = {}

    def add_figure(self, name: str, figure: go.Figure) -> None:
        """Register a Plotly figure in the dashboard.

        Parameters
        ----------
        name : str
            Identifier for the figure.
        figure : plotly.graph_objects.Figure
            The figure to display.
        """
        self._figures[name] = figure

    def add_data(self, name: str, data: Any) -> None:
        """Register arbitrary data in the dashboard.

        Parameters
        ----------
        name : str
            Identifier for the data.
        data : Any
            The data object (mesh, DataFrame, array, etc.).
        """
        self._data[name] = data

    def add_mesh(
        self,
        name: str,
        mesh: pv.DataSet,
        scalars: Optional[str] = None,
        scalars_title: Optional[str] = None,
    ) -> None:
        """Add a mesh with optional scalar field as a contour figure.

        Parameters
        ----------
        name : str
            Identifier for this mesh entry.
        mesh : pv.DataSet
            The PyVista mesh.
        scalars : str, optional
            Scalar field to plot as a contour. If ``None``, only the
            mesh geometry is added.
        scalars_title : str, optional
            Title override for the scalar contour.
        """
        if scalars is not None and scalars in mesh.point_data:
            title = scalars_title or f"{scalars} — {name}"
            fig = plotly_contour_from_mesh(mesh, scalars, title=title)
            self._figures[f"{name}_contour"] = fig

        self._data[name] = mesh

    def to_html(
        self,
        filename: Union[str, Path] = "dashboard.html",
        *,
        include_plotly_js: bool = True,
        responsive: bool = True,
    ) -> Path:
        """Export all registered figures to a self-contained HTML dashboard.

        Parameters
        ----------
        filename : str or Path, optional
            Output HTML file. Defaults to ``"dashboard.html"``.
        include_plotly_js : bool, optional
            Embed the Plotly.js library for offline viewing.
        responsive : bool, optional
            Make the layout responsive to window size.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        out_path = Path(filename)
        if out_path.suffix != ".html":
            out_path = out_path.with_suffix(".html")

        html_parts: List[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append('<html lang="en">')
        html_parts.append("<head>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
        html_parts.append(f"<title>CFD Dashboard — {self.case_path.name}</title>")

        if include_plotly_js:
            html_parts.append(
                '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
            )
        html_parts.append(
            "<style>"
            "body { font-family: sans-serif; margin: 0; padding: 1em; "
            "background: #fafafa; }"
            "h1 { padding: 0.5em 0; border-bottom: 2px solid #333; }"
            "h2 { margin-top: 2em; }"
            ".grid { display: grid; grid-template-columns: repeat(auto-fill, "
            "minmax(600px, 1fr)); gap: 1em; }"
            ".figure { background: #fff; border: 1px solid #ddd; "
            "border-radius: 4px; padding: 0.5em; }"
            ".figure iframe, .figure div { width: 100% !important; }"
            "</style>"
        )
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append(f"<h1>CFD Dashboard — {self.case_path.name}</h1>")

        if self._data:
            html_parts.append("<h2>Mesh Summary</h2>")
            for name, mesh in self._data.items():
                if isinstance(mesh, pv.DataSet):
                    stats = mesh.n_points, mesh.n_cells
                    html_parts.append(
                        f"<p><strong>{name}</strong>: "
                        f"{stats[0]} points, {stats[1]} cells</p>"
                    )

        if self._figures:
            html_parts.append("<h2>Visualizations</h2>")
            html_parts.append("<div class='grid'>")
            for name, fig in self._figures.items():
                html_parts.append("<div class='figure'>")
                html_parts.append(fig.to_html(
                    include_plotlyjs=False,
                    full_html=False,
                    div_id=f"fig_{name}",
                ))
                html_parts.append(f"<p>{name}</p>")
                html_parts.append("</div>")
            html_parts.append("</div>")

        html_parts.append("</body>")
        html_parts.append("</html>")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info("Dashboard saved to %s", out_path)
        return out_path

    def to_streamlit(self) -> None:
        """Render the dashboard in a Streamlit app.

        This method should be called from within a Streamlit context.
        Requires the ``streamlit`` package.

        Example
        -------
        ```python
        from foampilot.postprocess import CFDDashboard

        dashboard = CFDDashboard("myCase")
        dashboard.add_mesh("internal", mesh, scalars="T")
        dashboard.to_streamlit()
        ```
        """
        try:
            import streamlit as st
        except ImportError as exc:
            raise ImportError(
                "streamlit is required for to_streamlit(). "
                "Install with: pip install streamlit"
            ) from exc

        st.set_page_config(
            page_title=f"CFD Dashboard — {self.case_path.name}",
            layout="wide",
        )
        st.title(f"CFD Dashboard — {self.case_path.name}")

        if self._data:
            st.header("Mesh Summary")
            for name, mesh in self._data.items():
                if isinstance(mesh, pv.DataSet):
                    col1, col2 = st.columns(2)
                    col1.metric("Points", mesh.n_points)
                    col2.metric("Cells", mesh.n_cells)

        if self._figures:
            st.header("Visualizations")
            for name, fig in self._figures.items():
                st.subheader(name)
                st.plotly_chart(fig, use_container_width=True)