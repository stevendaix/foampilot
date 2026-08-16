"""Geometry mapping and visualization utilities for urban CFD tests."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from foampilot.urban.model.urban_model import UrbanModel
from foampilot.urban.simplification.cfd_simplifier import CFDGeometry
from foampilot.urban.model.domain import CFDDomain, WindFrame


class GeometryMapper:
    """Map and visualize urban CFD geometry for validation."""

    def __init__(self, urban: UrbanModel, geometry: Optional[CFDGeometry] = None):
        self.urban = urban
        self.geometry = geometry

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the urban geometry."""
        buildings = self.urban.buildings()
        bbox = self.urban.bbox()

        heights = [b.height for b in buildings]
        areas = [b.area for b in buildings]

        summary = {
            "n_buildings": len(buildings),
            "bbox": {
                "xmin": bbox[0],
                "ymin": bbox[1],
                "zmin": bbox[2],
                "xmax": bbox[3],
                "ymax": bbox[4],
                "zmax": bbox[5],
            },
            "height_stats": {
                "min": min(heights) if heights else 0.0,
                "max": max(heights) if heights else 0.0,
                "mean": sum(heights) / len(heights) if heights else 0.0,
            },
            "area_stats": {
                "min": min(areas) if areas else 0.0,
                "max": max(areas) if areas else 0.0,
                "total": sum(areas),
            },
        }

        if self.geometry is not None:
            summary["domain_box"] = {
                "xmin": self.geometry.domain_box[0],
                "ymin": self.geometry.domain_box[1],
                "zmin": self.geometry.domain_box[2],
                "xmax": self.geometry.domain_box[3],
                "ymax": self.geometry.domain_box[4],
                "zmax": self.geometry.domain_box[5],
            }
            summary["cfd_buildings"] = len(self.geometry.buildings)

        return summary

    def save_summary(self, path: Path) -> None:
        """Save geometry summary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def plot_footprints(self, ax=None, title: str = "Building footprints"):
        """Plot building footprints on XY plane."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        for i, b in enumerate(self.urban.buildings()):
            color = colors[i % len(colors)]
            coords = list(b.footprint.exterior.coords)
            poly = MplPolygon(coords, closed=True, fill=True,
                             facecolor=color, edgecolor="black", alpha=0.6)
            ax.add_patch(poly)

        bbox = self.urban.bbox()
        ax.set_xlim(bbox[0] - 5, bbox[3] + 5)
        ax.set_ylim(bbox[1] - 5, bbox[4] + 5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_domain(self, ax=None, title: str = "CFD domain"):
        """Plot CFD domain box with buildings."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Polygon as MplPolygon
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if self.geometry is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
            domain_rect = Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
                label="CFD domain"
            )
            ax.add_patch(domain_rect)

        for i, b in enumerate(self.urban.buildings()):
            coords = list(b.footprint.exterior.coords)
            poly = MplPolygon(coords, closed=True, fill=True,
                             facecolor="blue", edgecolor="black", alpha=0.5)
            ax.add_patch(poly)

        if self.geometry is not None:
            bbox = self.geometry.domain_box
            ax.set_xlim(bbox[0] - 10, bbox[3] + 10)
            ax.set_ylim(bbox[1] - 10, bbox[4] + 10)
        else:
            bbox = self.urban.bbox()
            ax.set_xlim(bbox[0] - 10, bbox[3] + 10)
            ax.set_ylim(bbox[1] - 10, bbox[4] + 10)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_3d(self, ax=None, title: str = "3D geometry"):
        """Plot 3D view of buildings and domain."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        for b in self.urban.buildings():
            coords = list(b.footprint.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]

            base_z = b.ground_z
            roof_z = b.roof_z

            bottom = [(x, y, base_z) for x, y in coords]
            top = [(x, y, roof_z) for x, y in coords]

            faces = []
            for i in range(len(coords) - 1):
                j = (i + 1) % (len(coords) - 1)
                faces.append([bottom[i], bottom[j], top[j], top[i]])

            n = len(coords) - 1
            bottom_face = [bottom[i] for i in range(n)]
            top_face = [top[i] for i in range(n)]
            faces.append(bottom_face)
            faces.append(top_face)

            poly3d = Poly3DCollection(faces, alpha=0.7,
                                     facecolor="lightblue", edgecolor="black")
            ax.add_collection3d(poly3d)

        if self.geometry is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
            domain_corners = [
                [xmin, ymin, zmin], [xmax, ymin, zmin],
                [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax],
                [xmax, ymax, zmax], [xmin, ymax, zmax],
            ]
            domain_edges = [
                [domain_corners[0], domain_corners[1]],
                [domain_corners[1], domain_corners[2]],
                [domain_corners[2], domain_corners[3]],
                [domain_corners[3], domain_corners[0]],
                [domain_corners[4], domain_corners[5]],
                [domain_corners[5], domain_corners[6]],
                [domain_corners[6], domain_corners[7]],
                [domain_corners[7], domain_corners[4]],
                [domain_corners[0], domain_corners[4]],
                [domain_corners[1], domain_corners[5]],
                [domain_corners[2], domain_corners[6]],
                [domain_corners[3], domain_corners[7]],
            ]
            for edge in domain_edges:
                xs, ys, zs = zip(*edge)
                ax.plot(xs, ys, zs, "r--", linewidth=2, alpha=0.5)

        all_x = [b.footprint.exterior.coords[0][0] for b in self.urban.buildings()]
        all_y = [b.footprint.exterior.coords[0][1] for b in self.urban.buildings()]
        all_z = [b.ground_z for b in self.urban.buildings()]

        if self.geometry is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.geometry.domain_box
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
        else:
            if all_x:
                ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
                ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
                ax.set_zlim(0, max(b.roof_z for b in self.urban.buildings()) + 5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)

        return ax

    def save_plots(self, output_dir: Path, prefix: str = "geometry") -> List[Path]:
        """Save all geometry plots to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            self.plot_footprints(ax=ax)
            path = output_dir / f"{prefix}_footprints.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

            fig, ax = plt.subplots(figsize=(10, 10))
            self.plot_domain(ax=ax)
            path = output_dir / f"{prefix}_domain.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            self.plot_3d(ax=ax)
            path = output_dir / f"{prefix}_3d.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        except ImportError:
            pass

        return saved
