"""Clear visualizations for the local deformation validation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def plot_branch_only(root: Path, output: Path, branch_id: int = 2) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, label in zip(axes, ("reference", "deformed")):
        mesh = trimesh.load(root / label / f"branch_{branch_id:02d}.stl", force="mesh")
        # Render a clean orthographic projection of the actual surface.
        for face in mesh.faces[::2]:
            xy = mesh.vertices[face][:, :2]
            ax.fill(xy[:, 0], xy[:, 1], color="#4c78a8" if label == "reference" else "#d62728", alpha=0.16, linewidth=0)
        ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], s=0.15, color="#1f2937", alpha=0.15)
        ax.set_aspect("equal")
        ax.set_title(f"Branche {branch_id} — {label}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.2)
    fig.savefig(output / "04_branch_02_clean_projection.png", dpi=220)
    plt.close(fig)


def plot_scale_profile(contract_path: Path, output: Path, branch_id: int = 2) -> None:
    data = json.loads(contract_path.read_text())
    branch = next(b for b in data["deformed_analysis"]["branches"] if b["branch_id"] == branch_id)
    s = np.array([section["abscissa"] for section in branch["sections"]])
    scale = np.array([section.get("metadata", {}).get("local_deformation_scale", 1.0) for section in branch["sections"]])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s, scale, color="#d62728", linewidth=2)
    ax.fill_between(s, 1.0, scale, color="#d62728", alpha=0.20)
    ax.axhline(1.0, color="#333333", linewidth=1)
    ax.set_xlabel("Abscisse de branche")
    ax.set_ylabel("Facteur radial")
    ax.set_title(f"Profil de déformation local — branche {branch_id}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "05_deformation_scale_profile_branch_02.png", dpi=220)
    plt.close(fig)


def plot_pyvista(root: Path, output: Path, branch_id: int = 2) -> None:
    import pyvista as pv
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 900))
    for index, label in enumerate(("reference", "deformed")):
        mesh = pv.read(str(root / label / f"branch_{branch_id:02d}.stl")).triangulate().clean()
        plotter.subplot(0, index)
        plotter.add_mesh(mesh, color="#4c78a8" if label == "reference" else "#d62728", smooth_shading=True, show_edges=False, opacity=1.0)
        plotter.add_text(f"Branche {branch_id} — {label}", font_size=14)
        plotter.add_axes()
        plotter.view_isometric()
        plotter.camera.zoom(1.2)
    plotter.link_views()
    plotter.screenshot(str(output / "06_branch_02_pyvista_surface.png"), transparent_background=False)
    plotter.close()


def main() -> None:
    root = Path("examples/medical_build/outputs/reconstructed_local_deformation")
    output = root / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    plot_branch_only(root, output)
    plot_scale_profile(root / "../local_deformation_real/aorta_complex_branch2.json", output)
    plot_pyvista(root, output)
    print(json.dumps({"output": str(output), "images": [
        "04_branch_02_clean_projection.png",
        "05_deformation_scale_profile_branch_02.png",
        "06_branch_02_pyvista_surface.png",
    ]}, indent=2))


if __name__ == "__main__":
    main()
