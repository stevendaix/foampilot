#!/usr/bin/env python3
"""Plot VoxCity buildings directly from the raw HDF5 file.

This is a verification/visualization tool that reads the voxelized
building footprints and heights from the VoxCity HDF5 output and
plots them in matplotlib.

It is intentionally independent from the OpenFOAM pipeline so you
can compare the original VoxCity data with the simplified CFD mesh.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


def load_voxcity_buildings(hdf5_path: str):
    """Extract building footprints and heights from the VoxCity HDF5.

    Returns a list of dicts with keys: id, height, footprint (rows, cols).
    """
    buildings = []
    with h5py.File(hdf5_path, "r") as f:
        if "voxcity" not in f:
            return buildings
        vox = f["voxcity"]
        if "building_id" not in vox or "building_height" not in vox:
            return buildings

        ids = vox["building_id"][:]
        heights = vox["building_height"][:]
        unique_ids = np.unique(ids[ids > 0])

        for bid in unique_ids:
            mask = ids == bid
            if not np.any(mask):
                continue
            rows, cols = np.where(mask)
            if len(rows) < 4:
                continue
            h = float(heights[mask].mean())
            buildings.append({
                "id": int(bid),
                "height": h,
                "rows": rows,
                "cols": cols,
            })

    return buildings


def plot_voxcity_h5(
    hdf5_path: str,
    output_path: str = "voxcity_h5_map.png",
    show_domain: bool = True,
    cell_size: float = 5.0,
):
    """Plot VoxCity buildings from the raw HDF5 file."""
    buildings = load_voxcity_buildings(hdf5_path)
    if not buildings:
        print("No buildings found in HDF5.")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(buildings)))

    for idx, b in enumerate(buildings):
        rows, cols = b["rows"], b["cols"]
        fxmin = cols.min() * cell_size
        fymin = rows.min() * cell_size
        fxmax = (cols.max() + 1) * cell_size
        fymax = (rows.max() + 1) * cell_size

        xmin = min(xmin, fxmin)
        ymin = min(ymin, fymin)
        xmax = max(xmax, fxmax)
        ymax = max(ymax, fymax)

        rect = Rectangle(
            (fxmin, fymin),
            fxmax - fxmin,
            fymax - fymin,
            facecolor=colors[idx],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
            label=f"ID {b['id']}: {b['height']:.1f}m",
        )
        ax.add_patch(rect)

    margin_x = (xmax - xmin) * 0.1
    margin_y = (ymax - ymin) * 0.1
    domain_xmin = xmin - margin_x
    domain_ymin = ymin - margin_y
    domain_xmax = xmax + margin_x
    domain_ymax = ymax + margin_y

    if show_domain:
        domain = Rectangle(
            (domain_xmin, domain_ymin),
            domain_xmax - domain_xmin,
            domain_ymax - domain_ymin,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Domaine HDF5",
        )
        ax.add_patch(domain)

    ax.set_xlim(domain_xmin - margin_x * 0.5, domain_xmax + margin_x * 0.5)
    ax.set_ylim(domain_ymin - margin_y * 0.5, domain_ymax + margin_y * 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"VoxCity HDF5 brut — {len(buildings)} bâtiments\nGrille {cell_size} m")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path} ({len(buildings)} buildings)")


def main():
    parser = argparse.ArgumentParser(description="Plot VoxCity HDF5 buildings")
    parser.add_argument("hdf5", help="Path to voxcity.h5")
    parser.add_argument("-o", "--output", default="voxcity_h5_map.png", help="Output image")
    parser.add_argument("--cell-size", type=float, default=5.0, help="Voxel size in meters")
    parser.add_argument("--no-domain", action="store_true", help="Hide domain box")
    args = parser.parse_args()

    plot_voxcity_h5(
        args.hdf5,
        output_path=args.output,
        show_domain=not args.no_domain,
        cell_size=args.cell_size,
    )


if __name__ == "__main__":
    main()
