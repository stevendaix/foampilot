"""Generate visual validation images for the UrbGEN foampilot port."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

from foampilot.urban.generation import UrbGENConfig, generate_urbgen

COLORS = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#E45756", 4: "#72B7B2", 5: "#B279A2", 7: "#FF9DA6"}
NAMES = {0: "I", 1: "L", 2: "T", 3: "H", 4: "C", 5: "Plus", 7: "Courtyard"}


def _draw_geom(ax, geom, **kwargs):
    if geom.is_empty:
        return
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in geoms:
        ax.add_patch(MplPolygon(list(poly.exterior.coords), closed=True, **kwargs))


def render_case(name: str, config: UrbGENConfig, output_dir: Path) -> Path:
    site = Polygon([(0, 0), (180, 0), (180, 130), (0, 130)])
    result = generate_urbgen(site, config)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    _draw_geom(ax, site, facecolor="#F2F2F2", edgecolor="#333333", linewidth=2, alpha=0.9)
    _draw_geom(ax, result.buildable_site, facecolor="none", edgecolor="#777777", linewidth=1.5, linestyle="--")
    for p in result.podium_footprints:
        _draw_geom(ax, p, facecolor="#D9D9D9", edgecolor="#555555", linewidth=1, alpha=0.9)
    for p, code in zip(result.tower_footprints, result.tower_typologies):
        _draw_geom(ax, p, facecolor=COLORS.get(code, "#999999"), edgecolor="#222222", linewidth=0.8, alpha=0.85)
        ax.text(p.centroid.x, p.centroid.y, NAMES.get(code, str(code)), ha="center", va="center", fontsize=7, color="white", weight="bold")
    ax.set_aspect("equal")
    ax.margins(0.04)
    ax.autoscale_view()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"UrbGEN / {name} — {len(result.tower_footprints)} tours, BCR={result.actual_bcr:.3f}, FAR={result.actual_far:.3f}")
    ax.grid(True, alpha=0.2)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=NAMES[k], markersize=8) for k, c in COLORS.items() if k in set(result.tower_typologies)]
    if handles:
        ax.legend(handles=handles, title="Typologie", loc="upper right")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"urbgen_{name.lower().replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("urbgen_validation_images"))
    args = parser.parse_args()
    cases = {
        "random_typologies": UrbGENConfig(bcr=0.16, far=2.5, setback=8, tower_typology_mode=6, podium_floors=2, height_variation=0.25, seed=42),
        "courtyard": UrbGENConfig(bcr=0.20, far=2.0, setback=8, tower_typology_mode=7, courtyard_break_count=4, courtyard_break_width=8, podium_floors=0, seed=42),
        "edge_aligned": UrbGENConfig(bcr=0.14, far=3.0, setback=8, tower_typology_mode=3, align_towers_to_edge=True, move_to_boundary=True, podium_floors=2, seed=42),
    }
    for name, config in cases.items():
        print(render_case(name, config, args.output))


if __name__ == "__main__":
    main()
