"""Create visual diagnostics for the real complex-aorta deformation campaign."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
import trimesh


def load_contract(path: Path):
    return json.loads(path.read_text())


def plot_surfaces(root: Path, output: Path) -> None:
    fig = plt.figure(figsize=(16, 8))
    for column, label in enumerate(("reference", "deformed"), start=1):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        for stl in sorted((root / label).glob("branch_*.stl")):
            mesh = trimesh.load(stl, force="mesh")
            faces = mesh.faces[::8]
            verts = mesh.vertices
            polys = [[verts[i] for i in face] for face in faces]
            is_target = stl.name == "branch_02.stl"
            color = "#d62728" if is_target and label == "deformed" else ("#ff9896" if is_target else "#4c78a8")
            alpha = 0.65 if is_target else 0.10
            ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=color, alpha=alpha, linewidths=0.05))
        ax.set_title(f"Aorte complexe — {label}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((1, 1, 1.4))
        ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    fig.savefig(output / "01_aorta_reference_vs_deformed_3d.png", dpi=180)
    plt.close(fig)


def plot_section(contract: dict, output: Path, branch_id: int = 2) -> None:
    source_analysis = json.loads(Path(contract["analysis_contract"]).read_text())
    deformed_analysis = json.loads((Path(contract["root"]) / "../local_deformation_real/aorta_complex_branch2.json").resolve().read_text())["deformed_analysis"]
    reference_branch = next(b for b in source_analysis["branches"] if b["branch_id"] == branch_id)
    deformed_branch = next(b for b in deformed_analysis["branches"] if b["branch_id"] == branch_id)
    scales = [section.get("metadata", {}).get("local_deformation_scale", 1.0) for section in deformed_branch["sections"]]
    station = int(np.argmax(scales))
    raw = np.asarray(reference_branch["sections"][station]["points"], dtype=float)
    deformed = np.asarray(deformed_branch["sections"][station]["points"], dtype=float)
    center = np.asarray(reference_branch["sections"][station]["center"], dtype=float)
    normal = np.asarray(reference_branch["sections"][station]["normal"], dtype=float)
    binormal = np.asarray(reference_branch["sections"][station]["binormal"], dtype=float)
    raw_local = np.column_stack(((raw - center) @ normal, (raw - center) @ binormal))
    deformed_local = np.column_stack(((deformed - center) @ normal, (deformed - center) @ binormal))
    scale = scales[station]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(raw_local[:, 0], raw_local[:, 1], "o-", ms=2, label="Référence")
    ax.plot(deformed_local[:, 0], deformed_local[:, 1], "o-", ms=2, label=f"Déformée (échelle {scale:.4f})")
    ax.set_aspect("equal")
    ax.set_title(f"Section maximale — branche {branch_id}, station {station}")
    ax.set_xlabel("coordonnée selon normal")
    ax.set_ylabel("coordonnée selon binormal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "02_section_maximum_branch_02.png", dpi=180)
    plt.close(fig)


def plot_graph(analysis: dict, output: Path) -> None:
    graph = nx.Graph()
    for branch in analysis["branches"]:
        graph.add_edge(branch["source_cap_id"], branch["target_cap_id"], branch_id=branch["branch_id"], length=branch["length"])
    pos = nx.spring_layout(graph, seed=7)
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(graph, pos, node_color=["#d95f02" if graph.degree(n) > 1 else "#1b9e77" for n in graph.nodes], node_size=700, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=2.0, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    labels = {(u, v): f"b{d['branch_id']}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax, font_size=8)
    ax.set_title("Graphe NetworkX — branches et caps")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "03_networkx_vascular_graph.png", dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path("examples/medical_build/outputs/reconstructed_local_deformation")
    output = root / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((root / "reconstruction_comparison.json").read_text())
    comparison["analysis_contract"] = "/tmp/medical_build_complex_prebuild/analysis_contract.json"
    comparison["root"] = str(root)
    analysis = load_contract(Path(comparison["analysis_contract"]))
    plot_surfaces(root, output)
    plot_section(comparison, output)
    plot_graph(analysis, output)
    (output / "visualization_manifest.json").write_text(json.dumps({
        "images": [
            "01_aorta_reference_vs_deformed_3d.png",
            "02_section_maximum_branch_02.png",
            "03_networkx_vascular_graph.png",
        ],
        "source": comparison["analysis_contract"],
    }, indent=2))
    print(json.dumps({"output": str(output), "images": 3}, indent=2))


if __name__ == "__main__":
    main()
