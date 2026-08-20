from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np


def build_graph(data, topology):
    graph = nx.Graph()
    endpoint_node = {}
    for node in topology.get("nodes", []):
        node_id = int(node["node_id"])
        graph.add_node(node_id, center=node["center"], degree_semantic=node["degree"], sides=node["sides"])
        for member in node["members"]:
            endpoint_node[(int(member["branch_id"]), member["side"])] = node_id
    for branch in data["branches"]:
        branch_id = int(branch["branch_id"])
        source = endpoint_node.get((branch_id, "first"), f"b{branch_id}_first")
        target = endpoint_node.get((branch_id, "last"), f"b{branch_id}_last")
        sections = branch.get("sections", [])
        centers = np.asarray([section["center"] for section in sections], float)
        length = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0
        graph.add_edge(source, target, branch_id=branch_id, length=length, sections=len(sections))
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.sections.read_text())
    patch = json.loads(args.patch_report.read_text())
    topology = json.loads(args.topology.read_text())
    graph = build_graph(data, topology)
    components = [sorted(component, key=str) for component in nx.connected_components(graph)]
    terminal_nodes = topology.get("terminal_nodes", [])
    junction_nodes = topology.get("junction_nodes", [])

    network_report = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": len(components),
        "connected": len(components) == 1,
        "components": components,
        "terminal_nodes": terminal_nodes,
        "junction_nodes": junction_nodes,
        "node_degrees": {str(node): int(degree) for node, degree in graph.degree()},
        "edges": [
            {"source": str(u), "target": str(v), **{key: value for key, value in attributes.items()}}
            for u, v, attributes in graph.edges(data=True)
        ],
    }

    report = {
        "branches": [],
        "topology": network_report,
        "patches": patch.get("patches", []),
    }

    print("\n=== MEDICAL_BUILD / TOPOLOGIE NETWORKX ===")
    print(f"Nœuds NetworkX       : {network_report['nodes']}")
    print(f"Arêtes NetworkX       : {network_report['edges']}")
    print(f"Composantes connexes  : {network_report['connected_components']}")
    print(f"Réseau connecté       : {'oui' if network_report['connected'] else 'NON'}")
    print(f"Nœuds terminaux       : {terminal_nodes}")
    print(f"Nœuds internes        : {junction_nodes}")
    print("\nArêtes / segments :")
    for edge in network_report["edges"]:
        print(f"  branche {edge['branch_id']:>2}: {edge['source']} -> {edge['target']} length={edge['length']:.3f} sections={edge['sections']}")

    print("\n=== BRANCHES ===")
    for branch in data["branches"]:
        sections = branch.get("sections", [])
        centers = np.asarray([section["center"] for section in sections], float)
        radii = np.asarray([section.get("equivalent_radius", 0.0) for section in sections], float)
        areas = np.asarray([section.get("area", 0.0) for section in sections], float)
        length = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0
        row = {
            "branch_id": branch["branch_id"],
            "source_cap_id": branch.get("source_cap_id"),
            "target_cap_id": branch.get("target_cap_id"),
            "sections": len(sections),
            "length": length,
            "radius_min": float(radii.min()) if len(radii) else 0.0,
            "radius_max": float(radii.max()) if len(radii) else 0.0,
            "area_min": float(areas.min()) if len(areas) else 0.0,
            "area_max": float(areas.max()) if len(areas) else 0.0,
        }
        report["branches"].append(row)
        print(f"branche {row['branch_id']:>2}: sections={row['sections']:>3} longueur={row['length']:8.3f} rayon=[{row['radius_min']:.3f}, {row['radius_max']:.3f}]")

    print("\n=== SORTIES CFD ===")
    for item in patch.get("patches", []):
        print(f"{item.get('name', '?'):>10}: aire={item.get('area', 0.0):8.3f} cellules={item.get('cells', item.get('cap_faces', 0)):>6} centre={np.round(item.get('center', [0, 0, 0]), 3).tolist()}")

    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
