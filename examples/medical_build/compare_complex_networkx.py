from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
CENTERLINES = ROOT / 'examples/medical_build/case_complex/analysis/centerlines.vtp'
OUT = ROOT / 'examples/medical_build/outputs'
TOL = 2.0


def cluster(points, tolerance):
    centers = []
    labels = []
    for point in points:
        point = np.asarray(point, dtype=float)
        distances = [np.linalg.norm(point - c) for c in centers]
        if distances and min(distances) <= tolerance:
            idx = int(np.argmin(distances))
            centers[idx] = (centers[idx] + point) / 2.0
        else:
            idx = len(centers)
            centers.append(point)
        labels.append(idx)
    return np.asarray(centers), labels


def main():
    cl = pv.read(CENTERLINES)
    endpoints = []
    cells = []
    for cell_id in range(cl.n_cells):
        part = cl.extract_cells([cell_id])
        pts = np.asarray(part.points, dtype=float)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        cells.append({'cell_id': cell_id, 'points': pts.tolist(), 'length': length})
        endpoints.extend([pts[0], pts[-1]])

    node_centers, labels = cluster(endpoints, TOL)
    graph = nx.MultiGraph()
    for node_id, center in enumerate(node_centers):
        graph.add_node(node_id, xyz=center.tolist())
    for i, item in enumerate(cells):
        source, target = labels[2*i], labels[2*i+1]
        graph.add_edge(source, target, key=item['cell_id'], cell_id=item['cell_id'], length=item['length'])

    degrees = dict(graph.degree())
    terminal_nodes = [n for n, d in degrees.items() if d == 1]
    junction_nodes = [n for n, d in degrees.items() if d >= 3]
    result = {
        'input': str(CENTERLINES),
        'endpoint_tolerance': TOL,
        'vtk_cells': cl.n_cells,
        'vtk_points': cl.n_points,
        'networkx_nodes': graph.number_of_nodes(),
        'networkx_edges': graph.number_of_edges(),
        'connected_components': nx.number_connected_components(graph),
        'is_tree': nx.is_tree(nx.Graph(graph)),
        'terminal_nodes': terminal_nodes,
        'junction_nodes': junction_nodes,
        'node_degrees': {str(k): int(v) for k, v in degrees.items()},
        'nodes': [{'node_id': int(n), 'xyz': graph.nodes[n]['xyz'], 'degree': int(graph.degree(n))} for n in graph.nodes],
        'edges': [{'cell_id': int(data['cell_id']), 'source': int(u), 'target': int(v), 'length': data['length']} for u, v, _, data in graph.edges(keys=True, data=True)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'complex_networkx_topology.json').write_text(json.dumps(result, indent=2))

    plt.figure(figsize=(12, 8))
    pos = {n: (graph.nodes[n]['xyz'][1], graph.nodes[n]['xyz'][2]) for n in graph.nodes}
    nx.draw_networkx_edges(graph, pos, edge_color='steelblue', width=3)
    nx.draw_networkx_nodes(graph, pos, node_color=['crimson' if n in junction_nodes else 'darkgreen' if n in terminal_nodes else 'gold' for n in graph.nodes], node_size=700)
    nx.draw_networkx_labels(graph, pos, labels={n: f'N{n}\nd={degrees[n]}' for n in graph.nodes}, font_size=9)
    edge_labels={(u,v): f"cell {data['cell_id']}" for u,v,_,data in graph.edges(keys=True,data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title('NetworkX topology — complex VMTK centerlines')
    plt.axis('equal'); plt.axis('off'); plt.tight_layout()
    plt.savefig(OUT / 'complex_networkx_graph_2d.png', dpi=180); plt.close()

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000)); plotter.set_background('white')
    colors = ['red','orange','yellow','green','blue','purple','cyan','magenta']
    for i in range(cl.n_cells):
        plotter.add_mesh(cl.extract_cells([i]), color=colors[i % len(colors)], line_width=6, render_lines_as_tubes=True, label=f'cell {i}')
    node_poly = pv.PolyData(node_centers)
    node_poly['degree'] = np.asarray([degrees[i] for i in range(len(node_centers))])
    plotter.add_mesh(node_poly, render_points_as_spheres=True, point_size=18, color='black')
    plotter.add_point_labels(node_centers, [f'N{i} d={degrees[i]}' for i in range(len(node_centers))], font_size=12, shape_color='white', text_color='black')
    plotter.add_legend(bcolor='white', face='rectangle')
    plotter.add_text('Original complex centerlines + NetworkX nodes', font_size=15, color='black')
    plotter.camera_position = 'iso'
    plotter.show(screenshot=str(OUT / 'complex_networkx_overlay_3d.png'), auto_close=True)
    print(json.dumps({k: result[k] for k in ['vtk_cells','networkx_nodes','networkx_edges','connected_components','terminal_nodes','junction_nodes']}, indent=2))


if __name__ == '__main__':
    main()
