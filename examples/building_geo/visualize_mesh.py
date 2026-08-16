#!/usr/bin/env python3
"""
Visualize OpenFOAM mesh and save images.

Usage:
    PYTHONPATH=../../foampilot/src python3 visualize_mesh.py \
        --case /tmp/voxcity_cached_demo \
        --output /tmp/mesh_vis.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import numpy as np
import matplotlib.pyplot as plt


def read_points_file(filepath: Path):
    """Read OpenFOAM points file."""
    import re
    content = filepath.read_text()
    
    match = re.search(r'\n(\d+)\n\(', content)
    if not match:
        return np.array([])
    
    start = match.start()
    array_content = content[start:]
    
    depth = 0
    end_idx = -1
    for i in range(len(array_content)):
        if array_content[i] == '(':
            depth += 1
        elif array_content[i] == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    
    if end_idx == -1:
        return np.array([])
    
    points_str = array_content[:end_idx+1]
    points = []
    for line in points_str.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            line = line[1:-1]
            parts = line.split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    
    return np.array(points)


def read_faces_file(filepath: Path):
    """Read OpenFOAM faces file."""
    import re
    content = filepath.read_text()
    
    match = re.search(r'\n(\d+)\n\(', content)
    if not match:
        return []
    
    start = match.start()
    array_content = content[start:]
    
    depth = 0
    end_idx = -1
    for i in range(len(array_content)):
        if array_content[i] == '(':
            depth += 1
        elif array_content[i] == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    
    if end_idx == -1:
        return []
    
    faces_str = array_content[:end_idx+1]
    faces = []
    
    # Pattern: nVerts(v0 v1 v2 ...)
    face_pattern = re.compile(r'(\d+)\(([^)]+)\)')
    for match in face_pattern.finditer(faces_str):
        try:
            n_verts = int(match.group(1))
            verts_str = match.group(2)
            verts = [int(v) for v in verts_str.split()]
            faces.append((n_verts, verts))
        except ValueError:
            pass
    
    return faces


def read_boundary_file(filepath: Path):
    """Read OpenFOAM boundary file."""
    content = filepath.read_text()
    
    patches = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (line and not line.startswith('//') and 
            not line.startswith('(') and not line.startswith(')') and 
            not line.startswith('FoamFile') and not line.startswith('{') and 
            not line.startswith('}') and not line.startswith('*') and
            not line.startswith('/*') and not line.startswith('*/') and
            not line.startswith('\\') and not line.startswith('=') and
            not line.startswith('|') and not line.startswith('//')):
            
            patch_name = line
            if not patch_name or len(patch_name) < 2:
                i += 1
                continue
                
            patch_type = "patch"
            n_faces = 0
            start_face = 0
            
            j = i + 1
            while j < len(lines):
                jline = lines[j].strip()
                if jline.startswith('type'):
                    patch_type = jline.split()[1].rstrip(';')
                elif jline.startswith('nFaces'):
                    n_faces = int(jline.split()[1].rstrip(';'))
                elif jline.startswith('startFace'):
                    start_face = int(jline.split()[1].rstrip(';'))
                elif jline.startswith('}'):
                    break
                j += 1
            
            patches.append((patch_name, patch_type, n_faces, start_face))
            i = j + 1
            continue
        i += 1
    
    return patches


def load_mesh(case_dir: Path):
    """Load OpenFOAM polyMesh."""
    polyMesh_dir = case_dir / "constant" / "polyMesh"
    
    points_file = polyMesh_dir / "points"
    faces_file = polyMesh_dir / "faces"
    owner_file = polyMesh_dir / "owner"
    neighbour_file = polyMesh_dir / "neighbour"
    boundary_file = polyMesh_dir / "boundary"
    
    if not all(f.exists() for f in [points_file, faces_file, owner_file, neighbour_file, boundary_file]):
        raise FileNotFoundError("Missing polyMesh files")
    
    points = read_points_file(points_file)
    faces = read_faces_file(faces_file)
    import re
    
    owner = []
    with open(owner_file) as f:
        content = f.read()
        match = re.search(r'\n(\d+)\n\(', content)
        if match:
            start = match.start()
            array_content = content[start:]
            depth = 0
            end_idx = -1
            for i in range(len(array_content)):
                if array_content[i] == '(':
                    depth += 1
                elif array_content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                owner_str = array_content[:end_idx+1]
                for line in owner_str.split('\n'):
                    line = line.strip()
                    if line.startswith('(') and line.endswith(')'):
                        line = line[1:-1]
                        if line:
                            try:
                                owner.append(int(line))
                            except ValueError:
                                pass
    
    neighbour = []
    with open(neighbour_file) as f:
        content = f.read()
        match = re.search(r'\n(\d+)\n\(', content)
        if match:
            start = match.start()
            array_content = content[start:]
            depth = 0
            end_idx = -1
            for i in range(len(array_content)):
                if array_content[i] == '(':
                    depth += 1
                elif array_content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                neighbour_str = array_content[:end_idx+1]
                for line in neighbour_str.split('\n'):
                    line = line.strip()
                    if line.startswith('(') and line.endswith(')'):
                        line = line[1:-1]
                        if line:
                            try:
                                neighbour.append(int(line))
                            except ValueError:
                                pass
    
    boundary = read_boundary_file(boundary_file)
    
    return {
        'points': points,
        'faces': faces,
        'owner': owner,
        'neighbour': neighbour,
        'boundary': boundary,
    }


def plot_mesh(mesh, output_path: Path, title: str = "OpenFOAM Mesh"):
    """Plot mesh overview with patches."""
    points = mesh['points']
    faces = mesh['faces']
    boundary = mesh['boundary']
    
    n_patches = len(boundary)
    n_cells = len(set(mesh['owner']))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{title}\n{len(points)} points, {n_cells} cells, {n_patches} patches", 
                 fontsize=14)
    
    # Top view (XY)
    ax1 = axes[0, 0]
    ax1.scatter(points[:, 0], points[:, 1], c='lightblue', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Front view (XZ)
    ax2 = axes[0, 1]
    ax2.scatter(points[:, 0], points[:, 2], c='lightcoral', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Front View (XZ)')
    ax2.grid(True, alpha=0.3)
    
    # Side view (YZ)
    ax3 = axes[1, 0]
    ax3.scatter(points[:, 1], points[:, 2], c='lightgreen', s=1, alpha=0.6)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (YZ)')
    ax3.grid(True, alpha=0.3)
    
    # Patch summary
    ax4 = axes[1, 1]
    patch_counts = [(b[0], b[2]) for b in boundary]
    names = [p[0] for p in patch_counts]
    counts = [p[1] for p in patch_counts]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax4.barh(names, counts, color=colors)
    ax4.set_xlabel('Number of Faces')
    ax4.set_title('Patches')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (name, count) in enumerate(patch_counts):
        ax4.text(count + max(counts)*0.02, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Mesh visualization saved to {output_path}")
    plt.close()


def print_mesh_stats(mesh):
    """Print mesh statistics."""
    points = mesh['points']
    faces = mesh['faces']
    owner = mesh['owner']
    boundary = mesh['boundary']
    
    n_points = len(points)
    n_faces = len(faces)
    n_cells = len(set(owner))
    n_internal = len(mesh['neighbour'])
    n_boundary = n_faces - n_internal
    
    print(f"\n{'='*60}")
    print(f"Mesh Statistics")
    print(f"{'='*60}")
    print(f"  Points: {n_points}")
    print(f"  Faces: {n_faces}")
    print(f"  Cells: {n_cells}")
    print(f"  Internal faces: {n_internal}")
    print(f"  Boundary faces: {n_boundary}")
    print(f"  Patches: {len(boundary)}")
    
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    print(f"  Bounding box: ({xmin:.1f}, {ymin:.1f}, {zmin:.1f}) -> ({xmax:.1f}, {ymax:.1f}, {zmax:.1f})")
    print(f"  Domain size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f}")
    
    print(f"\n  Patches:")
    for name, ptype, nfaces, start in boundary:
        print(f"    {name:15s}: {nfaces:5d} faces  ({ptype})")


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenFOAM mesh")
    parser.add_argument("--case", required=True, help="OpenFOAM case directory")
    parser.add_argument("--output", default="mesh_visualization.png", help="Output image")
    args = parser.parse_args()
    
    case_dir = Path(args.case)
    output_path = Path(args.output)
    
    print(f"Loading mesh from {case_dir}...")
    mesh = load_mesh(case_dir)
    
    print_mesh_stats(mesh)
    
    print(f"\nGenerating visualization...")
    plot_mesh(mesh, output_path, title=f"Mesh: {case_dir.name}")
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
