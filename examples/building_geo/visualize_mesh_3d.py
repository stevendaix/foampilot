#!/usr/bin/env python3
"""
Visualize OpenFOAM mesh with matplotlib 3D, coloring boundary patches.

This is more reliable than PyVista off-screen rendering on headless systems.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_points_file(filepath: Path):
    """Read OpenFOAM points file."""
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
    face_pattern = re.compile(r'(\d+)\(([^)]+)\)')
    for match in face_pattern.finditer(faces_str):
        try:
            n_verts = int(match.group(1))
            verts = [int(v) for v in match.group(2).split()]
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
        if (line and not line.startswith(('//', '(', ')', 'FoamFile', '{', '}', '*', 
                                         '/*', '*/', '\\', '=', '|', 'format', 'class',
                                         'object', 'location', '// *')) and len(line) > 2):
            patch_name = line
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


def create_patch_meshes(points, faces, boundary):
    """Create separate face lists for each patch."""
    patch_faces = {}
    
    for patch_name, patch_type, n_faces, start_face in boundary:
        if n_faces == 0:
            continue
        
        faces_list = []
        for i in range(n_faces):
            face_idx = start_face + i
            if face_idx < len(faces):
                n_verts, verts = faces[face_idx]
                faces_list.append(verts)
        
        if faces_list:
            patch_faces[patch_name] = faces_list
    
    return patch_faces


def visualize_mesh(points, patch_faces, output_path: Path, case_name: str):
    """Create matplotlib 3D visualization."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    patch_colors = {
        'inlet': 'blue',
        'outlet': 'red',
        'top': 'green',
        'ground': 'saddlebrown',
        'side_left': 'orange',
        'side_right': 'gold',
        'buildings': 'dimgray',
        'patch0': 'lightgray',
        'unknown': 'white',
    }
    
    # Plot each patch
    for patch_name, faces_list in patch_faces.items():
        color = patch_colors.get(patch_name, 'white')
        
        for verts in faces_list[:100]:  # Limit for performance
            face_coords = points[verts]
            poly = Poly3DCollection([face_coords], alpha=0.8, 
                                   facecolor=color, edgecolor='black', linewidth=0.3)
            ax.add_collection3d(poly)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'OpenFOAM Mesh: {case_name}', fontsize=14)
    
    # Set equal aspect ratio
    if len(points) > 0:
        xmin, ymin, zmin = points.min(axis=0)
        xmax, ymax, zmax = points.max(axis=0)
        
        max_range = max(xmax-xmin, ymax-ymin, zmax-zmin) / 2.0
        mid_x = (xmax + xmin) / 2.0
        mid_y = (ymax + ymin) / 2.0
        mid_z = (zmax + zmin) / 2.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    legend_elements = []
    for patch_name in patch_faces.keys():
        color = patch_colors.get(patch_name, 'white')
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=patch_name))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Mesh visualization saved to {output_path}")
    plt.close()


def print_mesh_stats(points, faces, boundary):
    """Print mesh statistics."""
    print(f"\n{'='*60}")
    print(f"Mesh Statistics")
    print(f"{'='*60}")
    print(f"  Points: {len(points)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Patches: {len(boundary)}")
    
    if len(points) > 0:
        xmin, ymin, zmin = points.min(axis=0)
        xmax, ymax, zmax = points.max(axis=0)
        print(f"  Bounding box: ({xmin:.1f}, {ymin:.1f}, {zmin:.1f}) -> ({xmax:.1f}, {ymax:.1f}, {zmax:.1f})")
        print(f"  Domain size: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f}")
    
    print(f"\n  Patches:")
    for name, ptype, nfaces, start in boundary:
        print(f"    {name:15s}: {nfaces:5d} faces  ({ptype})")


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenFOAM mesh with matplotlib 3D")
    parser.add_argument("--case", required=True, help="OpenFOAM case directory")
    parser.add_argument("--output", default="mesh_visualization.png", help="Output image")
    args = parser.parse_args()
    
    case_dir = Path(args.case)
    output_path = Path(args.output)
    
    polyMesh_dir = case_dir / "constant" / "polyMesh"
    points_file = polyMesh_dir / "points"
    faces_file = polyMesh_dir / "faces"
    boundary_file = polyMesh_dir / "boundary"
    
    if not all(f.exists() for f in [points_file, faces_file, boundary_file]):
        raise FileNotFoundError("Missing polyMesh files")
    
    print(f"Loading mesh from {case_dir}...")
    points = read_points_file(points_file)
    faces = read_faces_file(faces_file)
    boundary = read_boundary_file(boundary_file)
    
    print_mesh_stats(points, faces, boundary)
    
    print(f"\nCreating matplotlib 3D visualization...")
    patch_faces = create_patch_meshes(points, faces, boundary)
    
    print(f"Generating visualization...")
    visualize_mesh(points, patch_faces, output_path, case_dir.name)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
