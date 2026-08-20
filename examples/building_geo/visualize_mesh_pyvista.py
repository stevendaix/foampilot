#!/usr/bin/env python3
"""
Visualize OpenFOAM mesh with PyVista, coloring boundary patches.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import re
import numpy as np
import pyvista as pv


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


def create_patch_mesh(points, faces, boundary):
    """Create a single PyVista mesh with patches as cell data."""
    # Map face index -> patch name
    face_to_patch = {}
    for patch_name, patch_type, n_faces, start_face in boundary:
        for i in range(n_faces):
            face_idx = start_face + i
            face_to_patch[face_idx] = patch_name
    
    # Build unified face array
    pv_faces = []
    patch_ids = []
    patch_name_list = []
    
    patch_to_id = {name: idx for idx, (name, _, _, _) in enumerate(boundary) if _}
    
    for face_idx, (n_verts, verts) in enumerate(faces):
        pv_faces.extend([n_verts] + verts)
        patch_name = face_to_patch.get(face_idx, 'unknown')
        patch_ids.append(patch_to_id.get(patch_name, -1))
        patch_name_list.append(patch_name)
    
    mesh = pv.PolyData(points, np.array(pv_faces))
    mesh.cell_data['patch_id'] = np.array(patch_ids)
    mesh.cell_data['patch_name'] = patch_name_list
    
    return mesh, boundary


def visualize_mesh(mesh, boundary, output_path: Path, case_name: str):
    """Create PyVista visualization."""
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    plotter.set_background('white')
    
    patch_colors = {
        'inlet': 'blue',
        'outlet': 'red',
        'top': 'green',
        'ground': 'brown',
        'side_left': 'orange',
        'side_right': 'yellow',
        'buildings': 'black',
        'patch0': 'lightgray',
        'unknown': 'white',
    }
    
    # Create color array
    patch_names = [p[0] for p in boundary]
    colors = [patch_colors.get(name, 'white') for name in patch_names]
    
    # Add mesh colored by patch
    plotter.add_mesh(
        mesh,
        scalars='patch_id',
        cmap='Set1',
        show_edges=True,
        edge_color='black',
        line_width=0.5,
        opacity=1.0,
    )
    
    # Add legend manually
    for idx, (name, ptype, nfaces, start) in enumerate(boundary):
        if nfaces > 0:
            color = patch_colors.get(name, 'white')
            plotter.add_text(f"{name}: {nfaces} faces", 
                            position=(10, 10 + idx * 25),
                            font_size=10,
                            color=color)
    
    # Set camera
    if mesh.n_points > 0:
        center = mesh.center
        plotter.camera_position = [
            (center[0] + 100, center[1] + 100, center[2] + 60),
            center,
            (0, 0, 1),
        ]
    
    plotter.add_axes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.render()
    plotter.screenshot(str(output_path))
    print(f"PyVista visualization saved to {output_path}")
    plotter.close()


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
    parser = argparse.ArgumentParser(description="Visualize OpenFOAM mesh with PyVista")
    parser.add_argument("--case", required=True, help="OpenFOAM case directory")
    parser.add_argument("--output", default="mesh_pyvista.png", help="Output image")
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
    
    print(f"\nCreating PyVista mesh...")
    mesh, boundary = create_patch_mesh(points, faces, boundary)
    
    print(f"Generating visualization...")
    visualize_mesh(mesh, boundary, output_path, case_dir.name)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
