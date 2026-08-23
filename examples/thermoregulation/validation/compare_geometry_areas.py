from pathlib import Path
import sys
import numpy as np
import trimesh

root = Path('/home/ubuntu/foampilot')
stl = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'examples/thermoregulation/makehuman/openfoam_cube_case/constant/triSurface/human.stl'
mesh = trimesh.load_mesh(stl, process=False)
print(f'STL={stl}')
print(f'vertices={len(mesh.vertices)} faces={len(mesh.faces)}')
print(f'bounds_min={mesh.bounds[0]} bounds_max={mesh.bounds[1]}')
print(f'extent={mesh.extents} height_y={mesh.extents[1]:.8f} m')
print(f'stl_area={mesh.area:.8f} m2')
print(f'watertight={mesh.is_watertight} winding_consistent={mesh.is_winding_consistent}')
print(f'volume={mesh.volume:.8f} m3')
