#!/usr/bin/env python3
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
background = ROOT / "background"
hull = ROOT / "hull"
region = background / "0" / "hull"
region.mkdir(parents=True, exist_ok=True)
mesh_link = region / "polyMesh"
if mesh_link.exists() or mesh_link.is_symlink():
    if mesh_link.is_symlink() or mesh_link.is_file():
        mesh_link.unlink()
    else:
        shutil.rmtree(mesh_link)
mesh_link.symlink_to(hull / "constant" / "polyMesh", target_is_directory=True)

patches = ("oversetHull", "hullMidPlane", "hull")

def header(src: str, object_name: str, location: str) -> str:
    src = re.sub(r'location\s+"[^"]*"\s*;', f'location    "{location}";', src, count=1)
    src = re.sub(r'object\s+\w+\s*;', f'object      {object_name};', src, count=1)
    return src.split("dimensions", 1)[0]

def make_field(name: str, dimensions: str, internal: str, patch_value: str, kind: str, location: str) -> str:
    template = (background / "0" / name).read_text()
    text = header(template, name, location)
    if kind == "U":
        wall = "        type fixedValue;\n        value uniform (0 0 0);"
        symmetry = "        type symmetry;"
    else:
        wall = f"        type fixedValue;\n        value uniform {patch_value};"
        symmetry = "        type symmetry;"
    boundary = (
        "    oversetHull\n    {\n        type zeroGradient;\n    }\n"
        "    hullMidPlane\n    {\n" + symmetry + "\n    }\n"
        "    hull\n    {\n" + wall + "\n    }\n"
    )
    return (text + f"dimensions      {dimensions};\n\n"
            f"internalField   uniform {internal};\n\n"
            "boundaryField\n{\n" + boundary + "}\n")

fields = {
    "U": ("[0 1 -1 0 0 0 0]", "(0 0 0)", "(0 0 0)"),
    "p_rgh": ("[1 -1 -2 0 0 0 0]", "0", "0"),
    "alpha.water": ("[0 0 0 0 0 0 0]", "0", "0"),
    "k": ("[0 2 -2 0 0 0 0]", "1e-06", "1e-06"),
    "omega": ("[0 0 -1 0 0 0 0]", "1", "1"),
    "epsilon": ("[0 2 -3 0 0 0 0]", "1e-06", "1e-06"),
    "nut": ("[0 2 -1 0 0 0 0]", "0", "0"),
    "zoneID": ("[0 0 0 0 0 0 0]", "0", "0"),
}
solver_fields = hull / "0"
solver_fields.mkdir(parents=True, exist_ok=True)
for name, (dims, internal, patch_value) in fields.items():
    (region / name).write_text(make_field(name, dims, internal, patch_value, name, "0/hull"))
    (solver_fields / name).write_text(make_field(name, dims, internal, patch_value, name, "0"))
print(f"Prepared donor region hull at {region} and solver fields at {solver_fields} with {len(patches)} patches")
