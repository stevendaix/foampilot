#!/usr/bin/env python3
"""
Variant CHT Électronique — Peu d'ailettes (3 fins)

Compare avec le cas baseline (5 ailettes) pour étudier
l'impact du nombre d'ailettes sur la résistance thermique
et la température de la puce.

Usage::

    cd examples/cht/electronique_few_fins
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "foampilot" / "src"))

import gmsh
from foampilot.mesh.gmsh_mesher import GmshMesher
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter
from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
)

chip_lx = 10e-3
chip_ly = 10e-3
chip_lz = 1e-3

hs_base_lx = 20e-3
hs_base_ly = 20e-3
hs_base_lz = 2e-3
fin_height = 15e-3
fin_thickness = 1e-3
fin_spacing = 3e-3
n_fins = 3

domain_lx = 50e-3
domain_ly = 30e-3
domain_lz = 30e-3

inlet_velocity = 1.0
inlet_temperature = 293.15
chip_heat_flux = 5000.0

chip_kappa = 150.0
chip_rho = 2330.0
chip_cp = 700.0

hs_kappa = 205.0
hs_rho = 2700.0
hs_cp = 900.0

lc_min = 1e-3
lc_max = 5e-3
lc_fin = 5e-4
lc_chip = 5e-4

case_path = Path(__file__).resolve().parent
gmsh_model_name = "electronique_few_fins"


class SimpleParent:
    def __init__(self, case_path):
        self.case_path = case_path


gmsh.initialize()
gmsh.model.add(gmsh_model_name)

parent = SimpleParent(case_path)
mesh = GmshMesher(parent, model_name=gmsh_model_name)

air_tag = mesh.add_air_domain(0, 0, 0, domain_lx, domain_ly, domain_lz)

hs_tag = mesh.add_heatsink_fins(
    x0=(domain_lx - hs_base_lx) / 2,
    y0=(domain_ly - hs_base_ly) / 2,
    z0=0,
    base_lx=hs_base_lx,
    base_ly=hs_base_ly,
    base_lz=hs_base_lz,
    fin_height=fin_height,
    fin_thickness=fin_thickness,
    fin_spacing=fin_spacing,
    n_fins=n_fins,
    name="heatsink",
)

chip_x = (domain_lx - chip_lx) / 2
chip_y = (domain_ly - chip_ly) / 2
chip_z = hs_base_lz + fin_height
chip_tag = mesh.add_chip(chip_x, chip_y, chip_z, chip_lx, chip_ly, chip_lz)

fluid_tags = mesh.subtract_solids_from_fluid(
    fluid_tag=air_tag,
    solid_tags=[hs_tag, chip_tag],
    fluid_name="air",
)

mesh.assign_electronique_patches(domain_lx, domain_ly, domain_lz)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.option.setNumber("Mesh.MshFileVersion", 2)

refine_regions = [
    ((chip_x + chip_lx / 2, chip_y + chip_ly / 2, chip_z + chip_lz / 2),
     max(chip_lx, chip_ly, chip_lz) * 0.5, lc_chip),
    ((chip_x + chip_lx / 2, domain_ly / 2, hs_base_lz + fin_height / 2),
     fin_height * 0.5, lc_fin),
]

for center, radius, lc in refine_regions:
    entities = gmsh.model.getEntitiesInBoundingBox(
        center[0] - radius, center[1] - radius, center[2] - radius,
        center[0] + radius, center[1] + radius, center[2] + radius,
    )
    if entities:
        gmsh.model.mesh.setSize(entities, lc)

gmsh.model.mesh.generate(3)

region_map = {
    "air": "fluid",
    "heatsink": "heatsink",
    "chip": "chip",
}

exporter = DirectOpenFOAMExporter(case_path)
written_dirs = exporter.export_multi_region(region_map)

print(f"Direct export completed — {len(written_dirs)} region(s) written:")
for d in written_dirs:
    print(f"  -> {d}")

fluid_region = FluidRegion(
    name="fluid",
    temperature=inlet_temperature,
    velocity=(inlet_velocity, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

chip_region = SolidRegion(
    name="chip",
    temperature=350.0,
    thermal_conductivity=chip_kappa,
    density=chip_rho,
    specific_heat=chip_cp,
)

heatsink_region = SolidRegion(
    name="heatsink",
    temperature=300.0,
    thermal_conductivity=hs_kappa,
    density=hs_rho,
    specific_heat=hs_cp,
)

regions = [fluid_region, chip_region, heatsink_region]

interfaces = [
    CoupledInterface(
        name="air_chip",
        fluid_region="fluid",
        solid_region="chip",
        heat_transfer_coefficient=50.0,
    ),
    CoupledInterface(
        name="air_heatsink",
        fluid_region="fluid",
        solid_region="heatsink",
        heat_transfer_coefficient=50.0,
    ),
]

solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=regions,
    interfaces=interfaces,
    region_solvers={
        "fluid": "fluid",
        "chip": "solid",
        "heatsink": "solid",
    },
    turbulence_model="kOmegaSST",
)

solver.setup_case()
solver.write_case()

print("\n" + "=" * 60)
print("Cas CHT Électronique (3 ailettes) créé avec succès")
print("=" * 60)
print(f"  Dossier du cas : {case_path}")
print(f"  Régions         : {[r.name for r in regions]}")
print(f"  Nombre d'ailettes : {n_fins}")
print(f"  Vitesse entrée  : {inlet_velocity} m/s")
print(f"  Flux puce       : {chip_heat_flux} W/m²")
print("=" * 60)

gmsh.finalize()