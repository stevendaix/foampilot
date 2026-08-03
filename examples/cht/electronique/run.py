#!/usr/bin/env python3
"""
Refroidissement d'Électronique CHT — Exemple foampilot + Gmsh

Ce script démontre comment utiliser foampilot pour simuler le refroidissement
d'un composant électronique (puce sur dissipateur à ailettes) avec transfert
thermique conjugué (CHT) multi-régions.

Le maillage est construit via les méthodes partagées de GmshMesher
(dans foampilot/mesh/gmsh_mesher.py) pour garantir la réutilisabilité.

Workflow:
  1. Définition paramétrique des dimensions (puce, dissipateur, domaine fluide)
  2. Construction géométrique via GmshMesher (add_air_domain, add_heatsink_fins,
     add_chip, subtract_solids_from_fluid)
  3. Attribution des groupes physiques (volumes + surfaces) via
     assign_electronique_patches
  4. Génération du maillage tétraédrique
  5. Export direct multi-régions vers OpenFOAM (polyMesh)
  6. Configuration CHT avec chtMultiRegionFoam
  7. Écriture des fichiers de cas

Usage::

    cd examples/cht/electronique
    python run.py

Le cas produit utilise les régions :
  - air      (fluide, convection forcée)
  - chip     (solide, silicium, source de chaleur)
  - heatsink (solide, aluminium, dissipation par ailettes)

Références comparatives :
  - ChipHX tutorial (KIT, BwUniCluster) — plaque + cylindres (pins)
  - circuitBoardCooling (OpenFOAM tutorials) — CHT avec baffles 3D
  - SimScale Rectangular Fins — validation CHT dissipateur à ailettes
  - SimScale LED COP — gestion thermique puce sur plaque
  - Gin Tonic (Holzmann CFD) — cas d'entraînement CHT OpenFOAM
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
from foampilot.utilities.manageunits import ValueWithUnit

# ============================================================================
# 1. Paramètres géométriques et physiques
# ============================================================================

# — Dimensions de la puce (silicium) —
chip_lx = 10e-3
chip_ly = 10e-3
chip_lz = 1e-3

# — Dimensions du dissipateur —
hs_base_lx = 20e-3
hs_base_ly = 20e-3
hs_base_lz = 2e-3
fin_height = 15e-3
fin_thickness = 1e-3
fin_spacing = 3e-3
n_fins = 5

# — Domaine fluide (air) —
domain_lx = 50e-3
domain_ly = 30e-3
domain_lz = 30e-3

# — Conditions d'écoulement —
inlet_velocity = 1.0
inlet_temperature = 293.15
chip_heat_flux = 5000.0

# — Propriétés matériaux —
chip_kappa = 150.0
chip_rho = 2330.0
chip_cp = 700.0

hs_kappa = 205.0
hs_rho = 2700.0
hs_cp = 900.0

# — Paramètres de maillage —
lc_min = 1e-3
lc_max = 5e-3
lc_fin = 5e-4
lc_chip = 5e-4

# — Chemins —
case_path = Path(__file__).resolve().parent
gmsh_model_name = "electronique_cht"


# ============================================================================
# 2. Initialisation Gmsh et construction géométrique via GmshMesher
# ============================================================================

class SimpleParent:
    """Parent minimal pour GmshMesher (fournit case_path)."""
    def __init__(self, case_path):
        self.case_path = case_path


gmsh.initialize()
gmsh.model.add(gmsh_model_name)

parent = SimpleParent(case_path)
mesh = GmshMesher(parent, model_name=gmsh_model_name)

# — Domaine fluide (boîte englobante) —
air_tag = mesh.add_air_domain(0, 0, 0, domain_lx, domain_ly, domain_lz)

# — Dissipateur avec ailettes —
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

# — Puce (silicium) posée sur le dissipateur —
chip_x = (domain_lx - chip_lx) / 2
chip_y = (domain_ly - chip_ly) / 2
chip_z = hs_base_lz + fin_height
chip_tag = mesh.add_chip(chip_x, chip_y, chip_z, chip_lx, chip_ly, chip_lz)

# — Soustraction des solides du domaine fluide —
fluid_tags = mesh.subtract_solids_from_fluid(
    fluid_tag=air_tag,
    solid_tags=[hs_tag, chip_tag],
    fluid_name="air",
)

# ============================================================================
# 3. Groupes physiques — surfaces (patches frontières)
# ============================================================================

mesh.assign_electronique_patches(domain_lx, domain_ly, domain_lz)

# ============================================================================
# 4. Maillage tétraédrique
# ============================================================================

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.option.setNumber("Mesh.MshFileVersion", 2)

# Raffinement local près de la puce et des ailettes
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

# ============================================================================
# 5. Export direct multi-régions vers OpenFOAM polyMesh
# ============================================================================

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

# ============================================================================
# 6. Configuration CHT avec foampilot
# ============================================================================

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

# ============================================================================
# 7. Résumé
# ============================================================================

print("\n" + "=" * 60)
print("Cas CHT Électronique créé avec succès")
print("=" * 60)
print(f"  Dossier du cas : {case_path}")
print(f"  Régions         : {[r.name for r in regions]}")
print(f"  Solveur         : chtMultiRegionFoam")
print(f"  Interfaces CHT  : {[i.name for i in interfaces]}")
print(f"  Maillage        : {len(written_dirs)} région(s) exportée(s)")
print(f"  Vitesse entrée  : {inlet_velocity} m/s")
print(f"  Flux puce       : {chip_heat_flux} W/m²")
print(f"  Nombre d'ailettes : {n_fins}")
print(f"\n  Pour lancer la simulation :")
print(f"    cd {case_path}")
print(f"    chtMultiRegionFoam")
print("=" * 60)

gmsh.finalize()