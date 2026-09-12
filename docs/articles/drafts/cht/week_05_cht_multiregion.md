# Conjugate Heat Transfer with OpenFOAM: From Gmsh to chtMultiRegionFoam in Python

*How to configure a multi-region conjugate heat transfer simulation with foampilot — from geometry creation in Gmsh to post-processing across coupled fluid and solid domains.*

---

## Introduction: The CHT Challenge

Conjugate Heat Transfer (CHT) is one of the most common cases in industrial CFD. A copper heatsink cooled by air. The copper conducts heat, the air convects it. Two fluids, two regions, a coupled interface.

**The problem**: OpenFOAM handles this via `chtMultiRegionFoam`, but the configuration is daunting: regionalized `constant/`, regionalized `0/`, interfaces, thermophysicalProperties…

**The promise**: With foampilot, you define two regions in Python, and everything else is generated.

---

## The Manual Workflow (and Why It's Hard)

1. Create `constant/fluid/`, `constant/solid/`
2. Create `0/fluid/`, `0/solid/`
3. Define interfaces in `constant/regionInterfaces/`
4. Configure `controlDict` with `regionSolvers`
5. Edit thermophysical properties by region
6. Verify field consistency between regions

**Message**: One error in an interface = silent crash of `chtMultiRegionFoam`.

---

## The foampilot Solution: Define, Don't Configure

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

# Define regions
fluid = FluidRegion(
    name="air",
    temperature=300,
    turbulence_model="kOmegaSST"
)
solid = SolidRegion(
    name="copper",
    temperature=350,
    thermal_conductivity=400
)

# Define interface
interface = CoupledInterface(
    name="interface_fluid_solid",
    region_1="air",
    region_2="copper"
)

# Configure solver
solver = ChtSolver(
    case_path="./cht_case",
    regions=[fluid, solid],
    interfaces=[interface]
)

solver.setup_case()
solver.run_simulation(nb_proc=4)
```

**Message**: 15 lines replace hours of manual configuration.

---

## Gmsh Mesh → CHT

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("cht_case")

# Create fluid volume (air)
air = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gid_air = gmsh.model.addPhysicalGroup(3, [air], name="air")
gmsh.model.setPhysicalName(3, gid_air, "air")

# Create solid volume (copper)
copper = gmsh.model.occ.addBox(0.4, 0.4, 0, 0.2, 0.2, 0.1)
gmsh.model.occ.synchronize()
gid_cu = gmsh.model.addPhysicalGroup(3, [copper], name="copper")
gmsh.model.setPhysicalName(3, gid_cu, "copper")

# Mesh and export
gmsh.model.mesh.generate(3)
exporter = DirectOpenFOAMExporter("./cht_case")
exporter.export_multi_region()
gmsh.finalize()
```

---

## Under the Hood

The automatic steps:
1. Creation of `constant/air/`, `constant/copper/`
2. Writing of `thermophysicalProperties` by region
3. Creation of `0/air/T`, `0/copper/T`, etc.
4. Generation of interfaces in `constant/regionInterfaces/`
5. Injection of `regionSolvers` into `controlDict`

---

## Multi-Region Post-Processing

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

# Compare temperatures
for name, mesh in regions.items():
    if "T" in mesh.point_data:
        print(f"{name}: T_max = {mesh.point_data['T'].max():.1f} K")
```

---

## Conclusion: CHT Made Accessible

In the next article, I'll explain how to present such a project to make it adopted by the community.

**Resources:**
- Repository: [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation: [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article under improvement — version 1.0*
