# Direct OpenFOAM Reader for PyVista: Architecture and Physical Quantity Extraction

*How foampilot reads OpenFOAM results directly into PyVista, without conversion, and extracts essential physical quantities like y+, wall shear stress, and strain rate.*

---

## Introduction: The foamToVTK Bottleneck

You have a running simulation. You want to visualize results. You run `foamToVTK`, wait 10 minutes, open Paraview. But what if you could visualize directly from Python, without conversion, and extract the physical quantities you care about?

**Promise**: With `OpenFOAMDirectReader` and PyVista, it's possible. And you can calculate y+, wall shear stress, strain rate — directly in the script.

---

## Part 1: Classic Workflow (and Its Limits)

1. Run OpenFOAM simulation
2. Run `foamToVTK` to convert to VTK
3. Open Paraview
4. Load VTK file
5. Configure visualization

**Problems**:
- Conversion time (sometimes long for large cases)
- Disk space doubled (OpenFOAM + VTK)
- Non-automatable workflow
- No scriptability for physical quantities

---

## Part 2: The Solution: Direct Reading

```python
import pyvista as pv
from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader

reader = OpenFOAMDirectReader(case_path="./poiseuille")
mesh = reader.read(time_step=100)

# Direct visualization
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="U", cmap="viridis")
plotter.show()
```

**Advantages**:
- No conversion
- Direct field access
- Scriptable
- Integrable in reports

---

## Part 3: Under the Hood — Reader Architecture

### 3.1 Reading OpenFOAM field files

The reader parses `volVectorField` and `volScalarField` files:

```python
# FoamFile header extraction
# boundaryField parsing
# Internal field reconstruction
```

### 3.2 Mesh reconstruction from polyMesh

```python
# Reading points, faces, owner, neighbour
# Building cell connectivity
# Patch extraction
```

### 3.3 Time step management

```python
# Listing time directories
# Selecting specific time step
# Lazy loading for memory efficiency
```

---

## Part 4: Extracting Physical Quantities

### 4.1 y+ (dimensionless wall distance)

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=1.52e-5
)

print(f"y+ min = {mesh_yp.point_data['y_plus'].min():.2f}")
print(f"y+ max = {mesh_yp.point_data['y_plus'].max():.2f}")
```

**Physics**: y+ should be < 1 for boundary layer resolved models (k-omega SST), or between 30 and 300 for wall function models (k-epsilon).

### 4.2 Wall shear stress

```python
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=1.52e-5, 
    wall_normal=[0, 1, 0]
)

tau_w = mesh_wss.cell_data["wall_shear_stress"]
print(f"τ_w max = {tau_w.max():.4f} Pa")

# Analytical validation for plane Poiseuille
tau_w_analytical = 0.5 * rho * G * H
print(f"Analytical τ_w = {tau_w_analytical:.4f} Pa")
```

### 4.3 Strain rate

```python
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
print(f"Max strain rate = {mesh_sr.point_data['strain_rate'].max():.4f} 1/s")
```

---

## Part 5: CHT Multi-Region

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

for name, region_mesh in regions.items():
    if "T" in region_mesh.point_data:
        print(f"{name}: T_max = {region_mesh.point_data['T'].max():.1f} K")
```

---

## Part 6: Performance

- Read time for 1M cells: ~1.5s
- Memory: ~500 MB for 1M cell case
- Comparison with foamToVTK + VTK reader
- Lazy loading strategy

---

## Conclusion: From Mesh to Results in One Pipeline

Visualization and post-processing are just steps. In the next article, we tackle a complete multi-region CHT case where all these techniques combine.

**Resources:**
- Repository: [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation: [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article under improvement — version 1.0*
